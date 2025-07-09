import rl_zoo3.train
import os
import gymnasium
import numpy as np
import jax
import torch
from rl_zoo3.export import make_dummy_obs
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, DroQ, CrossQ
from rl_zoo3.utils import ALGOS, get_latest_run_id
import argparse
import openvino as ov

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Env", type=str, required=True)
parser.add_argument("--algo", help="RL Algorithm", default="td3", type=str, required=False)
parser.add_argument("--exp_id", help="Experiment ID", type=int, required=False)
parser.add_argument("--output", help="Target directory", type=str, required=True)
parser.add_argument("--squash", help="Squash output between -1 and 1 (default False)", action="store_true")
parser.add_argument("--enjoy", help="Enjoy the torch module", action="store_true")
args = parser.parse_args()

print(f"Loading env {args.env}")
env = gymnasium.make(args.env)

latest_exp_id = get_latest_run_id(f"logs/{args.algo}/", args.env)
exp_id = args.exp_id if args.exp_id is not None else latest_exp_id
print(f"Loading model {args.algo}, env {args.env}, exp_id {exp_id}")
model_fname = f"logs/{args.algo}/{args.env}_{exp_id}/best_model.zip"

print(f"Loading model {model_fname}")
if args.algo != "crossq":
    raise NotImplementedError("Only CrossQ is currently supported")
model = ALGOS[args.algo].load(model_fname, env=env)


def jax_to_torch(tensor: jax.Array) -> torch.Tensor:
    """
    Converts a jax tensor (Array) to a torch tensor
    """    
    return torch.tensor(np.array(tensor))


def load_batch_norm(params: dict, batch_stats: dict) -> torch.nn.BatchNorm1d:
    """
    Translate a JAX Batch norm to Torch
    """    
    features = params["bias"].shape[0]

    bn = torch.nn.BatchNorm1d(features, momentum=0.99, eps=0.001)
    values = bn.state_dict()
    values["weight"] = jax_to_torch(params["scale"])
    values["bias"] = jax_to_torch(params["bias"])
    values["running_mean"] = jax_to_torch(batch_stats["mean"])
    values["running_var"] = jax_to_torch(batch_stats["var"])

    bn.load_state_dict(values)

    return bn


def load_dense(params: dict) -> torch.nn.Linear:
    """
    Translates a JAX Dense layer to Torch
    """    
    in_features, out_features = params["kernel"].shape

    dense = torch.nn.Linear(in_features, out_features)
    values = dense.state_dict()
    values["weight"] = jax_to_torch(params["kernel"].T)
    values["bias"] = jax_to_torch(params["bias"])

    dense.load_state_dict(values)

    return dense


class TorchActor(torch.nn.Module):
    """
    Jax Actor translated to PyTorch
    This is based on CrossQ actor model, the architecture is as following:

    - BatchRenorm
    - Dense
    - ReLu
    - BatchRenor
    - Dense
    - ReLu
    [...] repeated for hidden layers
    - BatchRenorm
    - Dense
    - Tanh (output layer)
    - If not squashed, the output is rescaled to the action space

    The activation function has to be ReLu
    """    

    def __init__(self, policy, squash: bool = False):
        super().__init__()
        jax_params = policy.actor_state.params
        batch_stats = policy.actor_state.batch_stats
        self.action_space = policy.action_space
        self.squash = squash

        self.action_low = torch.tensor(self.action_space.low)
        self.action_high = torch.tensor(self.action_space.high)

        layers = []
        for k in range(len(policy.actor.net_arch)):
            layers += [
                load_batch_norm(jax_params[f"BatchRenorm_{k}"], batch_stats[f"BatchRenorm_{k}"]),
                load_dense(jax_params[f"Dense_{k}"]),
                torch.nn.ReLU()
            ]

        k = len(policy.actor.net_arch)
        layers += [
            load_batch_norm(jax_params[f"BatchRenorm_{k}"], batch_stats[f"BatchRenorm_{k}"]),
            load_dense(jax_params[f"Dense_{k}"]),
            torch.nn.Tanh()
        ]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        if not self.squash:
            output = self.action_low + (output + 1) * 0.5 * (self.action_high - self.action_low)

        return output


mlp = TorchActor(model.policy, args.squash)
mlp.eval()

if args.enjoy:
    env = gymnasium.make(args.env, render_mode="human")
    with torch.no_grad():
        # Enjoy the torch module
        for episode in range(100_000):
            obs, infos = env.reset()
            finished, truncated = False, False
            returns = 0

            while not finished and not truncated:
                obs_torch = torch.tensor(obs).unsqueeze(0)
                action_torch = mlp(torch.tensor(obs_torch))
                action = action_torch.detach().numpy().squeeze()

                # That would be the code to use with the native JAX module (for debugging purpose)
                # action, _ = model.predict(obs, deterministic=True)
                # print(action)

                obs, rewards, finished, truncated, infos = env.step(action)
                returns += rewards

            print(f"Episode {episode} returns {returns}")
else:
    obs = make_dummy_obs(env)
    directory = f"{args.output}/{args.algo}/{args.env}_{exp_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    actor_fname = f"{directory}/{args.env}_actor.onnx"
    print(f"Exporting actor model to {actor_fname}")
    torch.onnx.export(mlp, obs, actor_fname, opset_version=11)

    print("Exporting models for OpenVino...")
    
    #### Old way to export model to OpenVino IR using Model Optimizer (mo) ####
    # input_shape = ",".join(map(str, obs.shape))
    # os.system(f"mo --input_model {actor_fname} --input_shape [{input_shape}] --compress_to_fp16=False --output_dir {args.output}")

    input_shape = (obs.shape, ov.Type.f32)

    ov_model_actor = ov.convert_model(input_model=actor_fname, input=input_shape)
    ov.save_model(ov_model_actor, f"{directory}/{args.env}_actor.xml", compress_to_fp16=False)
