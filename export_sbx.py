import rl_zoo3.train
import importlib
import os
import gymnasium
import numpy as np
import jax
import torch
from rl_zoo3.export import make_dummy_obs
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, DroQ, CrossQ
from rl_zoo3.utils import ALGOS, get_latest_run_id
import gymnasium as gym
import argparse
import openvino as ov
import gym_footsteps_planning

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
parser.add_argument(
    "--gym-packages",
    type=str,
    nargs="+",
    default=[],
    help="Additional external Gym environment package modules to import",
)
parser.add_argument("--export-critic", help="Export the critic network as well", action="store_true")
args = parser.parse_args()

for env_module in args.gym_packages:
    print(f"Importing gym module {env_module}")
    m = importlib.import_module(env_module)
    print(m)
    print(m.__file__)

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


def load_batch_norm_vectorized(params: dict, batch_stats: dict, critic_idx: int = 0) -> torch.nn.BatchNorm1d:
    """
    Translate a JAX Batch norm to Torch for vectorized critics
    """    
    # Extract parameters for specific critic index
    features = params["bias"].shape[1]  # Skip the first dimension (critic index)

    bn = torch.nn.BatchNorm1d(features, momentum=0.99, eps=0.001)
    values = bn.state_dict()
    values["weight"] = jax_to_torch(params["scale"][critic_idx])
    values["bias"] = jax_to_torch(params["bias"][critic_idx])
    values["running_mean"] = jax_to_torch(batch_stats["mean"][critic_idx])
    values["running_var"] = jax_to_torch(batch_stats["var"][critic_idx])

    bn.load_state_dict(values)

    return bn


def load_dense_vectorized(params: dict, critic_idx: int = 0) -> torch.nn.Linear:
    """
    Translates a JAX Dense layer to Torch for vectorized critics
    """    
    # Extract parameters for specific critic index
    _, in_features, out_features = params["kernel"].shape  # Skip the first dimension (critic index)

    dense = torch.nn.Linear(in_features, out_features)
    values = dense.state_dict()
    values["weight"] = jax_to_torch(params["kernel"][critic_idx].T)
    values["bias"] = jax_to_torch(params["bias"][critic_idx])

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
                torch.nn.ReLU(),
            ]

        k = len(policy.actor.net_arch)
        layers += [
            load_batch_norm(jax_params[f"BatchRenorm_{k}"], batch_stats[f"BatchRenorm_{k}"]),
            load_dense(jax_params[f"Dense_{k}"]),
            torch.nn.Tanh(),
        ]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        if not self.squash:
            output = self.action_low + (output + 1) * 0.5 * (self.action_high - self.action_low)

        return output


class TorchCritic(torch.nn.Module):
    """
    Jax Critic translated to PyTorch
    This is based on CrossQ critic model, the architecture is as following:

    - Concatenate observation and action
    - BatchRenorm_0 -> Dense_0 -> ReLU
    - BatchRenorm_1 -> Dense_1 -> ReLU  
    - BatchRenorm_2 -> Dense_2 (output layer with linear activation)

    The activation function is ReLU for hidden layers, linear for output
    """    

    def __init__(self, policy, critic_idx: int = 0):
        super().__init__()
        # CrossQ uses a vectorized critic, we access VmapCritic_0
        jax_params = policy.qf_state.params["VmapCritic_0"]
        batch_stats = policy.qf_state.batch_stats["VmapCritic_0"]

        layers = []
        
        # Layer 1: BatchRenorm_0 -> Dense_0 -> ReLU
        layers += [
            load_batch_norm_vectorized(jax_params["BatchRenorm_0"], batch_stats["BatchRenorm_0"], critic_idx),
            load_dense_vectorized(jax_params["Dense_0"], critic_idx),
            torch.nn.ReLU()
        ]
        
        # Hidden layers: BatchRenorm_k -> Dense_k -> ReLU
        for k in range(1, len(policy.qf.net_arch)):
            layers += [
                load_batch_norm_vectorized(jax_params[f"BatchRenorm_{k}"], batch_stats[f"BatchRenorm_{k}"], critic_idx),
                load_dense_vectorized(jax_params[f"Dense_{k}"], critic_idx),
                torch.nn.ReLU()
            ]
        
        # Output layer: BatchRenorm_final -> Dense_final (no activation)
        final_k = len(policy.qf.net_arch)
        layers += [
            load_batch_norm_vectorized(jax_params[f"BatchRenorm_{final_k}"], batch_stats[f"BatchRenorm_{final_k}"], critic_idx),
            load_dense_vectorized(jax_params[f"Dense_{final_k}"], critic_idx)
        ]
        
        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs, action):
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=1)
        return self.net(x)


mlp = TorchActor(model.policy, args.squash)
mlp.eval()

# Initialize critic if needed
if args.export_critic:
    print("Preparing critic export...")
    # CrossQ uses VmapCritic structure with n_critics
    num_critics = model.policy.n_critics
    print(f"Found {num_critics} critics to export")

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
    ov.save_model(ov_model_actor, f"{directory}/{args.env}_actor.xml")

    if args.export_critic:
        # Export each critic (CrossQ uses VmapCritic structure with n_critics)
        num_critics = model.policy.n_critics
        
        for critic_idx in range(num_critics):
            mlpc = TorchCritic(model.policy, critic_idx)
            mlpc.eval()

            critic_fname = f"{directory}/{args.env}_critic_{critic_idx}.onnx"
            print(f"Exporting critic model {critic_idx} to {critic_fname}")
            
            # Create dummy inputs for the critic (observation and action)
            dummy_action = torch.randn(1, env.action_space.shape[0])
            torch.onnx.export(mlpc, (obs, dummy_action), critic_fname, opset_version=11)

            # OpenVINO export for critic
            input_shape = [(obs.shape, ov.Type.f32), (dummy_action.shape, ov.Type.f32)]
            ov_model_critic = ov.convert_model(input_model=critic_fname, input=input_shape)
            ov.save_model(ov_model_critic, f"{directory}/{args.env}_critic_{critic_idx}.xml")
