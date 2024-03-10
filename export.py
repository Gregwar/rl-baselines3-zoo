import os
import gymnasium as gym
import rl_zoo3.import_envs
import importlib
import torch as th
import argparse
from rl_zoo3.export import onnx_export, make_dummy_obs
from rl_zoo3.utils import ALGOS, get_latest_run_id


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Env", type=str, required=True)
parser.add_argument("--algo", help="RL Algorithm", default="td3", type=str, required=False)
parser.add_argument("--exp_id", help="Experiment ID", type=int, required=False)
parser.add_argument("--output", help="Target directory", type=str, required=True)
parser.add_argument("--squash", help="Squash output between -1 and 1 (default False)", action="store_true")
parser.add_argument(
    "--gym-packages",
    type=str,
    nargs="+",
    default=[],
    help="Additional external Gym environment package modules to import",
)
args = parser.parse_args()
device = th.device("cpu")

for env_module in args.gym_packages:
    importlib.import_module(env_module)

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

print(f"Loading env {args.env}")
env = gym.make(args.env)

latest_exp_id = get_latest_run_id("logs/td3/", args.env)
exp_id = args.exp_id if args.exp_id is not None else latest_exp_id
print(f"Loading model {args.algo}, env {args.env}, exp_id {exp_id}")
model_fname = f"logs/{args.algo}/{args.env}_{exp_id}/best_model.zip"

print(f"Loading model {model_fname}")
model = ALGOS[args.algo].load(model_fname, env=env, custom_objects=custom_objects, device=device)

actor_fname = f"{args.output}/{args.env}_actor.onnx"
value_fname = f"{args.output}/{args.env}_value.onnx"
print(f"Exporting actor model to {actor_fname}")
onnx_export(env, model, actor_fname, value_fname, args.squash)

print("Exporting models for OpenVino...")
obs = make_dummy_obs(env)
input_shape = ",".join(map(str, obs.shape))
os.system(f"mo --input_model {actor_fname} --input_shape [{input_shape}] --compress_to_fp16=False --output_dir {args.output}")
os.system(f"mo --input_model {value_fname} --input_shape [{input_shape}] --compress_to_fp16=False --output_dir {args.output}")
