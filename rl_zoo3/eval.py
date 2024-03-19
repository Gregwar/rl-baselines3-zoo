import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
import gymnasium as gym
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from stable_baselines3.common.env_util import make_vec_env
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize


def evaluate() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("-n", "--episodes", help="number of episodes", default=128, type=int)
    args = parser.parse_args()

    algo = args.algo
    folder = args.folder

    _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            args.env,
            args.load_best,
            False,
            False,
        )
    
    env = gym.make(args.env)
    model = ALGOS[args.algo].load(model_path, env)
    
    print(f"Evaluating the model for {args.episodes} episodes...")
    rewards = []
    gamma = 0.998
    for k in tqdm(range(args.episodes)):
        obs, _ = env.reset()
        end = False
        result = 0
        step = 0

        while not end:
            action, _ = model.predict(obs)
            obs, reward, done, trucated, infos = env.step(action)
            result += reward * gamma ** step

            end = done or trucated
            step += 1

        rewards.append(result)
        print(f"Reward (mean): {np.mean(rewards)}")

if __name__ == "__main__":
    evaluate()