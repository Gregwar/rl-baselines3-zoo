import gymnasium as gym
from torchinfo import summary
import torch as th
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy

device = th.device("cpu")


def unsquash_action(action, env):
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("Policy is squashing but the action space is not continuous")
    low, high = th.tensor(env.action_space.low, device=device), th.tensor(env.action_space.high, device=device)
    return low + (0.5 * (action + 1.0) * (high - low))


class TD3Actor(th.nn.Module):
    def __init__(self, env: gym.Env, policy: TD3Policy, squash_output: bool = False):
        super(TD3Actor, self).__init__()

        self.env = env
        self.features_extractor = policy.actor.features_extractor
        self.mu = policy.actor.mu
        self.policy = policy
        self.squash_output = squash_output

    def forward(self, obs):
        features = self.features_extractor(obs)
        action = self.mu(features)

        if self.policy.squash_output and not self.squash_output:
            action = unsquash_action(action, self.env)

        return action


class TD3PolicyValue(th.nn.Module):
    def __init__(self, policy: TD3Policy, actor_model: th.nn.Module):
        super(TD3PolicyValue, self).__init__()

        self.actor = actor_model
        self.critic = policy.critic

    def forward(self, obs):
        action = self.actor(obs)
        critic_features = self.critic.features_extractor(obs)
        return self.critic.q_networks[0](th.cat([critic_features, action], dim=1))


class SACActor(th.nn.Module):
    def __init__(self, env: gym.Env, policy: SACPolicy, squash_output: bool = False):
        super(SACActor, self).__init__()

        self.env = env
        self.features_extractor = policy.features_extractor
        self.policy = policy
        self.squash_output = squash_output
        self.actor_model = th.nn.Sequential(
            policy.actor.features_extractor, policy.actor.latent_pi, policy.actor.mu, th.nn.Tanh()
        )

    def forward(self, obs):
        action = self.actor_model(obs)

        if self.policy.squash_output and not self.squash_output:
            action = unsquash_action(action, self.env)

        return action


class SACPolicyValue(th.nn.Module):
    def __init__(self, policy: SACPolicy, actor_model: th.nn.Module):
        super(SACPolicyValue, self).__init__()

        self.actor_model = actor_model
        self.critic = policy.critic

    def forward(self, obs):
        action = self.actor_model(obs)
        critic_features = self.critic.features_extractor(obs)

        return self.critic.q_networks[0](th.cat([critic_features, action], dim=1))


def make_dummy_obs(env) -> th.Tensor:
    obs, _ = env.reset()
    obs = th.Tensor(obs, device=device)
    obs = preprocess_obs(obs, env.observation_space).unsqueeze(0)

    return obs


def onnx_export(env: gym.Env, model, actor_filename: str, value_filename=None, squash_output: bool = False):
    obs = make_dummy_obs(env)

    if isinstance(model, TD3):
        actor_model = TD3Actor(env, model.policy, squash_output)
        if value_filename is not None:
            value_model = TD3PolicyValue(model.policy, actor_model)
    elif isinstance(model, SAC) or isinstance(model, TQC):
        if model.use_sde:
            raise NotImplementedError("SDE not supported")

        actor_model = SACActor(env, model.policy, squash_output)
        if value_filename is not None:
            value_model = SACPolicyValue(model.policy, actor_model)
    else:
        raise ValueError(f"Unsupported model {type(model)}")

    print("Actor model")
    summary(actor_model)
    th.onnx.export(actor_model, obs, actor_filename, opset_version=11)

    if value_filename is not None:
        print("Value model")
        summary(value_model)
        th.onnx.export(value_model, obs, value_filename, opset_version=11)
