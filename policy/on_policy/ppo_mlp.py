
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Categorical
from gym.spaces import Box, Discrete

from policy.util import init_


class GaussianActor(nn.Module):

    def __init__(self, obs_space, act_space, hidden_sizes=(64, 64)):
        super(GaussianActor, self).__init__()

        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        self.high = act_space.high[0]
        self.low = act_space.low[0]

        self.mu = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[1], act_dim)),
            nn.Tanh()
        )
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, obs, act=None):
        dist = self._distribution(obs)
        logp = None
        if act is not None:
            logp = self._log_prob_from_distribution(dist, act)
        return dist, logp

    def _distribution(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        mu = self.high * self.mu(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    @staticmethod
    def _log_prob_from_distribution(dist, act):
        return dist.log_prob(act).sum(dim=-1)


class CategoricalActor(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(64, 64)):
        super(CategoricalActor, self).__init__()

        obs_dim = obs_space.shape[0]
        act_dim = act_space.n

        self.pi = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[1], act_dim)),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, act=None):
        dist = self._distribution(obs)
        logp = None
        if act is not None:
            logp = self._log_prob_from_distribution(dist, act)
        return dist, logp

    def _distribution(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        act_prob = self.pi(obs)
        return Categorical(act_prob)

    @staticmethod
    def _log_prob_from_distribution(dist, act):
        return dist.log_prob(act)


class Critic(nn.Module):
    def __init__(self, obs_space, hidden_sizes=(64, 64)):
        super(Critic, self).__init__()

        obs_dim = obs_space.shape[0]
        self.q = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[1], 1))
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.q(obs).squeeze(dim=-1)


class ActorCritic(nn.Module):

    def __init__(self, obs_space, act_space, hidden_sizes=(64, 64)):
        super(ActorCritic, self).__init__()

        self.critic = Critic(obs_space, hidden_sizes)

        if isinstance(act_space, Box):
            self.actor = GaussianActor(obs_space, act_space, hidden_sizes)
        elif isinstance(act_space, Discrete):
            self.actor = CategoricalActor(obs_space, act_space, hidden_sizes)

    @torch.no_grad()
    def step(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        dist = self.actor._distribution(obs)
        action = dist.sample()
        # Last axis sum needed for Torch Normal distribution
        logp = self.actor._log_prob_from_distribution(dist, action)
        val = self.critic(obs)
        return action.numpy(), val.numpy(), logp.numpy()

    @torch.no_grad()
    def act(self, obs):
        return self.step(obs)[0]


# if __name__ == '__main__':
#     import gym
#
#     act_buf = []
#     obs_buf = []
#     logp_buf = []
#     rew_buf = []
#     new_logp = []
#     env = gym.make('LunarLander-v2')
#     # env = gym.make('BipedalWalker-v3')
#     obs_space = env.observation_space
#     act_space = env.action_space
#     ac = ActorCritic(obs_space, act_space)
#
#     o = env.reset()
#     for _ in range(10):
#         obs_buf.append(o)
#         a, logp = ac.act(o)
#         o, r, d, _ = env.step(a)
#         val, logp_new = ac(o, a)
#
#         logp_buf.append(logp)
#         act_buf.append(a)
#         rew_buf.append(r)
#         new_logp.append(logp_new)

    # obs_buf = torch.as_tensor(obs_buf, dtype=torch.float32)
    # act_buf = torch.as_tensor(np.array(act_buf), dtype=torch.float32)
    # V, lop_new = ac(obs_buf, act_buf)
