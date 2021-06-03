import torch
import torch.nn as nn

from policy.util import init_


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=(64, 64)):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            init_(nn.Linear(state_dim, hidden_sizes[0])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[1], action_dim)),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.actor(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64)):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            init_(nn.Linear(state_dim + action_dim, hidden_sizes[0])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            init_(nn.Linear(hidden_sizes[1], 1))
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.critic(sa).squeeze(dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(64, 64)):
        super(ActorCritic, self).__init__()

        state_dim = obs_space.shape[0]
        action_dim = act_space.shape[0]
        max_action = act_space.high[0]

        self.pi = Actor(state_dim, action_dim, max_action, hidden_sizes)
        self.q1 = Critic(state_dim, action_dim, hidden_sizes)
        self.q2 = Critic(state_dim, action_dim, hidden_sizes)

    @torch.no_grad()
    def act(self, state):
        return self.pi(state).numpy()
