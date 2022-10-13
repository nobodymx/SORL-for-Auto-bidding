import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import numpy as np
import torch
import gin
from sdk.Common.Utils import draw_base_actions


# @gin.configurable
# class Critic(nn.Module):
#     def __init__(self, dim_observation, dim_action):
#         super(Critic, self).__init__()
#         self.dim_observation = dim_observation
#         self.dim_action = dim_action

#         self.FC1 = nn.Linear(self.dim_observation, 10)
#         self.FC2 = nn.Linear(10+self.dim_action, 50)
#         self.FC3 = nn.Linear(50, 10)
#         self.FC4 = nn.Linear(10, 1)

#     # obs: batch_size * obs_dim
#     def forward(self, obs, acts):
#         obs = F.relu(self.FC1(obs))
#         combined = torch.cat([obs, acts], 1)
#         result = F.relu(self.FC2(combined))
#         result = F.relu(self.FC3(result))
#         return self.FC4(result)


@gin.configurable
class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 100)
        self.FC2 = nn.Linear(100, 10)
        self.FC3 = nn.Linear(10, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))  # constrain to 0~10
        return (result + 1) * 5


@gin.configurable
class BGActors:
    def __init__(self, dim_obs, dim_actions, proportion=0.1, fixed_random_seed=0):
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.proportion = proportion

        torch.random.manual_seed(fixed_random_seed)
        # actors and critics and their targets
        self.actors = Actor(self.num_of_states, self.num_of_actions)

        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.actors.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # base actions
        self.base_actions = np.array([5.00, 5.22, 5.64, 6.24, 7.07, 7.15, 7.25, 7.23,
                                      6.85, 6.96, 6.94, 7.12, 6.81, 6.73, 6.22, 5.92,
                                      5.83, 5.23, 4.96, 4.87, 4.87, 4.76, 4.72, 4.32,
                                      4.92, 4.82, 4.71, 4.23, 4.43, 4.52, 4.12, 5.23,
                                      4.93, 4.94, 4.95, 4.82, 4.86, 4.98, 4.78, 4.88,
                                      5.12, 5.23, 4.24, 4.14, 4.53, 4.65, 4.52, 5.85,
                                      5.81, 4.99, 5.92, 5.82, 5.95, 5.72, 5.91, 5.52,
                                      5.73, 5.93, 5.87, 5.98, 5.70, 5.12, 5.71, 5.91,
                                      5.98, 5.23, 5.63, 5.86, 6.21, 6.13, 6.12, 6.24,
                                      6.32, 6.98, 7.12, 7.23, 7.35, 7.89, 8.12, 8.31,
                                      8.01, 7.98, 7.64, 7.24, 7.38, 6.24, 6.03, 5.99,
                                      5.98, 5.79, 6.10, 5.85, 5.55, 5.35, 5.25, 5.34
                                      ])
        # self.draw_base_actions()

    def draw_base_actions(self):
        draw_base_actions(self.base_actions)

    def take_actions(self, states, mode="behavior"):
        time_step = int(states[0] * 48 + 48) - 1
        # print(time_step)
        states = torch.Tensor(states).type(self.FloatTensor)
        # print(states)
        actions = self.actors(states)
        # off-policy
        actions = actions.cpu().data.numpy()
        if mode == "behavior":
            base_action = self.base_actions[time_step]
            actions = self.proportion * actions + (1 - self.proportion) * base_action
            # bounds
            actions = actions if actions >= 0 else 0
            actions = actions if actions <= 10 else 10
        elif mode == "target":
            base_action = self.base_actions[time_step]
            actions = self.proportion * actions + (1 - self.proportion) * base_action
            # bounds
            actions = actions if actions >= 0 else 0
            actions = actions if actions <= 10 else 10
        # print(actions)
        return actions
