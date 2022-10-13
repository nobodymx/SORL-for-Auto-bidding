import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import numpy as np
import torch
import gin
from sdk.Common.Utils import random_sampling, normalize_state, draw_q_value_utils, normalize_state_batch


@gin.configurable
class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.FC1 = nn.Linear(self.dim_observation, 100)
        self.FC1_ = nn.Linear(self.dim_action, 100)
        self.FC2 = nn.Linear(100+100, 100)
        self.FC3 = nn.Linear(100, 10)
        # self.FC5 = nn.Linear(100, 100)
        # self.FC6 = nn.Linear(100, 100)
        self.FC4 = nn.Linear(10, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs = F.relu(self.FC1(obs))
        acts = F.relu(self.FC1_(acts))
        combined = torch.cat([obs, acts], 1)
        result = F.relu(self.FC2(combined))
        result = F.relu(self.FC3(result))
        # result = F.relu(self.FC5(result))
        # result = F.relu(self.FC6(result))
        return self.FC4(result)


@gin.configurable
class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 10)
        self.FC2 = nn.Linear(10, 10)
        self.FC3 = nn.Linear(10, dim_action)
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))  # constrain to 0~10
        return (result + 1) * 5


@gin.configurable
class DDPG:
    def __init__(self, dim_obs, dim_actions, gamma, tau, critic_lr, actor_lr, buffer_size, sample_size, network_random_seed=1,
                Q_train_ite=2, Actor_train_ite=1, explore_noise=0.1):
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.Q_train_ite = Q_train_ite
        self.Actor_train_ite = Actor_train_ite
        self.explore_noise = explore_noise
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        torch.random.manual_seed(network_random_seed)
        # actors and critics and their targets
        self.actors = Actor(self.num_of_states, self.num_of_actions)
        self.critics = Critic(self.num_of_states, self.num_of_actions)

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.GAMMA = gamma
        self.tau = tau
        self.num_of_steps = 0

        self.var = 1  # 修改过
        self.critic_optimizer = Adam(self.critics.parameters(),
                                     lr=self.critic_lr)
        self.actor_optimizer = Adam(self.actors.parameters(),
                                    lr=self.actor_lr)

        self.num_of_episodes = 0

        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.actors.cuda()
            self.critics.cuda()
            self.actors_target.cuda()
            self.critics_target.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # replay buffer
        self.replay_buffer = {"states": np.zeros((buffer_size, self.num_of_states)),
                              "actions": np.zeros((buffer_size, self.num_of_actions)),
                              "rewards": np.zeros((buffer_size, 1)),
                              "next_states": np.zeros((buffer_size, self.num_of_states)),
                              "terminal": np.zeros((buffer_size, 1))}
        self.buffer_pointer = 0
        self.if_full = False
        self.buffer_size = buffer_size
        self.sample_size = sample_size

    def store_experience(self, states, actions, rewards, next_states, terminal, values):
        self.replay_buffer["states"][self.buffer_pointer] = deepcopy(states)
        self.replay_buffer["actions"][self.buffer_pointer] = deepcopy(actions)
        self.replay_buffer["rewards"][self.buffer_pointer] = deepcopy(rewards)
        self.replay_buffer["next_states"][self.buffer_pointer] = deepcopy(next_states)
        self.replay_buffer["terminal"][self.buffer_pointer] = deepcopy(terminal)

        self.buffer_pointer += 1
        if self.buffer_pointer == self.buffer_size:
            self.if_full = True
            self.buffer_pointer = 0

    def sample_experience(self):
        sample_index_list = random_sampling()
        experience_samples = np.zeros((self.sample_size,
                                       self.num_of_states + self.num_of_actions + 1 + self.num_of_states + 1))
        index = 0
        for i in sample_index_list:
            experience_samples[index, 0:self.num_of_states] = deepcopy(self.replay_buffer["states"][i])
            experience_samples[index, self.num_of_states:self.num_of_states + self.num_of_actions] = deepcopy(
                self.replay_buffer["actions"][i])
            experience_samples[index, self.num_of_states + self.num_of_actions] = deepcopy(
                self.replay_buffer["rewards"][i])
            experience_samples[index,
            self.num_of_states + self.num_of_actions + 1:self.num_of_states + self.num_of_actions + 1 + self.num_of_states] = deepcopy(
                self.replay_buffer["next_states"][i])
            experience_samples[index, self.num_of_states + self.num_of_actions + 1 + self.num_of_states] = deepcopy(
                self.replay_buffer["terminal"][i])
            index += 1
        return experience_samples

    def train(self):
        """
        use this function only when the buffer is full
        just sample
        :return:
        """
        # torch.autograd.set_detect_anomaly(True)
        if self.if_full:
            # increase training episode
            self.num_of_episodes += 1
            if self.num_of_episodes >= 2000:
                self.critic_optimizer = Adam(self.critics.parameters(),
                                     lr=0.00001)
            
            samples = torch.Tensor(self.sample_experience()).type(self.FloatTensor)

            # extract states, actions, rewards, next_states
            states = samples[:, 0:self.num_of_states]
            # print(states)
            actions = samples[:, self.num_of_states:(self.num_of_states + self.num_of_actions)]
            # print(actions)
            rewards = samples[:, self.num_of_actions + self.num_of_states].unsqueeze(1)
            next_states = samples[:,
                          self.num_of_states + self.num_of_actions + 1:self.num_of_states + self.num_of_actions + 1 + self.num_of_states]
            terminal = samples[:, self.num_of_states + self.num_of_actions + 1 + self.num_of_states]

            # norm_states = normalize_state_batch(states)
            # norm_next_states = normalize_state_batch(next_states)
            # print(states)
            # print(next_states)

            actions = actions / 5 - 1
            # print(actions)
            # rewards = (rewards - 10)
            """
            train critic network
            """
            
            # calculate the actions
            next_actions = self.actors_target(next_states)
            # print(next_actions)

            next_actions = next_actions / 5 - 1
            # print(next_actions)
            # calculate target values
            target_Q = self.critics_target(next_states, next_actions) * terminal.reshape(len(terminal), 1)
            target_Q = rewards + self.GAMMA * target_Q
            print(target_Q)
            for i in range(self.Q_train_ite):
                # calculate current Q
                current_Q = self.critics(states, actions)
                # print(current_Q[0])
                loss_Q = torch.nn.MSELoss()(current_Q, target_Q.detach())
                # empty the optimizer
                self.critic_optimizer.zero_grad()
                # backwards
                loss_Q.backward()
                # update
                self.critic_optimizer.step()

                print(loss_Q.cpu().data.numpy())

            """
            train actor network
            """
            for i in range(self.Actor_train_ite):
                actions_this_agent = self.actors(states)
                actions_this_agent = actions_this_agent / 5 - 1
                loss_A = -self.critics(states, actions_this_agent)
                loss_A = loss_A.mean()
                self.actor_optimizer.zero_grad()
                loss_A.backward()
                self.actor_optimizer.step()

                print(loss_A.cpu().data.numpy())

            # check the Q value
            self.draw_q_value()

            return loss_Q.cpu().data.numpy(), loss_A.cpu().data.numpy()
        else:
            return None, None
    
    def draw_q_value(self, num=100, state=[96,1500,0],path=None):
        state_set = [[96, 1500, 0],
        [48, 750, 750],
        [20, 100, 1400]]
        actions = np.linspace(-1,1,num)
        # print(actions)
        for i in range(len(state_set)):
            state = normalize_state(deepcopy(state_set[i]))
            q_values = []
            for j in range(num):
                # print(actions[j])
                q_values.append(deepcopy(self.critics(torch.Tensor([state]).type(self.FloatTensor),
                            torch.Tensor([[actions[j]]]).type(self.FloatTensor)).cpu().data.numpy()[0]))
            print(state_set[i][0])
            draw_q_value_utils(q_values, time_step=96-state_set[i][0], path=path)

    def take_actions(self, states, mode="behavior"):
        states = torch.Tensor(states).type(self.FloatTensor)
        # print(states)
        actions = self.actors(states)
        # off-policy
        actions = actions.cpu().data.numpy()
        if mode == "behavior":
            actions += np.random.randn(1) * self.explore_noise
            actions = actions if actions >= 0 else 0
            actions = actions if actions <= 10  else 10
            # actions = np.random.rand()*10
        elif mode == "target":
            pass
        # print(actions)
        return actions
    
    def pre_train_Q(self, iteration=10000):
        if self.if_full:
            print("start pretrain Q network")
            states = torch.Tensor(self.replay_buffer["states"]).type(self.FloatTensor)
            actions = torch.Tensor(self.replay_buffer["actions"]).type(self.FloatTensor)
            # actions = actions / 5 - 1
            rewards = torch.Tensor(self.replay_buffer["rewards"]).type(self.FloatTensor)
            next_states = torch.Tensor(self.replay_buffer["next_states"]).type(self.FloatTensor)
            terminal = torch.Tensor(self.replay_buffer["terminal"]).type(self.FloatTensor)
            
            for ite in range(iteration):
                """
                train critic network
                """
                # calculate the actions
                next_actions = self.actors_target(next_states)

                # next_actions = next_actions / 5 - 1
                # calculate target values
                target_Q = self.critics_target(next_states, next_actions) * terminal.reshape(len(terminal), 1)
                target_Q = rewards + self.GAMMA * target_Q
                # print(target_Q[0])
                
                # calculate current Q
                current_Q = self.critics(states, actions)
                # print(current_Q[0])
                loss_Q = torch.nn.MSELoss()(current_Q, target_Q.detach())
                # empty the optimizer
                self.critic_optimizer.zero_grad()
                # backwards
                loss_Q.backward()
                # update
                self.critic_optimizer.step()

                print(loss_Q.cpu().data.numpy())
            self.draw_q_value()
        else:
            print("ERROR: buffer not full")

    def update_target(self):
        for target_param, source_param in zip(self.critics_target.parameters(),
                                              self.critics.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data)
        for target_param, source_param in zip(self.actors_target.parameters(),
                                              self.actors.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data)

    def save_net(self, save_path):
        try:
            torch.save(self.critics, save_path + "/critic.pkl")
            torch.save(self.actors, save_path + "/actor.pkl")
        except:
            print("save net failed: there is no such path")

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cuda:0'):
        try:
            self.critics = torch.load(load_path + "/critic.pkl", map_location="cpu")
            self.actors = torch.load(load_path + "/actor.pkl", map_location="cpu")
            print("model stored path " + next(self.critics.parameters()).device.type)
            self.actors_target = deepcopy(self.actors)
            self.critics_target = deepcopy(self.critics)
            self.critic_optimizer = Adam(self.critics.parameters(),
                                            lr=self.critic_lr)
            self.actor_optimizer = Adam(self.actors.parameters(),
                                        lr=self.actor_lr)
            
            # cuda usage
            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                self.actors.cuda()
                self.critics.cuda()
                self.actors_target.cuda()
                self.critics_target.cuda()
            print("model stored path " + next(self.critics.parameters()).device.type)
        except:
            print("load net failed: there is no such path")
