from tkinter.messagebox import NO
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import numpy as np
import torch
import gin
from sdk.Model.DDPG import Actor
from sdk.Common.Utils import random_sampling_personalized, normalize_state, draw_q_value_utils, save_Q_png_file_path


@gin.configurable
class CriticCQL(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(CriticCQL, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.FC1 = nn.Linear(self.dim_observation, 10)
        self.FC2 = nn.Linear(10 + dim_action, 50)
        self.FC3 = nn.Linear(50, 10)
        # self.FC3_ = nn.Linear(100, 10)
        self.FC4 = nn.Linear(10, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        result = F.relu(self.FC3(result))
        # result = F.relu(self.FC3_(result))
        return self.FC4(result)


@gin.configurable
class ActorCQL(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(ActorCQL, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 10)
        self.FC2 = nn.Linear(10, 10)
        self.FC3 = nn.Linear(10, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))  # constrain to 0~10
        return (result + 1) * 5


@gin.configurable
class CQL:
    def __init__(self, dim_obs, dim_actions, gamma, tau, critic_lr, actor_lr, buffer_size, sample_size, network_random_seed=1,
                 explore_noise=0.1, Q_train_ite=2, Actor_train_ite=1, cql_rate=0.0002,
                 cql_num_sample=10, ACTION_MAX=10, ACTION_MIN=0, TD_error_rate=0.5,
                 TD_switch="on", CQL_switch="off", explore_size=10000):
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        
        self.explore_noise = explore_noise
        self.Q_train_ite = Q_train_ite
        self.Actor_train_ite = Actor_train_ite
        self.cql_rate = cql_rate
        self.TD_error_rate = TD_error_rate
        self.cql_num_sample = cql_num_sample
        self.ACTION_MAX = ACTION_MAX
        self.ACTION_MIN = ACTION_MIN
        self.network_random_seed = network_random_seed
        self.explore_size = explore_size

        
        self.TD_switch = TD_switch
        self.CQL_switch = CQL_switch


        print("TD loss switch: " + self.TD_switch)
        print("CQL loss switch: " + self.CQL_switch)

        self.Q_png_save_path = save_Q_png_file_path(self.TD_error_rate, self.cql_rate,  0,
                                                    self.TD_switch, self.CQL_switch, "off", self.explore_size) + "_random_seed_" + str(self.network_random_seed) + ".png" 


        self.Q_loss_valid = True
        if self.TD_switch != "on" and self.CQL_switch != "on" and self.VCQL_switch != "on":
            self.Q_loss_valid = False

        torch.random.manual_seed(self.network_random_seed)
        # actors and critics and their targets
        self.actors = ActorCQL(self.num_of_states, self.num_of_actions)
        self.critics = CriticCQL(self.num_of_states, self.num_of_actions)

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.GAMMA = gamma
        self.tau = tau
        self.num_of_steps = 0

        self.var = 1  # 修改过
        self.critic_optimizer = Adam(self.critics.parameters(),
                                     lr=critic_lr)
        self.actor_optimizer = Adam(self.actors.parameters(),
                                    lr=actor_lr)

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
        self.train_episode = 0

    def store_experience(self, states, actions, rewards, next_states, terminal):
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
        self.buffer_size = len(self.replay_buffer["states"])
        self.sample_size = int(0.2 * len(self.replay_buffer["states"]))
        sample_index_list = random_sampling_personalized(sample_number=self.sample_size,
                                                         buffer_size=self.buffer_size)
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
            if not self.Q_loss_valid:
                print("ERROR: Q loss not valid")
                return None, None
            self.train_episode += 1
            samples = torch.Tensor(self.sample_experience()).type(self.FloatTensor)

            # extract states, actions, rewards, next_states
            states = samples[:, 0:self.num_of_states]
            actions = samples[:, self.num_of_states:(self.num_of_states + self.num_of_actions)]
            rewards = samples[:, self.num_of_actions + self.num_of_states].unsqueeze(1)
            next_states = samples[:,
                          self.num_of_states + self.num_of_actions + 1:self.num_of_states + self.num_of_actions + 1 + self.num_of_states]
            terminal = samples[:, self.num_of_states + self.num_of_actions + 1 + self.num_of_states]

            actions = actions / 5 - 1
            # rewards = (rewards - 10)
            """
            train critic network
            """
            # calculate the actions
            next_actions = self.actors_target(next_states)

            next_actions = next_actions / 5 - 1
            # calculate target values
            target_Q = self.critics_target(next_states, next_actions) * terminal.reshape(len(terminal), 1)
            target_Q = target_Q * self.GAMMA + rewards
            for i in range(self.Q_train_ite):
                # initialize Q loss
                loss_Q = 0
                # TD target term: calculate current Q
                if self.TD_switch == "on":
                    print("TD switch on")
                    current_Q = self.critics(states, actions)
                    loss_Q += self.TD_error_rate * torch.nn.MSELoss()(current_Q,
                                                  target_Q.detach())

                # CQL terms: conservative penalty & rewards for in-distribution actions
                if self.CQL_switch == "on":
                    print("CQL switch on")
                    cql_states = torch.stack([states for i in range(self.cql_num_sample)]).reshape(
                        self.cql_num_sample * self.sample_size,
                        self.num_of_states).type(self.FloatTensor)
                    cql_sample_actions = torch.linspace(self.ACTION_MIN / 5 - 1, self.ACTION_MAX / 5 - 1, self.cql_num_sample).type(self.FloatTensor)
                    cql_sample_actions = torch.stack([cql_sample_actions for i in range(self.sample_size)]).permute(1,
                                                                                                                    0).reshape(
                        self.cql_num_sample * self.sample_size, 1).type(self.FloatTensor)

                    part_1 = self.critics(cql_states, cql_sample_actions).reshape(self.cql_num_sample,
                                                                            self.sample_size).permute(1, 0)
                    
                    # part_1 /= 100
                    part_1 = torch.log(torch.sum(torch.exp(part_1 / 100), 1).unsqueeze(1))
                    part_2 = -self.critics(states, actions)
                    loss_cql = self.cql_rate * (part_1 + part_2)

                    loss_Q += self.cql_rate * loss_cql.mean()
                    
                # empty the optimizer
                self.critic_optimizer.zero_grad()
                loss_Q.backward()
                self.critic_optimizer.step()

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
            # check the Q value
            self.draw_q_value(path=self.Q_png_save_path)
            return loss_Q.cpu().data.numpy(), loss_A.cpu().data.numpy()
        else:
            return None, None

    def draw_q_value(self, num=100, state_set=[96, 1500, 0], path=None):
        state_set = [[96, 1500, 0]]
        # [48, 750, 750],
        # [20, 100, 1400]]
        actions = np.linspace(-1, 1, num)
        # print(actions)
        for i in range(len(state_set)):
            state = normalize_state(deepcopy(state_set[i]))
            q_values = []
            for j in range(num):
                # print(actions[j])
                q_values.append(deepcopy(self.critics(torch.Tensor([state]).type(self.FloatTensor),
                                                    torch.Tensor([[actions[j]]]).type(
                                                        self.FloatTensor)).cpu().data.numpy()[0]))
            print(state_set[i][0])
            draw_q_value_utils(q_values, time_step=96 - state_set[i][0], path=path)

    def take_actions(self, states, mode="behavior"):
        states = torch.Tensor(states).type(self.FloatTensor)
        actions = self.actors(states)
        # off-policy
        actions = actions.cpu().data.numpy()
        if mode == "behavior":
            actions += np.random.randn(1) * self.explore_noise
            actions = actions if actions >= 0 else 0
        elif mode == "target":
            pass
        return actions

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
            torch.save(self.critics, save_path + "/critic_seed_" + str(self.network_random_seed) + ".pkl")
            torch.save(self.actors, save_path + "/actor_seed_" + str(self.network_random_seed) + ".pkl")
           
        except:
            print("save net failed: there is no such path")

    def load_net(self, load_path="saved_model/fixed_initial_budget"):
        try:
            self.critics = torch.load(load_path + "/critic_"+ str(self.network_random_seed) +".pkl", map_location=torch.device('cpu'))
            self.actors = torch.load(load_path + "/actor_"+ str(self.network_random_seed) +".pkl", map_location=torch.device('cpu'))
        
            self.actors_target = deepcopy(self.actors)
            self.critics_target = deepcopy(self.critics)
            self.critic_optimizer = Adam(self.critics.parameters(),
                                         lr=self.critic_lr)
            self.actor_optimizer = Adam(self.actors.parameters(),
                                        lr=self.actor_lr)
            # cuda usage
            if self.use_cuda:
                self.actors.cuda()
                self.critics.cuda()
                self.actors_target.cuda()
                self.critics_target.cuda()
        except:
            print("load net failed: there is no such path")
