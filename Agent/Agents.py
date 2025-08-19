from tkinter import N
from tkinter.messagebox import NO
import gin
from copy import deepcopy
from sdk.Common.Utils import normalize_state
import numpy as np
from sdk.Model.BCQ import BCQ
from sdk.Model.V_CQL import VCQL
from sdk.Model.CQL import CQL
from sdk.Model.DDPG import DDPG
from sdk.Model.USCB import USCB


@gin.configurable
class Agents:
    def __init__(self, num_agent, algorithm_name, bgactors, dim_obs=3, representation_index=0,
                 stop_threshold=0.1, load_algorithm_background_actor=False,
                 net_load_path=None, iteration=0, explore_flag=False, explore_policy=None,
                 load_path_initial_safe_policy="saved_models/sorl/initial_safe_policy/actor.pkl",
                 load_path_phi="saved_models/sorl/iteration_", network_random_seed=0, phi_name="VCQL",
                 ):
        """
        :param num_agent:
        :param algorithm:
        :param mode:
        :param represent_index:
        """
        self.num_agent = num_agent
        self.dim_obs = dim_obs
        self.current_state = np.zeros((self.num_agent, self.dim_obs))
        self.actions = []
        self.budget = 0
        self.representation_index = representation_index
        self.stop_threshold = stop_threshold
        self.iteration = iteration

        print("INFO: algorithm name: " + algorithm_name)
        if algorithm_name == "BCQ":
            self.algorithm = BCQ()
            self.algorithm.load_base_actor_network()
        elif algorithm_name == "CQL":
            self.algorithm = CQL()
        elif algorithm_name == "VCQL":
            self.algorithm = VCQL()
            # load old Q network
            if iteration == 0:
                self.algorithm.load_old_Q_network(load_path="saved_models/sorl/initial_safe_policy/critic.pkl")
            else:
                self.algorithm.load_old_Q_network(load_path=load_path_phi + str(iteration - 1) + "/" + phi_name + "/critic_seed_" + str(network_random_seed) + ".pkl")
        elif algorithm_name == "DDPG":
            self.algorithm = DDPG()
        elif algorithm_name == "USCB":
            self.algorithm = USCB()


        self.back_ground_algorithms = [bgactors() for i in range(self.num_agent)]
        if load_algorithm_background_actor:
            for i in range(self.num_agent):
                self.back_ground_algorithms[i].load_net(load_path=net_load_path)

        if explore_flag:
            self.explore_policy = explore_policy()
            self.iteration = iteration
            phi_path = load_path_phi+ str(self.iteration-1) + "/" + phi_name + "/critic_seed_" + str(network_random_seed) + ".pkl" if self.iteration > 0 else "saved_models/sorl/initial_safe_policy/critic.pkl"
            self.explore_policy.load_nets(phi_path=phi_path,
                                          u_v_path=load_path_initial_safe_policy)
            
            self.explore_indicator = "off" if self.iteration == 0 else "on"

    def reset(self, initial_state, budget):
        self.current_state = initial_state
        self.budget = budget

    def take_actions(self, mode="train", test_index=None, train_index=None, sorl_index=None):
        """
        if there is no budget, then kick off the bidding
        :return:
        """
        self.actions = []
        if mode == "test":
            if test_index is None:
                print("ERROR: there is no test campaign")
            else:
                for i in range(self.num_agent):
                    if self.current_state[i, 1] <= self.stop_threshold:
                        self.actions.append(-1)
                    else:
                        if i == test_index:
                            self.actions.append(
                                self.algorithm.take_actions(normalize_state(deepcopy(self.current_state[i])),
                                                            mode="target"))
                        else:
                            self.actions.append(
                                self.back_ground_algorithms[i].take_actions(
                                    normalize_state(deepcopy(self.current_state[i])),
                                    mode="target"))
        elif mode == "train":
            if train_index is None:
                print("ERROR: there is no train campaign")
            else:
                for i in range(self.num_agent):
                    if self.current_state[i, 1] <= self.stop_threshold:
                        self.actions.append(-1)
                    else:
                        if i == train_index:
                            # if self.algorithm.if_full:
                            self.actions.append(
                                self.algorithm.take_actions(normalize_state(deepcopy(self.current_state[i])),
                                                            mode="behavior"))

                        else:
                            temp_action = self.back_ground_algorithms[i].take_actions(
                                normalize_state(deepcopy(self.current_state[i])),
                                mode="target")
                            self.actions.append(temp_action)
        elif mode == "train_vas":
            if train_index is None:
                print("ERROR: there is no train campaign")
            else:
                if self.current_state[1] <= self.stop_threshold:
                    self.actions = -1
                else:
                    self.actions = self.algorithm.take_actions(normalize_state(deepcopy(self.current_state)),
                                                               mode="behavior")
        elif mode == "test_vas":
            if test_index is None:
                print("ERROR: there is no train campaign")
            else:
                if self.current_state[1] <= self.stop_threshold:
                    self.actions = -1
                else:
                    self.actions = self.algorithm.take_actions(normalize_state(deepcopy(self.current_state)),
                                                               mode="target")
        elif mode == "sorl":
            if sorl_index is None:
                print("ERROR: there is no SORL campaign")
            else:
                for i in range(self.num_agent):
                    if self.current_state[i, 1] <= self.stop_threshold:
                        self.actions.append(-1)
                    else:
                        if i == sorl_index:
                            # if self.algorithm.if_full:
                            self.actions.append(
                                self.explore_policy.explore_action(self.current_state[i],
                                                                   explore_flag=self.explore_indicator))
                        else:
                            temp_action = self.back_ground_algorithms[i].take_actions(
                                normalize_state(deepcopy(self.current_state[i])),
                                mode="target")
                            self.actions.append(temp_action)
        else:
            print("no such mode!")
        return self.actions

    def transitions_no_store(self, next_state):
        self.current_state = deepcopy(next_state)

    def transitions(self, rewards, next_state, terminal, values, train_index=None, mode="ras_train"):
        """
        :param train_index:
        :param terminal:
        :param rewards: dimension: num_agent
        :param next_state: dimension: num_agent
        :return:
        """
        if train_index is None:
            print("ERROR: train index is None")
        else:
            if mode == "ras_train":
                if self.actions[train_index] >= 0:
                    self.algorithm.store_experience(normalize_state(deepcopy(self.current_state[train_index])),
                                                    self.actions[train_index],
                                                    rewards[train_index],
                                                    normalize_state(deepcopy(next_state[train_index])),
                                                    terminal[train_index],
                                                    values)
            elif mode == "vas_train":
                if self.actions >= 0:
                    self.algorithm.store_experience(normalize_state(deepcopy(self.current_state)),
                                                    self.actions,
                                                    rewards,
                                                    normalize_state(deepcopy(next_state)),
                                                    terminal,
                                                    values)
        self.current_state = deepcopy(next_state)

    def train(self):
        Q_loss, A_loss = self.algorithm.train()
        if Q_loss is None:
            print("buffer not full")
        else:
            print("Q_loss -- %f, A_loss -- %f" % (Q_loss, A_loss))
        return Q_loss, A_loss

    def pre_train_Q_network(self, iteration=10000):
        self.algorithm.pre_train_Q(iteration=iteration)
