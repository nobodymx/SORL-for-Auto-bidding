"""
    This file is for the tool functions used in other codes
"""
from tkinter.messagebox import NO
import gin
import numpy as np
from matplotlib import pyplot as plt


@gin.configurable
def random_sampling(sample_number, buffer_size):
    """
    :return: list with form [ , , , , ,]
    """
    list_ = []
    count = 0
    while count < sample_number:
        random_number = np.random.randint(0, buffer_size, 1).tolist()[0]
        if random_number not in list_:
            list_.append(random_number)
            count += 1
    return list_


def random_sampling_personalized(sample_number, buffer_size):
    list_ = []
    count = 0
    while count < sample_number:
        random_number = np.random.randint(0, buffer_size, 1).tolist()[0]
        if random_number not in list_:
            list_.append(random_number)
            count += 1
    return list_


@gin.configurable
def normalize_state(state, num_step=48, max_budget=200):
    state[0] = (state[0] - num_step / 2) / (num_step / 2)
    state[1] = (state[1] - max_budget / 2) / (max_budget / 2)
    state[2] = (state[2] - max_budget / 2) / (max_budget / 2)
    return state


@gin.configurable
def normalize_state_batch(state, num_step=48, max_budget=200):
    state[:, 0] = (state[:, 0] - num_step / 2) / (num_step / 2)
    state[:, 1] = (state[:, 1] - max_budget / 2) / (max_budget / 2)
    state[:, 2] = (state[:, 2] - max_budget / 2) / (max_budget / 2)
    return state


def draw_rewards(x, rewards, path=None):
    plt.plot(x, rewards, 'b', label='Rewards', linewidth=2.0)
    # plt.xlim(0, epi)
    # plt.ylim((200,500))
    if path is None:
        plt.savefig('results/rewards/episode_mean_rewards.png')
    else:
        plt.savefig(path)
    plt.show()
    plt.close()


def draw_ROI(ROI):
    """
    :param ROI:
    :return:
    """
    x = [i for i in range(len(ROI))]
    plt.bar(x, ROI)
    plt.ylim((0, max(ROI)))
    plt.savefig("results/bar.png")
    plt.show()
    plt.close()


def draw_actions(actions, epi, rewards):
    """
    :param ROI:
    :return:
    """
    # print(actions)
    plt.plot([i for i in range(len(actions))], actions, 'b', label='Rewards', linewidth=2.0)
    plt.text(48, np.max(actions), 'rewards:' + str(round(rewards, 3)))
    plt.savefig("results/actions/actions_%d.png" % epi)
    plt.show()
    plt.close()


def softmax(x):
    e_x = x - np.max(x)
    while np.max(e_x) > 10:
        e_x /= 10
    return np.exp(e_x) / np.sum(np.exp(e_x))


def draw_q_value_utils(q_values, time_step=0, path=None):
    plt.plot([i / 10 for i in range(len(q_values))], q_values, 'b', label='Q Values', linewidth=2.0)
    if path is None:
        plt.savefig("results/network_value/Q_value_ddpg_init_poor_policy_tau_0_3_0_85_not_divide_%d.png" % time_step)
    else:
        plt.savefig(path)
    plt.show()
    plt.close()


def draw_base_actions(base_actions):
    plt.plot([i for i in range(len(base_actions))], base_actions, 'b', label='Q Values', linewidth=2.0)
    plt.savefig("results/base_actions.png")
    plt.show()
    plt.close()

def save_Q_png_file_path(td_rate, cql_rate, vcql_rate, td_switch, cql_switch, vcql_switch, explore_size):
    interm = ""
    interm_ = "rate_"
    if td_switch == "on":
        interm += "TD"
        interm_ += str(td_rate)
    if cql_switch == "on":
        interm += "_cql"
        interm_ += "_" + str(cql_rate)
    if vcql_switch == "on":
        interm += "_vcql"
        interm_ += "_" + str(vcql_rate)
    return "results/network_value/sorl/"+interm+"/"+interm_+"_explore_"+str(explore_size)

def save_reward_file_path(algorithm_name, td_rate, cql_rate, vcql_rate, td_switch, cql_switch, vcql_switch, explore_size):
    if algorithm_name == "VCQL" or algorithm_name == "CQL":
        interm = ""
        interm_ = "rate_"
        if td_switch == "on":
            interm += "TD"
            interm_ += str(td_rate)
        if cql_switch == "on":
            interm += "_cql"
            interm_ += "_" + str(cql_rate)
        if vcql_switch == "on":
            interm += "_vcql"
            interm_ += "_" + str(vcql_rate)
        return "results/rewards/sorl/"+interm+"/"+interm_+"_explore_"+str(explore_size)

    elif algorithm_name == "BCQ":
        return "results/rewards/sorl/BCQ/explore_"+str(explore_size)

    
    