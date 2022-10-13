import gin
import numpy as np
from sdk.Environment.RealAdvertisingSystem import RealAdvertisingSystem
from sdk.Environment.VirtualAdvertisingSystem import VirtualAdvertisingSystem
from sdk.Agent.Agents import Agents
from sdk.Common.Utils import draw_rewards, draw_ROI, save_reward_file_path
import pandas as pd
from copy import deepcopy
from sdk.Common.Utils import normalize_state
import os
from sdk.Evaluation.RPE import RPE
from sdk.Evaluation.OPE import OPE
from sdk.Model.CalculateOptimalR import OptimalRCalculator


@gin.configurable
def run_sorl(episode=100000,
             len_step=48,
             representation_index=0,
             representation_budget=1500,
             num_agent=100,
             dim_obs=3,
             target_update_epi=5,
             saved_trained_net=True,
             save_path="saved_model/sorl/iteration_",
             load_path="saved_model/sorl/iteration_",
             min_budget=10,
             max_budget=200,
             store_ranking_log=False,
             fixed_random_seed=1,
             iteration=0,
             gpu='0',
             explore_size=1000,
             load_previous_iteration_network=False,
             evaluation_interval=100,
             network_random_seed=0,
             cql_rate=0.0002, vcql_rate=1, TD_error_rate=0.5,
             TD_switch="on", CQL_switch="off", VCQL_switch="off",
             algorithm_name="VCQL",
             ):
    # specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # initial budget
    np.random.seed(fixed_random_seed)
    budget = np.random.randint(min_budget, max_budget, num_agent)
    budget[representation_index] = representation_budget

    # initial env and agents
    ras = RealAdvertisingSystem()
    # evaluation methods
    rpe = RPE()
    ope = OPE()
    # create VAS
    vas = VirtualAdvertisingSystem(budget=budget)
    training_step_len = vas.ranking_log[representation_index]["step_cnt"][0]
    optimal_R = OptimalRCalculator()
    agents = Agents()
    
    if load_previous_iteration_network:
        # load parameters
        if iteration > 0 :
            agents.algorithm.load_net(load_path=load_path + str(iteration - 1))
        else:
            agents.algorithm.load_net(load_path="saved_models/sorl/initial_safe_policy")

    # save reward path
    rewards_save_path = save_reward_file_path(algorithm_name, TD_error_rate, cql_rate, vcql_rate, TD_switch, CQL_switch, VCQL_switch, explore_size)
    # initial state
    initial_state = np.zeros((num_agent, dim_obs))
    initial_state[:, 0] = len_step
    initial_state[:, 1] = budget

    initial_state_test = np.zeros(dim_obs)
    initial_state_test[0] = len_step
    initial_state_test[1] = budget[representation_index]

    save_path = save_path + str(iteration) + "/" + algorithm_name

    # abortion condition
    latest_rewards = []
    abortion_flag = False

    """
    ------------------------------------------------------------------------------------
    online explorations
    ------------------------------------------------------------------------------------
    """

    data_buffer = {"states": [],
                   "actions": [],
                   "next_states": [],
                   "rewards": [],
                   "terminal": []}
    # loading previous offline data
    for i in range(iteration):
        states = pd.read_excel("data/offline_dataset/iteration_" + str(i) + "/states.xlsx",
                               sheet_name="Sheet1").values[:, 1::]
        actions = pd.read_excel("data/offline_dataset/iteration_" + str(i) + "/actions.xlsx",
                                sheet_name="Sheet1").values[:, 1::]
        rewards = pd.read_excel("data/offline_dataset/iteration_" + str(i) + "/rewards.xlsx",
                                sheet_name="Sheet1").values[:, 1::]
        next_states = pd.read_excel("data/offline_dataset/iteration_" + str(i) + "/next_states.xlsx",
                                    sheet_name="Sheet1").values[:, 1::]
        terminal = pd.read_excel("data/offline_dataset/iteration_" + str(i) + "/terminal.xlsx",
                                 sheet_name="Sheet1").values[:, 1::]
        if i >= 1:
            data_buffer["states"] = np.concatenate([data_buffer["states"], states], 0)
            data_buffer["actions"] = np.concatenate([data_buffer["actions"], actions], 0)
            data_buffer["next_states"] = np.concatenate([data_buffer["next_states"], next_states], 0)
            data_buffer["rewards"] = np.concatenate([data_buffer["rewards"], rewards], 0)
            data_buffer["terminal"] = np.concatenate([data_buffer["terminal"], terminal], 0)
        else:
            data_buffer["states"] = states
            data_buffer["actions"] = actions
            data_buffer["next_states"] = next_states
            data_buffer["rewards"] = rewards
            data_buffer["terminal"] = terminal

    # data_buffer["states"] = pd.read_excel("data/offline_dataset/iteration_" + str(iteration) + "/states.xlsx",
    #                                       sheet_name="Sheet1").values[:, 1::]
    # data_buffer["actions"] = pd.read_excel("data/offline_dataset/iteration_" + str(iteration) + "/actions.xlsx",
    #                                        sheet_name="Sheet1").values[:, 1::]
    # data_buffer["rewards"] = pd.read_excel("data/offline_dataset/iteration_" + str(iteration) + "/rewards.xlsx",
    #                                        sheet_name="Sheet1").values[:, 1::]
    # data_buffer["next_states"] = pd.read_excel("data/offline_dataset/iteration_" + str(iteration) + "/next_states.xlsx",
    #                                            sheet_name="Sheet1").values[:, 1::]
    # data_buffer["terminal"] = pd.read_excel("data/offline_dataset/iteration_" + str(iteration) + "/terminal.xlsx",
    #                                         sheet_name="Sheet1").values[:, 1::]

    # stored data
    explore_states, explore_actions, explore_rewards, explore_next_states, explore_terminal = [], [], [], [], []

    complete_explore = False

    # online explorations for data collections
    for epi in range(episode):
        agents.reset(initial_state, budget)
        ras.reset(budget, store_ranking_log=store_ranking_log)
        for step in range(len_step):
            # store states
            explore_states.append(deepcopy(normalize_state(deepcopy(agents.current_state[representation_index]))))
            actions = agents.take_actions(mode="sorl", sorl_index=representation_index)
            # store actions
            explore_actions.append(deepcopy(actions[representation_index]))
            rewards, next_state, terminal, flag = ras.next_state(agents.current_state, actions, step,
                                                                 store_ranking_log=store_ranking_log)
            # store rewards
            explore_rewards.append(deepcopy(rewards[representation_index]))
            explore_terminal.append(deepcopy(terminal[representation_index]))

            agents.transitions_no_store(deepcopy(next_state))
            explore_next_states.append(deepcopy(normalize_state(deepcopy(agents.current_state[representation_index]))))

            # check if all the campaigns have run out of budgets
            if flag:
                print("terminate in advance")
                break

            if len(explore_states) == explore_size:
                complete_explore = True
                break
        if complete_explore:
            break

    if iteration >= 1:
        data_buffer["states"] = np.concatenate([data_buffer["states"], np.array(explore_states)], 0)
        data_buffer["actions"] = np.concatenate([data_buffer["actions"], np.array(explore_actions)], 0)
        data_buffer["next_states"] = np.concatenate([data_buffer["next_states"], np.array(explore_next_states)], 0)
        data_buffer["rewards"] = np.concatenate([data_buffer["rewards"], np.array(explore_rewards)], 0)
        data_buffer["terminal"] = np.concatenate([data_buffer["terminal"], np.array(explore_terminal)], 0)
    else:
        data_buffer["states"] = np.array(explore_states)
        data_buffer["actions"] = np.array(explore_actions)
        data_buffer["next_states"] = np.array(explore_next_states)
        data_buffer["rewards"] = np.array(explore_rewards)
        data_buffer["terminal"] = np.array(explore_terminal)

    agents.algorithm.if_full = True
    agents.algorithm.replay_buffer["states"] = deepcopy(data_buffer["states"])
    agents.algorithm.replay_buffer["actions"] = deepcopy(data_buffer["actions"])
    agents.algorithm.replay_buffer["rewards"] = deepcopy(data_buffer["rewards"])
    agents.algorithm.replay_buffer["next_states"] = deepcopy(data_buffer["next_states"])
    agents.algorithm.replay_buffer["terminal"] = deepcopy(data_buffer["terminal"])

    """
    ------------------------------------------------------------------------------------
    offline RL
    ------------------------------------------------------------------------------------
    """

    
    # rewards storage
    accumulated_rewards = []
    ope_value = []
    optimal_r, alpha_best, pv_sum = optimal_R.calculate_optimal_R(vas.ranking_log[representation_index],
                                                                  representation_budget)
    print("R* = %f" % optimal_r)
    print("best_alpha = %f" % alpha_best)

    # offline updating for target policies
    eva_step = 0
    for epi in range(episode):
        # train 
        Q_loss, A_loss = agents.train()
        # update the target
        if Q_loss is not None and epi % target_update_epi == 0:
            agents.algorithm.update_target()

        # evaluation

        if epi % evaluation_interval == 0:
            """
            evaluation with RAS -- RPE
            """
            acc_rewards, store_actions_ = rpe.evaluate(agents, ras, initial_state, budget, store_ranking_log, len_step,
                                                    representation_index)
            accumulated_rewards.append(deepcopy(acc_rewards[representation_index]))
            latest_rewards.append(deepcopy(acc_rewards[representation_index]))
            if len(latest_rewards) > 1000:
                latest_rewards.pop(0)
                # check if abortion condition has been met
                if (max(latest_rewards) - min(latest_rewards)) / min(latest_rewards) < 0.05:
                    abortion_flag = True
            

            print("rewards: %f" % (accumulated_rewards[-1]))
            # draw the reward
            x = [i for i in range(eva_step + 1)]
            if eva_step > 0:
                draw_rewards(x, accumulated_rewards,
                            rewards_save_path+"random_seed_%d_iteration_%d.png" % (network_random_seed, iteration))
                pd.DataFrame(np.array(accumulated_rewards)).to_excel(
                    rewards_save_path+"random_seed_%d_iteration_%d.xlsx" % (network_random_seed, iteration))
                # draw_actions(store_actions[epi], epi,
                #              sum(accumulated_reward[test_index, step_cnt]) / len(test_index))

            """
            evaluate with VAS -- OPE
            """
            ope_reward, R_optimal_R_ratio = ope.evaluate(agents, initial_state_test, budget, vas, training_step_len,
                                                        "test_vas", representation_index, optimal_R=optimal_r)

            print("OPE reward: %f, R/R* : %f" % (ope_reward, R_optimal_R_ratio))

            ope_value.append(deepcopy(R_optimal_R_ratio))
            x = [i for i in range(eva_step + 1)]

            if eva_step > 0:
                draw_rewards(x, ope_value, rewards_save_path+"random_seed_%d_OPE.png" % network_random_seed)
                pd.DataFrame(np.array(ope_value)).to_excel(
                    rewards_save_path+"random_seed_%d_OPE.xlsx" % network_random_seed)
            
            eva_step += 1
            
        if abortion_flag:
            print("INFO: the training has been converge")
            print("algorithm name :" + algorithm_name + ", random seed: "+str(network_random_seed))
            break
    if saved_trained_net:
        print("saved")
        agents.algorithm.save_net(save_path)
