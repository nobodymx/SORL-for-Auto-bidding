import gin
import numpy as np
from sdk.Environment.VirtualAdvertisingSystem import VirtualAdvertisingSystem
from sdk.Environment.RealAdvertisingSystem import RealAdvertisingSystem
from sdk.Agent.Agents import Agents
from sdk.Common.Utils import draw_rewards, draw_ROI
from sdk.Evaluation.RPE import RPE
from sdk.Evaluation.OPE import OPE
from sdk.Model.CalculateOptimalR import OptimalRCalculator
import pandas as pd
from copy import deepcopy
from sdk.Common.Utils import draw_rewards, draw_ROI, draw_actions
import os


@gin.configurable
def run_vas(episode=100000,
            collecting_data_episode=1,
            len_step=48,
            num_agent=100,
            dim_obs=3,
            test_mode="test",
            train_mode="train_vas",
            target_update_epi=5,
            saved_trained_net=True,
            save_path="saved_model/fixed_initial_budget_vbs",
            test_flag=True,
            min_budget=10,
            max_budget=200,
            store_ranking_log=False,
            network_random_seed=1,
            fixed_random_seed=0,
            evaluation_interval=50,
            gpu='0',
            representation_index=0,
            representation_budget=1500,
            pre_train_Q_iteration=100000,
            pre_train_Q_flag=False,
            algorithm_name="DDPG"):
    # specify GPU version
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # collecting stage 2 log
    print("start collecting log in stage 2")
    np.random.seed(fixed_random_seed)
    budget = np.random.randint(min_budget, max_budget, num_agent)
    budget[representation_index] = representation_budget
    ras = RealAdvertisingSystem()
    agents = Agents()
    # agents.algorithm.draw_q_value(path="results/network_value/initial_Q.png")

    # initial state
    initial_state = np.zeros((num_agent, dim_obs))
    initial_state[:, 0] = len_step
    initial_state[:, 1] = budget

    # accmulated rewards
    previous_rewards = []

    for epi in range(collecting_data_episode):

        agents.reset(initial_state, budget)
        ras.reset(budget)
        """
        start episode --------------------------------------------
        """
        acc = 0
        for step in range(len_step):
            print(step)
            # take actions
            actions = agents.take_actions(mode="test", test_index=representation_index)
            print(actions[0])
            # transit to next state
            rewards, next_state, terminal, flag = ras.next_state(agents.current_state, actions, step)
            print(rewards[0])
            acc += rewards[0]
            # check if all the campaigns have run out of budgets
            if flag:
                print("terminate in advance")
                break
            agents.transitions_no_store(next_state)
        print(acc)
        ras.store_ranking_log_after_one_episode()
    ras.store_ranking_log(data_path="virtual_advertising_stage_2_log")
    print("----------------------------------------------------")
    print("Finishing collecting log of stage 2")

    # training with the VAS
    # initial budget
    np.random.seed(fixed_random_seed)
    budget = np.random.randint(min_budget, max_budget, num_agent)
    budget[representation_index] = representation_budget
    # storage
    max_value = 0
    accumulated_reward = []
    ope_value = []
    # pretrain flag
    pretrain_Q_flag = pre_train_Q_flag

    # initial env and agents
    ras = RealAdvertisingSystem()
    vas = VirtualAdvertisingSystem(budget=budget)
    agents = Agents()
    step_cnt = 0

    store_actions = np.ones((episode, len_step)) * (-1)
    # initial state
    initial_state = np.zeros(dim_obs)
    initial_state[0] = len_step
    initial_state[1] = budget[representation_index]

    # initial state
    initial_state_test = np.zeros((num_agent, dim_obs))
    initial_state_test[:, 0] = len_step
    initial_state_test[:, 1] = budget

    Q_loss = None
    print("total time step %d" % vas.ranking_log[representation_index]["step_cnt"][0])
    training_step_len = vas.ranking_log[representation_index]["step_cnt"][0]

    # evalution method
    rpe = RPE()
    ope = OPE()
    optimal_R = OptimalRCalculator()
    optimal_r, alpha_best, pv_sum = optimal_R.calculate_optimal_R(vas.ranking_log[representation_index],
                                                                  representation_budget)
    print("R* = %f" % optimal_r)
    print("best_alpha = %f" % alpha_best)

    abortion_flag = False

    for epi in range(episode):
        agents.reset(initial_state, budget)

        """
        start episode --------------------------------------------
        """
        state = agents.current_state
        for step in range(training_step_len):
            print("epi: %d, step: %d" % (epi, step))
            agents.current_state = deepcopy(state)
            actions = agents.take_actions(mode=train_mode, train_index=representation_index)
            rewards, next_state, terminal, flag, values = vas.next_state(agents.current_state, actions, step,
                                                                         train_index=representation_index)
            # rewards = rewards / 10
            state = deepcopy(next_state)
            agents.transitions(rewards, next_state, terminal, values, train_index=representation_index,
                               mode="vas_train")
            temp_flag = flag
            if agents.algorithm.buffer_pointer % evaluation_interval == 0:
                print(agents.algorithm.buffer_pointer)
                # train 
                Q_loss, A_loss = agents.train()
                # update the target
                if Q_loss is not None and step_cnt % target_update_epi == 0:
                    agents.algorithm.update_target()
                if Q_loss is not None and pretrain_Q_flag:
                    agents.pre_train_Q_network(iteration=pre_train_Q_iteration)
                    pretrain_Q_flag = False
                print("--------------------------------------")
                # test
                if (test_flag and (Q_loss is not None)) or (step_cnt == 0):
                    """
                    evaluation with RAS -- RPE
                    """
                    acc_rewards, store_actions_ = rpe.evaluate(agents, ras, initial_state_test, budget,
                                                               store_ranking_log, len_step,
                                                               representation_index, )
                    for i in range(len(store_actions_)):
                        store_actions[step_cnt, i] = store_actions_[i]
                    accumulated_reward.append(deepcopy(acc_rewards[representation_index]))
                    previous_rewards.append(deepcopy(acc_rewards[representation_index]))
                    if len(previous_rewards) > 1000:
                        previous_rewards.pop(0)
                        # check if abortion condition has been met
                        if (max(previous_rewards) - min(previous_rewards)) / min(previous_rewards) < 0.05:
                            abortion_flag = True

                    print("rewards: %f" % (accumulated_reward[-1]))
                    # draw the reward

                    x = [i for i in range(step_cnt + 1)]

                    if step_cnt > 0:
                        draw_rewards(x, accumulated_reward, "results/rewards/VAS/"+algorithm_name+"/random_seed_%d.png" % network_random_seed)
                        pd.DataFrame(np.array(accumulated_reward)).to_excel(
                            "results/rewards/VAS/"+algorithm_name+"/random_seed_%d.xlsx" % network_random_seed)
                        # draw_actions(store_actions[epi], epi,
                        #              sum(accumulated_reward[test_index, step_cnt]) / len(test_index))

                    """
                    evaluate with VAS -- OPE
                    """
                    ope_reward, R_optimal_R_ratio = ope.evaluate(agents, initial_state, budget, vas, training_step_len,
                                                                 train_mode, representation_index, optimal_R=optimal_r)

                    print("OPE reward: %f, R/R* : %f" % (ope_reward, R_optimal_R_ratio))

                    ope_value.append(deepcopy(R_optimal_R_ratio))
                    x = [i for i in range(step_cnt + 1)]

                    if step_cnt > 0:
                        draw_rewards(x, ope_value, "results/rewards/VAS/"+algorithm_name+"/random_seed_%d_OPE.png" % network_random_seed)
                        pd.DataFrame(np.array(ope_value)).to_excel(
                            "results/rewards/VAS/"+algorithm_name+"/random_seed_%d_OPE.xlsx" % network_random_seed)

                    if max_value < R_optimal_R_ratio:
                        max_value = R_optimal_R_ratio
                        if saved_trained_net:
                            print("saved")
                            agents.algorithm.save_net(save_path)

                    step_cnt += 1

            # check if the campaign have budgets that is smaller than threshold
            if temp_flag and step < training_step_len - 1:
                print("terminate in advance")
                break
        if abortion_flag:
            print("INFO: training in VAS complete!")
            break
        # # update the target
        # if epi % target_update_epi == 0 and Q_loss is not None:
        #     agents.algorithm.update_target()
