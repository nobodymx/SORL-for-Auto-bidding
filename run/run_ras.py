import gin
import numpy as np
from sdk.Environment.RealAdvertisingSystem import RealAdvertisingSystem
from sdk.Agent.Agents import Agents
from sdk.Common.Utils import draw_rewards, draw_ROI, draw_actions
from sdk.Evaluation.RPE import RPE
import pandas as pd
from copy import deepcopy
import torch
import os



@gin.configurable
def run_ras(episode=100000,
            len_step=48,
            num_agent=100,
            dim_obs=3,
            test_mode="test",
            train_mode="train",
            target_update_epi=5,
            saved_trained_net=True,
            save_path="saved_models/RAS",
            test_flag=True,
            min_budget=10,
            max_budget=200,
            store_ranking_log=False,
            random_seed=1,
            evaluation_interval=50,
            gpu='0',
            draw_actions_flag=True,
            representation_budget=1500,
            representation_index=0):
    # assign GPU index
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # initial budget
    np.random.seed(random_seed)
    budget = np.random.randint(min_budget, max_budget, num_agent)
    budget[representation_index] = representation_budget
    # storage
    max_value = 0
    accumulated_reward = []
    store_actions = np.ones((episode, len_step)) * (-1)

    # initial env and agents
    ras = RealAdvertisingSystem()
    ras_test = RealAdvertisingSystem()
    agents = Agents()
    evaluation_epoch = 0

    # initial state
    initial_state = np.zeros((num_agent, dim_obs))
    initial_state[:, 0] = len_step
    initial_state[:, 1] = budget

    # evaluation 
    rpe = RPE()

    Q_loss = None

    for epi in range(episode):

        agents.reset(initial_state, budget)
        ras.reset(budget, store_ranking_log=False)
        """
        start episode --------------------------------------------
        """
        state = initial_state
        print("initial state:")
        print(agents.current_state[0])
        for step in range(len_step):
            print("epi: %d, step: %d" % (epi, step))
            agents.current_state = deepcopy(state)
            actions = agents.take_actions(mode=train_mode, train_index=representation_index)
            rewards, next_state, terminal, flag = ras.next_state(agents.current_state, actions, step,
                                                                 store_ranking_log=False,
                                                                 representation_index=representation_index)
            state = deepcopy(next_state)
            agents.transitions(rewards, next_state, terminal, 0, train_index=representation_index)
            temp_flag = flag

            if agents.algorithm.buffer_pointer % evaluation_interval == 0:
                # print("evaluation epoch: %d"%evaluation_epoch)
                print(agents.algorithm.buffer_pointer)
                # train
                Q_loss, A_loss = agents.train()
                # update the target
                if Q_loss is not None and evaluation_epoch %target_update_epi == 0:
                    agents.algorithm.update_target()

                # evaluation
                if (test_flag and (Q_loss is not None)) or (evaluation_epoch == 0):
                    """
                    start episode --------------------------------------------
                    """
                    """
                    evaluate with RAS -- RPE
                    """
                    acc_rewards, store_actions_ = rpe.evaluate(agents, ras, initial_state, budget, store_ranking_log, len_step, test_mode,
                                                              representation_index)
                    for i in range(len(store_actions_)):
                        store_actions[evaluation_epoch, i] = store_actions_[i]
                    accumulated_reward.append(deepcopy(acc_rewards[representation_index])) 

                   
                    if max_value < accumulated_reward[-1] :
                        max_value = accumulated_reward[-1]
                        if saved_trained_net:
                            agents.algorithm.save_net(save_path)

                    print("rewards: %f" % (accumulated_reward[-1]))
                    # draw the reward
                    x = [i for i in range(evaluation_epoch + 1)]

                    
                    if evaluation_epoch > 0:
                        draw_rewards(x, accumulated_reward, path="results/rewards/RAS/random_seed_%d.png"%random_seed)
                        pd.DataFrame(np.array(accumulated_reward)).to_excel("results/rewards/RAS/random_seed_%d.xlsx" % random_seed)
    
                        if draw_actions_flag:
                            draw_actions(store_actions[evaluation_epoch], evaluation_epoch,
                                        accumulated_reward[-1])

                    evaluation_epoch += 1
            # check if all the campaigns have run out of budgets
            if temp_flag and step < len_step - 1:
                print("terminate in advance")
                break
        # update the target
        if epi % target_update_epi == 0 and Q_loss is not None:
            agents.algorithm.update_target()
