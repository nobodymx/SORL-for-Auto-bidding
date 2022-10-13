import gin
import numpy as np
from sdk.Environment.RealAdvertisingSystem import RealAdvertisingSystem
from sdk.Environment.VirtualAdvertisingSystem import VirtualAdvertisingSystem
from sdk.Agent.Agents import Agents
from sdk.Common.Utils import draw_rewards, draw_ROI
import pandas as pd
from copy import deepcopy
import torch
from sdk.Model.OnlineExplorationPolicy import OnlineExplorationPolicy
from sdk.Evaluation.OPE import OPE
from sdk.Evaluation.RPE import RPE
from sdk.Model.CalculateOptimalR import OptimalRCalculator
import os



@gin.configurable
def run_evaluations(evaluation_episode=100000,
                    len_step=48,
                    representation_index=0,
                    representation_budget=1500,
                    num_agent=100,
                    dim_obs=3,
                    min_budget=10,
                    max_budget=200,
                    fixed_random_seed=1,
                    load_path="saved_models/VAS",
                    gpu='0',
                    take_action_mode="test",
                    ):
    # specify GPU version
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # initial budget
    np.random.seed(fixed_random_seed)
    budget = np.random.randint(min_budget, max_budget, num_agent)
    budget[representation_index] = representation_budget
    # storage
    rpe_rewards = np.zeros(evaluation_episode)
    ope_R_R_optimal = np.zeros(evaluation_episode)
    # store_actions = np.ones((episode, len_step)) * (-1)

    # initial env and agents
    ras = RealAdvertisingSystem()
    vas = VirtualAdvertisingSystem(budget=budget)
    agents = Agents()
    # load policy parameters to be evaluated
    model_device = "cpu" if gpu == "cpu" else "cuda:" + gpu
    agents.algorithm.load_net(load_path=load_path, device=model_device)
    # evalution class
    ope = OPE()
    rpe = RPE()
    optimal_R = OptimalRCalculator()
    optimal_r, alpha_best, pv_sum = optimal_R.calculate_optimal_R(vas.ranking_log[representation_index], representation_budget)
    print("R* = %f" % optimal_r)
    print("best_alpha = %f" % alpha_best)
    print("win pv sum = %f" % pv_sum)
    print("---------------------------------")

    # initial state
    initial_state = np.zeros((num_agent, dim_obs))
    initial_state[:, 0] = len_step
    initial_state[:, 1] = budget

    # initial state for OPE
    # initial state
    initial_state_OPE = np.zeros(dim_obs)
    initial_state_OPE[0] = len_step
    initial_state_OPE[1] = budget[representation_index]


    for epi in range(evaluation_episode):
        """
        start episode --------------------------------------------
        """
        agents.reset(initial_state, budget)
        ras.reset(budget, store_ranking_log=False)

        """
        evaluation with RAS -- RPE
        """
        acc_rewards, store_actions_ = rpe.evaluate(agents, ras, initial_state, budget, False, len_step,
                                                    representation_index, mode=take_action_mode)
        
        rpe_rewards[epi] = deepcopy(acc_rewards[representation_index])

        print("evaluation epi: %d, RPE rewards: %f" % (epi, rpe_rewards[epi]))
        # draw the reward
        x = [i for i in range(epi + 1)]

        # if epi > 0:
        #     draw_rewards(x, rpe_rewards, "results/rewards/evaluations/"+policy_type+"_random_seed_%d_RPE.png" % random_seed)
        #     pd.DataFrame(np.array(rpe_rewards)).to_excel("results/rewards/evalutions/"+policy_type+"_random_seed_%d_RPE.xlsx" % random_seed)
            

        """
        evaluate with VAS -- OPE
        """
        ope_reward, R_optimal_R_ratio = ope.evaluate(agents, initial_state_OPE, budget, vas, len_step, 
                                                        representation_index, optimal_R=optimal_r,
                                                        take_action_mode=take_action_mode)
        
        print("evaluation epi: %d, OPE rewards, %f, R/R* : %f" % (epi, ope_reward, R_optimal_R_ratio))
        ope_R_R_optimal[epi] = deepcopy(R_optimal_R_ratio)
        # draw the reward
        x = [i for i in range(epi + 1)]

        # if epi > 0:
        #     draw_rewards(x, ope_R_R_optimal, "results/rewards/evaluations/"+policy_type+"_random_seed_%d_OPE.png" % random_seed)
        #     pd.DataFrame(np.array(ope_R_R_optimal)).to_excel("results/rewards/evalutions/"+policy_type+"_random_seed_%d_OPE.xlsx" % random_seed)
            



