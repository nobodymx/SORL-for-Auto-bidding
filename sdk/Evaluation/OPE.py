

from statistics import mode


class OPE:
    def __init__(self):
        """
        evaluate the policy with the RAS
        """
        pass
    
    def evaluate(self, agents, initial_state, budget, env, training_step_len, 
                representation_index, optimal_R=1, take_action_mode="test"):
        
        agents.reset(initial_state, budget)

        """
        start episode --------------------------------------------
        """
        acc_rewards = 0
        for step in range(training_step_len):
            actions = agents.take_actions(mode=take_action_mode, test_index=representation_index,
                                            train_index=representation_index,
                                            sorl_index=representation_index)
            rewards, next_state, terminal, flag, value = env.next_state(agents.current_state, actions, step,
                                                                 train_index=representation_index)
            acc_rewards += rewards
            agents.transitions_no_store(next_state)
            # check if the campaign have budgets that is smaller than threshold
            if flag and step<training_step_len-1:
                print("terminate in advance")
                break
        R_optimal_R_ratio = acc_rewards / optimal_R
        return acc_rewards, R_optimal_R_ratio