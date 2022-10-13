from copy import deepcopy


class RPE:
    def __init__(self):
        """
        evaluate the policy with the RAS
        """
        pass

    def evaluate(self, agents, env, initial_state, budget, store_ranking_log, len_step, 
                representation_index, online_rate_threshold=0.6, mode="test"):
        # store actions and rewards
        store_actions = []
        acc_rewards = 0
        agents.reset(initial_state, budget)
        env.reset(budget, store_ranking_log=store_ranking_log)
        for step in range(len_step):
            actions = agents.take_actions(mode=mode, test_index=representation_index,
                                                        train_index=representation_index,
                                                        sorl_index=representation_index)
            try:
                store_actions.append(deepcopy(actions[representation_index].item()))
            except:
                print(representation_index)
                store_actions.append(deepcopy(actions[representation_index]))
            rewards, next_state, terminal, flag = env.next_state(agents.current_state, actions, step,
                                                                    store_ranking_log=store_ranking_log)
            
            acc_rewards += rewards
            agents.transitions_no_store(deepcopy(next_state))

            # check if all the campaigns have run out of budgets
            if flag:
                print("terminate in advance")
                break
            if actions[representation_index] < 0:
                print("terminate in advance")
                break
        print(store_actions)
        # punish if the online rate is smaller than a certain threshold
        print("online rate: " + str(step/(len_step-1)))
        if step / (len_step - 1) < online_rate_threshold:
            acc_rewards *= 0.1
        return acc_rewards, store_actions