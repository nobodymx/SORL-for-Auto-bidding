import os.path
from copy import deepcopy
import gin
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@gin.configurable
class VirtualAdvertisingSystem:
    def __init__(self, path="data/", log_name="virtual_advertising_stage_2_log",
                 num_campaign=100, num_epi=1, num_step=48, budget=None,
                 terminate_threshold=0.1, reserve_price=0.01):
        self.num_campaign = num_campaign
        self.path = path
        self.log_name = log_name
        self.ranking_log = []
        self.num_epi = num_epi
        self.T = num_step
        self.budget = budget
        self.terminate_threshold = terminate_threshold
        self.reserve_price = reserve_price

        self.ranking_log = [{"campaign_id": i,
                             "episode_cnt": 0,
                             "step_cnt": [],
                             "pv_index": [],
                             "pv_value": [],
                             "market_price": [],
                             "is_win": []
                             }
                            for i in range(self.num_campaign)]

        if os.path.exists(self.path + self.log_name):
            for i in range(self.num_campaign):
                for epi in range(self.num_epi):
                    campaign_data = pd.read_excel(self.path + self.log_name + '/campaign_' + str(i) + "/epi_" + str(epi) + ".xlsx", sheet_name="Sheet1")
                    self.ranking_log[i]["episode_cnt"] += 1
                    self.ranking_log[i]["pv_index"].append([])
                    self.ranking_log[i]["pv_value"].append([])
                    self.ranking_log[i]["market_price"].append([])
                    self.ranking_log[i]["is_win"].append([])
                    self.ranking_log[i]["step_cnt"].append(deepcopy(len(campaign_data)))
                    for step in range(len(campaign_data)):
                        self.ranking_log[i]["pv_index"][epi].append(deepcopy(eval(campaign_data["pv_index"][step])))
                        self.ranking_log[i]["pv_value"][epi].append(deepcopy(eval(campaign_data["pv_value"][step])))
                        self.ranking_log[i]["market_price"][epi].append(deepcopy(eval(campaign_data["market_price"][step])))
                        self.ranking_log[i]["is_win"][epi].append(deepcopy(eval(campaign_data["is_win"][step])))

        else:
            print("ERROR: there is no data path")

        self.epi_use = 0

    def draw_pv_pool(self, campaign_id=0, epi=0, step=0, pv_num=47, policy_prove_time=10, displace_dist=100):
        ax = plt.subplot(111)
        involve_pv = self.ranking_log[campaign_id]["pv_index"][epi][step]
        involve_pv = [i - displace_dist for i in involve_pv]
        data = np.zeros((policy_prove_time, pv_num))
        data[:, involve_pv] = 1

        cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=.8, dark=0.45, )
        # sns.palplot(sns.cubehelix_palette(8, start=2, rot=0.05, reverse=True))
        g = sns.heatmap(data=data, linewidths=1, cmap=cmap)
        g.xaxis.set_ticks_position('top')
        g.set_xticklabels([str(i + 1) for i in range(pv_num)])
        plt.savefig("results/virtual_pv_pool.png", dpi=400)
        plt.show()

    def draw_real_pv_pool(self, campaign_id=0, step=0, pv_num=47, displace_dist=100, epi_list=None):
        ax = plt.subplot(111)
        data = np.zeros((len(epi_list), pv_num))
        cnt = 0
        for epi in epi_list:
            involve_pv = self.ranking_log[campaign_id]["pv_index"][epi][step]
            involve_pv = [i - displace_dist for i in involve_pv]
            data[cnt, involve_pv] = 1
            cnt += 1

        cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=.8, dark=0.45, )
        # sns.palplot(sns.cubehelix_palette(8, start=2, rot=0.05, reverse=True))
        g = sns.heatmap(data=data, linewidths=1, cmap=cmap)
        g.xaxis.set_ticks_position('top')
        g.set_xticklabels([str(i + displace_dist) for i in range(pv_num)], rotation=90)
        g.set_yticklabels(epi_list, rotation=90)
        plt.xlabel("PV Index")
        plt.ylabel("Training Step")
        plt.savefig("results/real_pv_pool.png", dpi=400)
        plt.show()

    def set_log_epi_use(self, log_epi_use):
        self.epi_use = log_epi_use

    def next_state(self, state, bid, time_step, train_index=None):
        """
        :param train_index:
        :param state:
        :param bid_set:
        :param time_step:
        :return:
        """
        rewards = 0
        next_state = deepcopy(state)
        # time left update
        next_state[0] -= 1
        bid_ = 0
        if bid < 0:
            bid_ = -1
        else:
            try:
                bid_ = bid.item()
            except:
                bid_ = bid
        bid = deepcopy(bid_)
        if time_step < self.ranking_log[train_index]["step_cnt"][0] - 1:
            terminal = 1
        else:
            terminal = 0
        
        if bid < 0:
            terminal = 0
            return rewards, next_state, terminal, True, 0
        
        # each pv
        num_pv = len(self.ranking_log[train_index]["pv_index"][self.epi_use][time_step])
        
        for pv in range(num_pv):
            pv_value = self.ranking_log[train_index]["pv_value"][self.epi_use][time_step][pv]
            c_bid = bid * pv_value
            market_price = self.ranking_log[train_index]["market_price"][self.epi_use][time_step][pv]
            if c_bid > (market_price-self.reserve_price) and next_state[1]-market_price>0:
                rewards += pv_value
                next_state[1] -= market_price
            
        if next_state[1] < self.terminate_threshold:
            terminal = 0
        next_state[2] = self.budget[train_index] - next_state[1]
        if len(self.ranking_log[train_index]["pv_index"][self.epi_use]) - 1 <= time_step:
            terminal = 0

        flag_terminate = True if terminal <= 0 else False

        value = self.calculate_value(bid, time_step, rewards, train_index, next_state[1])
        # print(value)
        return rewards, next_state, terminal, flag_terminate, value
    
    def calculate_value(self, bid, time_step, rewards, train_index, budget_left):
        value = rewards
        for t in range(time_step+1, self.ranking_log[train_index]["step_cnt"][self.epi_use]):
            # each pv
            num_pv = len(self.ranking_log[train_index]["pv_index"][self.epi_use][t])
            
            for pv in range(num_pv):
                pv_value = self.ranking_log[train_index]["pv_value"][self.epi_use][t][pv]
                c_bid = bid * pv_value
                market_price = self.ranking_log[train_index]["market_price"][self.epi_use][t][pv]
                if c_bid > (market_price-self.reserve_price) and budget_left-market_price>0:
                    value += pv_value
                    budget_left -= market_price
                
            if budget_left < self.terminate_threshold:
                break
        return value
            