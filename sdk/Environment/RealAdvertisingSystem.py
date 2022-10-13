import os.path
import gin
import numpy as np
from copy import deepcopy
import time
import pandas as pd


@gin.configurable
class RealAdvertisingSystem:
    def __init__(self, num_of_campaign, reserve_price, T, fixed_random_seed, ranking_noise=0.00, pre_ranking_rate=0.5,
                 min_num=5, terminate_threshold=0.1,
                 min_pv_num=5, max_pv_num=50, market_price_discount=0.85):
        """
        Note that although this is not actually Real Advertising System (RAS), it can approximate the main parts of many RAS.
        It includes:
            --- stage 1 
            --- stage 2
        """
        self.num_of_campaign = num_of_campaign
        self.reserve_price = reserve_price
        self.T = T
        self.budget = 0
        self.ranking_noise = ranking_noise
        self.pre_ranking_rate = pre_ranking_rate
        self.min_num = min_num
        self.terminate_threshold = terminate_threshold
        self.min_pv_num = min_pv_num
        self.max_pv_num = max_pv_num
        self.market_price_discount = market_price_discount

        self.fixed_random_seed = fixed_random_seed
        np.random.seed(self.fixed_random_seed)

        # pv info

        self.pv_stream = np.random.randint(self.min_pv_num, self.max_pv_num, self.T)  # pv number
        self.num_of_pv = np.sum(self.pv_stream)

        # self.pv_value = [np.random.rand(self.pv_stream[i], self.num_of_campaign) for i in range(self.T)]
        self.pv_pre_ranking_value = [np.random.rand(self.pv_stream[i], self.num_of_campaign) for i in range(self.T)]
        self.pv_ranking_value = [np.random.rand(self.pv_stream[i], self.num_of_campaign) for i in range(self.T)]

        self.ranking_log = [{"campaign_id": i,
                             "episode_cnt": 0,
                             "step_cnt": 0,
                             "pv_index": [],
                             "pv_value": [],
                             "bid_price": [],
                             "market_price": [],
                             "is_win": [],
                             "budget_left": []
                             }
                            for i in range(self.num_of_campaign)]

        self.pv_index_cnt = 0

        self.store_market_price = []
        self.epi_cnt = -1

    def _increase_episode_empty_list(self):
        for i in range(self.num_of_campaign):
            self.ranking_log[i]["pv_index"].append([])
            self.ranking_log[i]["pv_value"].append([])
            self.ranking_log[i]["bid_price"].append([])
            self.ranking_log[i]["market_price"].append([])
            self.ranking_log[i]["is_win"].append([])
            self.ranking_log[i]["budget_left"].append([])

    def _increase_step_empty_list(self):
        for i in range(self.num_of_campaign):
            cnt = self.ranking_log[i]["episode_cnt"]
            self.ranking_log[i]["pv_index"][cnt].append([])
            self.ranking_log[i]["pv_value"][cnt].append([])
            self.ranking_log[i]["bid_price"][cnt].append([])
            self.ranking_log[i]["market_price"][cnt].append([])
            self.ranking_log[i]["is_win"][cnt].append([])

    def _increase_ranking_log_episode_cnt(self):
        for i in range(self.num_of_campaign):
            self.ranking_log[i]["episode_cnt"] += 1
            self.ranking_log[i]["step_cnt"] = 0

    def _increase_ranking_log_step_cnt(self):
        for i in range(self.num_of_campaign):
            self.ranking_log[i]["step_cnt"] += 1

    def store_ranking_log_after_one_episode(self):
        self._increase_ranking_log_episode_cnt()

    def _add_ranking_log(self, campaign_id, pv_index, pv_value, bid_price, market_price, is_win):
        epi_cnt = self.ranking_log[campaign_id]["episode_cnt"]
        step_cnt = self.ranking_log[campaign_id]["step_cnt"]
        self.ranking_log[campaign_id]["pv_index"][epi_cnt][step_cnt].append(deepcopy(pv_index))
        self.ranking_log[campaign_id]["pv_value"][epi_cnt][step_cnt].append(deepcopy(pv_value))
        self.ranking_log[campaign_id]["bid_price"][epi_cnt][step_cnt].append(deepcopy(bid_price))
        self.ranking_log[campaign_id]["market_price"][epi_cnt][step_cnt].append(deepcopy(market_price))
        self.ranking_log[campaign_id]["is_win"][epi_cnt][step_cnt].append(deepcopy(is_win))

    def _add_ranking_log_budget(self, campaign_id, budget_left):
        epi_cnt = self.ranking_log[campaign_id]["episode_cnt"]
        self.ranking_log[campaign_id]["budget_left"][epi_cnt].append(deepcopy(budget_left))

    def reset(self, budget, store_ranking_log=True):
        self.budget = budget
        self.pv_index_cnt = 0
        if store_ranking_log:
            self.store_market_price.append(deepcopy(np.zeros(sum(self.pv_stream))))
            self.epi_cnt += 1
            self._increase_episode_empty_list()
            if len(self.ranking_log[0]["pv_value"]) - 1 > self.ranking_log[0]["episode_cnt"]:
                self._increase_ranking_log_episode_cnt()

    def cost(self, bid_set, represent_index, time_step):
        """
        :param time_step:
        :param represent_index:
        :param bid_set: R^self.num_of_campaign
        :return:
        """
        x = np.zeros(self.num_of_campaign)
        winner = np.argmax(bid_set)
        x[winner] = 1
        virtual_bid_set = bid_set
        virtual_bid_set[winner] = -1
        cost = np.max(virtual_bid_set)
        values = x * self.pv_pre_ranking_value[time_step]
        return winner, cost, values[represent_index]

    def next_state(self, state, bid_set, time_step, store_ranking_log=True, representation_index=0):
        """
        :param state:
        :param bid_set:
        :param time_step:
        :return:
        """
        rewards = np.zeros(self.num_of_campaign)
        next_state = deepcopy(state)
        # time left update
        next_state[:, 0] -= 1
        bid_set_ = np.zeros(self.num_of_campaign)
        for i in range(self.num_of_campaign):
            if bid_set[i] < 0:
                bid_set_[i] = -1
            else:
                try:
                    bid_set_[i] = bid_set[i].item()
                except:
                    bid_set_[i] = bid_set[i]
        bid_set = deepcopy(bid_set_)
        if time_step < self.T - 1:
            terminal = np.ones(self.num_of_campaign)
        else:
            terminal = np.zeros(self.num_of_campaign)
        for i in range(self.num_of_campaign):
            if bid_set[i] < 0:
                terminal[i] = 0

        # increase ranking log space
        if store_ranking_log:
            self._increase_step_empty_list()

        for i in range(self.pv_stream[time_step]):

            x = np.zeros(self.num_of_campaign)

            # consider running out of budget
            """
            bidding for the whole chain
            """
            # c_bid = bid_set * self.pv_value[time_step][i]
            a_bid = deepcopy(bid_set)

            """
            pre-ranking stage
            """
            pre_ranking_ecpm = a_bid * self.pv_pre_ranking_value[time_step][i]
            # add some noise, approximate the complex pre-ranking models
            for j in range(len(pre_ranking_ecpm)):
                if pre_ranking_ecpm[j] > 0:
                    pre_ranking_ecpm[j] += np.random.randn() * self.ranking_noise
                    pre_ranking_ecpm[j] = max(pre_ranking_ecpm[j], 0.1)  # not smaller than 0
            # calculate sorting
            pre_ranking_result = self.pre_ranking_sort(pre_ranking_ecpm)
            if pre_ranking_result is None:
                print("INFO: no campaign into ranking stage")
                if store_ranking_log:
                    for cam in range(self.num_of_campaign):
                        self._add_ranking_log_budget(campaign_id=cam, budget_left=next_state[cam, 1])
                    self._increase_ranking_log_step_cnt()
                terminal = np.zeros(self.num_of_campaign)
                return rewards, next_state, terminal, True
            """
            ranking stage
            """
            ranking_ecpm = a_bid * self.pv_ranking_value[time_step][i]
            for k in range(len(ranking_ecpm)):
                if pre_ranking_result[k] > 0:
                    # add some noise, approximate the complex ranking models
                    ranking_ecpm[k] += np.random.randn() * self.ranking_noise
                    ranking_ecpm[k] = max(ranking_ecpm[k], 0.1)  # not smaller than 0
                else:
                    ranking_ecpm[k] = -1

            win_index, market_price, next_state = self.ranking_sort(deepcopy(ranking_ecpm), deepcopy(next_state))
            # IBOO: bid higher than market price in the VAS may not win, following a probability
            market_price *= self.market_price_discount

            
            if store_ranking_log:
                self.store_market_price[self.epi_cnt][self.pv_index_cnt] = market_price

            # print("winner for time step %d in pv %d -- %d" % (time_step, i, winner))
            x[win_index] = 1

            # store to the log
            if store_ranking_log:
                for l in range(self.num_of_campaign):
                    # only ranking log
                    if ranking_ecpm[l] > 0:
                        is_win = 1 if l == win_index else 0
                        self._add_ranking_log(l, self.pv_index_cnt, self.pv_ranking_value[time_step][i, l], bid_set[l],
                                              market_price, is_win)

            # cumulated rewards
            rewards += x * self.pv_ranking_value[time_step][i]

            # increase pv index cnt
            self.pv_index_cnt += 1

            # if all campaign have run out of budgets
            run_out_of_budget_cnt = 0
            for k in range(self.num_of_campaign):
                if next_state[k, 1] < self.terminate_threshold:
                    run_out_of_budget_cnt += 1
                    terminal[k] = 0
                    bid_set[k] = -1
            if run_out_of_budget_cnt >= self.num_of_campaign - 0.1:
                if store_ranking_log:
                    for cam in range(self.num_of_campaign):
                        self._add_ranking_log_budget(campaign_id=cam, budget_left=next_state[cam, 1])
                    self._increase_ranking_log_step_cnt()
                terminal = np.zeros(self.num_of_campaign)
                return rewards, next_state, terminal, True

        if store_ranking_log:
            for cam in range(self.num_of_campaign):
                self._add_ranking_log_budget(campaign_id=cam, budget_left=next_state[cam, 1])
            self._increase_ranking_log_step_cnt()
        # consumed budget
        for i in range(self.num_of_campaign):
            next_state[i, 2] = self.budget[i] - next_state[i, 1]
        
        if next_state[representation_index, 1]< self.terminate_threshold:
            return rewards, next_state, terminal, True
        else:
            return rewards, next_state, terminal, False

    def pre_ranking_sort(self, ecpm):
        sort_index = np.argsort(ecpm)

        # calculate active number
        active_num = 0
        for i in range(len(ecpm)):
            if ecpm[i] > 0:
                active_num += 1
        if active_num == 0:
            print("WARNING: no active campaign in pre_ranking")
            return None

        # calculate through number
        if active_num * self.pre_ranking_rate >= self.min_num:
            through_num = int(active_num * self.pre_ranking_rate)
        elif active_num >= self.min_num:
            through_num = self.min_num
        else:
            through_num = active_num

        mask = np.zeros(len(ecpm))
        for i in range(through_num):
            mask[sort_index[-(i + 1)]] = 1
        return mask

    def ranking_sort(self, ecpm, next_state):
        # flag_terminal = False
        # strict budget
        while True:
            win_index = np.argmax(ecpm)
            ecpm[win_index] = -1
            market_price = np.max(ecpm) + self.reserve_price
            if next_state[win_index, 1] < market_price:
                if sum(ecpm) <= -self.num_of_campaign + 0.1:
                    break
                else:
                    continue
            else:
                # reduce budget
                market_price = self.reserve_price if market_price < 0 else market_price
                next_state[win_index, 1] -= market_price
                break

        # soft budget
        # win_index = np.argmax(ecpm)
        # ecpm[win_index] = -1
        # market_price = np.max(ecpm) + self.reserve_price
        # market_price = self.reserve_price if market_price < 0 else market_price
        # next_state[win_index, 1] -= market_price

        return win_index, market_price, next_state

    def store_ranking_log(self, root_path="data/", data_path=None, store_epi=None):
        # check the root path
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if data_path is None:
            data_path = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
            if not os.path.exists(root_path + data_path):
                os.makedirs(root_path + data_path)

        for i in range(self.num_of_campaign):
            if not os.path.exists(root_path + data_path + "/campaign_" + str(i)):
                os.makedirs(root_path + data_path + "/campaign_" + str(i))
            # os.makedirs(root_path + current_time + "/campaign_" + str(i))
            epi_cnt = self.ranking_log[i]["episode_cnt"]
            if store_epi is None:
                for epi in range(epi_cnt):
                    data = {"pv_index": self.ranking_log[i]["pv_index"][epi],
                            "pv_value": self.ranking_log[i]["pv_value"][epi],
                            "bid_price": self.ranking_log[i]["bid_price"][epi],
                            "market_price": self.ranking_log[i]["market_price"][epi],
                            "is_win": self.ranking_log[i]["is_win"][epi],
                            "budget_left": self.ranking_log[i]["budget_left"][epi]}
                    df = pd.DataFrame(data)
                    # df.to_excel(root_path + current_time + "/campaign_" + str(i) + "/epi_" + str(epi) + ".xlsx")
                    df.to_excel(
                        root_path + data_path + "/campaign_" + str(i) + "/epi_" + str(epi) + ".xlsx")
            else:
                data = {"pv_index": self.ranking_log[i]["pv_index"][store_epi],
                        "pv_value": self.ranking_log[i]["pv_value"][store_epi],
                        "bid_price": self.ranking_log[i]["bid_price"][epi],
                        "market_price": self.ranking_log[i]["market_price"][store_epi],
                        "is_win": self.ranking_log[i]["is_win"][store_epi],
                        "budget_left": self.ranking_log[i]["budget_left"][store_epi]}
                df = pd.DataFrame(data)
                # df.to_excel(root_path + current_time + "/campaign_" + str(i) + "/epi_" + str(store_epi) + ".xlsx")
                df.to_excel(
                    root_path + data_path + "/campaign_" + str(i) + "/epi_" + str(store_epi) + ".xlsx")

        print("INFO: saved ranking log.")

    def store_market_price_fn(self, data_path=None):
        if data_path is None:
            data_path = "real_bidding_ranking_log"
        pd.DataFrame(np.array(self.store_market_price)).to_excel("data/" + data_path + "/market_price.xlsx")
