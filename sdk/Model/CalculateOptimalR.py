from cmath import cos
import numpy as np


class OptimalRCalculator:
    def __init__(self):
        pass

    def calculate_optimal_R(self, log, budget):
        """
        log: dict
        """
        pv_value = log["pv_value"][0]
        market_price = log["market_price"][0]
        step_count = log["step_cnt"][0]

        pv_value_all = []
        market_price_all = []
        for i in range(step_count):
            pv_value_all += pv_value[i]
            market_price_all += market_price[i]
        pv_value_all = np.array(pv_value_all)
        market_price_all = np.array(market_price_all)
        total_pv_num = len(market_price_all)


        buffer = np.zeros((total_pv_num, 4)) # cost performance, market price, value, budget
        buffer[:, 1] = market_price_all
        buffer[:, 2] = pv_value_all
        buffer[:, 3] = budget

        buffer[:, 0] = np.divide(buffer[:, 2], buffer[:, 1])
        buffer = buffer[buffer[:, 0].argsort()[::-1]]  # ranking according to the cost performance

        alpha_best = -1
        pv_sum = 0
        optimal_R = 0
        cost_sum = 0
        for i in range(total_pv_num):
            cost_perf, cost, value, budget = buffer[i]
            cost_sum += cost
            optimal_R += value
            pv_sum += 1
            if cost_sum > budget:
                cost_sum -= cost
                optimal_R -= value
                break
            alpha_best = 1 / cost_perf
        
        return optimal_R, alpha_best, pv_sum




        