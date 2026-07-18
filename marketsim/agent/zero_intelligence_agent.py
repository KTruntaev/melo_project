import random
import numpy as np
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL, CDA, MELO
from typing import List
import torch
import math


class ZIAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, pv_var: float, eta: float = 1.0, cda_proportion=1,
                            melo_proportion=0, meloMarket=None, order_tracker=None):
        self.agent_id = agent_id
        self.market = market
        self.meloMarket = meloMarket
        self.order_tracker = order_tracker
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = PrivateValues(q_max, pv_var)
        #Rand generated on entry
        self.meloPV = 0
        self.position = 0
        self.meloPosition = 0
        self.shade = shade
        self.cash = 0
        self.melo_profit = 0
        self.eta = eta
        self.melo_trades = []
        # Standing MELO order (at most one; None when no live MELO order)
        self.melo_order = None

        #In case we want the agent to act different based on the market
        self.cda_proportion = cda_proportion
        self.melo_proportion = melo_proportion

    def generate_pv(self):
        #Generate new private values
        self.pv = PrivateValues(self.q_max, self.pv_var)

    def generate_melo_pv(self):
        pass

    def get_id(self) -> int:
        return self.agent_id

    # def noisy_obs(self):
    #     mean, r, T = self.market.get_info()
    #     t = self.market.get_time()
    #     val = self.market.get_fundamental_value()
    #     ot = val + np.random.normal(0,np.sqrt(self.obs_noise))

    #     rho_noisy = (1-r)**(t-self.prev_arrival_time)
    #     rho_var = rho_noisy ** 2

    #     prev_estimate = (1-rho_noisy)*mean + rho_noisy*self.prev_obs_mean
    #     prev_var =  rho_var * self.prev_obs_var + (1 - rho_var) / (1 - (1-r)**2) * int(self.market.fundamental.shock_std ** 2)

    #     curr_estimate = self.obs_noise / (self.obs_noise + prev_var) * prev_estimate + prev_var / (self.obs_noise + prev_var) * ot
    #     curr_var = self.obs_noise * prev_var / (self.obs_noise + prev_var)

    #     rho = (1-r)**(T-self.prev_arrival_time)

    #     self.prev_arrival_time = T
    #     self.prev_obs_mean = curr_estimate
    #     self.prev_obs_var = curr_var

        # return (1 - rho) * mean + rho * curr_estimate

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate
        # return estimate + np.random.normal(0, np.sqrt(3e5))

    def melo_order_has_min_surplus(self, estimate):
        # The standing order still achieves the minimum required surplus Rmin
        # (self.shade[0]) against the current fundamental estimate
        if self.melo_order.order_type == BUY:
            return self.melo_order.price <= estimate - self.shade[0]
        else:
            return self.melo_order.price >= estimate + self.shade[0]

    def withdraw_melo_order(self):
        if self.meloMarket is not None:
            self.meloMarket.withdraw_all(self.agent_id, self.order_tracker)
        self.melo_order = None

    def take_action(self, side, market = CDA, seed = 0):
        # Estimate new private, fundamental value
        estimate = self.estimate_fundamental()

        # Manage existing MELO order: keep it as long as it still achieves the
        # minimum surplus; withdraw it when redirected to CDA or the fundamental
        # estimate has moved against it
        if self.melo_order is not None:
            if market == CDA or not self.melo_order_has_min_surplus(estimate):
                self.withdraw_melo_order()
            else:
                return []

        # estimate = self.estimate_fundamental() + np.random.normal(0, np.sqrt(1e6))
        #print(f"MY ZI shade is! {self.shade}")

        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread*random.random() + self.shade[0]
        # a = self.pv.value_for_exchange(self.position, BUY)
        # b = self.pv.value_for_exchange(self.position, SELL)

        if side == BUY:
            price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
        else:
            price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset

        # print("ACTION WAS TAKEN AT TIME ",t, " by agent, ", self.agent_id, "at price: ", price, "on side: ", side)
        if self.eta != 1.0:
            if side == BUY:
                surplus = price - estimate
                best_price = self.market.order_book.get_best_ask()
                if (price - best_price) > self.eta*surplus and best_price != np.inf:
                    price = best_price
            else:
                surplus = estimate - price
                best_price = self.market.order_book.get_best_bid()
                if (best_price - price) > self.eta*surplus and best_price != np.inf:
                    price = best_price

        order = Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=self.market.get_time(),
            order_type=side,
            order_id=random.randint(1, 10000000)
        )
        # print(f'It is timestep {t} and I am assigned {side}. I am making an order with price {price} since my estimate is {estimate} ')
        # print(f'My current position is {self.position} and my private values are {self.pv.values}')

        if market == MELO:
            self.melo_order = order

        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p
        # A fill means the agent's single outstanding order executed, so any
        # cached MELO order is no longer live
        self.melo_order = None

    def __str__(self):
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.pv = PrivateValues(self.q_max, self.pv_var)
        self.melo_order = None
