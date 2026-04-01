"""
MELO Agent that can allocate trading between CDA and MELO markets.
"""
import random
import numpy as np
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL, CDA, MELO
from typing import List, Optional, Dict, Any, Tuple
import torch
import math


class MeloAgent(Agent):
    """
    MELO Agent that can trade in both CDA and MELO markets with specific allocation proportions.
    """
    
    def __init__(self, 
                agent_id: int,
                market: Market,
                meloMarket: Market = None,
                q_max: int = 10,
                pv_var: float = 5e6,
                shade: List = [0,0],
                cda_proportion: float = 0.5,
                melo_proportion: float = 0.5,
                order_quantity: int = 5):
        """
        Initialize a MELO agent.
        
        Args:
            agent_id: Agent ID
            market: CDA market instance
            meloMarket: MELO market instance
            q_max: Maximum inventory quantity
            pv_var: Private value variance
            cda_proportion: Proportion of trades allocated to CDA market
            melo_proportion: Proportion of trades allocated to MELO market
            order_quantity: Fixed quantity for orders
        """
        self.agent_id = agent_id
        self.market = market
        self.shade = shade
        self.meloMarket = meloMarket
        self.q_max = q_max
        self.pv_var = pv_var
        self.cda_proportion = cda_proportion
        self.melo_proportion = melo_proportion
        self.order_quantity = order_quantity
        
        # Initialize inventory and cash
        self.inventory = 0
        self.cash = 0
        self.melo_profit = 0
        
        # Track orders
        self.active_orders = {}
        
        #Rand generated on entry
        self.meloPV = 0
        self.position = 0
        self.meloPosition = 0
        self.melo_profit = 0
        self.melo_pv_history = []
        self.pv = PrivateValues(q_max, pv_var)

    def generate_pv(self):
        #Generate new private values
        # self.pv = PrivateValues(self.q_max, self.pv_var)
        pass

    def generate_melo_pv(self):
        self.meloPV = torch.randn(1) * torch.sqrt(torch.tensor(self.pv_var))
        # print("GENERATED PV is: ", self.meloPV)

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate
        # return estimate + np.random.normal(0, np.sqrt(3e5))

    def take_action(self, side: bool, market) -> List[Order]:
        t = self.market.get_time()
        self.generate_melo_pv()
        # print("ACTION WAS TAKEN AT TIME ",t, " by agent, ", self.agent_id, "at PV: ", self.meloPV )
        # if side == BUY:
        price = self.estimate_fundamental() + self.meloPV
            # price = midpoint + self.meloPv[0]
        # else:
        #     price = self.estimate_fundamental() + self.meloPV
            # price = midpoint - self.meloPv[0]
        
        if market == CDA:
            #print(f"MY shade is! {self.shade}")
            spread = self.shade[1] - self.shade[0]
            valuation_offset = spread*random.random() + self.shade[0]
            if side == BUY:
                price -= valuation_offset
            else:
                price += valuation_offset
        
        order = Order(  
            price=price.item(),
            quantity=5,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000) # TODO: change to a trackable id
        )
        return [order]

    def record_trade(self, side, quantity) -> None:
        # print("----------------------")
        # print("Trade was done at ",self.market.get_time(), " by agent, ", self.agent_id, "at PV: ", self.meloPV)
        # print("----------------------")
        if side == BUY:
            self.melo_pv_history.append((quantity,self.meloPV))
        else:
            self.melo_pv_history.append((-quantity, self.meloPV))

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.inventory)

    def reset(self):
        self.inventory = 0
        self.cash = 0
        self.pv = PrivateValues(self.q_max, self.pv_var)

    def receive_market_event(self, time: float, fundamental: float):
        """
        Process a market event.
        
        Args:
            time: Current time
            fundamental: Current fundamental value
        """
        # Cancel old orders if any
        self.cancel_orders()
        
        # Skip if inventory is at max in either direction
        if abs(self.inventory) >= self.q_max:
            return
        
        # Decide whether to send an order and to which market
        market_choice = random.random()
        
        # Sample private value
        pv = np.random.normal(0, np.sqrt(self.pv_var))
        
        # Set price as fundamental + private value
        price = fundamental + pv
        
        # Decide whether to buy or sell
        if self.inventory > 0:  # More likely to sell if inventory is positive
            is_buy = random.random() < 0.3
        elif self.inventory < 0:  # More likely to buy if inventory is negative
            is_buy = random.random() < 0.7
        else:  # Equal chance if inventory is zero
            is_buy = random.random() < 0.5
        
        direction = BUY if is_buy else SELL
        quantity = self.order_quantity
        
        # Choose market based on allocation proportions
        if market_choice < self.cda_proportion:
            # Send to CDA market
            if self.market is not None:
                order_id = self.market.submit_order(
                    agent_id=self.agent_id,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    time=time
                )
                self.active_orders[order_id] = {
                    "market": "CDA",
                    "direction": direction,
                    "price": price,
                    "quantity": quantity
                }
        elif market_choice < (self.cda_proportion + self.melo_proportion):
            # Send to MELO market
            if self.meloMarket is not None:
                order_id = self.meloMarket.submit_order(
                    agent_id=self.agent_id,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    time=time
                )
                self.active_orders[order_id] = {
                    "market": "MELO",
                    "direction": direction,
                    "price": price,
                    "quantity": quantity
                }
    
    def receive_execution(self, order_id: str, price: float, quantity: int, time: float):
        """
        Process a trade execution.
        
        Args:
            order_id: Order ID
            price: Execution price
            quantity: Executed quantity
            time: Execution time
        """
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            
            if order["direction"] == BUY:
                self.inventory += quantity
                self.cash -= price * quantity
            else:  # SELL
                self.inventory -= quantity
                self.cash += price * quantity
            
            # Remove from active orders if fully executed
            if quantity >= order["quantity"]:
                del self.active_orders[order_id]
            else:
                order["quantity"] -= quantity
    
    def receive_melo_execution(self, order_id: str, price: float, quantity: int, time: float):
        """
        Process a MELO trade execution.
        
        Args:
            order_id: Order ID
            price: Execution price
            quantity: Executed quantity
            time: Execution time
        """
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            
            if order["direction"] == BUY:
                self.inventory += quantity
                self.melo_profit -= price * quantity
            else:  # SELL
                self.inventory -= quantity
                self.melo_profit += price * quantity
            
            # Remove from active orders if fully executed
            if quantity >= order["quantity"]:
                del self.active_orders[order_id]
            else:
                order["quantity"] -= quantity
    
    def cancel_orders(self):
        """Cancel all active orders."""
        for order_id, order in list(self.active_orders.items()):
            if order["market"] == "CDA" and self.market is not None:
                self.market.cancel_order(order_id)
            elif order["market"] == "MELO" and self.meloMarket is not None:
                self.meloMarket.cancel_order(order_id)
        
        self.active_orders = {}
    
    def get_total_profit(self, fundamental: float):
        """
        Calculate total profit.
        
        Args:
            fundamental: Current fundamental value
            
        Returns:
            Total profit including mark-to-market value
        """
        mark_to_market = self.inventory * fundamental
        return self.cash + mark_to_market
    
    def get_melo_profit(self):
        """
        Get profit from MELO market.
        
        Returns:
            MELO market profit
        """
        return self.melo_profit

