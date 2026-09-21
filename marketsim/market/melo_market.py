from collections import defaultdict
from marketsim.event.event_queue import EventQueue
from marketsim.fourheap.fourheap import FourHeap
from marketsim.fundamental.fundamental_abc import Fundamental
from marketsim.market.market import Market
from marketsim.fourheap.melo_fourheap import MELOFourHeap

class MeloMarket(Market):
    def __init__(self, fundamental: Fundamental, time_steps, holding_period):
        super().__init__(fundamental, time_steps)
        self.holding_period = holding_period
        self.order_book = MELOFourHeap(holding_period)
        self.orders_rec = []
        self._step_count = 0

    def clear_market(self):
        new_orders = self.order_book.market_clear()
        self.matched_orders += new_orders
        return new_orders

    def withdraw_all(self, agent_id: int, order_tracker):
        self.order_book.withdraw_all(agent_id, order_tracker)

    def step(self, order_tracker, midprice):
        self._step_count += 1
        current_time = self.get_time()
        orders = self.event_queue.step()
        
        return_val = -1
        # Process new orders
        for order in orders:
            if order.quantity <= 0:
                continue
            order_placement = self.order_book.insert(order, order_tracker, current_time, midprice)
            if order_placement == 1:
                return_val = current_time

        return return_val
        
    def reset(self, fundamental=None):
        self.order_book = MELOFourHeap(self.holding_period)
        self.matched_orders = []
        self.event_queue = EventQueue()
        self._step_count = 0
        if fundamental is not None:
            self.fundamental = fundamental
