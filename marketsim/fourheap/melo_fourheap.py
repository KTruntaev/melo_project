from collections import deque, defaultdict
from typing import List
from marketsim.fourheap import constants
from marketsim.fourheap.fourheap import FourHeap
from marketsim.fourheap.order import Order, CancelledOrder, MatchedOrder
from marketsim.fourheap.order_queue import OrderQueue
import math
import copy
import numpy as np

class MELOFourHeap(FourHeap):
    """
    A four-heap data structure implementation for M-ELO (Midpoint Extended Limit Order) orders.
    
    M-ELO orders require a mandatory holding period before becoming eligible for matching.
    The structure maintains separate queues for orders in different states:
    - Eligibility queues: Orders waiting to first cross the midpoint.
    - Holding queues (``*_holding_queue``): Orders in the holding period. Holding completes
      linearly — an order here is never removed for a BBO shift; the only exits
      are completion (-> active) or agent withdrawal.
    - Active queues: Orders past the holding period, ordered by completion time.
      An active order executes only while on the correct side of the BBO
      (buy >= mid, sell <= mid); otherwise it waits in place, keeping its age,
      and is never demoted back to eligibility. Exits only via matching/withdrawal.
    - Matched queues: Orders that have executed. (buy_cancelled/sell_cancelled are
      retained for API compatibility but are no longer populated — nothing is
      cancelled for a BBO move.)
    
    Attributes:
        curr_best_bid (float): Current best bid price in the market
        curr_best_ask (float): Current best ask price in the market
        midpoint (float): Current midpoint price ((best_bid + best_ask) / 2)
        order_timestamps (dict): Mapping of order IDs to their timestamp
        time (int): Current time step
        holding_period (int): Required resting time before an order becomes active
    """

    def __init__(self, holding_period=50, plus_one=False):
        """
        Initialize the M-ELO four-heap structure.

        Args:
            holding_period (int): Mandatory waiting period before orders become eligible
        """
        super().__init__(plus_one)
        self.order_timestamps = {}
        self.holding_period = holding_period

        #cancelled_queues due to midpoint rise during holding
        self.buy_cancelled: List[(CancelledOrder, float)] = []
        self.sell_cancelled: List[(CancelledOrder, float)] = []

        #matched_queues
        self.buy_matched_orders: List[MatchedOrder] = []
        self.sell_matched_orders: List[MatchedOrder] = []

        #active queues
        self.buy_active_queue: deque[Order] = deque([])
        self.sell_active_queue: deque[Order] = deque([])

        #Holding period: orders rest here (with their entry time) until the hold elapses
        self.buy_holding_queue: deque[(Order, int)] = deque([])
        self.sell_holding_queue: deque[(Order, int)] = deque([])

        #Waiting for midpoint cross before entering the holding period (sorted by price)
        self.buy_eligibility_queue = OrderQueue(is_max_heap=True)
        self.sell_eligibility_queue = OrderQueue(is_max_heap=False)

        #TODO: DELETE AFTER. THIS IS JUST FOR DATA TRACKING
        self.removed_eligibility = 0
        self.removed_holding = 0
        self.removed_active = 0


    def insert(self, order: Order, order_tracker: dict, current_time: int, midprice: float):
        """
        Insert a new order into the appropriate queue based on its price relative to midpoint.
        
        Args:
            order (Order): The order to be inserted
            order_tracker (dict): Dictionary to track order status
            current_time (int): Current simulation time
        """
        self.agent_id_map[order.agent_id].append(order.order_id)

        #If crosses midpoint on placement
        if order.order_type == constants.BUY:
            if not math.isnan(midprice) and order.price >= midprice:
                self.buy_holding_queue.append((order, current_time))
                order_tracker[order.order_id] = "holding"
                return 1
            else:
                self.buy_eligibility_queue.add_order(order)
                order_tracker[order.order_id] = "eligibility"
                return 0
        else:
            if not math.isnan(midprice) and order.price <= midprice:
                self.sell_holding_queue.append((order, current_time))
                order_tracker[order.order_id] = "holding"
                return 1
            else:
                self.sell_eligibility_queue.add_order(order)
                order_tracker[order.order_id] = "eligibility"
                return 0
    
    def update_active_queue(self, curr_time, order_tracker):
        """
        Advance the holding period: move orders from the holding queue to the
        active queue once they have rested for ``holding_period``,
        in FIFO order.

        Holding periods complete **linearly**: once an order has entered the
        holding queue it is guaranteed to finish its hold and become active,
        regardless of any BBO/midpoint shifts in the meantime. Orders are never
        removed from the holding queue for crossing (or un-crossing) the BBO —
        the only exits from holding are completion (-> active) or an explicit
        agent withdrawal.
        """
        # print("CALLED AT THIS TIMESTEP", curr_time)
        while self.buy_holding_queue and curr_time - self.buy_holding_queue[0][1] >= self.holding_period:
            order = self.buy_holding_queue.popleft()
            self.buy_active_queue.append(order[0])
            order_tracker[order[0].order_id] = "active"
        
        while self.sell_holding_queue and curr_time - self.sell_holding_queue[0][1] >= self.holding_period:
            order = self.sell_holding_queue.popleft()
            self.sell_active_queue.append(order[0])
            order_tracker[order[0].order_id] = "active"

    def withdraw_all(self, agent_id: int, order_tracker):
        for order_id in self.agent_id_map[agent_id]:
            self.remove(order_id)
            order_tracker[order_id] += " withdrawn"
        self.agent_id_map[agent_id] = []

    def update_eligiblity_queue(self, curr_time, order_tracker, midprice):
        """
        Check orders in eligibility queues and move them to the holding queue if they cross midpoint.
        """
        order_moved = -1
        if not math.isnan(midprice):
            while True:
                peek_value = self.buy_eligibility_queue.peek()
                if peek_value == float('inf') or peek_value == float('-inf'):
                    break
                if peek_value >= midprice:
                    self.buy_holding_queue.append((self.buy_eligibility_queue.peek_order(), curr_time))
                    order_tracker[self.buy_eligibility_queue.peek_order().order_id] = "holding"
                    self.buy_eligibility_queue.remove(self.buy_eligibility_queue.peek_order_id())
                    order_moved = curr_time
                else:
                    break

            while True:
                peek_value = self.sell_eligibility_queue.peek()
                if peek_value == float('inf') or peek_value == float('-inf'):
                    break
                if peek_value <= midprice:
                    self.sell_holding_queue.append((self.sell_eligibility_queue.peek_order(), curr_time))
                    order_tracker[self.sell_eligibility_queue.peek_order().order_id] = "holding"
                    self.sell_eligibility_queue.remove(self.sell_eligibility_queue.peek_order_id())
                    order_moved = curr_time
                else:
                    break
        return order_moved

        
    def matching_orders(self, curr_time, order_tracker, midprice):
        """
        Match active M-ELO orders at the midpoint.

        An active order is *executable* only while it is on the correct side of
        the BBO — a buy priced at or above the midpoint, a sell at or below it.
        Among executable orders the oldest matches first (FIFO == holding-period
        completion order). Active orders on the wrong side of the BBO are **never
        removed**: they keep their position and age, and become matchable again
        if the midpoint moves back. Nothing is demoted to eligibility or
        cancelled for an adverse BBO move. Partial matches are supported; the
        remainder stays in place with its age.
        """
        #NOTE: The matched orders get split up by quantities so that the orders in the same indices of the matched_queue are matched.
        new_matched = [[], []]
        if math.isnan(midprice):
            return new_matched

        def oldest_executable(queue, is_buy):
            # Oldest active order (FIFO == holding-completion order) on the
            # correct side of the BBO. Non-crossing orders are skipped, not
            # removed, so they retain their age for future matching.
            for order in queue:
                crosses = order.price >= midprice if is_buy else order.price <= midprice
                if crosses:
                    return order
            return None

        while True:
            buy = oldest_executable(self.buy_active_queue, True)
            sell = oldest_executable(self.sell_active_queue, False)
            if buy is None or sell is None:
                break

            matched_quantity = min(buy.quantity, sell.quantity)
            if matched_quantity <= 0:
                break  # defensive: zero-quantity orders should never rest here

            buy_match_order = copy.deepcopy(buy)
            buy_match_order.quantity = matched_quantity
            order_tracker[buy.order_id] = "matched"
            new_matched[0].append(MatchedOrder(midprice, curr_time, buy_match_order))

            sell_match_order = copy.deepcopy(sell)
            sell_match_order.quantity = matched_quantity
            order_tracker[sell.order_id] = "matched"
            new_matched[1].append(MatchedOrder(midprice, curr_time, sell_match_order))

            buy.quantity -= matched_quantity
            sell.quantity -= matched_quantity
            # Fully-filled orders leave the book; partially-filled ones stay in
            # place (keeping their age); non-crossing orders are never touched.
            if buy.quantity == 0:
                self.buy_active_queue.remove(buy)
            if sell.quantity == 0:
                self.sell_active_queue.remove(sell)

        self.buy_matched_orders.extend(new_matched[0])
        self.sell_matched_orders.extend(new_matched[1])
        return new_matched

    def remove(self, order_id: int):
        for order in self.buy_active_queue:
            if order.order_id == order_id:
                self.buy_active_queue.remove(order)
                self.removed_active += 1
                return
        
        for order in self.sell_active_queue:
            if order.order_id == order_id:
                self.sell_active_queue.remove(order)
                self.removed_active += 1
                return
        
        for order_tuple in self.buy_holding_queue:
            if order_tuple[0].order_id == order_id:
                self.buy_holding_queue.remove(order_tuple)
                self.removed_holding += 1
                return
        
        for order_tuple in self.sell_holding_queue:
            if order_tuple[0].order_id == order_id:
                self.sell_holding_queue.remove(order_tuple)
                self.removed_holding += 1
                return
        
        if self.buy_eligibility_queue.contains(order_id):
            self.buy_eligibility_queue.remove(order_id)
            self.removed_eligibility += 1
            return
        
        if self.sell_eligibility_queue.contains(order_id):
            self.sell_eligibility_queue.remove(order_id)
            self.removed_eligibility += 1
            return

    def market_clear(self):
        matched_orders = self.buy_matched_orders + self.sell_matched_orders
        return matched_orders

    def observe(self) -> str:
        """
        Generate a string representation of the current heap state.
        
        Returns:
            str: Formatted string showing the state of all queues
        """
        s = '--------------\n'
        names = ['buy_matched', 'buy_unmatched', 'sell_matched', 'sell_unmatched']
        for i, heap in enumerate(self.heaps):
            s += names[i]
            s += '\n'
            # s += f'Top order_id: {heap.peek_order().order_id}\n'
            s += f'Top price: {abs(heap.peek())}\n'
            s += f'Number of orders: {heap.count()}\n\n\n'

        return s