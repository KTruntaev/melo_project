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
    - Eligibility queues: Orders waiting for midpoint cross
    - Activation queues: Orders that crossed midpoint but are in holding period
    - Active queues: Orders that completed holding period and are ready for matching
    - Matched/Cancelled queues: Orders that have been matched or cancelled
    
    Attributes:
        curr_best_bid (float): Current best bid price in the market
        curr_best_ask (float): Current best ask price in the market
        midpoint (float): Current midpoint price ((best_bid + best_ask) / 2)
        order_timestamps (dict): Mapping of order IDs to their timestamp
        time (int): Current time step
        holding_period (int): Required holding period before order activation
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

        #Pending wait time after activation
        self.buy_activation_queue: deque[(Order, int)] = deque([])
        self.sell_activation_queue: deque[(Order, int)] = deque([])

        #Waiting for midpoint cross for activation (sorted by price)
        self.buy_eligibility_queue = OrderQueue(is_max_heap=True)
        self.sell_eligibility_queue = OrderQueue(is_max_heap=False)

        #TODO: DELETE AFTER. THIS IS JUST FOR DATA TRACKING
        self.removed_eligibility = 0
        self.removed_activation = 0
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
                self.buy_activation_queue.append((order, current_time))
                order_tracker[order.order_id] = "activation"
                return 1
            else:
                self.buy_eligibility_queue.add_order(order)
                order_tracker[order.order_id] = "eligibility"
                return 0
        else:
            if not math.isnan(midprice) and order.price <= midprice:
                self.sell_activation_queue.append((order, current_time))
                order_tracker[order.order_id] = "activation"
                return 1
            else:
                self.sell_eligibility_queue.add_order(order)
                order_tracker[order.order_id] = "eligibility"
                return 0
    
    def update_active_queue(self, curr_time, order_tracker):
        """
        Move orders from activation queues to active queues if they've completed their holding period.
        Orders are processed in FIFO order within each queue.
        """
        # print("CALLED AT THIS TIMESTEP", curr_time)
        while self.buy_activation_queue and curr_time - self.buy_activation_queue[0][1] >= self.holding_period:
            order = self.buy_activation_queue.popleft()
            self.buy_active_queue.append(order[0])
            order_tracker[order[0].order_id] = "active"
        
        while self.sell_activation_queue and curr_time - self.sell_activation_queue[0][1] >= self.holding_period:
            order = self.sell_activation_queue.popleft()
            self.sell_active_queue.append(order[0])
            order_tracker[order[0].order_id] = "active"

    def withdraw_all(self, agent_id: int, order_tracker):
        for order_id in self.agent_id_map[agent_id]:
            self.remove(order_id)
            order_tracker[order_id] += " withdrawn"
        self.agent_id_map[agent_id] = []

    def update_eligiblity_queue(self, curr_time, order_tracker, midprice):
        """
        Check orders in eligibility queues and move them to activation queues if they cross midpoint.
        """
        order_moved = -1
        if not math.isnan(midprice):
            while True:
                peek_value = self.buy_eligibility_queue.peek()
                if peek_value == float('inf') or peek_value == float('-inf'):
                    break
                if peek_value >= midprice:
                    self.buy_activation_queue.append((self.buy_eligibility_queue.peek_order(), curr_time))
                    order_tracker[self.buy_eligibility_queue.peek_order().order_id] = "activation"
                    self.buy_eligibility_queue.remove(self.buy_eligibility_queue.peek_order_id())
                    order_moved = curr_time
                else:
                    break

            while True:
                peek_value = self.sell_eligibility_queue.peek()
                if peek_value == float('inf') or peek_value == float('-inf'):
                    break
                if peek_value <= midprice:
                    self.sell_activation_queue.append((self.sell_eligibility_queue.peek_order(), curr_time))
                    order_tracker[self.sell_eligibility_queue.peek_order().order_id] = "activation"
                    self.sell_eligibility_queue.remove(self.sell_eligibility_queue.peek_order_id())
                    order_moved = curr_time
                else:
                    break
        return order_moved

        
    def matching_orders(self, curr_time, order_tracker, midprice):
        """
        Match compatible orders from buy and sell active queues.
        
        Orders are matched at the midpoint price. Orders that would result in losses
        (buy orders above midpoint or sell orders below midpoint) are cancelled.
        Partial matches are supported, with remaining quantities staying in the queue.
        """
        #NOTE: The matched orders get split up by quantities so that the orders in the same indices of the matched_queue are matched.
        new_matched = [[], []]
        if not math.isnan(midprice):
            while self.buy_active_queue and self.sell_active_queue: 
                # Revalidate order queues before processing matches
                while self.buy_active_queue and midprice > self.buy_active_queue[0].price:
                    current_order = self.buy_active_queue.popleft()
                    self.buy_eligibility_queue.add_order(current_order)
                    #For tracking purposes
                    order_tracker[current_order.order_id] = "cancelled"
                    # TODO it doesn't get actually cancelled -> just moved to eligibility queue
                    self.buy_cancelled.append((current_order, midprice))

                while self.sell_active_queue and midprice < self.sell_active_queue[0].price:
                    current_order = self.sell_active_queue.popleft()
                    self.sell_eligibility_queue.add_order(current_order)
                    # TODO it doesn't get actually cancelled -> just moved to eligibility queue
                    self.sell_cancelled.append((current_order, midprice))
                    order_tracker[current_order.order_id] = "cancelled"

                # If after cancellations there are no matching orders, exit
                if not self.buy_active_queue or not self.sell_active_queue:
                    break

                buy_match_order = copy.deepcopy(self.buy_active_queue[0])
                order_tracker[buy_match_order.order_id] = "matched"
                sell_match_order = copy.deepcopy(self.sell_active_queue[0])
                order_tracker[sell_match_order.order_id] = "matched"
                matched_quantity = min(self.buy_active_queue[0].quantity, self.sell_active_queue[0].quantity)
                self.buy_active_queue[0].quantity -= matched_quantity
                self.sell_active_queue[0].quantity -= matched_quantity
                

                # Handle fully matched buy order
                if self.buy_active_queue[0].quantity == 0:
                    new_matched[0].append(MatchedOrder(midprice, curr_time, buy_match_order))
                    self.buy_active_queue.popleft()
                else:
                    buy_match_order.quantity = matched_quantity
                    new_matched[0].append(MatchedOrder(midprice, curr_time, buy_match_order))

                # Handle fully matched sell order
                if self.sell_active_queue[0].quantity == 0:
                    new_matched[1].append(MatchedOrder(midprice, curr_time, sell_match_order))
                    self.sell_active_queue.popleft()
                else:
                    sell_match_order.quantity = matched_quantity
                    new_matched[1].append(MatchedOrder(midprice, curr_time, sell_match_order))

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
        
        for order_tuple in self.buy_activation_queue:
            if order_tuple[0].order_id == order_id:
                self.buy_activation_queue.remove(order_tuple)
                self.removed_activation += 1
                return
        
        for order_tuple in self.sell_activation_queue:
            if order_tuple[0].order_id == order_id:
                self.sell_activation_queue.remove(order_tuple)
                self.removed_activation += 1
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