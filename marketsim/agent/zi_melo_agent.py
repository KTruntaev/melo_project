"""
ZI (small-lot) strategic agent for the M-ELO/CDA simulator.

Subclass of :class:`~marketsim.agent.melo_agent.MeloAgent`. Supplies the ZI
*valuation logic*: position-based private values, price = estimate + marginal
PV ± shade with optional ``eta`` price-improvement, a keep-or-withdraw rule for
the standing M-ELO order (keep while it still clears R_min), and a
position-based final payoff.
"""
import random
import numpy as np
from typing import List
from marketsim.agent.melo_agent import MeloAgent
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL, CDA, MELO


class ZIMeloAgent(MeloAgent):
    role = "ZI"
    default_order_quantity = 1

    def generate_pv(self):
        # ZI's position-based private values are fixed per episode (set at
        # construction / reset); this regenerates them for external callers.
        self.pv = PrivateValues(self.q_max, self.pv_var)

    def take_action(self, side, market=CDA, seed=0) -> List[Order]:
        estimate = self.estimate_fundamental()

        # Manage existing M-ELO order: keep it while it still clears R_min;
        # withdraw when redirected to CDA or the estimate has moved against it.
        if self.melo_order is not None:
            self.melo_orders_encountered += 1
            if market == CDA:
                self.withdraw_melo_order(reason="redirected_to_cda")
            elif not self.melo_order_has_min_surplus(estimate):
                self.withdraw_melo_order(reason="min_surplus_violated")
            else:
                return []

        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread * random.random() + self.shade[0]

        if side == BUY:
            price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
        else:
            price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset

        if self.eta != 1.0:
            if side == BUY:
                surplus = price - estimate
                best_price = self.cdaMarket.order_book.get_best_ask()
                if (price - best_price) > self.eta * surplus and best_price != np.inf:
                    price = best_price
            else:
                surplus = estimate - price
                best_price = self.cdaMarket.order_book.get_best_bid()
                if (best_price - price) > self.eta * surplus and best_price != np.inf:
                    price = best_price

        order = Order(
            price=price,
            quantity=self.order_quantity,
            agent_id=self.get_id(),
            time=self.cdaMarket.get_time(),
            order_type=side,
            order_id=random.randint(1, 10000000),
        )

        if market == MELO:
            self.melo_order = order
            self.melo_orders_placed += 1

        return [order]

    def final_value(self, fundamental_val) -> float:
        # Position-based valuation: realized cash + marked inventory + the
        # private value of the resting position.
        return self.get_pos_value() + self.position * fundamental_val + self.cash
