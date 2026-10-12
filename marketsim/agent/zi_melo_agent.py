"""
Unified ZI / MOBI agent for the M-ELO simulator.

This class merges the two agents that previously lived in
``melo_agent.py`` (``MeloAgent`` — the MOBI role) and
``zero_intelligence_agent.py`` (``ZIAgent`` — the ZI role). A single
``role`` switch selects between the two behaviours; each branch reproduces the
original agent's logic verbatim so that EGTA payoffs remain reproducible.

Role differences (everything else is shared):

  * ``role="MOBI"`` (large-lot, slow arrival)
      - order quantity defaults to 5
      - private value is a fresh i.i.d. draw each arrival (``meloPV``) and is
        booked per fill into ``melo_pv_history``; payoff uses that history
      - price = fundamental estimate + meloPV; shade applied on CDA only
      - always withdraws and resubmits its standing M-ELO order
  * ``role="ZI"`` (small-lot, quick arrival)
      - order quantity defaults to 1
      - private value is position-based (``PrivateValues``); payoff uses
        ``get_pos_value()``
      - price = estimate + marginal PV +/- shade, with optional ``eta`` shading
      - manages its standing M-ELO order (keeps while it still clears R_min)
        and records placement/removal tracking

Arrival *rate* is NOT a property of this class: the simulator schedules
MOBI vs ZI wake-ups from separate streams keyed by ``self.role`` /
``agent_roles`` (``lam_melo_mobi`` vs ``lam_melo_zi``).
"""
import random
import numpy as np
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL, CDA, MELO
from typing import List
import torch

MOBI = "MOBI"
ZI = "ZI"


class ZIMeloAgent(Agent):
    def __init__(self,
                 agent_id: int,
                 market: Market,
                 role: str = ZI,
                 meloMarket: Market = None,
                 order_tracker=None,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade: List = None,
                 cda_proportion: float = 0.5,
                 melo_proportion: float = 0.5,
                 order_quantity: int = None,
                 eta: float = 1.0):
        role = role.upper()
        if role not in (MOBI, ZI):
            raise ValueError(f"role must be 'MOBI' or 'ZI', got {role!r}")
        self.role = role

        self.agent_id = agent_id
        self.market = market
        self.meloMarket = meloMarket
        self.order_tracker = order_tracker
        self.q_max = q_max
        self.pv_var = pv_var
        self.shade = shade if shade is not None else [0, 0]
        self.eta = eta

        # Order size: large lot for MOBI, single unit for ZI, unless overridden.
        if order_quantity is None:
            order_quantity = 5 if role == MOBI else 1
        self.order_quantity = order_quantity

        # In case the agent is told to split activity between the two venues.
        self.cda_proportion = cda_proportion
        self.melo_proportion = melo_proportion

        # Shared position/cash bookkeeping (payoff uses position*fund + cash).
        self.position = 0
        self.cash = 0
        self.melo_profit = 0

        # Position-based private values (used for ZI pricing/payoff; harmless
        # to construct for MOBI, whose payoff uses melo_pv_history instead).
        self.pv = PrivateValues(q_max, pv_var)

        # Standing M-ELO order (at most one live at a time; None when none out).
        self.melo_order = None

        # --- MOBI private-value state ---
        # Fresh i.i.d. draw each arrival; melo_pv_history books (+/-qty, pv) per
        # fill so end_sim can value MOBI trades at the PV seen when traded.
        self.meloPV = 0
        self.melo_pv_history = []

        # --- ZI M-ELO order placement/removal tracking (testing/debugging) ---
        self.melo_orders_placed = 0
        self.melo_orders_removed = 0
        # Times take_action saw a standing M-ELO order on arrival (the case the
        # keep/withdraw logic acts on). kept = encountered - removed.
        self.melo_orders_encountered = 0
        # Each entry: (time, reason, queue_status_at_removal). reason is one of
        # "redirected_to_cda", "min_surplus_violated". queue_status is whatever
        # order_tracker recorded for the order just before withdrawal
        # ("eligibility", "activation", "active", or None if unknown).
        self.melo_removal_log = []

    # ------------------------------------------------------------------
    # Identity / shared helpers
    # ------------------------------------------------------------------
    def get_id(self) -> int:
        return self.agent_id

    def __str__(self):
        return f'ZI{self.agent_id}'

    def generate_pv(self):
        # Regenerate position-based private values. MOBI never relied on this
        # (its original was a no-op); kept here for the ZI role and external
        # callers that refresh PVs between episodes.
        if self.role == ZI:
            self.pv = PrivateValues(self.q_max, self.pv_var)

    def generate_melo_pv(self):
        # MOBI draws a fresh scalar private value on each arrival; ZI does not.
        if self.role == MOBI:
            self.meloPV = torch.randn(1) * torch.sqrt(torch.tensor(self.pv_var))

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1 - r) ** (T - t)

        estimate = (1 - rho) * mean + rho * val
        return estimate

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    # ------------------------------------------------------------------
    # M-ELO standing-order management
    # ------------------------------------------------------------------
    def melo_order_has_min_surplus(self, estimate):
        # The standing order still achieves the minimum required surplus Rmin
        # (self.shade[0]) against the current fundamental estimate.
        if self.melo_order.order_type == BUY:
            has_surplus = self.melo_order.price <= estimate - self.shade[0]
        else:
            has_surplus = self.melo_order.price >= estimate + self.shade[0]

        if not has_surplus:
            side_str = "BUY" if self.melo_order.order_type == BUY else "SELL"
            print(f"Removed order type {side_str} with price {self.melo_order.price} "
                  f"because private value is {estimate} and rmin is {self.shade[0]}")

        return has_surplus

    def withdraw_melo_order(self, reason="unspecified"):
        # Look up which queue the order sat in (eligibility/activation/active/
        # other) before withdrawal, for the removal log below. Only ZI agents
        # carry an order_tracker; MOBI passes None, in which case (matching the
        # original MeloAgent) the market withdrawal is a no-op and the simulator
        # is responsible for clearing the book before take_action.
        queue_status = None
        if self.order_tracker is not None and self.melo_order is not None:
            queue_status = self.order_tracker.get(self.melo_order.order_id)

        if self.meloMarket is not None and self.order_tracker is not None:
            self.meloMarket.withdraw_all(self.agent_id, self.order_tracker)

        self.melo_orders_removed += 1
        self.melo_removal_log.append((self.market.get_time(), reason, queue_status))
        self.melo_order = None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def take_action(self, side, market=CDA, seed=0) -> List[Order]:
        if self.role == MOBI:
            return self._take_action_mobi(side, market)
        return self._take_action_zi(side, market)

    def _take_action_mobi(self, side, market) -> List[Order]:
        t = self.market.get_time()
        # Arrival process: withdraw any standing M-ELO order before resubmitting.
        # MOBI agents have no keep logic (paper §3.1).
        if self.melo_order is not None:
            self.withdraw_melo_order()
        self.generate_melo_pv()

        price = self.estimate_fundamental() + self.meloPV

        if market == CDA:
            spread = self.shade[1] - self.shade[0]
            valuation_offset = spread * random.random() + self.shade[0]
            if side == BUY:
                price -= valuation_offset
            else:
                price += valuation_offset

        order = Order(
            price=price.item(),
            quantity=self.order_quantity,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000),  # TODO: change to a trackable id
        )
        if market == MELO:
            self.melo_order = order
        return [order]

    def _take_action_zi(self, side, market) -> List[Order]:
        # Estimate new private, fundamental value.
        estimate = self.estimate_fundamental()

        # Manage existing M-ELO order: keep it as long as it still achieves the
        # minimum surplus; withdraw it when redirected to CDA or the fundamental
        # estimate has moved against it.
        if self.melo_order is not None:
            self.melo_orders_encountered += 1
            if market == CDA:
                side_str = "BUY" if self.melo_order.order_type == BUY else "SELL"
                print(f"Removed order type {side_str} with price {self.melo_order.price} "
                      f"because agent was redirected to the CDA market")
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
                best_price = self.market.order_book.get_best_ask()
                if (price - best_price) > self.eta * surplus and best_price != np.inf:
                    price = best_price
            else:
                surplus = estimate - price
                best_price = self.market.order_book.get_best_bid()
                if (best_price - price) > self.eta * surplus and best_price != np.inf:
                    price = best_price

        order = Order(
            price=price,
            quantity=self.order_quantity,
            agent_id=self.get_id(),
            time=self.market.get_time(),
            order_type=side,
            order_id=random.randint(1, 10000000),
        )

        if market == MELO:
            self.melo_order = order
            self.melo_orders_placed += 1

        return [order]

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------
    def record_trade(self, side, quantity) -> None:
        # MOBI books the private value seen at trade time so end_sim can value
        # the trade. ZI payoff is position-based (get_pos_value) and records
        # nothing here. Only MOBI should have this exercised by the simulator.
        if self.role != MOBI:
            return
        if side == BUY:
            self.melo_pv_history.append((quantity, self.meloPV))
        else:
            self.melo_pv_history.append((-quantity, self.meloPV))

    def update_position(self, q, p):
        self.position += q
        self.cash += p
        # A fill means the agent's single outstanding order executed, so any
        # cached M-ELO order is no longer live.
        self.melo_order = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.position = 0
        self.cash = 0
        self.melo_profit = 0
        self.pv = PrivateValues(self.q_max, self.pv_var)
        self.melo_order = None
        # MOBI PV state
        self.meloPV = 0
        self.melo_pv_history = []
        # ZI tracking
        self.melo_orders_placed = 0
        self.melo_orders_removed = 0
        self.melo_orders_encountered = 0
        self.melo_removal_log = []
