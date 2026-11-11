"""
MOBI (large-lot) strategic agent for the M-ELO/CDA simulator.

Subclass of :class:`~marketsim.agent.melo_agent.MeloAgent`. Supplies the MOBI
*valuation logic*: a fresh scalar private value (``meloPV``) drawn each arrival
and booked per fill into ``melo_pv_history``; price = estimate + meloPV (shade
on CDA only); no keep logic (always resubmits its standing M-ELO order); payoff
values each trade at the PV seen when it traded.
"""
import random
import torch
from typing import List
from marketsim.agent.melo_agent import MeloAgent
from marketsim.fourheap.order import Order
from marketsim.fourheap.constants import BUY, SELL, CDA, MELO


class MobiMeloAgent(MeloAgent):
    role = "MOBI"
    default_order_quantity = 5

    def _init_role_state(self):
        # Fresh i.i.d. scalar PV per arrival; melo_pv_history books (±qty, pv)
        # per fill so final_value can value each trade at the PV seen when traded.
        self.meloPV = 0
        self.melo_pv_history = []

    def _reset_role_state(self):
        self.meloPV = 0
        self.melo_pv_history = []

    def generate_pv(self):
        self.meloPV = torch.randn(1) * torch.sqrt(torch.tensor(self.pv_var))

    def take_action(self, side, market=CDA, seed=0) -> List[Order]:
        t = self.cdaMarket.get_time()
        # Arrival process: drop any standing M-ELO order before resubmitting.
        # MOBI has no keep logic (paper §3.1).
        if self.melo_order is not None:
            self.withdraw_melo_order(reason="mobi_resubmit")
        self.generate_pv()

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
        # Cache the standing order only while it lives in M-ELO; a CDA order
        # leaves no standing M-ELO order (the simulator clears the book).
        self.melo_order = order if market == MELO else None
        return [order]

    def record_trade(self, side, quantity) -> None:
        # Book the private value seen at trade time so final_value can value it.
        if side == BUY:
            self.melo_pv_history.append((quantity, self.meloPV))
        else:
            self.melo_pv_history.append((-quantity, self.meloPV))

    def final_value(self, fundamental_val) -> float:
        return (self.position * fundamental_val + self.cash
                + sum(quantity * value for quantity, value in self.melo_pv_history))
