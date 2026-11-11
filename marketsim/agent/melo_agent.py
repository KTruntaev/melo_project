"""
Shared parent for the two strategic M-ELO/CDA agent types.

``MeloAgent`` holds everything the MOBI and ZI roles have in common — identity,
the fundamental-value estimate, position/cash bookkeeping, the per-arrival
``act()`` scaffolding, and the standing M-ELO order lifecycle (keep/withdraw
bookkeeping). The two subclasses supply the *valuation logic* that distinguishes
the roles:

  * :class:`~marketsim.agent.mobi_melo_agent.MobiMeloAgent` (MOBI, large-lot):
    a fresh scalar private value (``meloPV``) each arrival, booked per fill into
    ``melo_pv_history``; price = estimate + meloPV (shade on CDA only); always
    resubmits; payoff values each trade at the PV seen when it traded.
  * :class:`~marketsim.agent.zi_melo_agent.ZIMeloAgent` (ZI, small-lot):
    position-based ``PrivateValues``; price = estimate + marginal PV ± shade
    (optional ``eta`` shading); keeps a standing M-ELO order while it still
    clears R_min; payoff uses ``get_pos_value()``.

Both subclasses are plain ``MeloAgent`` instances to the simulator, which keeps
them together in a single arrivals/agents list and routes wake-ups by role via
``agent_roles`` (``lam_melo_mobi`` vs ``lam_melo_zi``).
"""
import random
from typing import List
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL, CDA, MELO


class MeloAgent(Agent):
    #: Role tag ("MOBI"/"ZI"); set by each subclass. The simulator reads it to
    #: pick the arrival stream and to gate role-specific booking.
    role = None
    #: Default order size when the caller does not override order_quantity.
    default_order_quantity = 1

    def __init__(self,
                 agent_id: int,
                 cdaMarket: Market,
                 meloMarket: Market = None,
                 melo_order_tracker=None,
                 q_max: int = 10,                 # Max position size
                 pv_var: float = 5e6,             # Variance used to draw private values
                 shade: List = None,
                 cda_proportion: float = 0.5,
                 melo_proportion: float = 0.5,
                 order_quantity: int = None,
                 eta: float = 1.0):
        self.agent_id = agent_id
        self.cdaMarket = cdaMarket
        self.meloMarket = meloMarket
        self.melo_order_tracker = melo_order_tracker
        self.q_max = q_max
        self.pv_var = pv_var
        self.shade = shade if shade is not None else [0, 0]
        self.eta = eta
        self.cda_proportion = cda_proportion
        self.melo_proportion = melo_proportion
        self.order_quantity = (order_quantity if order_quantity is not None
                               else self.default_order_quantity)

        # Shared position/cash bookkeeping (payoff uses position*fund + cash).
        self.position = 0
        self.cash = 0

        # Position-based private values. Constructed for BOTH roles (MOBI's
        # payoff uses meloPV instead, but the draw must still happen to keep the
        # RNG stream — and therefore payoffs — identical across roles).
        self.pv = PrivateValues(q_max, pv_var)

        # Standing M-ELO order (at most one live at a time; None when none out).
        self.melo_order = None

        # M-ELO order placement/removal tracking. withdraw_melo_order updates
        # these for either role; end_sim only aggregates them for ZI.
        self.melo_orders_placed = 0
        self.melo_orders_removed = 0
        self.melo_orders_encountered = 0   # standing-order arrivals (kept = encountered - removed)
        self.melo_removal_log = []         # (time, reason, queue_status_at_removal)

        self._init_role_state()

    # ------------------------------------------------------------------
    # Hooks the subclasses fill in
    # ------------------------------------------------------------------
    def _init_role_state(self):
        """Set up any role-specific private-value state (no-op by default)."""

    def _reset_role_state(self):
        """Reset any role-specific private-value state (no-op by default)."""

    def generate_pv(self):
        """Refresh this agent's private value(s) on arrival (role-specific)."""
        raise NotImplementedError

    def take_action(self, side, market=CDA, seed=0) -> List:
        """Price and build this arrival's order(s) (role-specific)."""
        raise NotImplementedError

    def final_value(self, fundamental_val) -> float:
        """Total end-of-simulation payoff for this agent (role-specific)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Identity / shared helpers
    # ------------------------------------------------------------------
    def get_id(self) -> int:
        return self.agent_id

    def __str__(self):
        return f'{self.role}{self.agent_id}'

    def estimate_fundamental(self):
        mean, r, T = self.cdaMarket.get_info()
        t = self.cdaMarket.get_time()
        val = self.cdaMarket.get_fundamental_value()
        rho = (1 - r) ** (T - t)
        return (1 - rho) * mean + rho * val

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    # ------------------------------------------------------------------
    # M-ELO standing-order management (shared)
    # ------------------------------------------------------------------
    def melo_order_has_min_surplus(self, estimate):
        # The standing order still achieves the minimum required surplus Rmin
        # (self.shade[0]) against the current fundamental estimate.
        if self.melo_order.order_type == BUY:
            return self.melo_order.price <= estimate - self.shade[0]
        return self.melo_order.price >= estimate + self.shade[0]

    def withdraw_melo_order(self, reason="unspecified"):
        # Record that the agent is dropping its standing M-ELO order. The
        # simulator owns the books and performs the actual removal; this only
        # updates the agent's own bookkeeping (queue status at drop time, the
        # removed counter, the removal log, and clearing the cached order).
        queue_status = None
        if self.melo_order_tracker is not None and self.melo_order is not None:
            queue_status = self.melo_order_tracker.get(self.melo_order.order_id)
        self.melo_orders_removed += 1
        self.melo_removal_log.append((self.cdaMarket.get_time(), reason, queue_status))
        self.melo_order = None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def act(self, time: int = None):
        """Single per-arrival entry point. Returns ``(venue, orders)``: an empty
        ``orders`` list means "keep my standing M-ELO order", a non-empty one
        means "replace". The agent manages its orders; the simulator steps and
        holds the markets."""
        venue = MELO if random.random() < self.melo_proportion else CDA
        side = random.choice([BUY, SELL])
        orders = self.take_action(side, venue)
        return venue, orders

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------
    def record_trade(self, side, quantity) -> None:
        """Book a fill. Default no-op (ZI); MOBI overrides to record its PV."""

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
        self.pv = PrivateValues(self.q_max, self.pv_var)
        self.melo_order = None
        self.melo_orders_placed = 0
        self.melo_orders_removed = 0
        self.melo_orders_encountered = 0
        self.melo_removal_log = []
        self._reset_role_state()
