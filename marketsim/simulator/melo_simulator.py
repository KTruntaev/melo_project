from marketsim.agent.agent import Agent
from marketsim.fourheap.constants import MELO
from marketsim.market.market import Market
from marketsim.market.melo_market import MeloMarket
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zi_melo_agent import ZIMeloAgent
import torch.distributions as dist
import torch
import math
from collections import defaultdict
from typing import Dict


def sample_arrivals(p, num_samples):
    # Ensure p is a float tensor - this fixes the torch.finfo error
    p_tensor = torch.tensor([float(p)], dtype=torch.float32)
    geometric_dist = dist.Geometric(p_tensor)
    return geometric_dist.sample((num_samples,)).squeeze()  # tensor of sampled gaps


class MELOSimulatorSampledArrival:
    """Discrete-time M-ELO / CDA market simulator.

    Responsibilities are deliberately narrow: the simulator **steps and holds
    the markets**. It owns the CDA order book (``market``) and the M-ELO book
    (``meloMarket``), advances the clock, delivers arrival events to agents,
    inserts the orders they hand back, and runs the market mechanics each tick
    (M-ELO eligibility/holding/activation queues and matching, CDA matching,
    fill booking).

    Deciding *what* to do with an order — which venue, which side, and whether
    to keep or replace a standing M-ELO order — belongs to the agent, which the
    simulator queries once per arrival via :meth:`ZIMeloAgent.act`. ``act``
    returns ``(venue, orders)``; a non-empty ``orders`` list means "replace"
    (the simulator withdraws the agent's live orders from both books, then
    inserts), and an empty list means "keep" (the books are left untouched).

    Every agent is strategic (MOBI or ZI) and is scheduled on a role-specific
    M-ELO arrival stream (``lam_melo_mobi`` / ``lam_melo_zi``). There are no
    background agents.
    """

    def __init__(self,
                 sim_time: int,
                 strategy_params=None,
                 # Role-based parameters for role symmetric games
                 role_strategy_counts=None,
                 role_names=None,
                 mean: float = 100,
                 r: float = .05,
                 shock_var: float = 10,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 eta: float = 1.0,
                 holding_period=50,
                 lam_melo_mobi: float = 1e-3,
                 lam_melo_zi: float = 6e-3,
                 profile_order=None,
                 ):

        if shade is None:
            shade = [250, 500]

        self.sim_time = sim_time
        self.time = 0
        self.holding_period = holding_period
        self.strategy_params = strategy_params

        # Role-based parameters for role symmetric games
        self.role_strategy_counts = role_strategy_counts
        self.role_names = role_names

        # Per-role M-ELO arrival intensities for strategic agents.
        self.lam_melo_mobi = lam_melo_mobi
        self.lam_melo_zi = lam_melo_zi

        self.arrivals_melo = defaultdict(list)
        self.timesteps_melo_updates = defaultdict(lambda: 0)

        self.arrivals_sampled = 10000
        # Separate arrival streams & indices per role (torch tensors).
        self.arrival_times_mobi = sample_arrivals(self.lam_melo_mobi, self.arrivals_sampled)
        self.arrival_times_zi = sample_arrivals(self.lam_melo_zi, self.arrivals_sampled)
        self.arrival_index_mobi = 0
        self.arrival_index_zi = 0

        self.best_buys = []
        self.best_asks = []
        self.timesteps_melo_trade = []
        self.num_orders = 0

        # ZI-specific M-ELO order placement/removal tracking. These are
        # recomputed (not accumulated) each time end_sim() runs, by aggregating
        # the per-agent counters/logs kept on each ZI-role ZIMeloAgent (see
        # melo_orders_placed/removed and melo_removal_log in zi_melo_agent.py).
        self.zi_orders_placed = 0
        self.zi_orders_removed = 0
        self.zi_removal_log = []  # list of (agent_id, time, reason, queue_status)

        self.melo_orders = []
        self.fundamental_estimates = []

        # Track mid-quote p_mid = (best_bid + best_ask)/2 over time. NaN if
        # either side empty.
        self.midpoints = []
        self.fundamentals = []

        fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
        self.market = Market(fundamental=fundamental, time_steps=sim_time)
        self.meloMarket = MeloMarket(fundamental, sim_time, self.holding_period)

        # All agents are strategic ZIMeloAgents. Agent ids are contiguous and
        # assigned in construction order, so the list index == agent_id.
        self.agents = []
        self.order_tracker = {}
        self.profile_order = profile_order  # may be None

        # Role mapping for fast lookup of the next-arrival stream in step().
        self.agent_roles: Dict[int, str] = {}

        # ------------------------------------------------------------------
        # Build strategic agents. Prefer the explicit per-player profile order
        # (guarantees agent-id ↔ profile alignment for EGTA); otherwise expand
        # the per-role strategy counts.
        # ------------------------------------------------------------------
        if profile_order is not None:
            for role_name, strategy_name in profile_order:
                self._add_strategic_agent(role_name, strategy_name, len(self.agents),
                                          q_max, pv_var, shade, eta)
        elif self.role_strategy_counts and self.role_names:
            for role_name in self.role_names:
                role_strategies = self.role_strategy_counts.get(role_name, {})
                for strategy_name, count in role_strategies.items():
                    for _ in range(count):
                        self._add_strategic_agent(role_name, strategy_name, len(self.agents),
                                                  q_max, pv_var, shade, eta)
        else:
            raise ValueError(
                "No agents to build: supply either `profile_order` or "
                "`role_strategy_counts` + `role_names`."
            )

    def _add_strategic_agent(self, role_name, strategy_name, agent_id,
                             q_max, pv_var, shade, eta):
        """Construct one strategic agent and schedule its first M-ELO wake-up."""
        params = self.strategy_params[strategy_name]
        agent_shade = params.get("shade", shade)

        if role_name == "MOBI":
            agent = ZIMeloAgent(
                agent_id=agent_id,
                role="MOBI",
                market=self.market,
                meloMarket=self.meloMarket,
                order_tracker=self.order_tracker,
                q_max=q_max,
                pv_var=pv_var,
                shade=agent_shade,
                cda_proportion=params["cda_proportion"],
                melo_proportion=params["melo_proportion"],
            )
            first_tick = self.arrival_times_mobi[self.arrival_index_mobi].item()
            self.arrival_index_mobi += 1
        else:  # ZI role
            agent = ZIMeloAgent(
                agent_id=agent_id,
                role="ZI",
                market=self.market,
                meloMarket=self.meloMarket,
                order_tracker=self.order_tracker,
                q_max=q_max,
                shade=agent_shade,
                pv_var=pv_var,
                eta=eta,
                cda_proportion=params["cda_proportion"],
                melo_proportion=params["melo_proportion"],
            )
            first_tick = self.arrival_times_zi[self.arrival_index_zi].item()
            self.arrival_index_zi += 1

        self.agents.append(agent)
        self.agent_roles[agent_id] = role_name
        self.arrivals_melo[first_tick].append(agent_id)

    def _schedule_next_arrival(self, agent_id):
        """Draw and enqueue this agent's next M-ELO wake-up from its role stream."""
        role_name = self.agent_roles.get(agent_id, "MOBI")
        if role_name == "MOBI":
            if self.arrival_index_mobi == self.arrivals_sampled:
                self.arrival_times_mobi = sample_arrivals(self.lam_melo_mobi, self.arrivals_sampled)
                self.arrival_index_mobi = 0
            next_time = self.arrival_times_mobi[self.arrival_index_mobi].item() + 1 + self.time
            self.arrival_index_mobi += 1
        else:  # ZI
            if self.arrival_index_zi == self.arrivals_sampled:
                self.arrival_times_zi = sample_arrivals(self.lam_melo_zi, self.arrivals_sampled)
                self.arrival_index_zi = 0
            next_time = self.arrival_times_zi[self.arrival_index_zi].item() + 1 + self.time
            self.arrival_index_zi += 1
        self.arrivals_melo[next_time].append(agent_id)

    def step(self):
        if self.time >= self.sim_time:
            self.end_sim()
            return

        self.meloMarket.event_queue.set_time(self.time)
        self.market.event_queue.set_time(self.time)

        melo_agents = self.arrivals_melo[self.time]
        for agent_id in melo_agents:
            agent: Agent = self.agents[agent_id]
            self.num_orders += 1

            # The agent manages its own orders; the simulator only executes the
            # decision against the books it holds.
            venue, orders = agent.act(self.time)
            if orders:
                # "Replace": clear this agent's live orders from both books, then
                # insert the new order into the chosen venue.
                self.meloMarket.withdraw_all(agent_id, self.order_tracker)
                self.market.withdraw_all(agent_id)
                if venue == MELO:
                    assert len(orders) == 1
                    self.meloMarket.add_orders(orders)
                    self.order_tracker[orders[0].order_id] = ""
                    self.melo_orders.append((self.time, orders[0].price, orders[0].order_type))
                else:
                    self.market.add_orders(orders)
            # else "keep": standing M-ELO order left untouched.

            self._schedule_next_arrival(agent_id)

        # Process new M-ELO orders. When melo_step_every_tick is True (the fix,
        # and the default) run every step so M-ELO orders are inserted into the
        # fourheap the same timestep they are placed. When False (the original
        # bug) fall back to running only when an agent arrives this tick — the
        # a/b harness sets this flag to reproduce the bug. Kept before the CDA
        # market.step() below so insert() sees the same pre-step midpoint.
        if getattr(self, 'melo_step_every_tick', True) or melo_agents:
            order_placement = self.meloMarket.step(self.order_tracker, self.market.order_book.midprice)
            if order_placement != -1:
                self.timesteps_melo_updates[order_placement + self.holding_period] = 1

        # Process CDA market orders
        prev_midpoint = self.market.order_book.midprice
        new_orders = self.market.step()
        current_midpoint = self.market.order_book.midprice
        if not math.isnan(current_midpoint):
            if math.isnan(prev_midpoint) or not math.isclose(self.market.order_book.midprice, prev_midpoint, rel_tol=1e-5):
                order_updated = self.meloMarket.order_book.update_eligiblity_queue(self.time, self.order_tracker, current_midpoint)
                if order_updated != -1:
                    self.timesteps_melo_updates[order_updated + self.holding_period] = 1
            melo_new_orders = self.meloMarket.order_book.matching_orders(self.time, self.order_tracker, current_midpoint)
            self.record_melo_trade(melo_new_orders)

        for matched_order in new_orders:
            agent_id = matched_order.order.agent_id
            # Record trade for MOBI agents only (per-fill private-value booking)
            if self.agents[agent_id].role == "MOBI":
                current_agent = self.agents[agent_id]
                current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                self.timesteps_melo_trade.append((self.time, matched_order.order.quantity, matched_order.order.order_type))
            quantity = matched_order.order.order_type * matched_order.order.quantity
            cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
            self.agents[agent_id].update_position(quantity, cash)

    def update_melo_market(self):
        self.meloMarket.order_book.update_active_queue(self.time, self.order_tracker)
        new_orders = self.meloMarket.order_book.matching_orders(self.time, self.order_tracker, self.market.order_book.midprice)
        self.record_melo_trade(new_orders)
        return

    def record_melo_trade(self, new_orders):
        if len(new_orders[0]) > 0:
            for side_orders in new_orders:
                for matched_order in side_orders:
                    agent_id = matched_order.order.agent_id
                    # Record trade for MOBI agents only (per-fill private-value booking)
                    if self.agents[agent_id].role == "MOBI":
                        current_agent = self.agents[agent_id]
                        current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                    quantity = matched_order.order.order_type * matched_order.order.quantity
                    cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                    self.agents[agent_id].update_position(quantity, cash)
        return

    def end_sim(self):
        fundamental_val = self.market.get_final_fundamental()
        values = {}
        positions = {}

        # Aggregate ZI-specific M-ELO placement/removal tracking from each agent.
        # Recomputed from scratch (not accumulated) so this stays correct even if
        # end_sim() is called more than once for the same run.
        self.zi_orders_placed = 0
        self.zi_orders_removed = 0
        self.zi_removal_log = []

        for agent_id in range(len(self.agents)):
            agent = self.agents[agent_id]
            if agent.role == "MOBI":
                values[agent_id] = agent.position * fundamental_val + agent.cash + sum(quantity * value for quantity, value in agent.melo_pv_history)
            else:
                values[agent_id] = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
            positions[agent_id] = agent.position
            if agent.role == "ZI":
                self.zi_orders_placed += agent.melo_orders_placed
                self.zi_orders_removed += agent.melo_orders_removed
                for (removal_time, reason, queue_status) in agent.melo_removal_log:
                    self.zi_removal_log.append((agent_id, removal_time, reason, queue_status))

        activation_queue = len(self.meloMarket.order_book.buy_activation_queue) + len(self.meloMarket.order_book.sell_activation_queue)
        active_queue = len(self.meloMarket.order_book.buy_active_queue) + len(self.meloMarket.order_book.sell_active_queue)
        elig_queue = self.meloMarket.order_book.buy_eligibility_queue.count() + self.meloMarket.order_book.sell_eligibility_queue.count()

        # A single flat dict keyed by name. The EGTA wrapper reads "payoffs" and
        # "midpoints"; the remaining keys are data-tracking artifacts for
        # analysis/tests.
        return {
            "payoffs": values,
            "positions": positions,
            "best_buys": self.best_buys,
            "best_asks": self.best_asks,
            "midpoints": self.midpoints,
            "timesteps_melo_trade": self.timesteps_melo_trade,
            "buy_cancelled": self.meloMarket.order_book.buy_cancelled,
            "sell_cancelled": self.meloMarket.order_book.sell_cancelled,
            "removed_eligibility": self.meloMarket.order_book.removed_eligibility,
            "removed_activation": self.meloMarket.order_book.removed_activation,
            "removed_active": self.meloMarket.order_book.removed_active,
            "melo_orders": self.melo_orders,
            "fundamental_estimates": self.fundamental_estimates,
            "num_orders": self.num_orders,
            "elig_queue": elig_queue,
            "activation_queue": activation_queue,
            "active_queue": active_queue,
            "zi_orders_placed": self.zi_orders_placed,
            "zi_orders_removed": self.zi_orders_removed,
            "zi_removal_log": self.zi_removal_log,
        }

    def run(self):
        for t in range(self.sim_time):
            if self.arrivals_melo[t]:
                try:
                    self.step()
                except KeyError:
                    print(self.arrivals_melo[self.time])
                    return self.market, self.meloMarket

            if self.timesteps_melo_updates[t] != 0:
                self.update_melo_market()

            self.fundamentals.append(self.market.get_fundamental_value())
            self.best_buys.append(self.market.order_book.get_best_bid())
            self.best_asks.append(self.market.order_book.get_best_ask())
            self.fundamental_estimates.append(self.agents[0].estimate_fundamental())

            # Record mid-quote for this timestep if both sides non-empty, else NaN.
            try:
                bid = self.best_buys[-1]
                ask = self.best_asks[-1]
                if (not math.isnan(bid)) and (not math.isnan(ask)):
                    self.midpoints.append((bid + ask) / 2.0)
                else:
                    self.midpoints.append(float('nan'))
            except Exception:
                # Fallback in rare edge cases where lists not yet populated
                self.midpoints.append(float('nan'))

            self.time += 1
        self.step()
