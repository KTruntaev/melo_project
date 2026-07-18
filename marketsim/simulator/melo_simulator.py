import random
from marketsim.agent.agent import Agent 
from marketsim.agent.hbl_agent import HBLAgent
from marketsim.fourheap.constants import BUY, SELL, MELO, CDA
from marketsim.market.market import Market
from marketsim.market.melo_market import MeloMarket
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.melo_agent import MeloAgent
import torch.distributions as dist
import torch
import math
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from typing import Dict


def sample_arrivals(p, num_samples):
    # Ensure p is a float tensor - this fixes the torch.finfo error
    p_tensor = torch.tensor([float(p)], dtype=torch.float32)
    geometric_dist = dist.Geometric(p_tensor)
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 10000 sampled time steps


class MELOSimulatorSampledArrival:
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_zi: int,
                 num_hbl: int,
                 num_strategic: int,
                 strategies = None,
                 strategy_counts = None,
                 strategy_params = None,
                 # New role-based parameters
                 role_strategy_counts = None,
                 role_names = None,
                 strategy_counts_background = None,
                 strategy_params_background = None,
                 strategies_background = None,
                 num_assets: int = 1,
                 lam: float = 0.1,
                 mean: float = 100,
                 r: float = .05,
                 shock_var: float = 10,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 eta: float = 1.0,
                 hbl_agent: bool = False,
                 lam_r: float = None,
                 holding_period = 50,
                 lam_melo = 0.1,               
                 lam_melo_mobi = 1e-3, 
                 lam_melo_zi = 6e-3, 
                 profile_order = None,
                 ):

        if shade is None:
            shade = [250, 500]
        if lam_r is None:
            lam_r = lam

        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.lam_r = lam_r
        self.lam_melo = lam_melo
        self.lam_melo_mobi = lam_melo_mobi
        self.lam_melo_zi = lam_melo_zi
        self.time = 0
        self.hbl_agent = hbl_agent
        self.holding_period = holding_period
        self.strategies = strategies
        self.strategy_counts = strategy_counts
        self.strategy_params = strategy_params

        # Role-based parameters for role symmetric games
        self.role_strategy_counts = role_strategy_counts
        self.role_names = role_names

        self.strategies_background = strategies_background
        self.strategy_counts_background = strategy_counts_background
        self.strategy_params_background = strategy_params_background

        self.arrivals = defaultdict(list)
        self.arrivals_melo = defaultdict(list)

        self.timesteps_melo_updates = defaultdict(lambda: 0)

        self.arrivals_sampled = 10000
        self.initial_arrivals = sample_arrivals(lam, self.num_agents)
        self.arrival_times = sample_arrivals(lam_r, self.arrivals_sampled)
        self.arrival_times_melo = sample_arrivals(lam_melo, self.arrivals_sampled)
        self.arrival_index_melo = 0
        self.arrival_index = 0

        self.best_buys = []
        self.best_asks = []
        self.timesteps_melo_trade = []
        self.timesteps_hbl_orders = []
        self.num_orders = 0
        
        # Debug print removed – it spammed the console each repetition

        #TODO: DATA TO DELETE LATER
        self.melo_orders = []
        self.fundamental_estimates = []

        # Track mid-quote p_mid = (best_bid + best_ask)/2 over time.  NaN if either side empty.
        self.midpoints = []

        fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
        self.market = Market(fundamental=fundamental, time_steps=sim_time)
        self.meloMarket = MeloMarket(fundamental, sim_time, self.holding_period)

        self.agents = {}

        self.order_tracker = {}
        self.profile_order = profile_order  # may be None

        self.fundamentals = []

        # ------------------------------------------------------------------
        # Prepare per-role arrival streams BEFORE strategic agent creation
        # ------------------------------------------------------------------

        # Per-role arrival intensities for strategic agents.
        self.lam_melo_mobi = lam_melo_mobi if lam_melo_mobi is not None else lam_melo
        self.lam_melo_zi   = lam_melo_zi   if lam_melo_zi   is not None else lam_melo

        # Separate arrival streams & indices (torch tensors → numpy later)
        self.arrival_times_mobi = sample_arrivals(self.lam_melo_mobi, self.arrivals_sampled)
        self.arrival_times_zi   = sample_arrivals(self.lam_melo_zi,   self.arrivals_sampled)
        self.arrival_index_mobi = 0
        self.arrival_index_zi   = 0

        # Role mapping for fast lookup in step()
        self.agent_roles: Dict[int, str] = {}

        # print("R", r, "SHOCK", shock_var)

        #Market is only passed in for access to fundamental. Melo doesn't care about that
        if not self.strategies_background:
            for agent_id in range(num_zi):
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1
                self.agents[agent_id] = (
                    ZIAgent(
                        agent_id=agent_id,
                        market=self.market,
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var,
                        eta=eta,
                        cda_proportion=1,
                        melo_proportion=0,
                    ))

            for agent_id in range(num_zi, num_zi + num_hbl):
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1

                self.agents[agent_id] = (
                    HBLAgent(
                        agent_id=agent_id,
                        market=self.market,
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var,
                        L=4,
                        arrival_rate=self.lam,
                        cda_proportion=1,
                        melo_proportion=0,
                    ))
        else:
            for strategy in self.strategies_background:
                count = strategy_counts_background[strategy]
                for i in range(count):
                    self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                    self.arrival_index += 1
                    params = self.strategy_params_background[strategy]
                    
                    self.agents[i] = (
                        ZIAgent(
                        agent_id=agent_id,
                        market=self.market,
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var,
                        eta=eta,
                        cda_proportion=1,
                        melo_proportion=0,
                    ))

            
        # Handle strategic agents - either role-based or legacy strategy-based
        strategic_agent_id = num_zi + num_hbl
        
        # If explicit player order is provided, respect it to guarantee ID profile alignment
        if profile_order is not None:
            for role_name, strategy_name in profile_order:
                params = self.strategy_params[strategy_name]
                agent_shade = params.get("shade", shade)

                if role_name == "MOBI":
                    self.agents[strategic_agent_id] = (
                        MeloAgent(
                            agent_id=strategic_agent_id,
                            market=self.market,
                            meloMarket=self.meloMarket,
                            q_max=q_max,
                            pv_var=pv_var,
                            shade=agent_shade,
                            cda_proportion=params["cda_proportion"],
                            melo_proportion=params["melo_proportion"],
                        ))
                else:  # ZI role
                    self.agents[strategic_agent_id] = (
                        ZIAgent(
                            agent_id=strategic_agent_id,
                            market=self.market,
                            q_max=q_max,
                            shade=agent_shade,
                            pv_var=pv_var,
                            eta=eta,
                            cda_proportion=params["cda_proportion"],
                            melo_proportion=params["melo_proportion"],
                            meloMarket=self.meloMarket,
                            order_tracker=self.order_tracker,
                        ))
                # --------------------------------------------------------------
                # Schedule first MELO wake-up using role-specific arrival stream
                # --------------------------------------------------------------

                if role_name == "MOBI":
                    first_tick = self.arrival_times_mobi[self.arrival_index_mobi].item()
                    self.arrival_index_mobi += 1
                else:  # ZI
                    first_tick = self.arrival_times_zi[self.arrival_index_zi].item()
                    self.arrival_index_zi += 1

                # Record mapping and enqueue first arrival
                self.agent_roles[strategic_agent_id] = role_name
                self.arrivals_melo[first_tick].append(strategic_agent_id)

                strategic_agent_id += 1

        elif self.role_strategy_counts and self.role_names:
            # Role-based strategic agent creation
            for role_name in self.role_names:
                role_strategies = self.role_strategy_counts.get(role_name, {})
                for strategy_name, count in role_strategies.items():
                    for _ in range(count):
                        # Schedule first arrival based on role-specific λ
                        if role_name == "MOBI":
                            next_tick = self.arrival_times_mobi[self.arrival_index_mobi].item()
                            self.arrival_index_mobi += 1
                        else:  # ZI
                            next_tick = self.arrival_times_zi[self.arrival_index_zi].item()
                            self.arrival_index_zi += 1

                        #self.arrivals_melo[next_tick].append(strategic_agent_id)

                        params = self.strategy_params[strategy_name]
                        agent_shade = params.get("shade", shade)


                        if role_name == "MOBI":
                            self.agents[strategic_agent_id] = (
                                MeloAgent(
                                    agent_id=strategic_agent_id,
                                    market=self.market,
                                    meloMarket=self.meloMarket,
                                    q_max=q_max,
                                    pv_var=pv_var,
                                    shade=agent_shade,
                                    cda_proportion=params["cda_proportion"],
                                    melo_proportion=params["melo_proportion"],
                                ))

                        else:  # ZI role
                            self.agents[strategic_agent_id] = (
                                ZIAgent(
                                    agent_id=strategic_agent_id,
                                    market=self.market,
                                    q_max=q_max,
                                    shade=agent_shade,
                                    pv_var=pv_var,
                                    eta=eta,
                                    cda_proportion=params["cda_proportion"],
                                    melo_proportion=params["melo_proportion"],
                                    meloMarket=self.meloMarket,
                                    order_tracker=self.order_tracker,
                                ))

                        # Record role mapping for efficient lookup during step()
                        self.agent_roles[strategic_agent_id] = role_name

                        # Schedule first MELO wake-up for this strategic agent
                        self.arrivals_melo[next_tick].append(strategic_agent_id)

                        strategic_agent_id += 1
                        
        elif not self.strategies:
            # Legacy: Create default strategic agents if no strategies specified
            for agent_id in range(num_strategic):
                self.arrivals_melo[self.arrival_times_melo[self.arrival_index_melo].item()].append(strategic_agent_id)
                self.arrival_index_melo += 1
                cda_proportion = 0
                melo_proportion = 1
                self.agents[strategic_agent_id] = (
                    MeloAgent(
                        agent_id=strategic_agent_id,
                    
                        market=self.market,
                        q_max=q_max,
                        pv_var=pv_var,
                        shade=shade,
                        cda_proportion=cda_proportion,
                        melo_proportion=melo_proportion
                    ))
                strategic_agent_id += 1
        else:
            for strategy in self.strategies:
                count = strategy_counts[strategy]
                for _ in range(count):
                    self.arrivals_melo[self.arrival_times_melo[self.arrival_index_melo].item()].append(strategic_agent_id)
                    self.arrival_index_melo += 1
                    params = self.strategy_params[strategy]
                    agent_shade = params.get("shade", shade)

                    self.agents[strategic_agent_id] = (
                        MeloAgent(
                            agent_id=strategic_agent_id,
                            #Not important which market
                            market=self.market,
                            meloMarket=self.meloMarket,
                            q_max=q_max,
                            pv_var=pv_var,
                            shade=agent_shade,
                            cda_proportion=params["cda_proportion"],
                            melo_proportion=params["melo_proportion"],
                        ))
                    strategic_agent_id += 1            

    def step(self):
        agents = self.arrivals[self.time]
        melo_agents = self.arrivals_melo[self.time]
        if self.time < self.sim_time:
            self.meloMarket.event_queue.set_time(self.time)
            self.market.event_queue.set_time(self.time)
            
            if agents:
                for agent_id in agents:
                    #Normal orderbook traders
                    agent: Agent = self.agents[agent_id]
                    self.market.withdraw_all(agent_id)  # TODO: replace this!
                    side = random.choice([BUY, SELL])
                    if random.random() < agent.melo_proportion:
                        marketSelection = MELO
                    else:
                        marketSelection = CDA
                        
                    if marketSelection == MELO:
                        #PLACE MELO
                        orders = agent.take_action(side, marketSelection) 
                        self.meloMarket.add_orders(orders)
                        
                        #TODO: add tracking of these orders
                    else:
                        #PLACE CDA ORDER
                        orders = agent.take_action(side, marketSelection)
                        self.market.add_orders(orders)

                    if isinstance(agent, HBLAgent):
                        self.timesteps_hbl_orders.append((self.time, side, orders[0].price))
                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = sample_arrivals(self.lam_r, self.arrivals_sampled)
                        self.arrival_index = 0
                    self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                    self.arrival_index += 1
                    
            if melo_agents:
                for agent_id in melo_agents:
                    agent = self.agents[agent_id]
                    self.meloMarket.withdraw_all(agent_id, self.order_tracker)
                    self.market.withdraw_all(agent_id)
                    self.num_orders += 1
                    # Check if the agent has strategy parameters (works for both MeloAgent and ZIAgent)
                    assert hasattr(agent, 'cda_proportion') and hasattr(agent, 'melo_proportion')
                    # Use the strategy parameters to determine market selection
                    if random.random() < agent.melo_proportion:
                        marketSelection = MELO
                    else:
                        marketSelection = CDA
                        
                    side = random.choice([BUY, SELL])
                    if marketSelection == MELO:
                        #PLACE MELO
                        orders = agent.take_action(side, marketSelection) 
                        self.meloMarket.add_orders(orders)
                        
                        assert len(orders) == 1

                        self.melo_orders.append((self.time, orders[0].price, orders[0].order_type))
                        self.order_tracker[orders[0].order_id] = ""

                    else:
                        #PLACE CDA ORDER
                        orders = agent.take_action(side, marketSelection)
                        self.market.add_orders(orders)
                            
                    #Determine which arrival stream applies for *next* wake-up
                    role_name = self.agent_roles.get(agent_id, "MOBI")
                    if role_name == "MOBI":
                        if self.arrival_index_mobi == self.arrivals_sampled:
                            self.arrival_times_mobi = sample_arrivals(self.lam_melo_mobi, self.arrivals_sampled)
                            self.arrival_index_mobi = 0
                        next_time = self.arrival_times_mobi[self.arrival_index_mobi].item() + 1 + self.time
                        self.arrivals_melo[next_time].append(agent_id)
                        self.arrival_index_mobi += 1
                    else:  # ZI
                        if self.arrival_index_zi == self.arrivals_sampled:
                            self.arrival_times_zi = sample_arrivals(self.lam_melo_zi, self.arrivals_sampled)
                            self.arrival_index_zi = 0
                        next_time = self.arrival_times_zi[self.arrival_index_zi].item() + 1 + self.time
                        self.arrivals_melo[next_time].append(agent_id)
                        self.arrival_index_zi += 1

                # Process new MELO market orders
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
                # Record trade for agents that have the record_trade method (MeloAgent)
                if hasattr(self.agents[agent_id], 'record_trade'):
                    current_agent = self.agents[agent_id]
                    current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                    self.timesteps_melo_trade.append((self.time, matched_order.order.quantity, matched_order.order.order_type))
                quantity = matched_order.order.order_type*matched_order.order.quantity
                cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                self.agents[agent_id].update_position(quantity, cash)

        else:
            self.end_sim()

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
                    # Record trade for agents that have the record_trade method (MeloAgent)
                    if hasattr(self.agents[agent_id], 'record_trade'):
                        current_agent = self.agents[agent_id]
                        current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                    quantity = matched_order.order.order_type*matched_order.order.quantity
                    cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                    self.agents[agent_id].update_position(quantity, cash)
        return

    def end_sim(self):
        fundamental_val = self.market.get_final_fundamental()
        values = {}
        positions = {}
        # melo_profits = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            if isinstance(agent, MeloAgent):
                values[agent_id] = agent.position*fundamental_val + agent.cash + sum(quantity * value for quantity, value in agent.melo_pv_history)
            else:
                values[agent_id] = agent.get_pos_value() + agent.position*fundamental_val + agent.cash
            positions[agent_id] = agent.position
        counts = Counter(self.order_tracker.values())
            
        # print(f'At the end of the simulation we get {values}')
        # print(f'MELO_ At the end of the simulation we get {melo_profits}')
        # input()
        activation_queue = len(self.meloMarket.order_book.buy_activation_queue) + len(self.meloMarket.order_book.sell_activation_queue)
        active_queue = len(self.meloMarket.order_book.buy_active_queue) + len(self.meloMarket.order_book.sell_active_queue)
        elig_queue = self.meloMarket.order_book.buy_eligibility_queue.count() + self.meloMarket.order_book.sell_eligibility_queue.count()
        return values, positions, self.best_buys, self.best_asks, self.midpoints, self.timesteps_melo_trade, self.timesteps_hbl_orders, self.meloMarket.order_book.buy_cancelled, self.meloMarket.order_book.sell_cancelled, self.meloMarket.order_book.removed_eligibility, self.meloMarket.order_book.removed_activation, self.meloMarket.order_book.removed_active, self.melo_orders, self.fundamental_estimates, self.num_orders, elig_queue, activation_queue, active_queue  # Return both CDA and MELO profits including midpoints

    def run(self):
        for t in range(self.sim_time):
            if self.arrivals[t] or self.arrivals_melo[t]:
                try:
                    self.step()
                except KeyError:
                    print(self.arrivals[self.time])
                    return self.market, self.meloMarket
            
            if self.timesteps_melo_updates[t] != 0:
                self.update_melo_market()

            # self.meloMarket.order_book.update_eligiblity_queue(self.time, self.order_tracker, self.market.order_book.midprice)
            # self.meloMarket.order_book.update_active_queue(self.time, self.order_tracker)
            # new_orders = self.meloMarket.order_book.matching_orders(self.time, self.order_tracker, self.market.order_book.midprice)
            # assert new_orders == [[], []]

            self.fundamentals.append(self.market.get_fundamental_value())
            self.best_buys.append(self.market.order_book.get_best_bid())
            self.best_asks.append(self.market.order_book.get_best_ask())
            self.fundamental_estimates.append(self.agents[0].estimate_fundamental())

            # Record mid-quote for this timestep if both sides non-empty, else NaN.
            try:
                bid = self.best_buys[-1]
                ask = self.best_asks[-1]
                if (not math.isnan(bid)) and (not math.isnan(ask)):
                    self.midpoints.append( (bid + ask) / 2.0 )
                else:
                    self.midpoints.append(float('nan'))
            except Exception:
                # Fallback in rare edge cases where lists not yet populated
                self.midpoints.append(float('nan'))

            self.time += 1
        self.step()


def sample_arrivals(p, num_samples):
    # Ensure p is a float tensor - this fixes the torch.finfo error 
    p_tensor = torch.tensor([float(p)], dtype=torch.float32)
    geometric_dist = dist.Geometric(p_tensor)
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps