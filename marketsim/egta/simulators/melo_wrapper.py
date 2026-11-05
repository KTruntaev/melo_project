"""
MELO simulator interface for EGTA with Role Symmetric Game support.
"""
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
from marketsim.egta.simulators.base import Simulator
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from marketsim.fourheap.constants import BUY, SELL
import concurrent.futures as cf 
import torch
import math

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (ImportError, AttributeError):

    pass
import time
from marketsim.egta.egta import Observation      # path you placed the patch in
ProfileKey = Tuple[Tuple[str, str], ...]  
import os
import sys
import warnings

# Keep progress bars visible in batch logs
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in reduce",
    category=RuntimeWarning,
    module="numpy.core.fromnumeric",
)


# Helper for parallel execution keep top level to pickle


_BASE_KWARGS: Optional[dict] = None


def _init_worker(base_kwargs):
    """Pool initializer – saves *base_kwargs* in a global variable."""
    global _BASE_KWARGS
    _BASE_KWARGS = base_kwargs


def _run_melo_single_rep(arg):
    """Run one MELO simulation repetition.

    Accepts either:
    1) a tuple ``(seed, sim_kwargs)`` – the modern and test-friendly path where
       all per-simulation parameters are supplied directly, or
    2) a bare integer ``seed`` – legacy/multiprocessing fallback where kwargs
       are provided once via the global `_BASE_KWARGS` set by `_init_worker`.
    """

    if isinstance(arg, tuple):
        seed, sim_kwargs = arg
    else:
        seed = arg
        if _BASE_KWARGS is None:
            raise RuntimeError("Worker not initialised with base kwargs")
        sim_kwargs = _BASE_KWARGS

    # Local seeding for determinism
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    # Run one MELO repetition (suppress verbose per-rep logging)
    sim = MELOSimulatorSampledArrival(**sim_kwargs)
    sim.run()

    results = sim.end_sim()
    raw_values = results["payoffs"]  # payoff dict keyed by agent id -> number/torch.Tensor
    midpoints = results["midpoints"]


    import numpy as _np
    if midpoints:
        arr = _np.asarray(midpoints, dtype=float)
        if _np.isfinite(arr).any():
            mid_mean = float(_np.nanmean(arr))
        else:
            mid_mean = float('nan')
    else:
        mid_mean = float('nan')

    values: Dict[Union[int,str], float] = {}
    for agent_id, payoff in raw_values.items():
        if isinstance(payoff, torch.Tensor):
            values[agent_id] = float(payoff.item())
        else:
            values[agent_id] = float(payoff)

    # store midpoint information under a non-numeric key so it is ignored by
    # payoff aggregation but still available to the caller
    values["__mid_mean"] = mid_mean

    return values 

class MeloSimulator(Simulator):
    """
    Interface to the MELO simulator for EGTA with Role Symmetric Game support.
    Now supports strategic ZI agents that can choose market allocation.
    """
    
    def __init__(self, 
                num_strategic_mobi: int = 10,
                num_strategic_zi: int = 30,
                sim_time: int = 10000,
                num_assets: int = 1,
                lam: float = 6e-3,
                mean: float = 1e6,
                r: float = 0.05,
                shock_var: float = 1e4,
                q_max: int = 10,
                pv_var: float = 5e6,
                shade: Optional[List[float]] = None,
                eta: float = 0.5,
                lam_melo: float = 1e-3,
                lam_r: Optional[float] = None,
                holding_period: int = 10,
                lam_melo_mobi: float = 1e-3,
                lam_melo_zi: float = 6e-3,
                # Background agents (non-strategic)
                num_background_zi: int = 0,
                num_background_hbl: int = 0,
                reps: int = 10000,
                # Role-specific strategies
                mobi_strategies: Optional[List[str]] = None,
                zi_strategies: Optional[List[str]] = None,
                # Mode control
                force_symmetric: bool = False,
                parallel: bool = False,
                log_profile_details: bool = False,
                num_workers: Optional[int] = None):
        """
        Initialize the MELO simulator interface for role symmetric games.
        
        Args:
            num_strategic_mobi: Number of strategic MOBI agents
            num_strategic_zi: Number of strategic ZI agents  
            sim_time: Simulation time
            num_assets: Number of assets
            lam: Arrival rate
            mean: Mean fundamental value
            r: Mean reversion rate
            shock_var: Shock variance
            q_max: Maximum quantity
            pv_var: Private value variance
            shade: Shade parameters
            eta: Eta parameter
            lam_r: Arrival rate for regular traders
            holding_period: Holding period
            lam_melo_mobi: Arrival rate for MOBI traders
            lam_melo_zi: Arrival rate for ZI traders
            num_background_zi: Number of non-strategic ZI agents
            num_background_hbl: Number of non-strategic HBL agents
            reps: Number of simulation repetitions
            mobi_strategies: Strategy names for MOBI role (if None, uses default)
            zi_strategies: Strategy names for ZI role (if None, uses default)
            force_symmetric: If True, forces symmetric game behavior (disables role symmetric mode)
            parallel: If True, use multiprocessing for parallel repetitions
            log_profile_details: If True, print profile summary after simulation
            num_workers: Number of worker threads for parallel execution
        """
        self.num_strategic_mobi = num_strategic_mobi
        self.num_strategic_zi = num_strategic_zi
        self.sim_time = sim_time
        self.profile_summaries: List[Dict[str, Any]] = []
        self.num_assets = num_assets
        self.lam = lam
        self.mean = mean
        self.r = r
        self.shock_var = shock_var
        self.q_max = q_max
        self.pv_var = pv_var
        # Global fallback shade – used by background agents and strategies
        # that *do not* specify their own "_shade…" suffix.
        self.lam_melo = lam_melo
        self.shade = shade or [250, 500]
        self.eta = eta
        self.lam_r = lam_r or lam
        self.holding_period = holding_period
        self.lam_melo_mobi = lam_melo_mobi
        self.lam_melo_zi = lam_melo_zi
        self.num_background_zi = num_background_zi
        self.num_background_hbl = num_background_hbl
        self.reps = reps
        self.order_quantity = 5  # Fixed order quantity for MOBI traders
        self.force_symmetric = force_symmetric
        self.parallel = parallel
        self.log_profile_details = log_profile_details
        self.num_workers = num_workers or 1
        
        # Define role names and player counts
        self.role_names = ["MOBI", "ZI"]
        self.num_players_per_role = [num_strategic_mobi, num_strategic_zi]
        
        # Define strategies for each role
        if mobi_strategies is None:
            raise ValueError("No MOBI (large-lot) strategies provided – please supply a non-empty list to `mobi_strategies`.")
        else:
            self.mobi_strategies = mobi_strategies
            
        if zi_strategies is None:
            raise ValueError("No ZI (small-lot) strategies provided – please supply a non-empty list to `zi_strategies`.")

        else:
            self.zi_strategies = zi_strategies
        
        self.strategy_names_per_role = [self.mobi_strategies, self.zi_strategies]
        
        # Define strategy parameters for both roles
        self.strategy_params = {}
        
        import re  # ── new: regex helper for shade suffix

        def _extract_shade(name: str, default: List[int]) -> List[int]:
            """Return [low,high] from "…_shadeLOW_HIGH" suffix, else *default*."""
            
            try:
                low_high = name.split("_shade", 1)[1]
                low, high = map(int, low_high.split("_", 1))
                return [low, high]
            except Exception:
                raise Exception("Can't get shade from strategy!")

        # MOBI strategy parameters  (now including per-strategy shade)
        for strategy in self.mobi_strategies:
            shade_pair = _extract_shade(strategy, self.shade)

            if "100_0" in strategy:
                self.strategy_params[strategy] = {
                    "cda_proportion": 1.0,
                    "melo_proportion": 0.0,
                    "shade": shade_pair,
                }
            elif "0_100" in strategy:
                self.strategy_params[strategy] = {
                    "cda_proportion": 0.0,
                    "melo_proportion": 1.0,
                    "shade": shade_pair,
                }
            

        # ZI strategy parameters (same allocation logic, incl. shade)
        for strategy in self.zi_strategies:
            shade_pair = _extract_shade(strategy, self.shade)

            if "100_0" in strategy:
                self.strategy_params[strategy] = {
                    "cda_proportion": 1.0,
                    "melo_proportion": 0.0,
                    "shade": shade_pair,
                }
            elif "0_100" in strategy:
                self.strategy_params[strategy] = {
                    "cda_proportion": 0.0,
                    "melo_proportion": 1.0,
                    "shade": shade_pair,
                }
           
        
        # In symmetric mode, remove role symmetric capabilities
        if self.force_symmetric:
            # Remove the get_role_info method to make EGTA treat this as a symmetric game
            if hasattr(self, 'get_role_info'):
                delattr(self, 'get_role_info')
    
        # store summaries so calling code can persist them
        self.profile_summaries: List[Dict[str, Any]] = []
    
    def get_num_players(self) -> int:
        """Get the total number of strategic players."""
        if self.force_symmetric:
            # In symmetric mode, return just the MOBI players (the strategic players we care about)
            return self.num_strategic_mobi
        else:
            # In role symmetric mode, return total strategic players
            return sum(self.num_players_per_role)
    
    def get_role_info(self) -> Tuple[List[str], List[int], List[List[str]]]:
        """Get role information for role symmetric games."""
        return self.role_names, self.num_players_per_role, self.strategy_names_per_role
    
    def get_strategies(self) -> List[str]:
        """Get all strategies across all roles (for backward compatibility)."""
        if self.force_symmetric:
            # In symmetric mode, return only MOBI strategies (the strategic players)
            return self.mobi_strategies
        else:
            # In role symmetric mode, return all strategies
            all_strategies = []
            for strategies in self.strategy_names_per_role:
                all_strategies.extend(strategies)
            return all_strategies
    
    def simulate_profile(
        self,
        profile: List[Tuple[str, str]],
        return_detailed: bool = False,   # kept for API compat (ignored)
    ) -> Observation:
        """
        Simulate *one* profile once and return an Observation containing:

        * canonical profile key
        * vector of player payoffs (mean over `reps` repetitions)
        * meta-data (runtime, seed)
        * valid flag (False if every repetition failed)
        """
        start_ts = time.time()
        

        # --------------------------------------------------------------
        # 1)  Build role-strategy counts (logic identical to old code)
        # --------------------------------------------------------------
        # (full role/symmetric detection code unchanged – snipped)
        #   → produces `role_strategy_counts`, `is_role_symmetric_profile`
        # --------------------------------------------------------------
        profile_roles = set(r for r, _ in profile)
        is_role_symmetric_profile = len(profile_roles) > 1 or "Player" not in profile_roles
        if is_role_symmetric_profile:
            role_strategy_counts = {r: Counter() for r in self.role_names}
            for role_name, strat in profile:
                role_strategy_counts[role_name][strat] += 1
        else:
            strategy_counts = Counter(s for _, s in profile)
            role_strategy_counts = {"MOBI": Counter(), "ZI": Counter()}
            for strat, c in strategy_counts.items():
                (role_strategy_counts["MOBI" if strat.startswith("MOBI_") else "ZI"])[strat] = c

        # --------------------------------------------------------------
        # 2)  Spawn repetitions (parallel or serial)
        # --------------------------------------------------------------
        # Background agents no longer exist in the M-ELO simulator; every agent
        # is strategic. These wrapper params are retained for API compatibility
        # but must be zero.
        num_background = self.num_background_zi + self.num_background_hbl
        if num_background:
            raise ValueError(
                "Background agents were removed from the M-ELO simulator; "
                f"got num_background_zi={self.num_background_zi}, "
                f"num_background_hbl={self.num_background_hbl} (both must be 0)."
            )
        base_kwargs = dict(
            sim_time=self.sim_time,
            mean=self.mean,
            r=self.r,
            shock_var=self.shock_var,
            q_max=self.q_max,
            pv_var=self.pv_var,
            shade=self.shade,
            eta=self.eta,
            holding_period=self.holding_period,
            lam_melo_mobi=self.lam_melo_mobi,
            lam_melo_zi=self.lam_melo_zi,
            role_strategy_counts=role_strategy_counts,
            strategy_params=self.strategy_params,
            role_names=self.role_names,
            profile_order=profile,
        )

        seeds = [random.randint(0, 2**32 - 1) for _ in range(self.reps)]
        iter_args = [(s, base_kwargs) for s in seeds]

        if self.parallel and self.reps > 1:
            import multiprocessing as mp
            # Use faster 'fork' start-method on Unix (not available on Mac="darwin" or Windows)
            if sys.platform.startswith("linux"):
                ctx = mp.get_context("fork")
            else:
                ctx = mp.get_context("spawn")
            n_workers = self._effective_workers()
            with cf.ProcessPoolExecutor(
                    mp_context=ctx,
                    max_workers=n_workers,
                    initializer=_init_worker,
                    initargs=(base_kwargs,),
                ) as pool:
                # Choose a chunksize so each worker receives several seeds per
                # task submission – reduces IPC overhead.
                chunk = 1  # smaller chunks yield smoother progress updates

                values_per_rep = list(
                    tqdm(
                        pool.map(_run_melo_single_rep, iter_args, chunksize=chunk),
                        total=self.reps,
                        desc="   reps",
                        unit="rep",
                        leave=False,
                        disable=not self.log_profile_details,
                        file=sys.stderr,
                        dynamic_ncols=True,
                    )
                )
        else:
            # serial fall-back: reuse the same simulator instance setup
            _init_worker(base_kwargs)
            values_per_rep = []
            for out in tqdm(map(_run_melo_single_rep, iter_args),
                            total=self.reps,
                            desc="   reps",
                            unit="rep",
                            leave=False,
                            disable=not self.log_profile_details,
                            file=sys.stderr,
                            dynamic_ncols=True):
                values_per_rep.append(out)

        # ------------------------------------------------------------------
        # Aggregate per-player payoffs AND collect midpoint summaries
        # ------------------------------------------------------------------
        aggregated_payoffs: List[float] = []
        mid_means: List[float] = []

        player_idx = 0
        for role_name, strat in profile:
            payoffs = []
            for values in values_per_rep:
                # pop midpoint statistic once per repetition (harmless to do many times)
                mm = values.get("__mid_mean")
                if mm is not None and not math.isnan(mm):
                    mid_means.append(mm)

                agent_id = num_background + player_idx
                if agent_id in values:
                    val = values[agent_id]
                    val = float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                    if not np.isnan(val) and not np.isinf(val):
                        payoffs.append(val)
            aggregated_payoffs.append(sum(payoffs) / len(payoffs) if payoffs else 0.0)
            player_idx += 1

        # Midpoint statistics across all repetitions (robust to all-NaN cases)
        if mid_means:
            arr_mid = np.asarray(mid_means, dtype=float)
            finite_mask = np.isfinite(arr_mid)
            if finite_mask.any():
                mp_mean = float(np.nanmean(arr_mid[finite_mask]))
                mp_std  = float(np.nanstd(arr_mid[finite_mask]))
            else:
                mp_mean = float('nan')
                mp_std  = float('nan')
        else:
            mp_mean = float('nan')
            mp_std  = float('nan')

        # --------------------------------------------------------------
        # 4)  Build and return the Observation
        # --------------------------------------------------------------
        wall_clock = time.time() - start_ts
        prof_key: ProfileKey = tuple(sorted((r, s) for r, s in profile))

        aux_data = {
            "runtime": wall_clock,
            "seed": seeds[0] if seeds else 0,
            "n_raw": len(values_per_rep),
            "mid_mean": mp_mean,
            "mid_std":  mp_std,
        }

        obs = Observation(
            profile_key=prof_key,
            payoffs=np.asarray(aggregated_payoffs, dtype=float),
            aux=aux_data,
        )
        return obs
    
    def simulate_profiles(
        self,
        profiles: List[List[Tuple[str, str]]],
        return_detailed: bool = False,
    ) -> List[Any]:
        """
        Simulate multiple role symmetric profiles.
        
        Args:
            profiles: List of role symmetric profiles
            return_detailed: If True, return detailed per-agent and per-role averages
            
        Returns:
            List of lists of (player_id, role_name, strategy_name, payoff) tuples
        """
        results = []
        for i, profile in enumerate(profiles):
            print(f"Simulating role symmetric profile {i+1}/{len(profiles)}: {profile}")
            profile_results = self.simulate_profile(profile, return_detailed=return_detailed)
            results.append(profile_results)
        return results 

    def _print_profile_summary(
        self,
        profile: List[Tuple[str, str]],
        per_agent: List[Tuple[str, str, str, float]],
        role_payoffs: Dict[str, List[float]],
        role_avg: Dict[str, float],
        profile_avg: float,
    ) -> None:
        """Prints counts per strategy and payoff stats for each role."""

        # count strategies per role
        role_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for role_name, strategy_name in [(r, s) for r, s in profile]:
            role_counts[role_name][strategy_name] += 1

        print("\n────────────────────────────────────────────")
        print("Profile summary:")
        for role_name, strat_counter in role_counts.items():
            counts_str = ", ".join([f"{k}:{v}" for k, v in strat_counter.items()])
            avg_pay = role_avg.get(role_name, 0.0)
            print(f"  Role {role_name} → [{counts_str}],  Avg Payoff: {avg_pay:.4f}")
            print(f"    Payoffs: {[round(p,4) for p in role_payoffs.get(role_name, [])]}")
        print(f"  Overall profile average: {profile_avg:.4f}")
        print("────────────────────────────────────────────")

        # keep structured copy for downstream persistence
        summary_dict = {
            "profile_counts": {r: dict(c) for r, c in role_counts.items()},
            "role_avg": role_avg,
            "role_payoffs": role_payoffs,
            "profile_avg": profile_avg,
        }
        self.profile_summaries.append(summary_dict) 

    # ------------------------------------------------------------------
    # Helper: determine worker count respecting SLURM allocation limits
    # ------------------------------------------------------------------
    def _effective_workers(self) -> int:
        """Return worker count obeying Slurm cpu limit and self.reps."""
        slurm_cores = os.environ.get("SLURM_CPUS_ON_NODE") \
            or os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cores is not None:
            available = int(slurm_cores)
        else:
            available = os.cpu_count() or 1
        return max(1, min(available, self.num_workers, self.reps)) 