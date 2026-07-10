import contextlib
import io
import random
import statistics

import numpy as np
import torch

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival as FixedSim
from marketsim.simulator.melo_simulator_old import MELOSimulatorSampledArrival as BuggySim
from marketsim.simulator.melo_simulator_cannonical import MELOSimulatorSampledArrival as CanonicalSim


SEEDS = [1, 7, 42, 100, 999]

QUIET_KWARGS = dict(
    num_background_agents=8,
    num_strategic=3,
    num_zi=5,
    num_hbl=3,
    sim_time=2000,
    lam=6e-3,
    mean=1e6,
    lam_melo=1e-3,
    r=0.0001,
    shock_var=1e6,
    q_max=10,
    pv_var=5e6,
    shade=[10, 30],
)

ACTIVE_KWARGS = dict(
    num_background_agents=20,
    num_strategic=10,
    num_zi=10,
    num_hbl=10,
    sim_time=20000,
    lam=6e-3,
    mean=1e6,
    lam_melo=5e-3,
    r=0.0001,
    shock_var=1e6,
    q_max=10,
    pv_var=5e6,
    shade=[10, 30],
    holding_period=5,
)


def _run(sim_cls, seed, kwargs):
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = sim_cls(**kwargs)
        sim.run()
    return sim


def _metrics(sim):
    ob = sim.meloMarket.order_book
    bg_zi = [aid for aid, a in sim.agents.items() if isinstance(a, ZIAgent)]
    return {
        "bg_zi_orders_in_book": sum(len(ob.agent_id_map.get(a, [])) for a in bg_zi),
        "total_orders_in_book": sum(len(v) for v in ob.agent_id_map.values()),
        "melo_matches":         len(ob.buy_matched_orders),
        "bg_zi_traded":         sum(1 for aid in bg_zi if sim.agents[aid].position != 0),
    }


def _summarize(sim_cls, kwargs):
    per_seed = [_metrics(_run(sim_cls, s, kwargs)) for s in SEEDS]
    return {
        k: (statistics.mean(d[k] for d in per_seed),
            statistics.pstdev(d[k] for d in per_seed))
        for k in per_seed[0]
    }


def _print_report(label, fixed, buggy, canonical):
    header = f"{'metric':>25s} | {'fixed':>18s} | {'buggy':>18s} | {'canonical':>18s}"
    print(f"\n[{label}]")
    print(header)
    print("-" * len(header))
    for k in fixed:
        fm, fs = fixed[k]
        bm, bs = buggy[k]
        cm, cs = canonical[k]
        print(f"{k:>25s} | {fm:8.2f} ± {fs:6.2f} | {bm:8.2f} ± {bs:6.2f} | {cm:8.2f} ± {cs:6.2f}")


def test_bug_impact_quiet_regime():
    fixed = _summarize(FixedSim, QUIET_KWARGS)
    buggy = _summarize(BuggySim, QUIET_KWARGS)
    canonical = _summarize(CanonicalSim, QUIET_KWARGS)
    _print_report("quiet regime — orders queue but no trades", fixed, buggy, canonical)

    assert buggy["bg_zi_orders_in_book"][0] == 0
    assert fixed["bg_zi_orders_in_book"][0] > 0
    assert fixed["total_orders_in_book"][0] > buggy["total_orders_in_book"][0]
    # canonical has ZIs on CDA (pre-fix defaults), so they never touch the MELO book
    assert canonical["bg_zi_orders_in_book"][0] == 0


def test_bug_impact_active_regime():
    fixed = _summarize(FixedSim, ACTIVE_KWARGS)
    buggy = _summarize(BuggySim, ACTIVE_KWARGS)
    canonical = _summarize(CanonicalSim, ACTIVE_KWARGS)
    _print_report("active regime — MELO matches actually happen", fixed, buggy, canonical)

    assert fixed["melo_matches"][0] > buggy["melo_matches"][0]
    assert fixed["bg_zi_traded"][0] > buggy["bg_zi_traded"][0]
    assert fixed["bg_zi_orders_in_book"][0] > buggy["bg_zi_orders_in_book"][0]
    # canonical: ZIs never route to MELO, so they never end up in the MELO book
    assert canonical["bg_zi_orders_in_book"][0] == 0
