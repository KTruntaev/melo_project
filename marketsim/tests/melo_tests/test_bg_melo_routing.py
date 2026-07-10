import contextlib
import io
import random

import numpy as np
import pytest
import torch

from marketsim.agent.melo_agent import MeloAgent
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival as FixedSim
from marketsim.simulator.melo_simulator_old import MELOSimulatorSampledArrival as BuggySim


SEED = 42
SEEDS = [1, 7, 42, 100, 999]
BASE_KWARGS = dict(
    sim_time=1000,
    lam=6e-3,
    mean=1e6,
    lam_melo=1e-3,
    r=0.0001,
    shock_var=1e6,
    q_max=10,
    num_hbl=0,
    pv_var=5e6,
    shade=[10, 30],
)


def _seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_and_run(sim_cls, num_zi=2, num_mobi=0, seed=SEED):
    _seed_everything(seed)
    sim = sim_cls(
        num_background_agents=num_mobi + num_zi,
        num_strategic=num_mobi,
        num_zi=num_zi,
        **BASE_KWARGS,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
    return sim


def _melo_orders_ever_inserted(sim):
    return sum(len(v) for v in sim.meloMarket.order_book.agent_id_map.values())


def _melo_queue_size(sim):
    ob = sim.meloMarket.order_book
    return (
        ob.buy_eligibility_queue.count()
        + ob.sell_eligibility_queue.count()
        + len(ob.buy_activation_queue)
        + len(ob.sell_activation_queue)
        + len(ob.buy_active_queue)
        + len(ob.sell_active_queue)
    )


def _melo_orders_scheduled(sim):
    return sum(len(v) for v in sim.meloMarket.event_queue.scheduled_activities.values())


@pytest.mark.parametrize("seed", SEEDS)
def test_buggy_sim_strands_bg_melo_orders(seed):
    sim = _build_and_run(BuggySim, seed=seed)
    scheduled = _melo_orders_scheduled(sim)
    inserted = _melo_orders_ever_inserted(sim)
    queued = _melo_queue_size(sim)

    assert scheduled > 0, (
        f"precondition failed (seed={seed}): no ZI MELO orders ever hit "
        "meloMarket.event_queue, so the bug can't be exercised. Check that ZI "
        "agents actually arrived and routed to MELO under this seed/config."
    )
    assert inserted == 0, (
        f"buggy sim (seed={seed}) should insert 0 MELO orders into the book "
        f"(MOBI=0, ZI-only); got {inserted} (of {scheduled} scheduled)"
    )
    assert queued == 0, (
        f"buggy sim (seed={seed}) should have 0 orders in any MELO order-book "
        f"queue; got {queued}"
    )

    print(sim.meloMarket.event_queue.scheduled_activities.values())


@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_sim_routes_bg_melo_orders_to_book(seed):
    sim = _build_and_run(FixedSim, seed=seed)
    scheduled = _melo_orders_scheduled(sim)
    inserted = _melo_orders_ever_inserted(sim)
    queued = _melo_queue_size(sim)

    assert scheduled > 0, (
        f"precondition failed (seed={seed}): no ZI MELO orders ever hit "
        "meloMarket.event_queue, so we can't tell whether the fix routed them "
        "anywhere. Check that ZI agents actually arrived and routed to MELO."
    )
    assert inserted > 0, (
        f"fixed sim (seed={seed}) should insert ZI-routed MELO orders into the "
        f"book; got {inserted} (of {scheduled} scheduled)"
    )
    assert queued > 0, (
        f"fixed sim (seed={seed}) should have at least some MELO orders sitting "
        f"in a book queue at end of sim; got {queued} (of {inserted} inserted)"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_fix_yields_more_melo_activity_than_bug(seed):
    fixed = _build_and_run(FixedSim, seed=seed)
    buggy = _build_and_run(BuggySim, seed=seed)
    assert _melo_orders_ever_inserted(fixed) > _melo_orders_ever_inserted(buggy), (
        f"fixed sim (seed={seed}) should insert strictly more MELO orders than "
        f"buggy sim; fixed={_melo_orders_ever_inserted(fixed)}, "
        f"buggy={_melo_orders_ever_inserted(buggy)}"
    )


def _build_buggy_sim_with_mobi():
    return BuggySim(
        num_background_agents=2,
        num_strategic=1,
        num_zi=1,
        **BASE_KWARGS,
    )


def test_mobi_arrival_drains_stranded_zi_melo_orders_under_bug():
    ZI_ID, MOBI_ID, TICK = 0, 1, 50
    _seed_everything()
    sim = _build_buggy_sim_with_mobi()
    sim.arrivals[TICK].append(ZI_ID)
    sim.arrivals_melo[TICK].append(MOBI_ID)

    snap = {}
    original_step = sim.meloMarket.step
    ob = sim.meloMarket.order_book
    eq = sim.meloMarket.event_queue

    def wrapped(*args, **kw):
        if not snap:
            snap["pending"] = len(eq.scheduled_activities[sim.time])
            snap["inserted_before"] = sum(len(v) for v in ob.agent_id_map.values())
            result = original_step(*args, **kw)
            snap["inserted_after"] = sum(len(v) for v in ob.agent_id_map.values())
            return result
        return original_step(*args, **kw)

    sim.meloMarket.step = wrapped
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()

    print(snap)

    assert snap["inserted_before"] == 0, snap
    assert snap["pending"] >= 2, snap
    assert snap["inserted_after"] == snap["pending"], snap


@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_sim_handles_mixed_mobi_and_zi_natural_arrivals(seed):
    num_zi, num_mobi = 2, 2
    sim = _build_and_run(FixedSim, num_zi=num_zi, num_mobi=num_mobi, seed=seed)

    zi_ids = [aid for aid, a in sim.agents.items() if isinstance(a, ZIAgent)]
    mobi_ids = [aid for aid, a in sim.agents.items() if isinstance(a, MeloAgent)]
    assert len(zi_ids) == num_zi and len(mobi_ids) == num_mobi, (
        f"precondition failed (seed={seed}): expected {num_zi} ZIs and "
        f"{num_mobi} MOBIs; got zi_ids={zi_ids}, mobi_ids={mobi_ids}"
    )

    zi_inserted = sum(len(sim.meloMarket.order_book.agent_id_map.get(a, [])) for a in zi_ids)
    mobi_inserted = sum(len(sim.meloMarket.order_book.agent_id_map.get(a, [])) for a in mobi_ids)

    assert zi_inserted > 0, (
        f"fixed sim (seed={seed}) should insert MELO orders from background ZIs "
        f"in a mixed regime; got {zi_inserted}"
    )
    assert mobi_inserted > 0, (
        f"fixed sim (seed={seed}) should insert MELO orders from strategic MOBIs "
        f"in a mixed regime; got {mobi_inserted}"
    )
