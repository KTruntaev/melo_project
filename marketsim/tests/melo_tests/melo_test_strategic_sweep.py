"""
Strategic-agent market-split sweep — a bite-sized companion to melo_test_1.py.

Unlike melo_test_1.py (which creates *background* ZI agents on the `if agents:`
step path and flips a per-arrival coin via a fractional melo_proportion), this test
reproduces how the *real* simulation / EGTA wrapper assigns markets:

  * Agents are STRATEGIC, built via role_strategy_counts + role_names +
    strategy_params, so ZI run through the `if melo_agents:` step path like
    production (melo_wrapper.py with num_background_zi = 0).
  * Strategies are PURE / "degenerate coins": each agent is assigned either
    all-lit (cda 1.0 / melo 0.0) or all-MELO (cda 0.0 / melo 1.0). The simulator's
    `random.random() < melo_proportion` draw then resolves deterministically, i.e.
    the agent "submits all orders to the chosen market" (paper §5.1).
  * The sweep varies the POPULATION split — how many of the NUM_ZI agents are
    assigned the all-MELO strategy — not a within-agent probability.

Run:  cd marketsim && PYTHONPATH=. python tests/melo_tests/melo_test_strategic_sweep.py
"""
import os, io, random, contextlib
from collections import Counter
import numpy as np
import torch

from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from marketsim.agent.zero_intelligence_agent import ZIAgent

# ------------------------- configuration -------------------------
REPS      = 20
SIM_TIME  = 2000
HP        = 50                      # holding period
SHADE     = [10, 30]

NUM_MOBI  = 10
MOBI_MELO = 5                       # how many MOBI assigned all-MELO (rest all-lit)

NUM_ZI    = 30
ZI_MELO_SWEEP = [0, 6, 12, 18, 24, 30]   # how many ZI assigned all-MELO (the swept split)

# ZI order-removal behavior on re-arrival. "intended" = paper's always-withdraw,
# which runs clean on the strategic path. "keep" = the new working-tree algorithm
# (keep while >= R_min) — expected to hit `assert len(orders)==1` on this path.
# "accumulate" = never withdraw (the original pile-up bug).
REMOVAL = "intended"
# -----------------------------------------------------------------

# Pure ("degenerate coin") strategy params. Names mirror the wrapper's 100_0 / 0_100.
STRATEGY_PARAMS = {
    "MOBI_100_0": {"cda_proportion": 1.0, "melo_proportion": 0.0, "shade": SHADE},
    "MOBI_0_100": {"cda_proportion": 0.0, "melo_proportion": 1.0, "shade": SHADE},
    "ZI_100_0":   {"cda_proportion": 1.0, "melo_proportion": 0.0, "shade": SHADE},
    "ZI_0_100":   {"cda_proportion": 0.0, "melo_proportion": 1.0, "shade": SHADE},
}


def build_sim(zi_melo):
    """One sim with `zi_melo` of the NUM_ZI agents assigned all-MELO (rest all-lit)."""
    role_strategy_counts = {
        "MOBI": Counter({"MOBI_0_100": MOBI_MELO, "MOBI_100_0": NUM_MOBI - MOBI_MELO}),
        "ZI":   Counter({"ZI_0_100": zi_melo,     "ZI_100_0": NUM_ZI - zi_melo}),
    }
    return MELOSimulatorSampledArrival(
        num_background_agents=NUM_MOBI + NUM_ZI,
        sim_time=SIM_TIME,
        num_zi=0, num_hbl=0,                              # no *background* agents -> all strategic
        num_strategic=NUM_MOBI + NUM_ZI,
        role_strategy_counts=role_strategy_counts,
        role_names=["MOBI", "ZI"],
        strategy_params=STRATEGY_PARAMS,
        lam=6e-3, lam_melo=1e-3, lam_melo_mobi=1e-3, lam_melo_zi=6e-3,
        mean=1e6, r=0.0001, shock_var=1e6, q_max=10, pv_var=5e6,
        shade=SHADE, holding_period=HP,
    )


def apply_removal(sim):
    for a in sim.agents.values():
        if not isinstance(a, ZIAgent):
            continue
        if REMOVAL == "intended":
            a.melo_order_has_min_surplus = lambda est: False          # always withdraw
        elif REMOVAL == "accumulate":
            a.melo_order_has_min_surplus = lambda est: False
            a.withdraw_melo_order = lambda reason="": None             # ...but make it a no-op
        # "keep" -> leave the default new algorithm in place


def collect(sim):
    zis = [a for a in sim.agents.values() if isinstance(a, ZIAgent)]
    zi_ids = {a.agent_id for a in zis}
    ob = sim.meloMarket.order_book
    matched = list(ob.buy_matched_orders) + list(ob.sell_matched_orders)
    zi_exec = len({(m.order.agent_id, m.order.order_id) for m in matched if m.order.agent_id in zi_ids})
    zi_placed = sum(getattr(a, "melo_orders_placed", 0) for a in zis)
    cda_exec = len({(m.order.agent_id, m.order.order_id) for m in sim.market.matched_orders})
    return dict(zi_placed=zi_placed, zi_exec=zi_exec, cda_exec=cda_exec,
                fill=(zi_exec / zi_placed) if zi_placed else float("nan"))


print(f"Strategic market-split sweep  ·  removal={REMOVAL}  ·  "
      f"{REPS} reps × {SIM_TIME} steps  ·  HP={HP}")
print(f"MOBI: {MOBI_MELO}/{NUM_MOBI} on MELO (pure)   ZI swept over {ZI_MELO_SWEEP} of {NUM_ZI} on MELO\n")
print(f"{'ZI MELO share':>14}{'placed':>10}{'MELO exec':>11}{'fill rate':>11}{'CDA exec':>10}{'crashed':>9}")

for zi_melo in ZI_MELO_SWEEP:
    data, crashed = [], 0
    for rep in range(REPS):
        random.seed(rep); np.random.seed(rep); torch.manual_seed(rep)
        sim = build_sim(zi_melo)
        apply_removal(sim)
        try:
            with contextlib.redirect_stdout(_DEVNULL):
                sim.run()
        except Exception as e:                            # e.g. assert len(orders)==1 for "keep"
            crashed += 1
            continue
        data.append(collect(sim))

    share = f"{zi_melo}/{NUM_ZI} ({100*zi_melo//NUM_ZI}%)"
    if data:
        placed = np.mean([d["zi_placed"] for d in data])
        mexec = np.mean([d["zi_exec"] for d in data])
        cexec = np.mean([d["cda_exec"] for d in data])
        fills = [d["fill"] for d in data if d["fill"] == d["fill"]]
        fill = f"{100*np.mean(fills):.1f}%" if fills else "n/a"
        print(f"{share:>14}{placed:>10.1f}{mexec:>11.1f}{fill:>11}{cexec:>10.1f}{crashed:>9}")
    else:
        print(f"{share:>14}{'—':>10}{'—':>11}{'—':>11}{'—':>10}{crashed:>9}  (all reps crashed)")