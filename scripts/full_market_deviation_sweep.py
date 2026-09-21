"""Run full-population MELO market-choice trials."""

import contextlib
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marketsim.simulator.melo_simulator import (  # noqa: E402
    MELOSimulatorSampledArrival as FixedSimulator,
)
from marketsim.simulator.melo_simulator_old import (  # noqa: E402
    MELOSimulatorSampledArrival as BuggySimulator,
)


MOBI = "MOBI"
ZI = "ZI"
MELO = "MELO"
LIT = "LIT"

STRATEGIES = {
    (MOBI, MELO): "MOBI_0_100_shade0_0",
    (MOBI, LIT): "MOBI_100_0_shade250_500",
    (ZI, MELO): "ZI_0_100_shade250_500",
    (ZI, LIT): "ZI_100_0_shade250_500",
}

STRATEGY_PARAMS = {
    STRATEGIES[(MOBI, MELO)]: {
        "cda_proportion": 0.0,
        "melo_proportion": 1.0,
        "shade": [0, 0],
    },
    STRATEGIES[(MOBI, LIT)]: {
        "cda_proportion": 1.0,
        "melo_proportion": 0.0,
        "shade": [250, 500],
    },
    STRATEGIES[(ZI, MELO)]: {
        "cda_proportion": 0.0,
        "melo_proportion": 1.0,
        "shade": [250, 500],
    },
    STRATEGIES[(ZI, LIT)]: {
        "cda_proportion": 1.0,
        "melo_proportion": 0.0,
        "shade": [250, 500],
    },
}


def run_trial(
    agents,
    simulator_cls=FixedSimulator,
    holding_period=0,
    seed=0,
    sim_time=10_000,
):
    """Run one simulation and return one payoff per input agent.

    ``agents`` is a list of ``(role, market)`` pairs. For example::

        agents = [(MOBI, MELO)] * 28 + [(ZI, LIT)] * 40
        payoffs = run_trial(agents, BuggySimulator, holding_period=120)

    The returned payoff list has the same order as ``agents``.
    """
    mobi_profile = []
    zi_markets = []
    for role, market in agents:
        try:
            strategy = STRATEGIES[(role, market)]
        except KeyError as exc:
            raise ValueError(f"Unknown agent choice: {(role, market)}") from exc

        if role == MOBI:
            mobi_profile.append((role, strategy))
        else:
            zi_markets.append(market)

    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch.manual_seed(seed)

    # The simulator currently contains extensive constructor and runtime prints.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            simulator = simulator_cls(
                num_background_agents=len(zi_markets),
                num_zi=len(zi_markets),
                num_hbl=0,
                num_strategic=len(mobi_profile),
                sim_time=sim_time,
                lam=6e-3,
                lam_r=6e-3,
                lam_melo=1e-3,
                lam_melo_mobi=1e-3,
                lam_melo_zi=6e-3,
                mean=1e6,
                r=1e-3,
                shock_var=1e6,
                q_max=10,
                pv_var=5e6,
                shade=[250, 500],
                eta=0.5,
                holding_period=holding_period,
                strategy_params=STRATEGY_PARAMS,
                role_names=[MOBI],
                profile_order=mobi_profile,
            )

            # ZIs must use the background arrival queue to exercise the bugged
            # path. Apply each ZI's fixed routing strategy after construction.
            for agent_id, market in enumerate(zi_markets):
                params = STRATEGY_PARAMS[STRATEGIES[(ZI, market)]]
                zi_agent = simulator.agents[agent_id]
                zi_agent.cda_proportion = params["cda_proportion"]
                zi_agent.melo_proportion = params["melo_proportion"]
                zi_agent.shade = params["shade"]

            simulator.run()

    payoff_by_agent = simulator.end_sim()[0]

    # Background ZIs receive IDs first, followed by strategic MOBIs. Restore
    # the caller's input order so the sweep can keep splitting the payoff list
    # into its MOBI and ZI sections.
    next_zi_id = 0
    next_mobi_id = len(zi_markets)
    payoff_agent_ids = []
    for role, _market in agents:
        if role == ZI:
            payoff_agent_ids.append(next_zi_id)
            next_zi_id += 1
        else:
            payoff_agent_ids.append(next_mobi_id)
            next_mobi_id += 1

    return [float(payoff_by_agent[agent_id]) for agent_id in payoff_agent_ids]


def run_sweep_trial(
    simulator_name,
    simulator_cls,
    mobi_lit,
    zi_lit,
    holding_period,
    seed,
):
    """Run one sweep configuration and return its CSV row."""
    mobi_melo = 28 - mobi_lit
    zi_melo = 40 - zi_lit
    agents = (
        [(MOBI, LIT)] * mobi_lit
        + [(MOBI, MELO)] * mobi_melo
        + [(ZI, LIT)] * zi_lit
        + [(ZI, MELO)] * zi_melo
    )

    payoffs = run_trial(
        agents,
        simulator_cls=simulator_cls,
        holding_period=holding_period,
        seed=seed,
    )
    mobi_payoffs = payoffs[:28]
    zi_payoffs = payoffs[28:]

    return {
        "simulator": simulator_name,
        "holding_period": holding_period,
        "seed": seed,
        "mobi_lit": mobi_lit,
        "mobi_melo": mobi_melo,
        "zi_lit": zi_lit,
        "zi_melo": zi_melo,
        "mobi_mean_payoff": np.mean(mobi_payoffs),
        "zi_mean_payoff": np.mean(zi_payoffs),
        "mobi_total_payoff": np.sum(mobi_payoffs),
        "zi_total_payoff": np.sum(zi_payoffs),
        "total_welfare": np.sum(payoffs),
    }


def main():
    print(os.cpu_count())
    holding_period = 120
    seed = 42
    workers = 24
    simulators = [
        ("buggy", BuggySimulator),
        ("fixed", FixedSimulator),
    ]

    output_dir = ROOT / "results" / "full_market_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"full_market_sweep_{timestamp}.csv"

    fieldnames = [
        "simulator",
        "holding_period",
        "seed",
        "mobi_lit",
        "mobi_melo",
        "zi_lit",
        "zi_melo",
        "mobi_mean_payoff",
        "zi_mean_payoff",
        "mobi_total_payoff",
        "zi_total_payoff",
        "total_welfare",
    ]

    trials = [
        (simulator_name, simulator_cls, mobi_lit, zi_lit, holding_period, seed)
        for simulator_name, simulator_cls in simulators
        for mobi_lit in range(29)
        for zi_lit in range(41)
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_sweep_trial, *trial) for trial in trials]

            for trial_number, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                writer.writerow(row)
                output_file.flush()
                print(
                    f"[{trial_number}/{len(trials)}] {row['simulator']}, "
                    f"MOBI lit={row['mobi_lit']}, ZI lit={row['zi_lit']}"
                )

    print(f"Sweep complete: {output_path}")


if __name__ == "__main__":
    main()
