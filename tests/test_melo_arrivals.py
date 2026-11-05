import types
import pytest
import numpy as np

import pathlib, sys, random, os


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marketsim.egta.simulators.melo_wrapper import MeloSimulator, _run_melo_single_rep
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival

# ---------------------------------------------------------------------------
# Helper profile builder
# ---------------------------------------------------------------------------

def make_simple_profile():
    """Return a minimal role-symmetric profile with one agent per strategy."""
    return [
        ("MOBI", "MOBI_0_100_shade0_0"),
        ("ZI",   "ZI_100_0_shade250_500"),
    ]


# ---------------------------------------------------------------------------
# Test 1: wrapper forwards lam_melo_mobi / lam_melo_zi correctly
# ---------------------------------------------------------------------------

def test_wrapper_forwards_per_role_lambda(monkeypatch):
    captured_kwargs = {}

    # Monkey-patch the helper to intercept kwargs without running the heavy sim
    def fake_run_single_rep(args):
        nonlocal captured_kwargs
        _, sim_kwargs = args
        captured_kwargs = sim_kwargs
        # Return dummy payoff dict (agent 0 & 1 with zero profit)
        return {0: 0.0, 1: 0.0}

    monkeypatch.setattr(
        "marketsim.egta.simulators.melo_wrapper._run_melo_single_rep",
        fake_run_single_rep,
    )

    # Build wrapper with distinctive lambdas
    wrapper = MeloSimulator(
        num_strategic_mobi=1,
        num_strategic_zi=1,
        sim_time=10,
        lam=1e-2,
        lam_r=1e-2,
        lam_melo_mobi=1e-4,
        lam_melo_zi=9e-4,
        mobi_strategies=["MOBI_0_100_shade0_0"],
        zi_strategies=["ZI_100_0_shade250_500"],
        reps=1,
        parallel=False,
        log_profile_details=False,
    )

    # Simulate a trivial profile
    wrapper.simulate_profile(make_simple_profile())

    assert captured_kwargs["lam_melo_mobi"] == pytest.approx(1e-4)
    assert captured_kwargs["lam_melo_zi"] == pytest.approx(9e-4)


# ---------------------------------------------------------------------------
# Test 2: MELOSimulatorSampledArrival initialises separate arrival streams
# ---------------------------------------------------------------------------

def test_arrival_streams_separated():
    rs_counts = {"MOBI": {"MOBI_0_100_shade0_0": 1}, "ZI": {"ZI_100_0_shade250_500": 1}}
    sim = MELOSimulatorSampledArrival(
        sim_time=1,
        role_strategy_counts=rs_counts,
        role_names=["MOBI", "ZI"],
        strategy_params={
            "MOBI_0_100_shade0_0": {"cda_proportion": 0.0, "melo_proportion": 1.0, "shade": [0, 0]},
            "ZI_100_0_shade250_500": {"cda_proportion": 1.0, "melo_proportion": 0.0, "shade": [250, 500]},
        },
        lam_melo_mobi=1e-4,
        lam_melo_zi=9e-4,
    )

    # The first scheduled times should differ on average; check ratio of means
    mean_mobi = sim.arrival_times_mobi.cpu().numpy().mean()
    mean_zi = sim.arrival_times_zi.cpu().numpy().mean()
    assert mean_zi < mean_mobi, "ZI arrival times should be faster (smaller)"

    # Role mapping should contain exactly 2 agents
    print(sim.agent_roles)
    assert len(sim.agent_roles) == 2

    assert list(sim.agent_roles.values()).count("MOBI") == 1
    assert list(sim.agent_roles.values()).count("ZI") == 1


# ---------------------------------------------------------------------------
# (Removed) Test 3: Legacy single-`lam_melo` fallback.
# The single-stream `lam_melo` parameter and its None-fallback to the per-role
# rates were purged from the M-ELO simulator; per-role `lam_melo_mobi` /
# `lam_melo_zi` are now the only arrival rates, so there is no fallback to test.
# ---------------------------------------------------------------------------