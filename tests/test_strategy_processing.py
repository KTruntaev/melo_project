import os, sys, pathlib
# Ensure project root is on PYTHONPATH when tests are run from within the tests directory.
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import importlib


import pytest
from marketsim.egta.simulators.melo_wrapper import MeloSimulator
from marketsim.simulator import    melo_simulator
import inspect

# Dynamically import the module (keeps test robust to package layout changes)
msim = importlib.import_module("marketsim.simulator.melo_simulator")
MELOSim = msim.MELOSimulatorSampledArrival

@pytest.fixture
def strategy_lists():
    mobi = [
        "MOBI_0_100_shade0_0",
        "MOBI_100_0_shade0_200",
        "MOBI_100_0_shade250_500",
    ]
    zi = [
        "ZI_100_0_shade100_500",
        "ZI_0_100_shade250_500",
        "ZI_100_0_shade250_500",
    ]
    return mobi, zi


def test_melo_simulator_strategy_params(strategy_lists):
    """Ensure each supplied strategy appears in strategy_params with correct CDA/MELO split."""
    mobi, zi = strategy_lists

    sim = MeloSimulator(
        num_strategic_mobi=len(mobi),
        num_strategic_zi=len(zi),
        sim_time=1,               # tiny for fast initialise
        lam=1e-2,
        lam_r=1e-2,
        lam_melo=1e-3,
        mean=1000,
        r=0.0,
        shock_var=1.0,
        q_max=1,
        pv_var=1.0,
        shade=[10, 30],
        holding_period=0,
        num_background_zi=0,
        num_background_hbl=0,
        reps=1,
        mobi_strategies=mobi,
        zi_strategies=zi,
        parallel=False,
    )

    # (1) All strategies are registered
    assert set(sim.get_strategies()) == set(mobi + zi)

    # (2) Allocation proportions parsed correctly from names
    for strat in mobi + zi:
        params = sim.strategy_params.get(strat)
        assert params is not None, f"Missing params for {strat}"

        # Check allocation proportions
        if "0_100" in strat:            # 100% MELO, 0% CDA
            assert params["melo_proportion"] == 1.0
            assert params["cda_proportion"] == 0.0
        elif "100_0" in strat:          # 100% CDA, 0% MELO
            assert params["cda_proportion"] == 1.0
            assert params["melo_proportion"] == 0.0
        else:
            pytest.fail(f"Unrecognised proportion pattern in {strat}")

        # Check that shade pair matches what is encoded in the name
        try:
            shade_suffix = strat.split("_shade", 1)[1]
            low, high = map(int, shade_suffix.split("_", 1))
            assert params["shade"] == [low, high], f"Shade mismatch for {strat}"
        except Exception:
            pytest.fail(f"Cannot parse shade from {strat}")

# ---------------------------------------------------------------------------
# Additional behaviour: holding-period propagation & profile simulation
# ---------------------------------------------------------------------------

def test_simulator_holding_period_and_profile(strategy_lists):
    """Simulator should store the holding_period and accept a full role-symmetric profile."""
    mobi, zi = strategy_lists

    holding = 7  # unique value to check propagation

    sim = MeloSimulator(
        num_strategic_mobi=len(mobi),
        num_strategic_zi=len(zi),
        sim_time=1,
        lam=1e-2,
        lam_r=1e-2,
        lam_melo=1e-3,
        mean=1000,
        r=0.0,
        shock_var=1.0,
        q_max=1,
        pv_var=1.0,
        shade=[10, 30],
        holding_period=holding,
        num_background_zi=0,
        num_background_hbl=0,
        reps=1,
        mobi_strategies=mobi,
        zi_strategies=zi,
        parallel=False,
    )

    # Verify the parameter travelled to the inner simulator.
    assert sim.holding_period == holding

    # Build a profile matching the required number of strategic players
    profile = [("MOBI", s) for s in mobi] + [("ZI", s) for s in zi]

    obs = sim.simulate_profile(profile)

    # Observation should have one payoff per strategic player
    assert len(obs.payoffs) == len(profile)
    # Profile key stored in observation should equal the sorted version we expect
    expected_key = tuple(sorted(profile))
    assert obs.profile_key == expected_key 


# ---------------------------------------------------------------------------
# Helper: breadth-first search for an attribute in a (potentially nested) object
# ---------------------------------------------------------------------------
def find_attr(obj, attr_name, max_depth=4):
    """Return list of (path, value) pairs where `attr_name` was found."""
    seen = set()
    queue = [([], obj, 0)]
    hits = []

    while queue:
        path, cur, depth = queue.pop(0)
        if id(cur) in seen or depth > max_depth:
            continue
        seen.add(id(cur))

        if hasattr(cur, attr_name):
            hits.append((".".join(path) or "<root>", getattr(cur, attr_name)))

        # Recurse into *public* attributes & items that look like objects
        for name, member in inspect.getmembers(cur):
            if name.startswith("_"):
                continue
            # skip callables (methods/functions) and basic types
            if callable(member) or isinstance(member, (int, float, str, bytes, bool, tuple, list, dict, set)):
                continue
            queue.append((path + [name], member, depth + 1))

    return hits


# ---------------------------------------------------------------------------
# Fixture with a *distinctive* holding period so we can find it later
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configured_sim():
    hp = 37          # unusual value to make the trace unambiguous
    sim = MeloSimulator(
        num_strategic_mobi=1,
        num_strategic_zi=1,
        sim_time=1,
        lam=1e-3,
        lam_r=1e-3,
        lam_melo=1e-4,
        mean=1000,
        r=0.0,
        shock_var=1.0,
        q_max=1,
        pv_var=1.0,
        shade=[10, 30],
        holding_period=hp,
        num_background_zi=0,
        num_background_hbl=0,
        reps=1,
        mobi_strategies=["MOBI_100_0_shade0_100"],
        zi_strategies=["ZI_100_0_shade0_100"],
        parallel=False,
    )
    return sim, hp


# ---------------------------------------------------------------------------
# Actual test
# ---------------------------------------------------------------------------
def test_holding_period_reaches_engine(configured_sim):
    sim, hp = configured_sim

    # Quick sanity check at wrapper level (already covered in your earlier test)
    assert sim.holding_period == hp

    # Now walk the object graph looking for *any* attribute named 'holding_period'
    hits = find_attr(sim, "holding_period")

    assert hits, "Could not locate `holding_period` anywhere underneath MeloSimulator"

    # Every occurrence we find should match the value we injected
    mismatches = [(p, v) for p, v in hits if v != hp]
    assert not mismatches, f"`holding_period` mismatches found: {mismatches}"

    # Optional: print locations when run with -s for easier debugging
    for p, v in hits:
        print(f"[trace] holding_period at {p or '<root>'} = {v}")



def find_attr(obj, attr_name, max_depth=3):
    """Return list of (dotted_path, value) where `attr_name` occurs."""
    queue = [([], obj, 0)]
    seen, hits = set(), []

    while queue:
        path, cur, depth = queue.pop(0)
        if id(cur) in seen or depth > max_depth:
            continue
        seen.add(id(cur))

        if hasattr(cur, attr_name):
            hits.append((".".join(path) or "<root>", getattr(cur, attr_name)))

        for name, member in inspect.getmembers(cur):
            if name.startswith("_") or callable(member):
                continue
            if isinstance(member, (int, float, str, bytes, bool, tuple, list, dict, set)):
                continue
            queue.append((path + [name], member, depth + 1))
    return hits


# ---------------------------------------------------------------------------
# Main test (robust version that doesn't assume attribute at root)
# ---------------------------------------------------------------------------
def test_holding_period_propagation_melo_simulator():
    HP = 91  # eye-catching value to trace

    sim = MELOSim(
        sim_time=1,
        profile_order=[("MOBI", "MOBI_0_100_shade0_0")],
        strategy_params={
            "MOBI_0_100_shade0_0": {"cda_proportion": 0.0, "melo_proportion": 1.0, "shade": [10, 30]},
        },
        mean=1000, r=0.0, shock_var=1.0,
        q_max=1, pv_var=1.0, shade=[10, 30],
        holding_period=HP,
        lam_melo_mobi=1e-4, lam_melo_zi=1e-4,
    )

    # 1) top-level simulator keeps the value
    assert sim.holding_period == HP

    # 2) find *any* occurrence inside MeloMarket
    hits = find_attr(sim.meloMarket, "holding_period", max_depth=2)
    assert hits, "No attribute named 'holding_period' found in MeloMarket tree"
    assert all(v == HP for _p, v in hits), f"Inconsistent values found: {hits}"

    # Optional: show trace with  pytest -s
    # for p, v in hits:
    #     print(f"[trace] {p or '<root>'}.holding_period = {v}")