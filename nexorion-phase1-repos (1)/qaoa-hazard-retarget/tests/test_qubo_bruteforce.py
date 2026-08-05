"""CF-1 experimental proof: encoded optimum == exact optimum, every instance.

Gate Review 1 blocker. Runs in continuous integration on randomly generated
small instances (within the enumeration bound of the frozen instance
definition) with fixed seed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np
from qubo_encoder import build_qubo, brute_force_optimum, qubo_energy

def test_ground_state_is_one_hot_optimum():
    rng = np.random.default_rng(2026)
    for _ in range(50):
        n = int(rng.integers(4, 13))
        hazard = rng.uniform(0, 10, n)
        divert = rng.uniform(0, 5, n)
        Q = build_qubo(hazard, divert)
        x_star, _ = brute_force_optimum(Q)
        i_true = int(np.argmin(hazard + divert))
        assert x_star.sum() == 1, "ground state must satisfy one-hot"
        assert int(np.argmax(x_star)) == i_true, "encoded optimum must match"

def test_penalty_enforces_feasibility():
    rng = np.random.default_rng(7)
    hazard, divert = rng.uniform(0, 10, 8), rng.uniform(0, 5, 8)
    Q = build_qubo(hazard, divert)
    n = 8
    feasible_best = min(qubo_energy(Q, np.eye(n)[i]) for i in range(n))
    assert qubo_energy(Q, np.zeros(n)) > feasible_best
    two_hot = np.zeros(n); two_hot[[0, 1]] = 1
    assert qubo_energy(Q, two_hot) > feasible_best
