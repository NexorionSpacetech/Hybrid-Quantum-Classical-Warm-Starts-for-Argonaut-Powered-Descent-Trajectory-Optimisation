"""Quadratic Unconstrained Binary Optimisation encoding of landing-site
selection — critical function CF-1.

Discrete subproblem (frozen instance NEXORION-FI-2026-A):
choose exactly one landing site i from the grid, minimising
    hazard_cost[i] + divert_cost[i]
subject to the one-hot (mutual exclusion) constraint sum_i x_i = 1.

Encoding: H(x) = sum_i (h_i + d_i) x_i + P * (sum_i x_i - 1)^2

Analytical claim (CF-1, to be documented in the derivation note): for
non-negative site costs c_i = h_i + d_i and penalty weight P > max_i c_i,
the ground state of H is the one-hot vector selecting the true optimum.
Sketch: the best one-hot state has energy c_min - P < 0; the all-zeros
state has energy 0; any k-hot state with k >= 2 has energy
sum-of-selected-costs + P*k*(k-2)  (zero penalty exactly at k=2), which is
bounded below by 2*c_min > c_min - P. The brute-force
suite below is the experimental proof of this claim (and caught an earlier,
weaker penalty bound that failed against the all-zeros state).

Experimental proof: brute-force enumeration on all instances small enough to
enumerate (tests/test_qubo_bruteforce.py) — every encoded optimum must match
the exact optimum.
"""
import numpy as np


def penalty_weight(site_costs: np.ndarray) -> float:
    """Sufficient penalty for exactness of the one-hot constraint (CF-1).

    P > max_i c_i guarantees the best one-hot state (energy c_min - P < 0)
    dominates the all-zeros state (energy 0) and every k-hot state, k >= 2.
    Assumes non-negative site costs, which holds for the frozen instance.
    """
    assert site_costs.min() >= 0, "frozen instance assumes non-negative costs"
    return float(site_costs.max()) + 1.0


def build_qubo(hazard_cost: np.ndarray, divert_cost: np.ndarray,
               penalty: float | None = None) -> np.ndarray:
    """Return the symmetric QUBO matrix Q with H(x) = x^T Q x + const.

    Expanding P*(sum x_i - 1)^2 = P*sum_i x_i^2 + 2P*sum_{i<j} x_i x_j
                                   - 2P*sum_i x_i + P
    and using x_i^2 = x_i for binaries:
      diagonal:     c_i + P - 2P = c_i - P
      off-diagonal: 2P (split symmetrically)
    """
    c = np.asarray(hazard_cost, float) + np.asarray(divert_cost, float)
    n = c.size
    P = penalty if penalty is not None else penalty_weight(c)
    Q = np.full((n, n), P)          # 2P split over Q[i,j] + Q[j,i]
    np.fill_diagonal(Q, c - P)
    return Q


def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ Q @ x)


def brute_force_optimum(Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact minimum by enumeration — tractable only for the small
    (enumerable) instance size fixed in the frozen instance definition."""
    n = Q.shape[0]
    assert n <= 20, "enumeration bound exceeded — use the small instance size"
    best_x, best_e = None, np.inf
    for k in range(2 ** n):
        x = np.array([(k >> i) & 1 for i in range(n)], float)
        e = qubo_energy(Q, x)
        if e < best_e:
            best_x, best_e = x, e
    return best_x, best_e
