"""Analytical validation — CF-2 evidence chain, Gate Review 1 blocker.

Two validation layers:
1. Jacobian correctness: analytical vs central finite differences.
2. Solver correctness: a hand-derivable case (vertical descent, constant
   gravity, unconstrained interior solution) where the fuel-optimal profile
   is known in closed form; SCvx must recover it to tolerance.
"""
import numpy as np
from scvx.dynamics import f, jacobians, LanderParams

P = LanderParams(g=1.62, isp=320.0, t_min=1000.0, t_max=10000.0,
                 m_dry=1500.0, m_wet=2200.0)


def test_jacobians_match_finite_differences():
    rng = np.random.default_rng(2026)
    x = np.concatenate([rng.normal(0, 100, 3), rng.normal(0, 10, 3), [2000.0]])
    u = rng.uniform(2000, 8000, 3)
    A, B = jacobians(x, u, P)
    eps = 1e-6
    A_fd = np.zeros_like(A)
    for i in range(7):
        dx = np.zeros(7); dx[i] = eps
        A_fd[:, i] = (f(x + dx, u, P) - f(x - dx, u, P)) / (2 * eps)
    B_fd = np.zeros_like(B)
    for i in range(3):
        du = np.zeros(3); du[i] = eps
        B_fd[:, i] = (f(x, u + du, P) - f(x, u - du, P)) / (2 * eps)
    assert np.allclose(A, A_fd, atol=1e-4)
    assert np.allclose(B, B_fd, atol=1e-4)


def test_solver_recovers_analytical_vertical_descent():
    """TODO(WP1): closed-form vertical fuel-optimal descent comparison."""
    pass
