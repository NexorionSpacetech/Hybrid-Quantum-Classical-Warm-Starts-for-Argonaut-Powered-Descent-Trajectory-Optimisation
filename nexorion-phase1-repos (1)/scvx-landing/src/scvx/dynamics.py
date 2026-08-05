"""Three-degree-of-freedom powered-descent dynamics with mass depletion.

State x = [r (3), v (3), m (1)]; control u = thrust vector T (3).
    r_dot = v
    v_dot = T/m + g
    m_dot = -||T|| / (Isp * g0)

The continuous dynamics are discretised for the Successive Convexification
(SCvx) loop via first-order-hold linearisation about the current reference
trajectory, yielding A_k, B_k, and the defect term for the convex subproblem.
"""
from dataclasses import dataclass
import numpy as np

G0 = 9.80665  # standard gravity for Isp definition, m/s^2


@dataclass(frozen=True)
class LanderParams:
    g: float          # local gravity magnitude, m/s^2 (lunar: 1.62)
    isp: float        # specific impulse, s
    t_min: float      # min thrust magnitude, N
    t_max: float      # max thrust magnitude, N
    m_dry: float      # dry mass, kg
    m_wet: float      # wet mass, kg


def f(x: np.ndarray, u: np.ndarray, p: LanderParams) -> np.ndarray:
    """Continuous-time dynamics x_dot = f(x, u)."""
    r, v, m = x[0:3], x[3:6], x[6]
    g_vec = np.array([0.0, 0.0, -p.g])
    r_dot = v
    v_dot = u / m + g_vec
    m_dot = -np.linalg.norm(u) / (p.isp * G0)
    return np.concatenate([r_dot, v_dot, [m_dot]])


def jacobians(x: np.ndarray, u: np.ndarray, p: LanderParams):
    """Analytical Jacobians A = df/dx, B = df/du about (x, u).

    Used by the SCvx linearisation. Validated against finite differences in
    tests/test_analytical.py — this validation is part of the CF-2 analytical
    evidence chain (integration argument for the warm-start).
    """
    m = x[6]
    u_norm = np.linalg.norm(u)
    A = np.zeros((7, 7))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 6] = -u / m**2
    B = np.zeros((7, 3))
    B[3:6, :] = np.eye(3) / m
    if u_norm > 1e-12:
        B[6, :] = -u / (u_norm * p.isp * G0)
    return A, B


def discretise_foh(x_ref, u_ref, p: LanderParams, dt: float):
    """First-order-hold discretisation about the reference trajectory.

    Returns per-node (A_k, B_minus_k, B_plus_k, defect_k) for the convex
    subproblem. TODO(WP1): implement via matrix exponential / integration of
    the state-transition matrix; verify defects vanish on a dynamically
    feasible reference.
    """
    raise NotImplementedError("WP1 Sub-task 1.2")
