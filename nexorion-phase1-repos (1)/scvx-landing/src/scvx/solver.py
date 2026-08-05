"""Successive Convexification (SCvx) solver for fuel-optimal powered descent.

Structure of the loop (per iteration):
  1. Linearise dynamics about the reference (dynamics.discretise_foh).
  2. Solve the convex subproblem (SOCP): fuel-optimal objective, linearised
     dynamics with virtual-control slack, trust region about the reference,
     thrust bounds via lossless convexification, glide-slope and pointing cone.
  3. Accept/reject step by comparing actual vs predicted cost reduction;
     update trust-region radius.
  4. Converge when defects and trust-region step fall below tolerance.

Warm-start hook (CF-2): `initial_reference` accepts an externally supplied
reference trajectory. The hybrid pipeline in qaoa-hazard-retarget passes the
QAOA-selected landing site's straight-line-to-site reference here; the
measured effect of that initialisation vs the default is the Phase 1 headline
quantity.
"""
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp  # noqa: F401  (subproblem assembly in WP1)

from .dynamics import LanderParams


@dataclass
class ScvxConfig:
    n_nodes: int = 40
    max_iter: int = 30
    tr_radius0: float = 10.0
    tol_defect: float = 1e-6
    tol_step: float = 1e-4
    seed: int = 2026  # fixed seed — pre-registration requirement


@dataclass
class ScvxResult:
    converged: bool
    iterations: int
    final_mass: float
    trajectory: np.ndarray
    controls: np.ndarray
    convergence_profile: list = field(default_factory=list)  # per-iter cost/defect


def solve(params: LanderParams, target_site: np.ndarray,
          config: ScvxConfig = ScvxConfig(),
          initial_reference: np.ndarray | None = None) -> ScvxResult:
    """Run the SCvx loop to the given landing site.

    TODO(WP1 Sub-task 1.2): assemble the SOCP subproblem, implement the
    accept/reject trust-region logic, and profile convergence across the
    frozen instance family. Validate against the analytical case in
    tests/test_analytical.py before any benchmarking (Gate Review 1).
    """
    raise NotImplementedError("WP1 Sub-task 1.2")
