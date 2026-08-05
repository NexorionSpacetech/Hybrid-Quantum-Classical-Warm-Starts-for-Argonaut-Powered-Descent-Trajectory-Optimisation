"""Quantum Approximate Optimisation Algorithm warm-start — CF-2.

Pipeline position: the QAOA solves the QUBO of qubo_encoder.py (landing-site
selection under hazard and divert cost); the selected site and its
straight-line reference trajectory are passed to the classical Successive
Convexification solver (scvx-landing) as `initial_reference`. The measured
effect of this initialisation vs the classical default, on the frozen
instance family, is the Phase 1 headline quantity.

Pre-registered settings (Execution Pack Section 6 — do not amend):
  backend: Qiskit FakeHeronV2 (or latest Heron calibration at 1 April 2026)
  noise:   depolarising, two-qubit error 3e-3
  ZNE:     linear extrapolation, scale factors {1, 2, 3},
           pulse-efficient transpilation
  stats:   100 Monte-Carlo runs/instance, 1e4 shots/evaluation,
           fixed seed, bootstrap 95% confidence intervals
  falsification: advantage declared absent if median fuel-margin
           improvement < 10% at two-qubit error 1e-2
"""
from dataclasses import dataclass
import numpy as np

SEED = 2026
SHOTS = 10_000
MC_RUNS = 100
ZNE_SCALES = (1, 2, 3)
TWO_QUBIT_ERROR = 3e-3
FALSIFICATION_ERROR = 1e-2
FALSIFICATION_THRESHOLD = 0.10  # median fuel-margin improvement


@dataclass
class QaoaResult:
    best_bitstring: np.ndarray
    energy: float
    depth_p: int
    raw_counts: dict
    calibration_snapshot: dict  # archived per hardware run (WP2.3)


def run_qaoa(Q: np.ndarray, p: int = 2, backend: str = "FakeHeronV2",
             seed: int = SEED, shots: int = SHOTS) -> QaoaResult:
    """Execute QAOA on the QUBO under the pre-registered protocol.

    TODO(WP2 Sub-task 2.1): Qiskit implementation — cost Hamiltonian from Q,
    standard mixer, COBYLA/SPSA parameter optimisation, noise model at
    TWO_QUBIT_ERROR, zero-noise extrapolation over ZNE_SCALES. Verify on
    noiseless simulation first; every run logs seed + full configuration.
    """
    raise NotImplementedError("WP2 Sub-task 2.1")


def site_to_reference(site_index: int, grid_shape: tuple, x0: np.ndarray,
                      n_nodes: int) -> np.ndarray:
    """Straight-line-to-site reference trajectory for the SCvx warm start.

    TODO(WP2 Sub-task 2.1): map site index -> surface coordinates on the
    frozen grid, interpolate states from x0, return the (n_nodes, 7) array
    consumed by scvx.solver.solve(initial_reference=...).
    """
    raise NotImplementedError("WP2 Sub-task 2.1")
