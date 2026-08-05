"""Pre-registered benchmark campaign runner — CF-3.

Compares, on the frozen instance family NEXORION-FI-2026-A:
  arm A: classical SCvx from the default initialisation
  arm B: hybrid — QAOA-selected site warm-start feeding SCvx

Primary quantity: fuel-margin improvement of arm B over arm A, median across
MC_RUNS Monte-Carlo runs per instance, with bootstrap 95% confidence
intervals. The falsification condition of qaoa_warmstart is evaluated
verbatim; secondary analyses must be labelled exploratory.

Every result record carries: instance_id, commit hash, seed, backend,
calibration snapshot (hardware runs), and full configuration — the
reproducibility requirements of Execution Pack Annex C.
"""
import json, subprocess, datetime
import numpy as np
from qaoa_warmstart import SEED, MC_RUNS, FALSIFICATION_THRESHOLD

def commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()

def bootstrap_ci(samples: np.ndarray, n_boot: int = 10_000,
                 seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(samples, samples.size, replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

def run_campaign(instance_family: list, hardware: bool = False) -> dict:
    """TODO(WP2 Sub-tasks 2.2/2.3): iterate instances x MC_RUNS, run both
    arms, persist one JSON record per run under results/ with the metadata
    above. Simulator campaign first (2.2); hardware subset confirmatory (2.3).
    """
    raise NotImplementedError("WP2 Sub-tasks 2.2 / 2.3")

def evaluate_falsification(median_improvement_at_1e2: float) -> str:
    """Verbatim pre-registered condition — do not amend."""
    if median_improvement_at_1e2 < FALSIFICATION_THRESHOLD:
        return "NULL RESULT: hypothesised quantum advantage declared absent."
    return "Advantage hypothesis survives the pre-registered condition."
