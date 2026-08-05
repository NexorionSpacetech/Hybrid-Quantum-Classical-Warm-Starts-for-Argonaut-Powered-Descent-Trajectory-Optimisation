"""Simulated three-arm benchmark scenario — NEXORION-PH1-SIM-2026-001.

STATUS: SYNTHETIC / ILLUSTRATIVE. This is a real, executed simulation of a
scaled-down pipeline, NOT the pre-registered Phase 1 campaign. Its numbers
must never be inserted into the TRL 3 Achievement Report.

Arms compared on the same frozen instances:
  A  CLASSICAL : greedy nearest-site initialisation (min divert cost),
                 then classical continuous refinement (SCvx proxy).
  B  HYBRID    : QAOA solves the site-selection QUBO; modal feasible sample
                 selects the site; classical continuous refinement follows.
  C  QUANTUM   : QAOA site selection only; NO classical refinement
                 (discrete solution used directly).

Physics/optimisation model (transparent, fixed before running):
  - Site cost c_i = hazard_i + divert_i (non-negative).
  - Fuel margin of a landing at site i:  margin_i = M0 - K * c_i   [percent]
  - Classical continuous refinement (trajectory shaping) adds +R percentage
    points; the quantum-only arm does not receive it.
  - QAOA: p=2 exact statevector simulation on n=12 sites (one qubit per
    site, 4096 states), parameters optimised by COBYLA on <E>.
  - Noise: global depolarising proxy — the output distribution is mixed with
    the uniform distribution, lambda = 1 - (1 - eps)^G, where G = p*n(n-1)/2
    is the two-qubit gate count of the fully-connected cost layers and eps
    the two-qubit error rate. Evaluated at eps = 3e-3 (nominal) and
    eps = 1e-2 (falsification setting), mirroring the pre-registered levels.
  - Site selection rule: mode of the feasible (one-hot) samples from 10^4
    shots; if no feasible sample is observed, the hybrid arm falls back to
    the classical greedy initialisation (see run()).
  - Statistics: 5 instances x 100 Monte-Carlo runs/arm/noise level,
    10^4 shots per run, fixed seeds, bootstrap 95% confidence intervals
    (10^4 resamples) on the median improvement over the classical arm.
"""
import sys, json, pathlib
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/home/claude/ph1/repos/qaoa-hazard-retarget/src")
from qubo_encoder import build_qubo  # the real, validated CF-1 encoder

# ----- fixed scenario constants (pre-committed before running) --------------
SEED = 2026
N_SITES = 12          # enumerable size of the frozen instance definition
N_INSTANCES = 5
MC_RUNS = 100
SHOTS = 10_000
P_LAYERS = 2
M0, K, R = 15.0, 0.8, 1.5        # margin model: base %, cost slope, refinement
EPS_LEVELS = {"nominal_3e-3": 3e-3, "falsification_1e-2": 1e-2}
N_BOOT = 10_000

rng_master = np.random.default_rng(SEED)


# ----- instance generation ---------------------------------------------------
def make_instance(rng):
    hazard = rng.uniform(0.0, 10.0, N_SITES)
    divert = rng.uniform(0.0, 5.0, N_SITES)
    return hazard, divert


def margin(site, cost):
    return M0 - K * cost[site]


# ----- exact QAOA statevector on the QUBO -----------------------------------
def all_energies(Q):
    n = Q.shape[0]
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    return np.einsum("si,ij,sj->s", X, Q, X), X


def apply_mixer(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        psi = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)
        a0, a1 = psi[:, 0, :].copy(), psi[:, 1, :].copy()
        psi[:, 0, :] = c * a0 + s * a1
        psi[:, 1, :] = s * a0 + c * a1
        psi = psi.reshape(-1)
    return psi


def qaoa_distribution(Q, p=P_LAYERS):
    n = Q.shape[0]
    E, _ = all_energies(Q)
    Escale = E / np.abs(E).max()

    def state(params):
        gammas, betas = params[:p], params[p:]
        psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), complex)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * Escale)
            psi = apply_mixer(psi, b, n)
        return psi

    def expval(params):
        pr = np.abs(state(params)) ** 2
        return float(pr @ E)

    best = None
    for trial in range(3):  # multistart
        x0 = np.random.default_rng(SEED + trial).uniform(0, np.pi, 2 * p)
        res = minimize(expval, x0, method="COBYLA",
                       options={"maxiter": 200, "rhobeg": 0.4})
        if best is None or res.fun < best.fun:
            best = res
    return np.abs(state(best.x)) ** 2, E


def noisy(probs, eps, n, p=P_LAYERS):
    G = p * n * (n - 1) // 2                 # ZZ couplings, fully connected
    lam = 1.0 - (1.0 - eps) ** G
    return (1 - lam) * probs + lam / probs.size, lam


def select_site(probs, rng, n):
    """Modal feasible (one-hot) sample from SHOTS shots; None if none seen."""
    samples = rng.choice(probs.size, size=SHOTS, p=probs)
    onehot_states = 1 << np.arange(n)         # integers with a single bit set
    feas = samples[np.isin(samples, onehot_states)]
    if feas.size == 0:
        return None
    vals, counts = np.unique(feas, return_counts=True)
    return int(np.log2(vals[np.argmax(counts)]))


# ----- campaign --------------------------------------------------------------
def bootstrap_ci(x, rng, n_boot=N_BOOT):
    meds = [np.median(rng.choice(x, x.size, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def run():
    results = {lvl: {"hybrid": [], "quantum": [], "classical_margin": [],
                     "hybrid_site_opt_rate": [], "quantum_infeasible": 0,
                     "lambda": None}
               for lvl in EPS_LEVELS}
    per_instance = []

    for k in range(N_INSTANCES):
        rng_i = np.random.default_rng(SEED + 100 + k)
        hazard, divert = make_instance(rng_i)
        cost = hazard + divert
        Q = build_qubo(hazard, divert)
        probs_ideal, _ = qaoa_distribution(Q)

        site_greedy = int(np.argmin(divert))          # classical default init
        site_opt = int(np.argmin(cost))
        m_classical = margin(site_greedy, cost) + R    # refined
        per_instance.append({
            "instance": k, "greedy_site": site_greedy, "opt_site": site_opt,
            "cost_greedy": float(cost[site_greedy]),
            "cost_opt": float(cost[site_opt]),
            "classical_margin_pct": float(m_classical),
            "ideal_margin_pct": float(margin(site_opt, cost) + R)})

        for lvl, eps in EPS_LEVELS.items():
            pn, lam = noisy(probs_ideal, eps, N_SITES)
            results[lvl]["lambda"] = float(lam)
            for run_id in range(MC_RUNS):
                rng_r = np.random.default_rng(SEED + 10_000 * k + run_id
                                              + int(eps * 1e6))
                s = select_site(pn, rng_r, N_SITES)
                if s is None:                          # infeasible output
                    results[lvl]["quantum_infeasible"] += 1
                    s_h = site_greedy                  # hybrid falls back
                    m_q = 0.0                          # mission-infeasible
                else:
                    s_h = s
                    m_q = margin(s, cost)              # NO refinement
                m_h = margin(s_h, cost) + R
                results[lvl]["hybrid"].append(100 * (m_h - m_classical) / m_classical)
                results[lvl]["quantum"].append(100 * (m_q - m_classical) / m_classical)
                results[lvl]["hybrid_site_opt_rate"].append(int(s_h == site_opt))

    rng_b = np.random.default_rng(SEED + 7)
    summary = {"per_instance": per_instance, "levels": {}}
    for lvl in EPS_LEVELS:
        h = np.array(results[lvl]["hybrid"]); q = np.array(results[lvl]["quantum"])
        summary["levels"][lvl] = {
            "lambda_mixing": results[lvl]["lambda"],
            "hybrid_median_improvement_pct": float(np.median(h)),
            "hybrid_ci95": bootstrap_ci(h, rng_b),
            "quantum_median_improvement_pct": float(np.median(q)),
            "quantum_ci95": bootstrap_ci(q, rng_b),
            "hybrid_opt_site_rate": float(np.mean(results[lvl]["hybrid_site_opt_rate"])),
            "quantum_infeasible_runs": results[lvl]["quantum_infeasible"],
            "n_datapoints_per_arm": int(h.size)}
    # falsification analogue, evaluated in the simulated world
    fal = summary["levels"]["falsification_1e-2"]["hybrid_median_improvement_pct"]
    summary["falsification_analogue"] = {
        "condition": "median hybrid improvement < 10% at eps = 1e-2",
        "measured_pct": fal, "triggered": bool(fal < 10.0)}
    return summary, results


if __name__ == "__main__":
    summary, raw = run()
    out = pathlib.Path("/home/claude/ph1/sim")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / "raw.npz",
             **{f"{lvl}_{arm}": np.array(raw[lvl][arm])
                for lvl in raw for arm in ("hybrid", "quantum")})
    print(json.dumps(summary["levels"], indent=2))
    print(json.dumps(summary["falsification_analogue"], indent=2))
