"""Extended simulated study — NEXORION-PH1-SIM-2026-002.
STATUS: SYNTHETIC / ILLUSTRATIVE. Executed simulation, not campaign data.

CLASSICAL MODEL (discrete layer methods, all deciding on MEASURED costs):
  C1 GREEDY   : nearest site (min divert) — information-poor baseline.
  C2 SIM-ANN  : classical simulated annealing on the measured QUBO.
  C3 EXACT    : exact enumeration on measured costs — classical ceiling.

QUANTUM SCENARIOS (three different methods, hybrid: each feeds refinement):
  Q1 QAOA     : depth-2 Quantum Approximate Optimisation Algorithm,
                parameters optimised (COBYLA, multistart).
  Q2 ADIABATIC: digitised adiabatic evolution — depth-8 Trotterised linear
                annealing schedule of H(s) = (1-s)Hx + s*Hc. Pure
                Hamiltonian-evolution method, no parameter optimisation.
  Q3 WS-QAOA  : warm-start QAOA — initial state biased toward the classical
                greedy solution (regularised bias), depth-2, optimised.

MONTE CARLO MODEL (layered uncertainty, aerospace dispersion style):
  L1 Instance dispersion : 5 frozen instances (hazard, divert fields).
  L2 Estimation noise    : measured hazard = true hazard + N(0, sigma_h);
                           ALL methods decide on measured, margins are
                           evaluated on TRUE costs. 40 dispersions/instance.
  L3 Hardware noise      : depolarising mixing at the two pre-registered
                           two-qubit error rates (quantum arms only).
  L4 Shot noise          : 10^4 shots per evaluation, modal feasible sample.
  L5 Statistics          : bootstrap 95% confidence intervals on medians.

Metrics: (a) true-margin improvement over C1; (b) rate of selecting the
measured-optimal site (approximation quality = scalability proxy).
"""
import sys, json, pathlib
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/home/claude/ph1/repos/qaoa-hazard-retarget/src")
from qubo_encoder import build_qubo

SEED = 2026
N_SITES = 12
N_INSTANCES = 5
N_DISPERSIONS = 40          # L2 Monte-Carlo dispersions per instance
SHOTS = 10_000
SIGMA_H = 1.5               # hazard estimation noise (0-10 scale)
M0, K, R = 15.0, 0.8, 1.5
EPS = {"nominal": 3e-3, "falsification": 1e-2}
N_BOOT = 10_000


# ---------------- shared machinery ------------------------------------------
def all_energies(Q):
    n = Q.shape[0]
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    return np.einsum("si,ij,sj->s", X, Q, X)


def apply_mixer(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        psi = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)
        a0, a1 = psi[:, 0, :].copy(), psi[:, 1, :].copy()
        psi[:, 0, :] = c * a0 + s * a1
        psi[:, 1, :] = s * a0 + c * a1
        psi = psi.reshape(-1)
    return psi


def circuit_probs(Escale, params, p, n, psi0):
    gammas, betas = params[:p], params[p:]
    psi = psi0.copy()
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * Escale)
        psi = apply_mixer(psi, b, n)
    return np.abs(psi) ** 2


def optimise(E, p, n, psi0, seed, maxiter=120, starts=2):
    Escale = E / np.abs(E).max()
    def expval(params):
        return float(circuit_probs(Escale, params, p, n, psi0) @ E)
    best = None
    for t in range(starts):
        x0 = np.random.default_rng(seed + t).uniform(0, np.pi, 2 * p)
        r = minimize(expval, x0, method="COBYLA",
                     options={"maxiter": maxiter, "rhobeg": 0.4})
        if best is None or r.fun < best.fun:
            best = r
    return circuit_probs(Escale, best.x, p, n, psi0)


def noisy(probs, eps, n, depth):
    G = depth * n * (n - 1) // 2
    lam = 1.0 - (1.0 - eps) ** G
    return (1 - lam) * probs + lam / probs.size


def modal_site(probs, rng, n):
    samples = rng.choice(probs.size, size=SHOTS, p=probs)
    onehot = 1 << np.arange(n)
    feas = samples[np.isin(samples, onehot)]
    if feas.size == 0:
        return None
    vals, counts = np.unique(feas, return_counts=True)
    return int(np.log2(vals[np.argmax(counts)]))


# ---------------- the three quantum methods ---------------------------------
def q1_qaoa(E, n, seed):
    psi0 = np.full(2 ** n, 1 / np.sqrt(2 ** n), complex)
    return q1_qaoa.depth, optimise(E, 2, n, psi0, seed)
q1_qaoa.depth = 2


def q2_adiabatic(E, n, seed):
    """Digitised adiabatic: linear ramp s_k = k/p over p Trotter steps of
    H(s) = (1-s) Hx + s Hc; gamma_k = s_k*dt, beta_k = (1-s_k)*dt. No
    parameter optimisation — pure Hamiltonian-evolution scenario."""
    p, dt = 8, 0.75
    Escale = E / np.abs(E).max()
    s = (np.arange(1, p + 1)) / p
    params = np.concatenate([s * dt, (1 - s) * dt])
    psi0 = np.full(2 ** n, 1 / np.sqrt(2 ** n), complex)
    return p, circuit_probs(Escale, params, p, n, psi0)


def q3_ws_qaoa(E, n, seed, greedy_site, bias=0.8):
    """Warm-start QAOA: product initial state biased toward the classical
    greedy one-hot string (regularised: qubit g at amplitude sqrt(bias),
    others at sqrt((1-bias)/4) toward |1>), then depth-2 optimised QAOA."""
    amps1 = np.full(n, np.sqrt(0.05))
    amps1[greedy_site] = np.sqrt(bias)
    amps0 = np.sqrt(1 - amps1 ** 2)
    # Build product state with index bit q <-> qubit q (matches apply_mixer
    # and the energy table's bit convention): ascending loop makes each new
    # qubit the high-order bit, so qubit q occupies bit q of the state index.
    psi0 = np.ones(1, complex)
    for q in range(n):
        psi0 = np.concatenate([amps0[q] * psi0, amps1[q] * psi0])
    return 2, optimise(E, 2, n, psi0, seed)


# ---------------- classical discrete methods --------------------------------
def c2_simulated_annealing(cost_meas, rng, iters=300, T0=5.0):
    n = cost_meas.size
    s = int(rng.integers(n))
    best, best_c = s, cost_meas[s]
    for t in range(iters):
        T = T0 * (1 - t / iters) + 1e-3
        cand = int(rng.integers(n))
        d = cost_meas[cand] - cost_meas[s]
        if d < 0 or rng.random() < np.exp(-d / T):
            s = cand
            if cost_meas[s] < best_c:
                best, best_c = s, cost_meas[s]
    return best


# ---------------- campaign ---------------------------------------------------
def margin_true(site, cost_true):
    return M0 - K * cost_true[site]


def bootstrap_ci(x, rng):
    meds = [np.median(rng.choice(x, x.size, replace=True))
            for _ in range(N_BOOT)]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def run():
    methods_q = {"Q1_qaoa": q1_qaoa, "Q2_adiabatic": q2_adiabatic,
                 "Q3_ws_qaoa": q3_ws_qaoa}
    out = {lvl: {m: {"impr": [], "opt_rate": []}
                 for m in ["C2_simann", "C3_exact", *methods_q]}
           for lvl in EPS}

    for k in range(N_INSTANCES):
        rng_i = np.random.default_rng(SEED + 100 + k)
        hazard_true = rng_i.uniform(0, 10, N_SITES)
        divert = rng_i.uniform(0, 5, N_SITES)
        cost_true = hazard_true + divert
        greedy = int(np.argmin(divert))

        for d in range(N_DISPERSIONS):
            rng_d = np.random.default_rng(SEED + 1000 * k + d)
            hazard_meas = np.clip(
                hazard_true + rng_d.normal(0, SIGMA_H, N_SITES), 0, None)
            cost_meas = hazard_meas + divert
            Q = build_qubo(hazard_meas, divert)
            E = all_energies(Q)
            opt_meas = int(np.argmin(cost_meas))
            m_c1 = margin_true(greedy, cost_true) + R

            # classical methods (noise-level independent; record under both)
            s_c2 = c2_simulated_annealing(cost_meas, rng_d)
            picks_cl = {"C2_simann": s_c2, "C3_exact": opt_meas}

            # quantum methods: ideal distribution once per dispersion
            dists = {}
            for name, fn in methods_q.items():
                if name == "Q3_ws_qaoa":
                    depth, pr = fn(E, N_SITES, SEED + 17 * d + k, greedy)
                else:
                    depth, pr = fn(E, N_SITES, SEED + 17 * d + k)
                dists[name] = (depth, pr)

            for lvl, eps in EPS.items():
                for name, site in picks_cl.items():
                    m = margin_true(site, cost_true) + R
                    out[lvl][name]["impr"].append(100 * (m - m_c1) / m_c1)
                    out[lvl][name]["opt_rate"].append(int(site == opt_meas))
                for name, (depth, pr) in dists.items():
                    pn = noisy(pr, eps, N_SITES, depth)
                    s = modal_site(pn, np.random.default_rng(
                        SEED + 77 * d + k + int(eps * 1e6)), N_SITES)
                    s_h = greedy if s is None else s
                    m = margin_true(s_h, cost_true) + R
                    out[lvl][name]["impr"].append(100 * (m - m_c1) / m_c1)
                    out[lvl][name]["opt_rate"].append(int(s_h == opt_meas))

    rng_b = np.random.default_rng(SEED + 7)
    summary = {"config": {"n_sites": N_SITES, "instances": N_INSTANCES,
                          "dispersions": N_DISPERSIONS, "sigma_hazard": SIGMA_H,
                          "shots": SHOTS, "eps": EPS},
               "levels": {}}
    for lvl in EPS:
        summary["levels"][lvl] = {}
        for m, d in out[lvl].items():
            x = np.array(d["impr"])
            summary["levels"][lvl][m] = {
                "median_improvement_pct": float(np.median(x)),
                "ci95": bootstrap_ci(x, rng_b),
                "opt_site_rate": float(np.mean(d["opt_rate"])),
                "n": int(x.size)}
    return summary, out


if __name__ == "__main__":
    summary, raw = run()
    outp = pathlib.Path("/home/claude/ph1/sim")
    (outp / "summary2.json").write_text(json.dumps(summary, indent=2))
    np.savez(outp / "raw2.npz",
             **{f"{lvl}_{m}": np.array(raw[lvl][m]["impr"])
                for lvl in raw for m in raw[lvl]})
    print(json.dumps(summary, indent=2))
