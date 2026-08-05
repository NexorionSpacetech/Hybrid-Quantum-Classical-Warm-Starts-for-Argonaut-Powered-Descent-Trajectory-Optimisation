import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from sim_scenario import (make_instance, build_qubo, qaoa_distribution, noisy,
                          select_site, margin, R, SEED, N_INSTANCES, N_SITES,
                          MC_RUNS)

NAVY, TEAL, RED, GREY = "#1F3864", "#2E86AB", "#C0392B", "#7f8c8d"

summary = json.load(open("summary.json"))
raw = np.load("raw.npz")

# ---------- Figure 1: three-arm comparison at the two pre-registered levels
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, lvl, ttl in zip(
        axes, ["nominal_3e-3", "falsification_1e-2"],
        ["Nominal noise (two-qubit error 3\u00d710\u207b\u00b3)",
         "Falsification setting (two-qubit error 1\u00d710\u207b\u00b2)"]):
    L = summary["levels"][lvl]
    arms = ["Classical\n(baseline)", "Hybrid\nquantum-classical", "Quantum\nonly"]
    med = [0.0, L["hybrid_median_improvement_pct"], L["quantum_median_improvement_pct"]]
    lo = [0.0, L["hybrid_ci95"][0], L["quantum_ci95"][0]]
    hi = [0.0, L["hybrid_ci95"][1], L["quantum_ci95"][1]]
    err = [np.array(med) - np.array(lo), np.array(hi) - np.array(med)]
    bars = ax.bar(arms, med, color=[GREY, TEAL, RED], width=0.6,
                  yerr=err, capsize=6, ecolor=NAVY)
    ax.axhline(0, color=NAVY, lw=1)
    ax.axhline(10, color=NAVY, lw=1, ls="--", alpha=0.6)
    ax.text(2.35, 10.4, "10% falsification\nthreshold", fontsize=8,
            color=NAVY, ha="right")
    for b, m in zip(bars, med):
        ax.text(b.get_x() + b.get_width()/2, m + (0.6 if m >= 0 else -1.6),
                f"{m:+.1f}%", ha="center", fontsize=10, fontweight="bold",
                color=NAVY)
    ax.set_title(ttl, fontsize=10, color=NAVY)
    ax.set_ylabel("Median fuel-margin improvement over classical (%)")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("SIMULATED SCENARIO \u2014 NOT CAMPAIGN DATA", fontsize=11,
             color=RED, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig1_three_arm.png", dpi=160)

# ---------- Figure 2: distributions
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, lvl, ttl in zip(
        axes, ["nominal_3e-3", "falsification_1e-2"],
        ["Nominal noise (3\u00d710\u207b\u00b3)", "Falsification setting (1\u00d710\u207b\u00b2)"]):
    h, q = raw[f"{lvl}_hybrid"], raw[f"{lvl}_quantum"]
    parts = ax.violinplot([h, q], showmedians=True, widths=0.7)
    for pc, col in zip(parts["bodies"], [TEAL, RED]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    parts["cmedians"].set_color(NAVY)
    for k in ("cbars", "cmins", "cmaxes"):
        parts[k].set_color(NAVY)
    ax.axhline(0, color=GREY, lw=1, ls=":")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Hybrid", "Quantum only"])
    ax.set_title(ttl, fontsize=10, color=NAVY)
    ax.set_ylabel("Improvement over classical (%)")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Run-level distributions (500 Monte-Carlo runs per arm) \u2014 SIMULATED",
             fontsize=11, color=RED, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig2_distributions.png", dpi=160)

# ---------- Figure 3: noise sweep
eps_grid = np.logspace(-4, np.log10(3e-2), 12)
hyb_med, qua_med = [], []
insts = []
for k in range(N_INSTANCES):
    rng_i = np.random.default_rng(SEED + 100 + k)
    hazard, divert = make_instance(rng_i)
    cost = hazard + divert
    Q = build_qubo(hazard, divert)
    probs, _ = qaoa_distribution(Q)
    insts.append((cost, int(np.argmin(divert)), probs))
for eps in eps_grid:
    hs, qs = [], []
    for k, (cost, greedy, probs) in enumerate(insts):
        m_cl = margin(greedy, cost) + R
        pn, _ = noisy(probs, eps, N_SITES)
        for run_id in range(30):
            rng_r = np.random.default_rng(SEED + 555*k + run_id + int(eps*1e7))
            s = select_site(pn, rng_r, N_SITES)
            s_h = greedy if s is None else s
            m_q = 0.0 if s is None else margin(s, cost)
            hs.append(100*(margin(s_h, cost) + R - m_cl)/m_cl)
            qs.append(100*(m_q - m_cl)/m_cl)
    hyb_med.append(np.median(hs)); qua_med.append(np.median(qs))

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.semilogx(eps_grid, hyb_med, "o-", color=TEAL, label="Hybrid quantum-classical")
ax.semilogx(eps_grid, qua_med, "s-", color=RED, label="Quantum only")
ax.axhline(0, color=GREY, lw=1.2, label="Classical baseline")
ax.axhline(10, color=NAVY, lw=1, ls="--", alpha=0.6)
ax.text(1.1e-4, 10.5, "10% falsification threshold", fontsize=8, color=NAVY)
for x, lbl in [(3e-3, "nominal"), (1e-2, "falsification")]:
    ax.axvline(x, color=NAVY, lw=0.8, ls=":", alpha=0.7)
    ax.text(x, ax.get_ylim()[0]+1, f" {lbl}", rotation=90, fontsize=8, color=NAVY)
ax.set_xlabel("Two-qubit gate error rate")
ax.set_ylabel("Median improvement over classical (%)")
ax.set_title("Advantage vs noise \u2014 SIMULATED SCENARIO", color=RED,
             fontweight="bold", fontsize=11)
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig3_noise_sweep.png", dpi=160)
print("figures written")
print("sweep hybrid medians:", [round(v,1) for v in hyb_med])
print("sweep quantum medians:", [round(v,1) for v in qua_med])
