import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, RED, GREY, GOLD, GREEN = "#1F3864", "#2E86AB", "#C0392B", "#7f8c8d", "#B8860B", "#2E8B57"
s = json.load(open("summary2.json"))

methods = ["C1_greedy", "C2_simann", "C3_exact", "Q1_qaoa", "Q2_adiabatic", "Q3_ws_qaoa"]
labels = ["C1 Greedy\n(baseline)", "C2 Simulated\nannealing", "C3 Exact\nenumeration",
          "Q1 QAOA\n(hybrid)", "Q2 Digitised\nadiabatic (hybrid)", "Q3 Warm-start\nQAOA (hybrid)"]
colors = [GREY, GREEN, GOLD, TEAL, "#6A5ACD", "#20B2AA"]

def vals(lvl):
    med, lo, hi = [0.0], [0.0], [0.0]
    for m in methods[1:]:
        d = s["levels"][lvl][m]
        med.append(d["median_improvement_pct"])
        lo.append(d["ci95"][0]); hi.append(d["ci95"][1])
    return np.array(med), np.array(lo), np.array(hi)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
for ax, lvl, ttl in zip(axes, ["nominal", "falsification"],
        ["Nominal noise (two-qubit error 3\u00d710\u207b\u00b3)",
         "Falsification setting (two-qubit error 1\u00d710\u207b\u00b2)"]):
    med, lo, hi = vals(lvl)
    err = [med - lo, hi - med]
    bars = ax.bar(range(6), med, color=colors, width=0.62,
                  yerr=err, capsize=5, ecolor=NAVY)
    ax.axhline(0, color=NAVY, lw=1)
    for i, m in enumerate(med):
        ax.text(i, m + (0.7 if m >= 0 else -1.9), f"{m:+.1f}%",
                ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title(ttl, fontsize=10, color=NAVY)
    ax.set_ylabel("Median true-margin improvement over C1 (%)")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Six-method comparison under the layered Monte-Carlo model \u2014 SIMULATED SCENARIO, NOT CAMPAIGN DATA",
             fontsize=11, color=RED, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig4_six_methods.png", dpi=160)

# approximation quality
fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = np.arange(5); w = 0.38
for off, lvl, col, lbl in [(-w/2, "nominal", TEAL, "Nominal (3\u00d710\u207b\u00b3)"),
                           (w/2, "falsification", RED, "Falsification (1\u00d710\u207b\u00b2)")]:
    rates = [s["levels"][lvl][m]["opt_site_rate"] * 100 for m in methods[1:]]
    bars = ax.bar(x + off, rates, w, color=col, alpha=0.85, label=lbl)
    for xi, r in zip(x + off, rates):
        ax.text(xi, r + 1.5, f"{r:.1f}%", ha="center", fontsize=8, color=NAVY)
ax.set_xticks(x); ax.set_xticklabels([l.replace("\n", " ") for l in labels[1:]], fontsize=8)
ax.set_ylabel("Measured-optimal site selected (% of runs)")
ax.set_title("Approximation quality \u2014 the scalability proxy \u2014 SIMULATED",
             color=RED, fontweight="bold", fontsize=11)
ax.legend(frameon=False); ax.set_ylim(0, 115)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig5_opt_rate.png", dpi=160)
print("ok")
