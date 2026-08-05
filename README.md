# Nexorion Spacetech — Phase 1 Code and Simulation Package

Hybrid quantum-classical warm-starts for Argonaut-class lunar powered-descent
and hazard-avoidance trajectory optimisation.

This bundle supports the Phase 1 (Technology Readiness Level 2 to 3)
feasibility study selected by the European Space Agency under the Open Space
Innovation Platform Call for Ideas: *Quantum Technology for Space Exploration*.
It contains **two public code repositories** and a **simulation package** of two
executed studies.

> **Reading guide.** Abbreviations are written in full on first use, then
> abbreviated. Registered product and company names (IBM Quantum, Q-CTRL,
> SparQ, Gurobi, GuSTO, ArgoNET) are kept as-is.

---

## 0. What is here

```
.
├── repos/
│   ├── scvx-landing/            Classical baseline (continuous layer)
│   └── qaoa-hazard-retarget/    Quantum layer + hybrid pipeline
└── simulation-package/          Two executed studies: code, data, figures
```

Two things are true at once, and it matters that you know which is which:

- **Validated and executed today.** The Quadratic Unconstrained Binary
  Optimisation (QUBO) encoding and its correctness proof; the lander dynamics
  and their analytical Jacobians; and both simulation studies with all their
  numbers and figures. These run and pass now.
- **Scaffolded, to be completed during the funded campaign.** The Successive
  Convexification (SCvx) subproblem solver loop and the Quantum Approximate
  Optimisation Algorithm (QAOA) circuit execution against real backends. These
  are present as documented skeletons that raise `NotImplementedError` at the
  exact points where campaign work begins.

This honesty is deliberate: reproducibility is the only currency at Technology
Readiness Level 3, and a reader must be able to tell a proven claim from a
planned one at a glance.

---

## 1. Repository: `scvx-landing`

The classical arm — the fuel-optimal powered-descent problem solved by
Successive Convexification. This is the baseline that any quantum contribution
must beat, and the continuous refinement stage the hybrid pipeline feeds into.

### Layout

```
scvx-landing/
├── src/scvx/
│   ├── dynamics.py              3-DOF powered-descent dynamics + Jacobians
│   └── solver.py                SCvx solver loop (skeleton)
├── benchmarks/
│   └── frozen_instance.yaml     NEXORION-FI-2026-A instance definition
├── tests/
│   └── test_analytical.py       Jacobian validation (passing)
├── environment.yml
├── Dockerfile
└── README.md
```

### `dynamics.py` — implemented and validated

Three-degree-of-freedom point-mass powered-descent dynamics with mass
depletion, over the seven-dimensional state (position, velocity, mass):

```
ṙ = v            v̇ = T/m + g            ṁ = −‖T‖ / (Isp · g₀)
```

- `f(x, u, p)` — continuous-time dynamics.
- `jacobians(x, u, p)` — the analytical state and control Jacobians A = ∂f/∂x
  and B = ∂f/∂u, used by the SCvx linearisation. **Validated against central
  finite differences** to a tolerance of 1e-4 in `test_analytical.py`.
- `discretise_foh(...)` — first-order-hold discretisation about the reference
  trajectory. **Scaffold** (`NotImplementedError`, Work Package 1 Sub-task 1.2):
  to be implemented via integration of the state-transition matrix.

### `solver.py` — scaffold

`solve(params, target_site, config, initial_reference)` runs the SCvx loop:
linearise → solve the convex second-order-cone subproblem (fuel objective,
lossless convexification of the thrust lower bound, virtual-control slack,
trust region) → accept/reject → repeat. The function signature, configuration
(`ScvxConfig`, fixed seed 2026), and result type (`ScvxResult` with a
convergence profile) are defined; the loop body is the campaign deliverable.

The key integration point is the **`initial_reference` argument** — this is the
hook where the quantum layer's chosen landing site enters as a warm start.

### `benchmarks/frozen_instance.yaml` — the credibility artefact

The instance **NEXORION-FI-2026-A**. Every benchmark claim refers to this file;
it is committed and tagged before any benchmarking, and it is immutable after
the tag. Bracketed values (`TODO`) are the Argonaut-class parameters fixed once
at kick-off. It defines the hazard-map family, lander dynamics parameters,
constraints, cost function, and three instance sizes — small (exactly
enumerable, for encoding validation), medium, and large (benchmark).

---

## 2. Repository: `qaoa-hazard-retarget`

The quantum arm — encodes the discrete landing-site selection problem, solves
it with a variational quantum algorithm, and composes the result with the
classical solver into the hybrid pipeline.

### Layout

```
qaoa-hazard-retarget/
├── src/
│   ├── qubo_encoder.py          QUBO encoding + exactness (implemented)
│   ├── qaoa_warmstart.py        QAOA warm-start + pre-registered constants
│   └── benchmark_runner.py      Campaign runner (helpers implemented)
├── tests/
│   └── test_qubo_bruteforce.py  Brute-force exactness suite (passing)
├── environment.yml
├── Dockerfile
└── README.md
```

### `qubo_encoder.py` — implemented and validated (Critical Function 1)

The discrete subproblem: choose exactly one landing site `i` minimising the
combined cost `c_i = hazard_i + divert_i`, subject to a one-hot constraint.
Encoded as a QUBO with a penalty term:

```
H(x) = Σᵢ cᵢ xᵢ  +  P · (Σᵢ xᵢ − 1)²
```

- `build_qubo(hazard, divert, penalty=None)` — returns the symmetric QUBO
  matrix.
- `penalty_weight(costs)` — the **proven sufficient penalty** `P > max_i c_i`.
  For non-negative costs this guarantees the ground state is the one-hot vector
  selecting the true optimum. (The proof sketch is in the module docstring; an
  earlier, weaker bound based on the *spread* of costs was caught failing
  against the all-zeros state by the test suite below — which is exactly why
  the suite is a blocking gate.)
- `brute_force_optimum(Q)` — exact enumeration, for validation at enumerable
  sizes only (asserts n ≤ 20).

### `test_qubo_bruteforce.py` — the Critical Function 1 experimental proof

Generates 50 randomised instances (4–12 sites, fixed seed) and asserts on every
one that the encoded ground state is one-hot **and** equals the independently
computed exact optimum, plus that the all-zeros and multi-hot states are
strictly dominated. **All pass.** This is the experimental half of the
Technology Readiness Level 3 evidence for Critical Function 1.

### `qaoa_warmstart.py` — constants real, execution scaffolded

The **pre-registered protocol constants are fixed here in code**, not just in
prose, so the executed and pre-registered protocols are verifiably identical:

```python
SEED = 2026;  SHOTS = 10_000;  MC_RUNS = 100
ZNE_SCALES = (1, 2, 3)                    # zero-noise extrapolation
TWO_QUBIT_ERROR = 3e-3                     # nominal noise
FALSIFICATION_ERROR = 1e-2                 # falsification setting
FALSIFICATION_THRESHOLD = 0.10             # <10% median improvement ⇒ null
```

`run_qaoa(...)` and `site_to_reference(...)` are **scaffolds**
(`NotImplementedError`, Work Package 2 Sub-task 2.1): the Qiskit circuit
construction, parameter optimisation, noise model, and zero-noise extrapolation
are the campaign deliverable. The falsification threshold is the honesty
mechanism — a median improvement below it is reported as a valid **null result**,
not hidden.

### `benchmark_runner.py` — helpers real, campaign loop scaffolded

`commit_hash()` and `bootstrap_ci(...)` (10,000-resample bootstrap 95%
confidence intervals) are implemented. `evaluate_falsification(...)` applies the
pre-registered condition verbatim. `run_campaign(...)` — the per-instance,
per-Monte-Carlo-run loop that persists one record per run with instance
identifier, commit hash, seed, and configuration — is the **scaffold** for
Work Package 2 Sub-tasks 2.2/2.3.

---

## 3. The hybrid pipeline (how the two repositories compose)

```
   hazard map ──► QUBO encoder ──► QAOA ──► selected landing site
   (frozen        (qaoa-hazard-  (qaoa-      │
    instance)      retarget)      hazard-    │  straight-line-to-site
                                  retarget)  ▼  reference trajectory
                                        SCvx solver  ◄── initial_reference
                                        (scvx-landing)     hook
                                             │
                                             ▼
                                    refined fuel-optimal trajectory
```

The **headline quantity** of Phase 1 is the fuel-margin difference between this
hybrid path and the classical baseline (SCvx from its default initialisation)
on the frozen instance family, with bootstrap confidence intervals, at the
pre-registered noise settings.

---

## 4. The simulation package

Two **executed** studies that rehearse the full analysis and reporting pipeline
on scaled-down models. Every number was produced by running the code with fixed
seeds — nothing is hand-entered.

> **Standing rule.** These are synthetic, illustrative studies — **not** the
> pre-registered campaign. Their numbers must never be inserted into the
> Technology Readiness Level 3 Achievement Report, which accepts numbers only
> from the tagged campaign dataset. Every figure is watermarked accordingly.

### Layout

```
simulation-package/
├── sim_scenario.py      Study 1: three-arm comparison
├── plots.py             Study 1: figures 1–3
├── summary.json         Study 1: statistics
├── raw.npz              Study 1: run-level data
├── sim_scenario2.py     Study 2: extended six-method study
├── plots2.py            Study 2: figures 4–5
├── summary2.json        Study 2: statistics
├── raw2.npz             Study 2: run-level data
└── fig1..fig5 .png      All five figures
```

### Study 1 — three-arm scenario (`sim_scenario.py`)

Compares **classical**, **hybrid quantum-classical**, and **quantum-only** on
the frozen 12-site instances. The quantum layer is an exact 12-qubit
depth-2 QAOA statevector simulation with COBYLA parameter optimisation; noise is
a depolarising proxy at the two pre-registered error rates; site selection is
the mode of the feasible one-hot samples from 10,000 shots.

Headline result (median improvement over classical, 95% confidence interval):

| Arm | Nominal (3×10⁻³) | Falsification (1×10⁻²) |
|---|---|---|
| Hybrid quantum-classical | **+8.5%** [+6.2, +10.3] | +0.0% [+0.0, +2.6] |
| Quantum only | −3.9% [−9.5, −2.2] | −11.5% [−12.3, −11.4] |

The ordering **hybrid > classical > quantum-only** holds across the noise sweep,
and the falsification analogue triggers at the high-noise setting — a rehearsed
null result.

### Study 2 — extended six-method study (`sim_scenario2.py`)

Adds an explicit **classical model** (three discrete methods: greedy,
simulated annealing, exact enumeration) and a **five-layer Monte-Carlo model**
where every method decides on *measured* hazard costs (truth + Gaussian
estimation noise) but every margin is scored on *truth*. Three **distinct
quantum methods** are compared:

| Method | Description | Depth |
|---|---|---|
| Q1 QAOA | Depth-2, optimised angles | 2 |
| Q2 Digitised adiabatic | Trotterised annealing schedule, no optimisation | 8 |
| Q3 Warm-start QAOA | Initial state biased to the classical incumbent | 2 |

Median true-margin improvement over the greedy baseline (nominal noise):

| Method | Improvement | Optimal-site rate |
|---|---|---|
| C2 Simulated annealing | +13.3% | 100% |
| C3 Exact enumeration | +13.3% | 100% |
| Q1 QAOA (hybrid) | +6.2% | 23.5% |
| Q2 Digitised adiabatic (hybrid) | −3.3% | 9.0% |
| Q3 Warm-start (hybrid) | +0.0% | 7.5% |

Key finding: **circuit depth is the currency of the noisy era** — the deeper,
more principled adiabatic method (528 two-qubit gates) is driven below baseline
by noise, while the shallow optimised QAOA is the only quantum method with
observed upside. The extended study document also surveys advanced techniques
(Quantum Amplitude Estimation, quantum machine learning, Hamiltonian methods)
with adoption verdicts.

---

## 5. Reproduce everything

### Repositories

```bash
# classical baseline
cd repos/scvx-landing
conda env create -f environment.yml && conda activate scvx-landing
PYTHONPATH=src pytest -q          # Jacobian validation

# quantum layer
cd ../qaoa-hazard-retarget
conda env create -f environment.yml && conda activate qaoa-hazard-retarget
PYTHONPATH=src pytest -q          # Critical Function 1 brute-force proof
```

Each repository also ships a `Dockerfile` (`docker build -t <name> . &&
docker run <name>` runs the test suite from a clean image).

### Simulation studies

```bash
cd simulation-package
pip install numpy scipy matplotlib
python sim_scenario.py    # regenerates summary.json + raw.npz
python plots.py           # regenerates figures 1–3
python sim_scenario2.py   # regenerates summary2.json + raw2.npz (few minutes)
python plots2.py          # regenerates figures 4–5
```

All seeds are fixed (base seed 2026); reruns reproduce the tables above.

---

## 6. Conventions and licence

- **Reference scheme:** `NEXORION-[PROGRAMME]-[TYPE]-[YEAR]-[NNN]`.
- **Frozen-instance discipline:** the tagged instance is immutable; every
  result cites the instance identifier and commit hash.
- **Pre-registration discipline:** benchmark methodology is fixed before results
  are seen; null results are reported honestly against the falsification
  condition.
- **Licence:** MIT for all code in both repositories.

---

## 7. Status checklist

**Implemented and validated**
- [x] QUBO encoding, proven penalty bound, brute-force suite (50 instances)
- [x] Lander dynamics + analytical Jacobians (finite-difference validated)
- [x] Frozen instance schema; pre-registered constants fixed in code
- [x] Bootstrap statistics; both simulation studies with figures

**Campaign deliverables (scaffolded)**
- [ ] SCvx first-order-hold discretisation and solver loop
- [ ] Analytical vertical-descent recovery test
- [ ] QAOA circuit execution, noise model, zero-noise extrapolation
- [ ] Simulator and IBM Quantum hardware campaigns
- [ ] Independent reproduction record
