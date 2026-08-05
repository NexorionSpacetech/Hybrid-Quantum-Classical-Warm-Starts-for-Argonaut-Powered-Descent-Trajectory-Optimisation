# scvx-landing

Classical Successive Convexification (SCvx) baseline for fuel-optimal lunar
powered descent — the classical arm of Nexorion Spacetech's Phase 1 benchmark
(European Space Agency Open Space Innovation Platform, Quantum Technology for
Space Exploration).

MIT licence. Part of the pre-registered Phase 1 (Technology Readiness Level
2 to 3) campaign; see the Execution Pack (NEXORION-PH1-EXEC-2026-001).

## Frozen instance
`benchmarks/frozen_instance.yaml` — instance NEXORION-FI-2026-A. Immutable
after the `fi-2026-a` tag. Every result cites the instance identifier and
commit hash.

## Reproduce
```bash
conda env create -f environment.yml && conda activate scvx-landing
export PYTHONPATH=$PWD/src
pytest -q            # analytical validation (Gate Review 1 evidence)
```
or `docker build -t scvx-landing . && docker run scvx-landing`.

## Status
- [ ] WP1.1 Frozen instance values fixed and tagged
- [ ] WP1.2 SOCP subproblem + trust-region loop; convergence profile
- [ ] WP1.2 Analytical vertical-descent recovery test passing
