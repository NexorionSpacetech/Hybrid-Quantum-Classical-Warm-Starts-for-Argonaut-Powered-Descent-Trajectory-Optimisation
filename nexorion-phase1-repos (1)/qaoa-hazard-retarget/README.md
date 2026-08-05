# qaoa-hazard-retarget

Hybrid quantum-classical warm-start for lunar powered-descent hazard-avoidance
retargeting — the quantum arm of Nexorion Spacetech's Phase 1 benchmark
(European Space Agency Open Space Innovation Platform, Quantum Technology for
Space Exploration). Pairs with the `scvx-landing` classical baseline.

MIT licence. Pre-registered protocol fixed in `src/qaoa_warmstart.py`
(constants) and Execution Pack NEXORION-PH1-EXEC-2026-001 Section 6 — the
methodology, statistics, and falsification condition may not be amended in
response to results. The result is published whether positive or null.

## Reproduce
```bash
conda env create -f environment.yml && conda activate qaoa-hazard-retarget
export PYTHONPATH=$PWD/src
pytest -q     # CF-1 brute-force validation (Gate Review 1 evidence)
```
or `docker build -t qaoa-hr . && docker run qaoa-hr`.

## Status
- [x] CF-1 QUBO encoding + brute-force validation suite
- [ ] WP2.1 QAOA implementation + warm-start integration (noiseless first)
- [ ] WP2.2 Simulator campaign (pre-registered protocol)
- [ ] WP2.3 Hardware subset on IBM Quantum (book runs in Month 1)
- [ ] WP3.3 Independent reproduction recorded
