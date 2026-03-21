# scDesign3 MIA Experiments

**IMPLEMENTED** as of 2026-03-14.  Supports both Gaussian (default) and vine copulas.

## Implementation

- `src/sdg/scdesign3/model.py` — `ScDesign3` class (wraps R scDesign3 via subprocess)
- `src/sdg/scdesign3/scdesign3.r` — R driver script (train / generate)
- `src/sdg/scdesign3/copula.py` — Copula parser (Gaussian + vine); same interface as scDesign2
- Registered in `src/sdg/run.py` under `"scdesign3"`

## Using scDesign3 in experiments

scDesign3 (Gaussian copula) plugs directly into the existing `run_experiment.py`
pipeline.  Its copula parser returns the same dict format as scDesign2, so
`attack_mahalanobis` and `attack_mahalanobis_no_aux` work unchanged.

Config key: `generator_name: scdesign3`
Copula type: set `copula_type: gaussian` (default) or `vine`.

## References

- sun2023scdesign3
- scDesign3 R package: https://github.com/SONGDONGYUAN1994/scDesign3
