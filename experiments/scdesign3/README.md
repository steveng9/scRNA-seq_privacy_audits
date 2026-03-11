# scDesign3 MIA Experiments

**NOT YET IMPLEMENTED** — planned as Experiment 2 in the paper revision.

## Plan

scDesign3 supports Gaussian copulas (default, same format as scDesign2) and Vine
copulas.  The Gaussian copula variant should work with the existing `parse_copula()`
function in `src/sdg/scdesign2/copula.py` with minimal adaptation.

### Steps to implement:
1. Add `src/sdg/scdesign3/model.py` wrapping the R `scDesign3` package
2. Register `"scdesign3"` in `src/sdg/run.py`
3. Add experiment configs here

### References
- sun2023scdesign3
- scDesign3 R package: https://github.com/SONGDONGYUAN1994/scDesign3
