"""
scDesign3 SDG — NOT YET IMPLEMENTED.

scDesign3 supports both Gaussian and Vine copulas (selectable via a `copula`
parameter: "gaussian" or "vine").  In its default Gaussian copula mode it
produces the same R object structure as scDesign2, so `parse_copula()` in
`sdg.scdesign2.copula` should be directly reusable.

To implement:
  1. Add model.py that wraps the R scDesign3 package.
  2. Register "scdesign3" in src/sdg/run.py under generator_classes.
  3. Add experiment configs under experiments/scdesign3/.

Reference: sun2023scdesign3
"""
