# MVN + MahalaMIA (copied from MIA_on_bulkRNAseq_CAMDA2026)

## Provenance

Copied 2026-09-23 from `~/MIA_on_bulkRNAseq_CAMDA2026` at commit
`88cdff7253179dad510b9ecc1ae3767f7234e306`
("FINDINGS 10i: the DP-edge estimator bug behind 10h, the fix, and first
sweep results").

Copied as-is, no logic changes:
- `paths.py`, `csvlock.py`, `datasets.py`, `runs.py`, `metrics.py`, `targets.py`
- `generators/base.py`, `generators/mvn.py`
- `attacks/base.py`, `attacks/mahalamia.py`

**Pruned** (rewritten, not copied) — `generators/__init__.py` and
`attacks/__init__.py` register only `mvn`/`MahalaMIA`, dropping the source
repo's `cvae`/`nd`/`pgm` generators and `mamamia`/`melomia_*` attacks, which
pull in heavier dependencies (torch, an external private-PGM repo) not needed
for the DP-MVN work this copy exists for. If those are ever needed here,
copy them the same way and add them back to the `__init__.py` imports.

## Why this exists

This repo (scRNA-seq_privacy_audits) is developing a formal donor-level DP
guarantee for scDesign2's Gaussian copula covariance matrix
(`src/sdg/dp/`). The MVN generator here does something structurally very
similar — fits a per-class mean vector + covariance matrix, samples from it —
but currently has **no formal DP mechanism** (its `noise_level` knob is an ad
hoc fidelity/privacy dial, not sensitivity-calibrated), and MahalaMIA is a
Mahalanobis-distance attack directly analogous to scMAMA-MIA. The plan is to
apply the same formal DP treatment developed for scDesign2 to MVN, as a
second, complementary paper: see `notes/DP_proof_bulk_mvn.txt` for the
proof-only writeup (implementation deliberately not started yet, pending
review — see that file).

## Verified working standalone (2026-09-23 smoke test)

```python
import sys
sys.path.insert(0, "experiments/bulk_rna_seq")
import mia.targets as T
from mia.attacks import MahalaMIA
from sklearn.metrics import roc_auc_score
import mia.datasets as D

T.build_target("BRCA", "mvn", 1, device="cpu", force=True)
scores = MahalaMIA().score("BRCA", "mvn", 1)
labels = D.membership_labels("BRCA", 1)
roc_auc_score(labels, scores)   # -> 0.921 (matches the source repo's
                                  #    documented "MVN leaks heavily" finding)
```

Reads real data from `~/data/CAMDA26` (env `CAMDA_DATA`) and
`~/CAMDA25_NoisyDiffusion` (env `CAMDA_ND_REPO`) — both already present on
this machine, no path changes needed (`paths.py` resolves everything
relative to env vars / this package's own location, copied unmodified).
Writes outputs to `experiments/bulk_rna_seq/artifacts/` (git-ignored, same
convention as the source repo).

## Not carried over

Anything specific to the CAMDA competition submission itself (scripts,
configs, results tables, the other three generators/attacks) stays in the
source repo. This is a working copy of the two pieces relevant to the DP
work, not a fork of the competition entry.
