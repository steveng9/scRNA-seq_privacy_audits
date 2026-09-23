# Low-rank donor-level DP for scDesign2 (`experiments/dp/lowrank/`)

Reduces dimensionality **before** noising the Gaussian copula's covariance
matrix, instead of noising the full G x G matrix and truncating after (the
existing v1/v2 mechanism, which only denoises — no sensitivity reduction).
See `src/sdg/dp/lowrank.py`'s module docstring for the full mechanism and,
importantly, for two rejected alternatives worked through in detail:

- **Noising the true top-r eigenvectors, then re-orthonormalizing** — the
  orthonormalization step is valid free post-processing, but there is no
  dataset-independent sensitivity bound to calibrate the *pre*-orthonormalization
  noise to (Davis-Kahan's bound depends on the eigengap, which can be
  arbitrarily small).
- **Projecting the already-fitted G x G matrix with a public orthonormal
  projection** — looks like it should avoid needing per-cell data, but the
  entrywise sensitivity argument breaks without an explicit **re-clip of each
  cell's projected vector**, which does require per-cell access.

The construction actually used: a **fixed, public, data-independent** random
projection `P` (semi-orthogonal, QR of a seeded Gaussian matrix), applied to
each cell's quantile-normal vector, **re-clipped**, then the r x r second
moment noised via the *existing, already-proven* `sensitivity.py` machinery
with `n_genes=r`. This requires the per-cell `quantile_normal` matrix, which
ordinary v1/v2 `.rds` files discard — hence `src/sdg/scdesign2/scdesign2_lowrank.r`,
an isolated fork of `scdesign2_v2.r` that additionally saves it.

## Bonus finding: v1/v2 clip_value bug

While deriving this mechanism's sensitivity bound from scratch, found that
the existing v1/v2 pipeline's `clip_value=3.0` doesn't match what the R
fitting code actually enforces (true bound ≈4.265) — under-calibrating sigma
by ~2x in every existing v1/v2 DP result. Fixed alongside this work; see
`notes/DP_clip_value_bug.txt` for the full writeup and which result
directories are (and aren't) affected.

## Files

- `train.py` — trains lowrank copulas (same fit as v2, `scdesign2_lowrank.r`
  additionally saves `quantile_normal`). Writes to
  `scdesign2/lowrank_no_dp/{nd}d/{trial}/models/`.
- `generate.py` — `--rank R --no-dp` (rank truncation only, no noise) or
  `--rank R --epsilon E [E...]` (rank + DP noise). Reuses the same trained
  `lowrank_no_dp` models for every (rank, epsilon) combination — only the
  noise-injection step differs.
- `run_evals.py` — quality (LISI/ARI/MMD) + MIA (Mahalanobis attack, quad
  mode: BB+aux/BB-aux x standard/Class-B) evaluation, isolated from the
  v1/v2/other sweep orchestration.
- `compare.py` — lowrank (rank x epsilon) vs. v2 (full-rank) markdown table.
- `sweep.sh` — end-to-end spot check: `ok`, 20 donors, trials 1-2, ranks
  `{5, 10, 20, 50}`, epsilons `{1, 100, 1e4, 1e6, 1e8}` (same epsilon grid as
  the v2 spot check, for direct comparability). Also regenerates the v2
  full-rank baseline at this scope with the corrected `TRUE_CLIP_VALUE`.

## Isolation contract

Same as `experiments/dp/v2/`: this directory's orchestration does not touch
`run_quality_evals.py`, `run_mia_sweep.py`, or the v1/v2 sweep scripts. The
only place this touches v2 at all is `sweep.sh`'s step 0, which regenerates
the *specific* `v2_eps_*/{ok}/20d/{1,2}` directories needed for an
apples-to-apples comparison column (using `experiments/dp/v2/generate.py`,
now fixed to use `TRUE_CLIP_VALUE` — unchanged otherwise). The stale,
wrong-clip_value results these replace were moved (not deleted) to
`~/data/scMAMAMIA/ok/scdesign2/_stale_clip3_backup/`.

## How to run

```bash
bash experiments/dp/lowrank/sweep.sh
```

Override the spot via env vars, e.g. `RANKS="10 30" EPSILONS="1 1000" bash sweep.sh`.

## Results

`results/spot_check_{dataset}_{nd}d.md`, written by `compare.py`.
