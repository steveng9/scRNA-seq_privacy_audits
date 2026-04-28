# Priority TODO — paper-revision sweep

Tracks work explicitly deferred from the 2026-04-28 baseline + quality
finalization sweep so we don't lose the threads. Sorted by descending priority.

---

## ✅ Completed in this round (2026-04-28)
- Batched MC, GAN-Leaks, GAN-Leaks-Cal in `src/attacks/baselines/batched_baselines.py`
  (exact match to unbatched optimised versions; verified to ≤1e-6 max diff).
- Subsampled-fit KDE for the DOMIAS baseline (max_fit=20k per side).
- `experiments/sdg_comparison/run_baselines_sweep.py` — memory-aware sweep over
  `{ok,aida,cg}/scdesign2/no_dp` at all donor counts up to 200d.
- Updated `src/run_baselines.py` to read donor splits from the shared
  `splits/` directory (matching the 2026-04-19 data layout).
- `run_quality_evals.py` now includes scdesign2/no_dp + zinbwave entries and
  exposes a `--force` flag so we can re-run after metric-code changes.

---

## 🔴 Defer-but-do-soon

### 1. 490d baseline attacks for `ok/scdesign2/no_dp`
Why: 200d already covers the table for the paper revision; 490d is the
"all donors" stress test that strengthens the donor-scaling story but is
not load-bearing for a single revision deadline.

How to apply when ready:
```
python experiments/sdg_comparison/run_baselines_sweep.py --nd 490 --skip-nd-above 490
```
Expected RAM: ~80-100 GB peak (HVG dense matrices for ~600k train+holdout
cells × 5000 genes ≈ 24 GB each, ×2 sides + synth + ref + working buffers).
Run serially, no other heavy jobs concurrently. Likely overnight.

### 2. Re-run quality metrics for `{ok,aida,cg}/sd2/no_dp` (MMD median-heuristic fix)
Why: existing CSVs were written 2026-03-25 03:00–04:57 UTC; the median-heuristic
gamma fix in `compute_mmd_optimized` landed at 2026-03-25 19:51 UTC
(commit `d9ae732`). MMD values in those CSVs were either ~2/n or ~0 — useless.

Scope updated 2026-04-28: ok is included too (originally trusted, but the
timestamp evidence shows ok is also pre-fix).

How to apply:
```
conda run -n tabddpm_ python experiments/sdg_comparison/run_quality_evals.py \
    --force --max-donors 200 \
    --dataset-filter ok/scdesign2/no_dp
conda run -n tabddpm_ python experiments/sdg_comparison/run_quality_evals.py \
    --force --max-donors 200 \
    --dataset-filter aida/scdesign2/no_dp
conda run -n tabddpm_ python experiments/sdg_comparison/run_quality_evals.py \
    --force --max-donors 200 \
    --dataset-filter cg/scdesign2/no_dp
```
ok: 35 trials. aida: 36 trials. cg: 27 trials. Total ~98 evaluations.
~2-5 minutes per trial at ≤50d, longer at 100/200d. OK to run with
`--workers 4`.

---

## 🟡 Lower priority, return-to list

### 3. Baseline attacks on non-scDesign2 SDG methods
Why: The paper's headline claim is about scDesign2; baselines on
scvi/scdiffusion/sd3g/sd3v/zinbwave/nmf would let us include comparison
rows in the baseline-AUC table — currently those columns are blank for
non-SD2 generators. Lower priority because the scMAMA-MIA scores already
cover those generators (BB-quad).

Scope when ready:
- `ok` and `aida` × {scvi, scdiffusion, scdesign3/gaussian, scdesign3/vine,
  zinbwave, nmf/no_dp} × {10d, 20d, 50d}.
- ~36 (dataset, nd) × 5 trials = 180 trials.
- Implementation: extend `SWEEP` in `run_baselines_sweep.py`.

### 4. Quality metrics for AIDA scDesign2-DP variants
Why: The paper table shows `ok/scdesign2/eps_*` rows. For AIDA we have
synth data for eps_1, eps_10, eps_100 (partial), but no quality results yet
and no synth for eps_1000+.

Scope when ready:
- Generate missing synth (eps_100 partial → 5 trials; eps_1000, eps_10000 →
  needs generation).
- Run `run_quality_evals.py --dataset-filter aida/scdesign2/eps_`.

### 5. Document baseline batching in CLAUDE.md / paper
Why: The Reviewer 9 ("Clarify Memory Constraints on Baseline Attacks")
requirement says we should *clearly* document why CAMDA2025 baselines
fail at >20 donors. After this sweep, we can write a short paragraph
showing: (a) the original implementation's O(n_test · n_synth) memory
profile, (b) our exact-batched fix for distance baselines, (c) the
documented subsample protocol for KDE.

Refs to cite: Silverman 1986; Politis–Romano–Wolf 1999; Hilprecht et al.
2019 (Monte Carlo MIA); Chen et al. 2020 (GAN-Leaks); van Breugel et al.
2023 (DOMIAS).

### 6. Quality metrics for OK1K 490d (sd2/no_dp)
Why: paper table currently uses 50d; reviewer rebuttal might cite 490d.
Status: 0/5 quality, 5/5 synth available.

How:
```
conda run -n tabddpm_ python experiments/sdg_comparison/run_quality_evals.py \
    --max-donors 490 --dataset-filter ok/scdesign2/no_dp
```
