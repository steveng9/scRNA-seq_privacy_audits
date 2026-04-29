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

## ✅ Fixes shipped 2026-04-29
- **Backed-mode load in `run_baselines.py`**: replaced `ad.read_h5ad(path)`
  with `ad.read_h5ad(path, backed="r")` followed by `.to_memory()` on the
  per-donor subsets. Fixes the OOM that killed all aida 50d/100d/200d trials
  in the first sweep (aida full h5ad is 57 GB; eager load + concurrency=3 →
  rc=-9 from kernel OOM killer within ~2 min of launch).
- **Removed redundant `gan_leaks_sc` baseline**: it was bit-for-bit identical
  to `gan_leaks` (both compute `exp(-min_g ||x-g||²)`) — verified on the
  aida/20d/t5 CSV (matching aucroc, AP, PR-AUC, TPR@FPR). Its implementation
  in `sc_baseline.py` was unbatched, called `np.save("X.npy", X)` /
  `np.save("Y.npy", Y)` on every chunk (creating a 5 GB Y.npy in repo root),
  and added ~45 min/trial. Dropped 2026-04-29; kept the cleaner
  `GAN_leaks_batched` from `batched_baselines.py` (with the Chen et al. 2020
  citation in the docstring).
- **`run_quality_evals.py --status`** flag added — mirrors
  `run_baselines_sweep.py --status` (dataset × donor-count completion grid).

---

## 🔴 Defer-but-do-soon

### 0. Re-run the 14 failed aida baseline trials (post-OOM fix)
After the in-flight 2026-04-28 sweep finishes (queue: ok 100d × 3 + ok 200d × 5),
relaunch baselines for the aida trials that were OOM-killed:
- aida/scdesign2/no_dp 50d × {1,2,3,4,5}
- aida/scdesign2/no_dp 100d × {1,2,3,4,5}
- aida/scdesign2/no_dp 200d × {1,2,3,4} (t5 has no synth)

The sweep script already detects completion via
`baselines_evaluation_results.csv`, so a plain re-launch will pick up only
the missing trials:
```
nohup setsid /home/golobs/miniconda3/envs/tabddpm_/bin/python \
    experiments/sdg_comparison/run_baselines_sweep.py \
    --dataset aida --max-concurrent 1 \
    > experiments/sdg_comparison/_baseline_logs/SWEEP_AIDA_RETRY.log 2>&1 < /dev/null &
```
Use `--max-concurrent 1` for aida specifically — even with backed-mode loading,
the in-memory subsets at 200d are large (~95k cells × 5k HVG dense = ~1.9 GB
per side, plus working buffers and KDE fits). Single-stream is safest.

---

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

### 2. ~~Re-run quality metrics for ALL stale (pre-MMD-fix) CSVs~~ — RESOLVED 2026-04-29
Spot-check on 2026-04-29 (5 trials covering the full stale window 2026-03-25
03:19–05:50 UTC) showed bit-for-bit agreement (max delta 1.1e-16) between
stale CSVs and fresh re-runs. The MMD median-heuristic fix was already in the
working tree before the formal commit `d9ae732` (2026-03-25 19:51 UTC).
Cutoff `MMD_FIX_TS` lowered to 2026-03-23 00:00 UTC; all 174 previously-stale
CSVs now reclassify as fresh. Re-running them was unnecessary.

Spot-check tool: `experiments/sdg_comparison/spot_check_stale_quality.py`
(retained for reference; runs the 5 representative trials with a /tmp results
dir and compares fresh values to stored ones without mutating originals).

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
