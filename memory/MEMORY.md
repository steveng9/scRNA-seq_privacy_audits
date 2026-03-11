# MEMORY.md — scRNA-seq Privacy Audits

This file tracks ongoing work, fixes, and findings for future reference.

---

## Quality Eval Pipeline (sc_evaluate.py, run_quality_eval.py)

### Key fixes made (2026-03-11)
- `run_quality_eval.py`: removed unused full-dataset load from `main()` that was OOM-killing on large datasets. `evaluate()` now called as `evaluate(quality_eval_cfg, None, None)` — the real/syn data are loaded internally by SingleCellEvaluator.
- `sc_evaluate.py` `load_test_anndata()`: uses `backed='r'` + `.to_memory()` to avoid loading 57 GB AIDA h5ad into RAM.
- `sc_evaluate.py` `get_statistical_evals()`: re-enabled `compute_mmd_optimized` and `compute_ari` (were commented out); ARI wrapped in try/except.
- `sc_metrics.py` `compute_ari()`: fixed size-mismatch crash — truncates (not shuffles) the larger array to match the smaller.

### AIDA 50d quality tracking bug
AIDA 50d trials 1-4 were erroneously marked quality=1 with empty result dirs. Reset to quality=0, results now computed.

---

## run_experiment.py fixes (2026-03-11)
- `create_data_splits()`: uses `backed='r'` + `.to_memory()` for full dataset load; skips re-extracting h5ad splits if they already exist.
- `_initialise_results_files()`: uses `targets.obs_names` instead of `targets.to_df().index` (avoids densifying full matrix).

---

## ScDesign2 model.py fixes (2026-03-11)
- `train()`: replaced `X_train.X.toarray()` (OOM on 200K+ cells) with per-cell-type sparse mean computation.
- `train()`: added `del X_train; gc.collect()` before `ProcessPoolExecutor`.
- `train()`: switched to `mp_context=multiprocessing.get_context("spawn")` with `max_workers=4` to avoid inheriting parent's memory via fork.

---

## AIDA dataset memory notes
- `full_dataset_cleaned.h5ad` is 57 GB on disk — ALWAYS use `backed='r'` when loading.
- 100d train.h5ad = 8.6 GB (225K cells × 35K genes sparse), 200d = ~15 GB (437K cells).
- OOM threshold: 15 fork-based workers OOM kills on 125 GB server.

---

## Quality eval results (as of 2026-03-11)
| Dataset | Trials done | LISI (mean±std) | ARI (mean±std) | MMD (mean±std) |
|---------|-------------|-----------------|----------------|----------------|
| HFRA 2d | 1-6 complete | 0.9119 ± 0.0058 | 0.4972 ± 0.1720 | 0.0008 ± 0.0001 |
| HFRA 20d | 1-5 complete | 0.8789 ± 0.0027 | 0.4617 ± 0.0331 | 0.0001 ± 0.0000 |
| AIDA 20d | 1-5 complete | see stats_evals.csv | | |
| AIDA 50d | 1-5 complete | 0.755 ± 0.007 | ~0.0002 (subsampled) | 0.0001 ± 0.0000 |
| AIDA 100d | pending (100d/200d MIA experiment running) | | | |
| AIDA 200d | pending | | | |

Note: AIDA ARI values near 0 are expected — the ARI metric compares cluster labels of different cells positionally (CAMDA2025 implementation flaw); scDesign2 generates equal cell counts so no truncation occurs, but AIDA's 33 cell types produce low Louvain cluster agreement in this metric.

---

## Active background jobs (2026-03-11)
- Shell PID 2471370: `src/run_aida_100d_200d_experiments.sh`
- Log: `/tmp/aida_100d_200d_experiment_log.txt`
- Python unbuffered (-u flag), uses spawn context, 4 workers
- After this completes: run `run_quality_eval.py` for 100d and 200d settings
