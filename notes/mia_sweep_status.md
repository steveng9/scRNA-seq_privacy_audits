# scMAMA-MIA Attack Sweep — Status & Notes

Last updated: 2026-04-12

---

## Sweep design

Each `run_experiment.py` job uses the **combined BB attack** (`run_both_bb: True`):
one job per (dataset, nd) computes BB+aux (tm:100) and BB-aux (tm:101) in a single
pass, sharing the synth-side Mahalanobis computation (copula parse + inversion +
d_s vector). This halves the job count vs. running them separately (160 jobs instead
of 320). The scDesign2 shadow copulas are fitted once on `synthetic.h5ad` and reused
for both variants.

---

## How to run

```bash
# Full sweep (all pending jobs):
nohup conda run --no-capture-output -n tabddpm_ \
    python experiments/sdg_comparison/run_mia_sweep.py \
    > /tmp/mia_sweep.log 2>&1 &

# Check progress (any time):
conda run -n tabddpm_ python experiments/sdg_comparison/run_mia_sweep.py --status

# Preview without running:
conda run -n tabddpm_ python experiments/sdg_comparison/run_mia_sweep.py --dry-run

# Narrow to one SDG or dataset:
conda run -n tabddpm_ python experiments/sdg_comparison/run_mia_sweep.py --sdg sd3g
conda run -n tabddpm_ python experiments/sdg_comparison/run_mia_sweep.py --dataset ok_scvi
```

The script is **idempotent** — safe to kill and restart anytime. It checks existing
results and only runs jobs that are still missing.

---

## Where results live

```
/home/golobs/data/{dataset_name}/{nd}d/{trial}/results/mamamia_results.csv
```

- `dataset_name` examples: `ok`, `aida`, `cg`, `ok_dp/eps_1`, `ok_sd3g`, `ok_scvi`, etc.
- `nd` = donor count (e.g., `10d`)
- `trial` = trial index 1–5

### Results CSV format

Rows are metrics (`auc`, `tpr@fpr=0.01`, etc.); columns are `metric` + one per
threat model:

| metric | tm:000  | tm:001  | tm:100  | tm:101  |
|--------|---------|---------|---------|---------|
| auc    | 0.72    | 0.65    | 0.68    | 0.60    |

Threat model codes:
- `000` = WB+aux  (white-box, with auxiliary data)
- `001` = WB-aux  (white-box, no auxiliary data)
- `100` = BB+aux  (black-box proxy scDesign2, with auxiliary data)  ← primary
- `101` = BB-aux  (black-box proxy, no auxiliary data)

For all non-scDesign2 SDGs the attack is **always black-box**: scDesign2 is fitted
as a proxy shadow model on the existing `synthetic.h5ad`. There is no white-box
access to the actual generator's copula.

---

## Current completion status (as of 2026-04-12 sweep launch)

### scDesign2

| Dataset | nd   | WB+aux | WB-aux | BB+aux | BB-aux |
|---------|------|--------|--------|--------|--------|
| ok      | 2–50 | ✓      | ✓      | ✓      | ✓      |
| ok      | 100  | ~4     | ✓      | ~4     | ✓      |
| ok      | 200  | ~4     | ✓      | ~4     | ✓      |
| aida    | 5–50 | ✓/~4   | ~1–2   | ✓      | ~1–2   |
| aida    | 100  | ·      | ·      | ✓      | ·      |
| aida    | 200  | ·      | ·      | ~4     | ·      |
| cg      | 2–20 | ✓      | ~1–2   | ✓      | ~1–2   |

The sweep will fill in all missing trials for WB-aux/BB-aux on aida and cg, and
the 1 missing trial on ok/100d and ok/200d.

### scDesign2 + DP (BB only; all 0/5 before this sweep)

| Dataset         | nd        | BB+aux | BB-aux |
|-----------------|-----------|--------|--------|
| ok_dp/eps_1     | 10,20,50  | ·      | ·      |
| ok_dp/eps_10    | 10,20,50  | ·      | ·      |
| ok_dp/eps_100   | 10,20,50  | ·      | ·      |
| ok_dp/eps_1000  | 10,20,50  | ·      | ·      |
| ok_dp/eps_10000 | 10,20,50  | ·      | ·      |

All 30 (dataset×nd×tm) combinations pending. Synthetic data for all is already
generated and verified.

### Other SDGs (all BB only; all 0/5 before this sweep)

| SDG             | Dataset    | nd          | Notes |
|-----------------|------------|-------------|-------|
| scDesign3-Gauss | ok_sd3g    | 2–200d      | Synthetic already generated (35/35 trials) |
| scDesign3-Vine  | ok_sd3v    | 10d, 20d    | Synthetic OK; **50d excluded** — see below |
| scVI            | ok_scvi    | 5–100d      | Synthetic generated (5/5 per nd) |
| scVI            | aida_scvi  | 10,20,50d   | Synthetic generated (5/5 per nd) |
| scDiffusion     | ok_scdiff  | 10,20,50d   | Synthetic generated (5/5 per nd) |
| scDiffusion     | aida_scdiff| 20,50d      | Synthetic generated (5/5 per nd) |

---

## PENDING: what still needs to be done after this sweep

### sd3v 50d — add back once generation is complete

As of 2026-04-12, `retrain_sd3v_1118hvg.py` is actively running, regenerating
ok_sd3v/50d with the fixed 1118-HVG set:
- Trial 1: **currently generating** (R vine copula fitting, 4 workers, ~100% CPU)
- Trials 2–5: old broken files from 2026-03-26 exist at the path but produce
  degenerate data (~20 nonzero genes). They will be overwritten as the retrain
  script progresses.

**Action required when complete:**
1. Verify all 5 trials: each `synthetic.h5ad` should be ~50–60 MB with >1000 expressed genes
2. Re-add `50` to the `ok_sd3v` donor_counts in `run_mia_sweep.py` (currently removed)
3. Run: `python experiments/sdg_comparison/run_mia_sweep.py --sdg sd3v`

### sd3v 100d — not yet generated

scDesign3-Vine 100d synthetic data has never been generated. After the 50d sweep
is complete, decide whether to generate 100d.

### aida sd3g / sd3v — not yet generated

`aida_sd3g` and `aida_sd3v` directories exist but contain no synthetic data.
Not in the current sweep.

---

## Memory / concurrency notes

- The MIA sweep runs **one `run_experiment.py` at a time**, using 4 parallel
  cell-type workers internally.
- As of sweep launch: sd3v 50d trial 1 R processes are using ~25 GB RAM; 96 GB
  available. No memory pressure concern.
- scDesign2-proxy attacks (used for all non-SD2 SDGs) fit a fresh scDesign2 copula
  on the existing `synthetic.h5ad`, then run the standard Mahalanobis-distance MIA.

---

## Note on interpretation

For scVI, scDiffusion, and sd3v, the BB attack is using scDesign2 as a **proxy**
shadow model — the structure of the actual generator is completely ignored. These
results measure "how much does this synthetic data leak, as measured by a scDesign2
copula fitted to it?" rather than exploiting the generator's own parameters. This is
the baseline; generator-specific attacks are planned next.
