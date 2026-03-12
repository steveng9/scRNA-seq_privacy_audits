# Handoff Note — DP Experiment (2026-03-12)

**Read this at the start of the next session.**

---

## What was just implemented

Differential privacy (DP) noise injection for the Gaussian copula is done and committed
to `main` (commit 7620374). See the four new files:

- `src/sdg/dp/sensitivity.py` — sensitivity math (read this alongside the formal proof)
- `src/sdg/dp/dp_copula.py` — `apply_gaussian_dp()`; the actual mechanism
- `src/attacks/scmamamia/attack_dp.py` — WB attack wrappers using DP-noised copula
- `experiments/dp/run_dp_sweep.py` — CLI sweep runner

---

## What to do next

### 1. Run the sweep on the server

The OneK1K 50d, 100d (and larger) completed trials live on the server, not locally.
Pull the new code there and run:

```bash
git pull
conda activate camda_conda
cd /path/to/ScMAMA-MIA

# 50d trial (find the right path for the 50d WB trial)
python experiments/dp/run_dp_sweep.py <path_to_50d_trial> \
    --epsilons 100 300 1000 3000 10000 30000 \
    --delta 1e-5 \
    --clip 2.0 \
    --seeds 0 1 2 3 4

# Then 100d
python experiments/dp/run_dp_sweep.py <path_to_100d_trial> \
    --epsilons 100 300 1000 3000 10000 30000 \
    --delta 1e-5 \
    --clip 2.0 \
    --seeds 0 1 2 3 4
```

**Why `--clip 2.0`:** With c=3.0 the sensitivity Δ_F is so large that noise dominates
even at ε=30000. With c=2.0, σ drops below 1.0 around ε=1000–3000 for the 100d dominant
cell type, so the AUC curve passes through the interesting transition region.

**Expected sanity check:** at ε=30000 with c=2.0, AUC should be close to the no-DP
baseline (~0.85 for 50d WB). If it's still near 0.5, the sensitivity bound is too
conservative and we need to investigate empirical sensitivity.

### 2. Trial directory structure

The sweep runner expects:
```
trial_dir/
  models/        ← WB copulas (.rds named by cell type, e.g. "0.rds")
  artifacts/aux/ ← aux shadow copulas (same naming)
  datasets/
    train.h5ad
    holdout.h5ad
```
The runner auto-detects whether HVGs are in `models/hvg.csv` or `artifacts/hvg.csv`.

### 3. After the sweep runs

- Results land in `trial_dir/results/dp_sweep_results.csv`
- A per-cell-type sensitivity report is at `trial_dir/results/dp_sensitivity_report.json`
- Next step: add a plotting function in `evaluation/plots.py` for the AUC-vs-ε curve,
  overlaid with the theoretical maximum AUC from `docs/dp_theoretical_limit.tex`

### 4. Open question Steven flagged

The sensitivity bound Δ_F = 4·k_max·c²·G / (N−k_max) is very conservative (assumes all
cells have all genes at the clipping boundary simultaneously). Steven is rightly skeptical
that this correctly represents the real sensitivity. Consider computing an **empirical
sensitivity** by actually re-fitting the copula with and without each donor and measuring
the true Frobenius norm change — this would tighten the bound and justify using lower
noise levels in the paper.

### 5. Formal proof

Steven wants to write a formal proof of the DP guarantee. The proof structure is in
`src/sdg/dp/sensitivity.py` (docstring). Key references:
- Dwork & Roth (2014), Theorem A.1 (Gaussian mechanism) and Proposition 2.1 (post-processing)
- Sheffet (2017) "Differentially Private Ordinary Least Squares" — sensitivity of X^TX
- Near & Abuah "Programming Differential Privacy" (Steven has read this)

Steven's existing informal MIA-bound analysis is in `docs/dp_theoretical_limit.tex` —
that is a *separate* result (given DP, what can an attacker achieve?) and is complementary
to the implementation proof.

---

## Summary of sensitivity numbers (for reference)

For OneK1K 100d, dominant cell type (n_cells=47933, k_max=1054, G≈400):

| c    | Δ_F   | σ at ε=100 | σ at ε=1000 | σ at ε=10000 |
|------|-------|------------|-------------|--------------|
| 3.0  | 323.8 | 15.69      | 1.57        | 0.157        |
| 2.0  | 143.9 | 6.97       | 0.70        | 0.070        |
| 1.0  | 36.0  | 1.74       | 0.17        | 0.017        |

Correlation matrix entries are in [-1, 1], so σ < ~0.3 is needed for the signal to
clearly dominate the noise.
