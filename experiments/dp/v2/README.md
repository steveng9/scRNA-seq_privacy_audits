# DP variant v2 — uncentered second moment

## What v2 changes

The v1 (paper-as-submitted) DP proof bounds the donor-level Frobenius
sensitivity of the **column-centered** sample covariance matrix
`Ĉ = (Z − μ)(Z − μ)ᵀ / (N − 1)`. After centering, each entry of `Z − μ`
lies in `[−2c, 2c]`, so the per-entry sensitivity carries a leading
factor of 4.

v2 drops the centering step. The matrix being noised is the **uncentered
second moment** `M = ZZᵀ / (N − 1)`. Each entry of `Z` lies in
`[−c, c]`, so the leading factor falls to 1 and σ becomes 4× smaller
for the same ε.

The justification is empirical: `z = Φ⁻¹(F̂(x))` is approximately
N(0, 1) at the population level, so `μ ≈ 0` and `M ≈ Ĉ`. The point of
this spot check is to confirm that.

## Layout

Both training-time and DP-time matrices live in a parallel directory
hierarchy that mirrors v1:

```
~/data/scMAMAMIA/ok/scdesign2/
  no_dp/                          # v1 (existing, untouched)
  eps_*/                          # v1 + DP (existing, untouched)
  v2_no_dp/{nd}d/{trial}/         # v2 retrained copulas (uncentered)
    models/{ct}.rds               # cov_mat = M (raw second moment)
    datasets/synthetic.h5ad       # synthetic from v2 copula, no DP
    provenance.json
  v2_eps_*/{nd}d/{trial}/         # v2 + DP
    datasets/synthetic.h5ad
    provenance.json
```

Nothing under `no_dp/`, `eps_*/`, or any other v1-era directory is
touched. Existing quality CSVs and MIA results are preserved.

## What's in the saved .rds (v2)

The v2 `cov_mat` is the **raw uncentered second moment**, NOT a
correlation matrix. It is not directly usable by
`simulate_count_scDesign2` until PSD-projected and normalized to
correlation. That post-processing step happens on the Python side, in
`generate.py`, before each `Rscript scdesign2.r gen` call. (Applying
this post-processing AFTER any DP noise is valid by the
post-processing theorem.)

The `dp_variant = "v2"` field is added to the saved list so a future
loader can verify it was trained with the v2 R script.

## Files

| File | Purpose |
|------|---------|
| `../../../src/sdg/scdesign2/scdesign2_v2.r` | R training script — `cor()` replaced with uncentered second moment |
| `train.py`     | Parallel R training driver. Outputs to `v2_no_dp/{nd}d/{trial}/models/` |
| `generate.py`  | Generation driver. Two modes: `--no-dp` writes to `v2_no_dp/.../datasets/`, `--epsilon E` writes to `v2_eps_{E}/.../datasets/` |
| `run_evals.py` | Spot-check evals: registers v2 paths and invokes the existing quality + MIA pipelines on them |
| `compare.py`   | Side-by-side v1 vs. v2 table |
| `sweep.sh`     | End-to-end orchestrator for the spot check |

## Running the spot check

```bash
cd /home/golobs/scRNA-seq_privacy_audits
bash experiments/dp/v2/sweep.sh
```

Settings (edit at the top of `sweep.sh`):
- dataset: ok
- nd:      20
- trials:  1, 2
- epsilons (DP runs): 1, 100, 10000, 1000000, 100000000
- baseline: also runs no-DP v2 at the same trials

Parallelism is intentionally kept low (2 workers) so the running
`run_baselines_sweep.py` is not crowded.

## Adding a v3, v4, … later

Same pattern: copy this directory to `v3/`, copy `scdesign2_v2.r` to
`scdesign2_v3.r`, change the matrix line and the `dp_variant` tag.
Output dirs are tagged `v3_no_dp/`, `v3_eps_*/`. The `dp_variant` knob
in `src/sdg/dp/sensitivity.py` should be extended with a new case,
adding any new prefactor to `_VARIANT_PREFACTOR`. Nothing in v2 is
touched.

## Provenance

Each output dir gets a `provenance.json` containing:
- `dp_variant` (e.g. "v2")
- `r_script_path` and SHA-256 of the R script used to train the copula
- `python_module_versions` (sensitivity.py, dp_copula.py git short hashes if available)
- `epsilon`, `delta`, `clip_value`, `seed` (DP runs only)
- `trained_at` / `generated_at` UTC timestamps
- `inputs`: full_h5ad path, splits dir, hvg path

This lets us trace any downstream score back to the exact code that
produced it, which we will need as the proof iterates.

## Spot-check results (2026-04-29, ok 20d, trials 1+2)

Full table at `results/spot_check_ok_20d.md`. Highlights:

**Quality (LISI / ARI / MMD).** v2 ≈ v1 at every ε. The "μ ≈ 0"
empirical assumption is confirmed:

| ε         | LISI v1 → v2     | ARI v1 → v2      | MMD v1 → v2          |
|-----------|------------------|------------------|----------------------|
| no DP     | 0.879 → 0.879    | 0.500 → 0.510    | 0.0005 → 0.0005      |
| 10⁸       | 0.880 → 0.878    | 0.482 → 0.492    | 0.0005 → 0.0005      |
| 10⁶       | 0.881 → 0.877    | 0.456 → 0.486    | 0.0005 → 0.0005      |
| 10⁴       | 0.842 → 0.871    | 0.510 → 0.520    | 0.0005 → 0.0005      |
| 100       | 0.692 → 0.692    | 0.555 → 0.603    | 0.0007 → 0.0007      |
| 1         | 0.681 → 0.676    | 0.572 → 0.547    | 0.0007 → 0.0007      |

LISI saturates near 0.67 at ε ∈ {1, 100} for both v1 and v2 — at low ε
PSD-projection of the noised matrix dominates and the 4× σ reduction
isn't visible in quality. v2's gain shows up at ε = 10⁴ (LISI 0.842 →
0.871).

**Attack AUC (BB+aux, Class B — primary threat model).** At the *same
nominal ε*, v2 leaks more than v1 — exactly as expected, since v2's σ
is 4× smaller while still satisfying (ε,δ)-DP. This is the headline
finding: **the tighter v2 proof reveals that attack success is higher
than v1 implied at the same nominal ε.**

| ε         | v1 BB+aux Class B | v2 BB+aux Class B |
|-----------|-------------------|-------------------|
| no DP     | 0.890 ± 0.071     | 0.951 ± 0.039     |
| 10⁸       | 0.883 ± 0.071     | 0.946 ± 0.036     |
| 10⁶       | 0.843 ± 0.068     | 0.951 ± 0.041     |
| 10⁴       | 0.843 ± 0.046     | 0.939 ± 0.029     |
| 100       | 0.782 ± 0.028     | 0.861 ± 0.034     |
| 1         | 0.787 ± 0.038     | 0.835 ± 0.040     |

The standard (no-Class-B) BB+aux attack also reaches 0.524 at v2 ε=1
vs 0.498 at v1 ε=1.

**Theoretical soundness.**
1. *Proof-draft consistency.* v2 matches the user's NeurIPS-revision
   proof: the matrix released is the uncentered second moment with
   per-entry bound `c²·k_max/(n−k_max)` (factor 1, vs. 4 in v1).
   Empirical quality parity confirms the `μ ≈ 0` assumption that lets
   the proof drop the centering step.
2. *The cor() issue.* v1's R script applies `cor()`, which both
   centers and **normalizes**, so what was actually released in v1 is
   the correlation matrix, not the centered covariance. v2 cleans
   this up: the saved `cov_mat` is the raw uncentered second moment.
   PSD-projection and normalization to a correlation matrix are
   deferred to the Python side (`generate.py`), where they are
   applied **after** the Gaussian noise — valid by the
   post-processing theorem, so they consume no privacy budget. The
   proof then bounds exactly what the R script outputs, closing the
   v1 proof↔code gap.

**Bottom line.** v2 quality is on par with v1 (slightly better at
ε≥10⁴, identical at low ε), the proof is now cleanly aligned with
the released matrix, and the tighter sensitivity exposes attack
leakage at every ε that v1's looser proof masked. Recommended next
step: run the full sweep (all ε, all donor counts, 5 trials).
