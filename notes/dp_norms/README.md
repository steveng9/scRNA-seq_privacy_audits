# DP noise / sensitivity tables

Two CSVs derived from the actual DP runs in `~/data/scMAMAMIA/{ds}/scdesign2/`.

## Sources of truth (constants)

| Constant | Value | Source |
|---|---|---|
| Entry-wise clip `c` | `3.0` | `experiments/dp/v2/generate.py:46`; `src/generators/gen_dp_quality_data.py:50` |
| `delta` | `1e-5` | same files (`DELTA`) |
| v1 sensitivity prefactor `K` | `4` (centered covariance) | `src/sdg/dp/sensitivity.py:77` |
| v2 sensitivity prefactor `K` | `1` (uncentered second moment) | same |
| Calibration | classical Gaussian: `σ = Δ_F · sqrt(2·ln(1.25/δ)) / ε` | `src/sdg/dp/sensitivity.py:165` |
| Per-cell-type Frobenius sensitivity | `Δ_F = K · c² · G · k_max / (n_cells - k_max)` | `frobenius_sensitivity()` |
| Composition | none in code — `ε` is per cell type | `apply_gaussian_dp` called once per ct |
| Noise application | `E_{ij} ~ N(0, σ²)` per entry; symmetrize `(E + Eᵀ)/2`; add to cov matrix | `src/sdg/dp/dp_copula.py:141-157` |

`sqrt(2·ln(1.25/1e-5)) ≈ 4.84596`.

## copula_dims.csv (1468 rows)

One row per (dataset, n_donors, trial, cell_type) — these are the **per-cell-type** scDesign2 fits in
`~/data/scMAMAMIA/{ds}/scdesign2/no_dp/{nd}/{trial}/models/{ct}.rds`.

Columns:
- `dataset` — `ok` or `aida`
- `n_donors` — `5d|10d|20d|50d|100d|200d|490d`
- `trial` — `1..6` (490d only has 1–5)
- `cell_type` — scDesign2's integer cell-type code
- `n_cells_in_copula` — `n_cell` field from the .rds (cells used to fit this copula)
- `n_train_cells_in_obs` — sanity check: `(full_obs.individual ∈ train) & (cell_type == ct)` row count; equals `n_cells_in_copula` when scDesign2 didn't filter further
- `n_genes_primary` — `len(gene_sel1)` = `G` (number of group-2 genes in the Gaussian copula)
- `k_max` — max cells any single train donor contributes to this cell type
- `has_cov_mat` — should be `True` everywhere (i.e., a Gaussian copula was actually fit)

## sigma_table.csv (29260 rows)

`copula_dims.csv` × `{v1, v2}` × `{1, 10, 100, 10³, 10⁴, 10⁵, 10⁶, 10⁷, 10⁸, 10⁹}`.

Columns add: `dp_variant`, `prefactor` (4 or 1), `clip_value`, `delta`, `epsilon`, `delta_F`, `sigma`.

`sigma` is the std of the Gaussian drawn for **every off-diagonal entry** of that cell type's covariance matrix at that ε. (Symmetrization halves the **effective** per-entry variance to `σ²/2`, but the draw itself is `N(0, σ²)`.)

## What ε means in the table

`epsilon` is the per-cell-type budget passed to `gaussian_noise_scale`. The codebase does **not** charge composition across cell types — so the actual basic-composition guarantee for the released artifact (tuple of T noisy correlation matrices) is `(T·ε, T·δ)`. Replacing this with zCDP composition is one of the two improvements in `notes/dp_vector_clipping_plan.md`; the other is swapping `K c² G` for `2 B²` once a vector clip `B` is chosen.

## Cross-checks

The v2 σ values for `ok 20d trial 1, ε=1` match the saved
`~/data/scMAMAMIA/ok/scdesign2/v2_eps_1/20d/1/provenance.json::per_cell_type_sigma` exactly.
