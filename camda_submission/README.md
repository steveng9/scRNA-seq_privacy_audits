# CAMDA 2025 Track II — PPML-Huskies Submission
## Team: PPML-Huskies

This package contains:
1. **Blue team** — three single-cell SDG methods (scDesign2+DP, scVI, ZINBWave)
2. **Red team** — scMAMA-MIA, a Mahalanobis-based membership inference attack

---

## Directory Layout

```
blue_team.py               Modified CAMDA blue team entry point
config.yaml                Blue team generator configuration
environment.yaml           Conda env for scDesign2+DP, ZINBWave, and the attack
environment_scvi.yaml      Secondary conda env for scVI only
install_r_packages.r       Installs required R packages (run once after conda setup)

ppml_generators/             Blue team generator source files
  scdesign2_dp.py          scDesign2 + Gaussian-mechanism DP (ε=100)
  zinbwave_gen.py          ZINBWave ZINB latent-factor model
  scvi_gen.py              scVI VAE (delegates to scvi_ conda env)
  run_scvi_standalone.py   scVI subprocess runner
  run_zinbwave_standalone.py  ZINBWave subprocess runner
  scdesign2.r              scDesign2 R implementation
  zinbwave.r               ZINBWave R implementation
  dp_copula.py             Gaussian DP noise injection
  sensitivity.py           DP sensitivity calculations

scMAMA-MIA/                Red team attack source files (self-contained)
  run_attack.py            Attack entry point (see usage below)
  attack.py                BB+aux Mahalanobis + Class B LLR attack functions
  scoring.py               Cell-level and donor-level aggregation
  copula.py                Gaussian copula parsing from scDesign2 .rds files
  cdf_utils.py             ZINB CDF and log-PMF utilities
  shadow_model.py          Shadow scDesign2 fitting for black-box mode
  scdesign2.r              scDesign2 R script (copy; self-contained)
  config.yaml              Attack configuration
```

---

## Setup

### 1. Create primary conda environment

```bash
conda env create -f environment.yaml
conda activate ppml-camda-env
```

### 2. Install R packages (required for scDesign2+DP, ZINBWave, and the attack)

```bash
Rscript install_r_packages.r
```

This installs: `scDesign2`, `zinbwave`, and their Bioconductor dependencies.
Expected time: 10–30 minutes.

### 3. (Optional) Create the scVI environment

Only needed when using the scVI generator (`generator_name: scvi`).

```bash
conda env create -f environment_scvi.yaml
```

---

## Blue Team: Running a Generator

Place `blue_team.py` and `ppml_generators/` in the CAMDA repo under
`src/generators/` (alongside the existing `models/` and `utils/` directories).

Edit `config.yaml` and set:
- `generator_name` to one of: `scdesign2_dp`, `scvi`, `zinbwave`
- `dir_list.home` and `dir_list.data` to your output and data directories

Then run from `src/generators/`:

```bash
conda activate ppml-camda-env
python blue_team.py run-singlecell-generator
```

### Generator options

| `generator_name` | Description | Env required |
|---|---|---|
| `scdesign2_dp` | scDesign2 Gaussian copula + Gaussian-mechanism DP (ε=100) | `ppml-camda-env` |
| `zinbwave`     | ZINBWave ZINB latent-factor model (Risso et al. 2018) | `ppml-camda-env` |
| `scvi`         | scVI variational autoencoder | `ppml-camda-env` + `ppml-camda-scvi` |

All three generators apply HVG selection (Scanpy: `min_mean=0.0125`, `max_mean=3`,
`min_disp=0.5`) before fitting.

---

## Red Team: Running the scMAMA-MIA Attack

The attack is **self-contained** in `scMAMA-MIA/` and does not require the CAMDA
repo to be installed.

### Requirements
The attack uses R (via rpy2) to parse scDesign2 `.rds` copula files.
Make sure the `ppml-camda-env` conda environment is active and R packages are installed.

### Inputs

| Argument | Required | Description |
|---|---|---|
| `--synth-h5ad` | Yes | Synthetic data from the target SDG |
| `--target-h5ad` | Yes | Target cells to score (obs: `individual`, `cell_type`) |
| `--train-donors` | Yes* | `.npy` file listing training donor IDs (= members) |
| `--aux-h5ad` | No | Auxiliary reference data; omit for BB-aux (weaker) |
| `--hvg-csv` | No | HVG mask CSV; auto-computed if absent |
| `--out-csv` | Yes | Output predictions CSV |

\*Optional if the target h5ad already has a `member` obs column (0/1).

### Example (BB+aux, recommended)

```bash
cd scMAMA-MIA/
conda activate ppml-camda-env

python run_attack.py \
  --synth-h5ad    /data/synthetic.h5ad \
  --target-h5ad   /data/target_cells.h5ad \
  --train-donors  /data/train_donors.npy \
  --aux-h5ad      /data/auxiliary.h5ad \
  --hvg-csv       /data/hvg.csv \
  --out-csv       results/predictions.csv
```

### Example (BB-aux, no auxiliary data required)

```bash
python run_attack.py \
  --synth-h5ad    /data/synthetic.h5ad \
  --target-h5ad   /data/target_cells.h5ad \
  --train-donors  /data/train_donors.npy \
  --out-csv       results/predictions_noaux.csv
```

The attack automatically falls back to BB-aux mode when `--aux-h5ad` is omitted.

### Output

`predictions.csv` contains one row per donor:
```
donor_id, true_membership, predicted_score
```
- `true_membership`: 1 if the donor was in the training set, 0 otherwise
- `predicted_score`: attack's estimated membership probability ∈ [0, 1]

If ground-truth labels are available, the script also prints **ROC AUC** to stdout.

---

## Attack Method: scMAMA-MIA (Black Box + auxiliary data "BB+aux" variant)

scMAMA-MIA adapts the MAMA-MIA framework (Golob et al.) to scRNA-seq data.

### Algorithm

1. **Shadow model** (black-box): fit scDesign2 Gaussian copulas on the synthetic
   data (`D_synth`) and optionally on auxiliary data (`D_aux`).

2. **Mahalanobis distance**: for each target cell, map gene expression to
   Uniform(0,1) via the ZINB marginal CDFs, then compute the Mahalanobis distance
   to the copula mean under both copulas.
   - Primary score: `log(d_aux) - log(d_synth)`
     (higher = closer to synth copula = more likely a member)

3. **Class B focal points**: for secondary genes (not in the copula covariance
   matrix), compute the log-likelihood ratio `Σ_g [log p_synth(x_g) - log p_aux(x_g)]`.
   This is the Neyman-Pearson optimal statistic under gene independence.
   Combined: `logit = log(d_aux/d_synth) + γ · LLR_B`  where `γ = 1/√n_secondary`.

4. **Donor aggregation**: z-score + sigmoid activation → average over each donor's
   cells → donor-level membership probability.

### Why it works

scDesign2's Gaussian copula directly encodes the gene-gene covariance structure of
the training data. Synthetic cells generated from a copula fit on a small training
set are "close" (in Mahalanobis distance) to that copula — and so are the original
training cells. Non-member cells (from a different donor split) are systematically
farther.

---

## References

- Golob et al. (2026). "Privacy Vulnerabilities in Synthetic Single-Cell RNA-Sequence Data."
  Submitted to RECOMB-Privacy 2026.
- Li et al. (2019). scDesign2. *Nature Computational Science*.
- Risso et al. (2018). ZINBWave. *Nature Communications*.
- Lopez et al. (2018). scVI. *Nature Methods*.
- Dwork & Roth (2014). "The Algorithmic Foundations of Differential Privacy."
