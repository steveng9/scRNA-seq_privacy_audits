# scVI MIA Experiments

Extends the scMAMA-MIA paper to attack **scVI** (Lopez et al. 2018), a VAE-based
single-cell RNA-seq synthetic data generator.  This addresses Reviewer 1 / 2 requests
to test beyond scDesign2/scDesign3.

## Why scVI

- Most widely cited scRNA-seq generative model (~8 000 citations)
- Architecturally orthogonal to copula-based methods: learns a continuous latent space
  via a VAE rather than fitting explicit marginal distributions
- Available via the `scvi-tools` pip package with a clean Python API
- Published attack strategy: ELBO-based MIA (higher ELBO → better fit → member)

## Attack Strategy

scVI trains a VAE: it learns p(x|z) (decoder) and q(z|x) (encoder).
The per-cell ELBO is:

    ELBO(x) = E_{q(z|x)}[log p(x|z)]  −  KL(q(z|x) ‖ p(z))

Members tend to have **higher ELBO** (the model overfits to training donors' expression
patterns).  We use ELBO as the cell-level membership score and aggregate to the donor
level via sigmoid + mean — the same pipeline as scMAMA-MIA.

Two threat model variants:
- **no-aux** (BB-aux analogue): uses only the target model's ELBO
- **+aux** (BB+aux analogue): calibrates using an auxiliary scVI model trained on D_aux

## Setup

```bash
# 1. Create the conda env (one-time)
bash experiments/scvi/setup_scvi_env.sh

# 2. Run an experiment (from repo root, in any env with anndata + scanpy)
conda run -n tabddpm_ python experiments/scvi/run_scvi_mia.py \
    --dataset  /path/to/data/ok/full_dataset_cleaned.h5ad \
    --out-dir  /path/to/results/scvi_mia/10d \
    --n-donors 10 \
    --n-trials 5 \
    --hvg-path /path/to/data/ok/hvg.csv
```

## File Layout

```
experiments/scvi/
├── README.md
├── setup_scvi_env.sh          # one-time environment setup
└── run_scvi_mia.py            # experiment runner (all trials)

src/sdg/scvi/
├── __init__.py
├── model.py                   # ScVI wrapper class (subprocess → scvi_ env)
└── run_scvi_standalone.py     # inner script (train / generate / score)

src/attacks/scvi_mia/
├── __init__.py
└── attack.py                  # ELBO aggregation → donor-level AUC
```

## Output

Each call to `run_scvi_mia.py` writes:

```
<out-dir>/
├── scvi_mia_results.csv       # per-trial AUC (no-aux + aux)
└── <trial>/
    ├── datasets/
    │   ├── train.h5ad
    │   ├── target.h5ad        # train + holdout cells, member-labelled
    │   └── aux.h5ad
    ├── models/
    │   ├── target/            # scVI model checkpoint
    │   └── aux/               # auxiliary scVI checkpoint
    ├── scores_target.npy      # per-cell ELBO from target model
    └── scores_aux.npy         # per-cell ELBO from aux model (if +aux)
```

## References

- Lopez et al. 2018 — scVI: Deep generative modelling for single-cell transcriptomics
- Carlini et al. 2022 — Membership inference attacks from first principles (ELBO MIA)
