# scDiffusion MIA Experiments

**Source**: Luo et al. 2024, *Bioinformatics* — https://doi.org/10.1093/bioinformatics/btae518
**Repo**: `/home/golobs/scDiffusion`  (cloned from github.com/EperLuo/scDiffusion)

## Architecture

scDiffusion is a latent diffusion model for scRNA-seq:

1. **VAE** — encodes cells (gene space) → 128-dim latent; trained from scratch
2. **DDPM backbone** — denoising diffusion in latent space (1000 steps, linear schedule)
3. **Classifier** (optional) — for conditional generation; not used here

Generation: sample latent from DDPM → decode through VAE → gene expression.

## Attack Strategy

Diffusion denoising loss MIA (Carlini et al. 2022 — "Membership Inference Attacks From First Principles"):

For each target cell:
1. Encode → latent z via VAE
2. At T random timesteps: add noise → compute `||ε - ε_θ(z_t, t)||²`
3. Score = `-mean_t[loss_t]`  (lower loss = better known to model = member)

No aux variant only (no calibrated aux model — unlike scMAMA-MIA).

## Setup

```bash
# One-time env setup
bash experiments/scdiffusion/setup_scdiffusion_env.sh

# Run experiment
conda run -n tabddpm_ python experiments/scdiffusion/run_scdiffusion_mia.py \
    --dataset  /home/golobs/data/ok/full_dataset_cleaned.h5ad \
    --out-dir  /home/golobs/data/scdiffusion_mia/10d \
    --n-donors 10 --n-trials 5 \
    --hvg-path /home/golobs/data/ok/hvg.csv
```

## Key Options

| Flag | Default | Notes |
|------|---------|-------|
| `--n-donors` | 10 | train = holdout donors |
| `--n-trials` | 5 | independent trials |
| `--vae-steps` | 150000 | VAE training iterations |
| `--diff-steps` | 300000 | diffusion backbone iterations |
| `--batch-size` | 512 | |
| `--save-interval` | 50000 | checkpoint frequency |
| `--n-score-times` | 50 | noise levels averaged for MIA |
| `--latent-dim` | 128 | VAE latent dimension |
| `--cell-type-col` | cell_type | obs column for cell type |
| `--individual-col` | individual | obs column for donor ID |
| `--hvg-path` | None | pre-computed HVG CSV |
| `--conda-env` | scdiff_ | |

## Standalone Script API

```bash
# In scdiff_ env:
python src/sdg/scdiffusion/run_scdiffusion_standalone.py train_vae \
    train.h5ad  vae_out/  --vae-steps 150000

python src/sdg/scdiffusion/run_scdiffusion_standalone.py train \
    train.h5ad  vae_out/model_seed=0_step=150000.pt  diff_out/  --diff-steps 300000

python src/sdg/scdiffusion/run_scdiffusion_standalone.py generate \
    train.h5ad  vae.pt  diff.pt  synth.h5ad  5000

python src/sdg/scdiffusion/run_scdiffusion_standalone.py score \
    target.h5ad  vae.pt  diff.pt  scores.npy  --n-score-times 50
```

## File Layout

```
experiments/scdiffusion/
├── README.md
├── setup_scdiffusion_env.sh
└── run_scdiffusion_mia.py

src/sdg/scdiffusion/
├── __init__.py
├── model.py                   (ScDiffusion wrapper class)
└── run_scdiffusion_standalone.py   (inner script, runs in scdiff_ env)

src/attacks/scdiffusion_mia/
├── __init__.py
└── attack.py                  (denoising loss aggregation → AUC)

/home/golobs/scDiffusion/      (upstream repo, read-only)
```

## Output Per Trial

```
<out-dir>/<trial>/
  datasets/
    train.h5ad          D_train cells
    target.h5ad         D_train + D_hold, member-labelled
  models/
    vae/  model_seed=0_step=150000.pt
    diff/ model300000.pt   (+ intermediates every save_interval steps)
  scores.npy            per-cell denoising-loss MIA scores
```

## Notes

- Training is slow: ~2–4 h per trial on a single A100 for 150k+300k steps
- For quick testing: `--vae-steps 5000 --diff-steps 10000 --save-interval 5000`
- `scdiff_` env pins torch==1.13.1+cu117 (required by scDiffusion)
- The env is isolated: the caller (tabddpm_ / camda_) dispatches via `conda run`
