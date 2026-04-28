# scMAMA-MIA: Privacy Attacks on Synthetic scRNA-seq Data

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Authors

- Steven Golob (corresponding author) - golobs@uw.edu
- Patrick McKeever
- Sikha Pentyala
- Martine De Cock
- Jonathan Peck

## Overview

scMAMA-MIA is a membership inference attack (MIA) against synthetic single-cell RNA-seq
(scRNA-seq) data generators.  It exploits the Gaussian copula structure fitted by
scDesign2 (and scDesign3) to infer which donors' cells were used to train a generator —
from the synthetic data alone.  For generators without copula structure (scVI, scDiffusion,
NMF), the attack uses scDesign2 as a proxy shadow model in black-box mode.

The framework supports multiple synthetic data generators (SDGs) and threat models
(white-box / black-box, with / without auxiliary data).

---

## Datasets

| Dataset | Cells | Donors | Cell types | Notes |
|---------|-------|--------|------------|-------|
| **OneK1K** | 1.26M | 981 | 14 | CAMDA 2025 / onek1k.org |
| **AIDA** | 1.06M | 508 | 33 | CZ CELLxGENE; includes sex + ethnicity metadata |
| **HFRA** | 108K | 22 | 9 | Fetal retina; small donor pool |

---

## Supported SDG Methods

| Method | Variant | Notes |
|--------|---------|-------|
| scDesign2 | no-DP, +DP (ε=10⁰–10⁹) | Primary attack target; Gaussian copula |
| scDesign3 | Gaussian, Vine copula | Proxy shadow model attack |
| scVI | — | Proxy shadow model attack |
| scDiffusion | — | Proxy shadow model attack |
| NMF | no-DP, +DP sweep | CAMDA 2024 co-winner; no copula structure |
| ZINBWave | no-DP | Risso et al. 2018; per-cell-type ZINB latent-factor model; proxy shadow model attack |

---

## Data Layout

All synthetic data lives under `~/data/scMAMAMIA/` with a clean hierarchy:

```
~/data/scMAMAMIA/
  {dataset}/                        # ok, aida, cg
    full_dataset_cleaned.h5ad
    hvg.csv  /  hvg_full.csv
    splits/{nd}d/{trial}/           # donor ID arrays — shared across ALL SDG variants
      train.npy / holdout.npy / auxiliary.npy
    aux_artifacts/{nd}d/{trial}/    # scDesign2 copulas fitted on aux donors — shared
      {cell_type}.rds  (one per cell type)
    scdesign2/
      no_dp/{nd}d/{trial}/
      eps_{e}/{nd}d/{trial}/        # DP variants
    scdesign3/
      gaussian/{nd}d/{trial}/
      vine/{nd}d/{trial}/
    scvi/no_dp/{nd}d/{trial}/
    scdiffusion/no_dp/{nd}d/{trial}/
    nmf/
      no_dp/{nd}d/{trial}/
      eps_{e}/{nd}d/{trial}/        # NMF DP sweep
    zinbwave/
      no_dp/{nd}d/{trial}/          # ZINBWave (10d, 20d, 50d for OneK1K)
```

Each SDG trial dir contains: `datasets/synthetic.h5ad`, `artifacts/`, `models/`, `results/`.
Donor splits and aux copulas are **not** stored per-SDG — only in the shared `splits/` and
`aux_artifacts/` dirs. All SDG variants for the same (dataset, nd, trial) share the same
aux copula fit, saving ~30–60 min of scDesign2 computation per trial.

---

## Installation

```bash
conda env create -f ENVIRONMENT.yaml
conda activate camda_conda
```

Additional conda environments for specific generators:
- `nmf_` — NMF generator (`src/sdg/nmf_generator/environment.yml`)
- `scvi_` — scVI
- `scdiff_` — scDiffusion

---

## Usage

### 1. Data Cleaning

```bash
python src/data/clean_data.py
```

### 2. Generate Synthetic Data

**All generators** (scDesign3, scVI, scDiffusion, NMF):
```bash
nohup conda run --no-capture-output -n tabddpm_ \
    python experiments/sdg_comparison/run_all.py --skip-hvg \
    > /tmp/sdg_generation.log 2>&1 &
```

**ZINBWave** (OneK1K, 10d/20d/50d, 5 trials each):
```bash
nohup bash experiments/sdg_comparison/run_zinbwave_generation.sh \
    > /tmp/zinbwave_generation.log 2>&1 &
# Monitor:  tail -f /tmp/zinbwave_generation.log
```

**Single ZINBWave trial**:
```bash
python experiments/sdg_comparison/generate_trial.py \
    --generator zinbwave \
    --dataset   /home/golobs/data/scMAMAMIA/ok/full_dataset_cleaned.h5ad \
    --splits-dir /home/golobs/data/scMAMAMIA/ok/splits/10d/1 \
    --out-dir    /home/golobs/data/scMAMAMIA/ok/zinbwave/no_dp/10d/1 \
    --hvg-path   /home/golobs/data/scMAMAMIA/ok/hvg_full.csv
```

**Single trial** (e.g., scDesign2, ok, 10 donors, trial 1):
```bash
python src/run_experiment.py <path-to-config>
```

**Check progress**:
```bash
python experiments/sdg_comparison/check_generated_data.py
```

### 3. Run MIA Attacks

**Full quad sweep** (all SDGs, all datasets, Class B + standard, WB + BB):
```bash
nohup python experiments/sdg_comparison/run_full_sweep.py \
    > /tmp/full_sweep.log 2>&1 &

python experiments/sdg_comparison/run_full_sweep.py --status        # completion grid
python experiments/sdg_comparison/run_full_sweep.py --dry-run       # preview jobs
python experiments/sdg_comparison/run_full_sweep.py --dataset ok    # single dataset
python experiments/sdg_comparison/run_full_sweep.py --sdg nmf       # single SDG filter
```

The sweep is memory-aware (scales parallelism by donor count), guards against OOM
(retries with halved workers), and skips jobs already complete in
`mamamia_results_classb.csv`.  Logs go to `experiments/sdg_comparison/_sweep_logs/`.

**Legacy sweep** (non-scDesign2 methods, standard Mahalanobis only):
```bash
python experiments/sdg_comparison/run_mia_sweep.py
python experiments/sdg_comparison/run_mia_sweep.py --status   # check completion
```

**Single experiment** (use a config YAML):
```bash
python src/run_experiment.py <path-to-config>
```

### 4. Quality Evaluation

```bash
python experiments/sdg_comparison/run_quality_evals.py
# Re-run after metric-code changes (e.g. the 2026-03-25 MMD median-heuristic fix):
python experiments/sdg_comparison/run_quality_evals.py --force --max-donors 200 \
    --dataset-filter ok/scdesign2/no_dp
```

### 4b. Baseline-MIA Sweep

DOMIAS-style baselines (MC, GAN-Leaks, GAN-Leaks-Cal, GAN-Leaks-SC, LOGAN-D1,
DOMIAS-KDE) across `{ok,aida,cg}/scdesign2/no_dp` up to 200d:

```bash
python experiments/sdg_comparison/run_baselines_sweep.py            # run sweep
python experiments/sdg_comparison/run_baselines_sweep.py --status   # check completion
python experiments/sdg_comparison/run_baselines_sweep.py --dry-run  # list pending
```

The distance baselines (MC, GAN-Leaks, GAN-Leaks-Cal) use exact batched
implementations in `src/attacks/baselines/batched_baselines.py` — bit-for-bit
match to the unbatched `domias.baselines_optimized` versions (verified to
≤1e-6 max diff) but with bounded RAM, so they scale to 200d. The DOMIAS-KDE
baseline subsamples its fit set (max_fit=20k per side, matching the DOMIAS
reference protocol; see `notes/PRIORITY_TODO.md` for citations).

### 5. Generate Tables and Figures

```bash
python experiments/sdg_comparison/make_mia_table.py          # LaTeX MIA AUC table (standard Mahalanobis)
python experiments/sdg_comparison/make_mia_table.py --classb # LaTeX MIA AUC table (Class B / Mahalanobis+LLR) → figures/mia_table_classb.tex
python experiments/sdg_comparison/make_quality_table.py   # LaTeX quality table
python experiments/sdg_comparison/make_sdg_umaps.py       # UMAP panels (all SDGs)
python experiments/sdg_comparison/make_nmf_umaps.py       # NMF-specific 3-panel UMAPs
```

---

## Repository Structure

```
scRNA-seq_privacy_audits/
├── src/
│   ├── run_experiment.py           # main MIA entry point
│   ├── sdg/                        # synthetic data generators
│   │   ├── scdesign2/              # scDesign2 (primary; Gaussian copula)
│   │   ├── scdesign3/              # scDesign3 (Gaussian + Vine copula)
│   │   ├── scvi/                   # scVI (VAE-based)
│   │   ├── scdiffusion/            # scDiffusion (diffusion model)
│   │   ├── nmf/                    # NMF wrapper (CAMDA 2024 co-winner)
│   │   ├── nmf_generator/          # upstream NMF repo (submodule)
│   │   └── zinbwave/               # ZINBWave (Risso et al. 2018)
│   ├── attacks/
│   │   ├── scmamamia/              # scMAMA-MIA (Mahalanobis + Class B LLR)
│   │   └── baselines/              # CAMDA2025 baseline MIAs
│   ├── evaluation/                 # quality metrics (LISI, ARI, MMD)
│   └── data/                       # CDF utils, donor splitting strategies
│
├── experiments/
│   ├── sdg_comparison/             # multi-SDG generation, attack, and table scripts
│   │   ├── run_full_sweep.py       # comprehensive quad sweep (all SDGs, all datasets)
│   │   ├── run_baselines_sweep.py  # DOMIAS baseline MIAs across sd2/no_dp
│   │   ├── run_all.py              # batch generation (all SDGs)
│   │   ├── run_mia_sweep.py        # legacy batch MIA sweep (standard Mahalanobis)
│   │   ├── run_quality_evals.py    # batch quality evaluation
│   │   ├── gen_nmf_dp_sweep.py     # NMF DP epsilon sweep
│   │   ├── make_mia_table.py       # LaTeX MIA AUC table
│   │   ├── make_quality_table.py   # LaTeX quality table
│   │   ├── make_sdg_umaps.py       # UMAP panels (all SDGs)
│   │   └── make_nmf_umaps.py       # NMF 3-panel UMAPs
│   ├── ablation/                   # Class B attack ablation framework
│   ├── dp/                         # differential privacy experiments
│   └── migrate_data.py             # one-time data layout migration (2026-04-18)
│
├── figures/                        # generated LaTeX tables and UMAP figures
└── CLAUDE.md                       # detailed implementation notes for Claude Code
```

---

## Key Design Decisions

**Attack variants:**
- `WB+aux` / `WB-aux` — white-box (copula from training data), with/without auxiliary data
- `BB+aux` / `BB-aux` — black-box (copula estimated from synthetic data), with/without auxiliary
- Class B enhancement: adds per-gene LLR evidence from secondary genes; `class_b_gamma="auto"` gives +0.14 AUC on ok 10d

**Evaluation:** ROC AUC at the donor level, averaged over 5 trials with different random donor splits.

**HVG selection:** `sc.pp.highly_variable_genes` with `min_mean=0.0125`, `max_mean=3`, `min_disp=0.5`.

---

## Citation

```bibtex
@misc{golob2025scmamamia,
  title={Privacy Vulnerabilities in Synthetic Single-Cell RNA-Sequence Data},
  author={Golob, Steven and McKeever, Patrick and Pentyala, Sikha and De Cock, Martine and Peck, Jonathan},
  year={2026},
  note={Under review at RECOMB-Privacy 2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Steven Golob — golobs@uw.edu
