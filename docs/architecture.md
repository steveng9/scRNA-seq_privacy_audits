# scMAMA-MIA: Repository Architecture

## Overview

This repository implements **scMAMA-MIA**, a membership inference attack (MIA) against
synthetic single-cell RNA-seq (scRNA-seq) generators. The codebase is organized around the
research framing — not the CAMDA competition structure it was forked from — and is designed
to support multiple synthetic data generators (SDGs) and the addition of differential
privacy (DP) to those generators.

---

## Directory Structure

```
camda_hpc/
├── configs/
│   ├── template.yaml               # template config for running MIA experiments
│   ├── onek1k/                     # per-dataset experiment configs
│   ├── aida/
│   └── hfra/
│
├── src/
│   ├── run_experiment.py           # MAIN ENTRY POINT: orchestrates the full pipeline
│   │
│   ├── sdg/                        # Synthetic Data Generators
│   │   ├── base.py                 # abstract SDG interface (train / generate / get_copula)
│   │   ├── run.py                  # SDG runner (instantiates and runs any registered SDG)
│   │   ├── scdesign2/
│   │   │   ├── model.py            # scDesign2 Python wrapper (trains via R subprocess)
│   │   │   ├── copula.py           # copula parsing, covariance building, config helpers
│   │   │   └── scdesign2.r         # R script: fit Gaussian copula + generate synthetic data
│   │   ├── scdesign3/              # PLACEHOLDER — not yet implemented
│   │   ├── scvae/                  # PLACEHOLDER — not yet implemented
│   │   └── utils/
│   │       └── prepare_data.py     # bulk RNA-seq data loader (Track I legacy)
│   │
│   ├── attacks/
│   │   ├── scmamamia/
│   │   │   ├── attack.py           # scMAMA-MIA attack algorithms
│   │   │   └── scoring.py          # cell→donor score aggregation + AUC computation
│   │   └── baselines/
│   │       └── sc_baseline.py      # CAMDA2025 baseline MIAs (GAN-Leaks, DOMIAS, etc.)
│   │
│   ├── evaluation/
│   │   ├── quality.py              # synthetic data quality metrics: LISI, ARI, MMD
│   │   └── plots.py                # figure generation
│   │
│   └── data/
│       ├── splits.py               # donor sampling strategies
│       ├── cdf_utils.py            # ZINB CDF, activation function, correlation utilities
│       └── clean_data.py           # raw dataset cleaning/preprocessing
│
├── experiments/
│   ├── scdesign2/                  # scDesign2 MIA experiments (current primary focus)
│   ├── scdesign3/                  # scDesign3 MIA experiments (Experiment 2 — future)
│   ├── dp/                         # Differential Privacy experiments (Experiment 1 — future)
│   ├── baselines/                  # CAMDA2025 baseline MIA comparisons
│   └── bulk_rna_seq/               # Track I bulk RNA-seq (NOT IMPLEMENTED — preserved for future)
│
└── data_splits/
    └── onek1k/                     # pre-split data (real/, synthetic/, split_indices/)
```

---

## How to Run a scMAMA-MIA Experiment

### Single experiment
```bash
python src/run_experiment.py <path-to-config>
```

### With verbose output
```bash
python src/run_experiment.py <path-to-config> --print
```

### Backward-compatible form (T/F server flag still accepted)
```bash
python src/run_experiment.py T <path-to-config> P
```

### Batch experiments
```bash
./create_experiment_config_files.sh  # generate configs
cd src/ && ./run_donor_level_mia.sh  # run all
```

---

## Pipeline Steps (in `run_experiment.py`)

1. **Configure** — load YAML, build path structure, look up next trial number from `tracking.csv`
2. **Split data** — sample train/holdout/aux donor sets
3. **Generate target synthetic data** — run the target SDG on the train set
4. **Generate focal-point copulas** — re-run the SDG on synthetic data (black-box) and aux data
5. **Run scMAMA-MIA attack** — per cell type, compute Mahalanobis distances, aggregate to donor scores
6. **Save results** — AUC saved to `results/mamamia_results.csv`, cell scores to `mamamia_all_scores.csv`
7. **Register trial** — mark this threat-model/trial combination complete in `tracking.csv`

---

## Adding a New SDG

1. Create `src/sdg/<new_sdg>/model.py` subclassing `BaseSingleCellDataGenerator` from `src/sdg/base.py`
2. Register it in `src/sdg/run.py` in the `generator_classes` dict
3. If it uses a Gaussian copula (e.g., scDesign3): `parse_copula()` in `src/sdg/scdesign2/copula.py` should work directly
4. If it uses a different architecture (e.g., scVAE): add a new attack module in `src/attacks/`
5. Add experiment configs under `experiments/<new_sdg>/`

## Adding Differential Privacy to an SDG

The natural injection points are in `src/sdg/scdesign2/copula.py`:
- Add calibrated noise to the fitted covariance matrix `R` (Gaussian/Laplace mechanism)
- Add noise to the marginal distribution parameters (pi, theta, mu)
- Expose an `epsilon` parameter in the SDG config

See `experiments/dp/` for the planned DP experiment structure.

---

## Threat Model Codes

Experiments are identified by a 3-bit threat model code stored in `tracking.csv`:

| Bit | Meaning                | 0 = yes        | 1 = no          |
|-----|------------------------|----------------|-----------------|
| 0   | white_box              | white-box      | black-box       |
| 1   | use_wb_hvgs            | use WB HVGs    | use synth HVGs  |
| 2   | use_aux                | use aux data   | no aux data     |

Primary threat model: `BB+aux` = `tm:110`

---

## Key Design Decisions

- **SDG-agnostic orchestrator**: `run_experiment.py` calls `run_sdg()` generically; all
  scDesign2-specific logic lives in `src/sdg/scdesign2/`
- **Copula parsing is decoupled from the attack**: `src/sdg/scdesign2/copula.py` exposes
  `parse_copula()` which converts R objects to numpy; the attack algorithms in
  `src/attacks/scmamamia/attack.py` operate on those numpy structures
- **Function registry instead of globals()**: `run_experiment.py` uses an explicit
  `FUNCTION_REGISTRY` dict for YAML-configured functions (CDFs, sampling strategies, etc.)
- **R script uses absolute paths**: `scdesign2/model.py` resolves the path to `scdesign2.r`
  via `__file__`, so the script is CWD-independent
- **Trial tracking**: `tracking.csv` allows resuming interrupted experiment batches without
  re-running completed trials
