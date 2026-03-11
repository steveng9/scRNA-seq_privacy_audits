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
from the synthetic data alone.

The framework supports multiple synthetic data generators (SDGs) and threat models
(white-box / black-box, with / without auxiliary data).  See `docs/architecture.md` for
the full design.

---

## Datasets

| Dataset | Cells | Donors | Cell types | Source |
|---------|-------|--------|------------|--------|
| **OneK1K** | 1.26M | 981 | 14 | CAMDA 2025 / onek1k.org |
| **AIDA** | 1.06M | 508 | 33 | CZ CELLxGENE |
| **HFRA** | 108K | 22 | 9 | CZ CELLxGENE |

See `docs/notes_about_OneK1K.txt`, `docs/notes_about_AIDA.txt`, `docs/notes_about_HFRA.txt` for details.

---

## Installation

```bash
conda env create -f ENVIRONMENT.yaml
conda activate camda_conda
```

---

## Usage

### 1. Data Cleaning

```bash
python src/data/clean_data.py
```

Edit the `data_dir` variable at the top of the file to point to your dataset location.

---

### 2. scMAMA-MIA Experiments

#### Single experiment

```bash
python src/run_experiment.py <path-to-config>           # local
python src/run_experiment.py T <path-to-config> P       # server, verbose
```

Use `configs/template.yaml` as a starting point for your config file.

The script:
1. Samples train / holdout / aux donor sets
2. Trains the SDG (scDesign2 by default) on the train set
3. Fits shadow copulas on synthetic data and aux data
4. Runs scMAMA-MIA for each cell type
5. Aggregates to donor-level AUC; saves results

Trial tracking is automatic — re-running the same config resumes incomplete trials
(tracked in `tracking.csv` within the experiment output directory).

#### Batch experiments

```bash
./create_experiment_config_files.sh     # generate configs for all sizes / datasets
./run_donor_level_mia.sh               # run all (edit paths for your machine)
```

---

### 3. Baseline MIA Experiments

Baselines require scMAMA-MIA trials to have already run (they reuse the same splits).

```bash
python src/run_baselines.py <path-to-config>
```

---

### 4. Quality Evaluation

```bash
python src/run_quality_eval.py <path-to-config>
```

---

## Repository Structure

```
camda_hpc/
├── configs/
│   └── template.yaml               # config template for MIA experiments
│
├── src/
│   ├── run_experiment.py           # MAIN ENTRY POINT
│   ├── sdg/                        # Synthetic Data Generators
│   │   ├── base.py                 # abstract SDG interface
│   │   ├── run.py                  # SDG runner (register new generators here)
│   │   ├── scdesign2/              # scDesign2 (primary SDG)
│   │   ├── scdesign3/              # placeholder — not yet implemented
│   │   └── scvae/                  # placeholder — not yet implemented
│   ├── attacks/
│   │   ├── scmamamia/              # scMAMA-MIA attack algorithms
│   │   └── baselines/              # CAMDA2025 baseline MIAs
│   ├── evaluation/                 # quality metrics (LISI, ARI, MMD)
│   └── data/                       # CDF utils, donor splitting strategies
│
├── experiments/
│   ├── scdesign2/                  # scDesign2 experiment configs + runner
│   ├── scdesign3/                  # future
│   ├── dp/                         # differential privacy experiments (HIGH PRIORITY)
│   ├── baselines/                  # CAMDA2025 baseline comparisons
│   └── bulk_rna_seq/               # Track I bulk RNA-seq (not implemented)
│
├── docs/
│   └── architecture.md             # full design doc
└── outputs/                        # legacy CAMDA competition output (see outputs/README.md)
```

For the full architecture and how to add new SDGs or DP, see `docs/architecture.md`.

---

## Citation

```bibtex
@misc{golob2025scmamamia,
  title={Privacy Vulnerabilities in Synthetic Single-Cell RNA-Sequence Data},
  author={Golob, Steven and McKeever, Patrick and Pentyala, Sikha and De Cock, Martine and Peck, Jonathan},
  year={2026},
  note={Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Steven Golob — golobs@uw.edu
