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

This repository contains all code needed to conduct privacy attack experiments on synthetic scRNA-seq data. While designed primarily for attacking scDesign2-generated data, the framework can be applied to any synthetic scRNA-seq dataset.

---

## Datasets

We investigate three scRNA-seq datasets:

### 1. OneK1K (CAMDA2025 Data)
- [Health Privacy Challenge Repository](https://github.com/PMBio/Health-Privacy-Challenge)
- [CAMDA 2025 Contest](https://bipress.boku.ac.at/camda2025/the-camda-contest-challenges/)
- [ELSA Health Privacy Challenge](https://elsa-ai.eu/elsa-health-privacy-challenge/)

### 2. Asian Immune Diversity Atlas (AIDA) - Version 1
- [CZ CELLxGENE Collection](https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508)

### 3. Human Fetal Retina Atlas (HFRA)
- [CZ CELLxGENE Collection](https://cellxgene.cziscience.com/collections/c11009b8-b113-4a99-9890-78b2f9df9d79)

**Note:** Additional dataset information can be found in:
- `./notes_about_OneK1K.txt`
- `./notes_about_AIDA.txt`
- `./notes_about_HFRA.txt`

---

## Installation

### Environment Setup

Create the Python environment using Miniconda:
```bash
conda env create -f ENVIRONMENT.yaml
conda activate <environment-name>
```

---

## Usage

### 1. Data Cleaning and Preparation

Clean the full datasets using:
```bash
python ./src/clean_data.py
```

**Important:** Edit the `data_dir` variable at the top of the file to point to your dataset location.

---

### 2. scMAMA-MIA Experiments

#### Single Experiment

1. **Create a configuration file** using the template:
```
   ./example_cfg_file_for_mamamia_experiments.yaml
```

2. **Run a single experiment:**
```bash
   cd ./src/generators/
   python ../mia_experiments_main.py F <path-to-config-file> P
```

   This script will:
   - Sample donors from the full dataset
   - Train scDesign2 on the sampled data
   - Generate synthetic data
   - Execute attacks according to the specified threat model

#### Batch Experiments

1. **Generate all config files:**
```bash
   ./create_experiment_config_files.sh
```
   This creates config files for all size configurations and datasets.

2. **Run the full experiment suite:**
```bash
   cd ./src/generators/
   ./run_donor_level_mia.sh
```

**Note:** Edit script paths to match your local machine configuration.

#### Experiment Tracking

A `tracking.csv` file automatically logs completed trials. When `mia_experiments_main.py` runs, it checks this file and initiates new trials for incomplete experiments.

---

### 3. Baseline MIA Experiments

Run baseline MIAs from the CAMDA2025 competition:

#### Single Baseline Experiment
```bash
cd ./src/mia/
python ../baseline_mias.py T <path-to-config-file>
```

#### Full Baseline Suite
```bash
./run_baseline_mias.sh
```

**Important:** Baseline experiments only execute for trials already completed by scMAMA-MIA. Run scMAMA-MIA experiments first.

---

### 4. Quality Evaluations

Evaluate synthetic data quality using two approaches:

#### CAMDA Competition Metrics
```bash
cd ./src/generators/
python ../simplified_quality_evaluation.py T <path-to-config-file> P
```

Or run all evaluations:
```bash
./run_quality_evals.sh
```


#### UMAP Visualization
Generate UMAPs comparing real and synthetic data:
```bash
python genUmap.py
```

**Important:** Quality experiments only execute for trials already completed by scMAMA-MIA. Run scMAMA-MIA experiments first.

---

## Directory Structure

The experiment configuration file template (`example_cfg_file_for_mamamia_experiments.yaml`) demonstrates the directory structure created for storing experimental results and artifacts.

---

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{golob2025scmamamia,
  title={Privacy Vulnerabilities in Synthetic Single-Cell RNA-Sequence Data},
  author={Golob, Steven and McKeever, Patrick and Pentyala, Sikha and De Cock, Martine and Peck, Jonathan},
  year={2025},
  note={Under review}
}
```

---

[//]: # (## License)

[//]: # ()
[//]: # ([Specify your license here - e.g., MIT, Apache 2.0, etc.])

[//]: # ()
[//]: # (---)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact Steven Golob at golobs@uw.edu or open an issue on this repository.

