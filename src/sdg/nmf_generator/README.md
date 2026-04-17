# SingleCellNMFGenerator

A toolkit for generating synthetic single-cell gene expression data using privacy-enhanced NMF (and other methods), with a simple CLI wrapper.

## Repository layout

- **config.yaml** — main configuration for datasets & generators  
- **environment.yml** — Conda environment specification   
- **src/generators/blue_team.py** — CLI wrapper  
- **src/generators/models/nmf_sampler.py** — DP-NMF generator implementation  

## Setup

1. Create and activate the conda environment:  
   ```bash
   conda env create -f environment.yml  
   conda activate sc-synthetic-demo
   ```
2. (Optional) If you prefer pip:
   ```bash
   pip install -r requirements.txt  
   ```
## Usage

### 1. Generate split indices & data splits

    python src/generators/blue_team.py generate-split-indices
    python src/generators/blue_team.py generate-data-splits

### 2. Run the single-cell generator

    python src/generators/blue_team.py run-singlecell-generator \
        --experiment_name demo_run \
        --dp none

## Method Overview

This pipeline generates synthetic single‐cell gene expression data with optional differential privacy:

1. **Data Splitting**  
   Create stratified train/test splits of your real dataset by donor.

2. **NMF Factorization**  
   Fit MiniBatchNMF on the train split to learn basis **H** and derive embeddings **W**.

3. **Differential Privacy (optional)**  
   Controlled by `--dp` flag:
   - **all**: noise on NMF + KMeans + summaries  
   - **nmf**: Gaussian noise on **H** only  
   - **kmeans**: Laplace noise on KMeans centroids only  
   - **sampling**: Laplace noise on cluster summaries only  
   - **none**: no privacy noise

4. **Synthetic Sampling**  
   Choose via `sampling_method` in `config.yaml`:
   - **ZINB**: zero-inflated negative binomial per cluster  
   - **Poisson**: simple Poisson draws from cluster means  

5. **Output**  
   Saves a compressed AnnData (`.h5ad`) under  
   `data_splits/{dataset}/synthetic/{generator}/{experiment_name}/onek1k_annotated_synthetic.h5ad`.

## User Controls

### CLI Commands

- `generate-split-indices`  
- `generate-data-splits`  
- `run-singlecell-generator`  
  - `--experiment_name <name>`: name of the output subfolder  
  - `--dp <all|nmf|kmeans|sampling|none>`: which DP mechanisms to apply  

### Key Config Entries (`config.yaml`)

- **Paths**:  
  `dir_list.home`,  
  `dataset_config.train_count_file`,  
  `dataset_config.test_count_file`

- **NMF Settings** (`nmf_sampler_config`):  
  `sample_fraction` (fraction of cells),  
  `seed`,  
  `n_components`,  
  `nmf_batch_size`,  
  `n_clusters`,  
  `n_synth_samples`,  
  `proportion_aware`

- **Sampling Method**:  
  `nmf_sampler_config.sampling_method`: `"zinb"` or `"poisson"`

- **Privacy Budgets** (`dp_config`):  
  `eps_nmf`,  
  `nmf_noise_scale`,  
  `eps_kmeans`,  
  `eps_summaries`,  
  `laplace_scale`

Use these flags and config options to customize splits, privacy levels, sampling distributions, and experiment metadata.

### References

1. CAMDA 2025 Health Privacy Challenge

2. Yazar S., Alquicira-Hernández J., Wing K., Senabouth A., Gordon G., Andersen S., Lu Q., Rowson A., Taylor T., Clarke L., Maccora L., Chen C., Cook A., Ye J., Fairfax K., Hewitt A., Powell J. "Single cell eQTL mapping identified cell type specific control of autoimmune disease." Science. (2022) (https://onek1k.org)

3. Lun ATL, McCarthy DJ, Marioni JC. “A step-by-step workflow for low-level analysis of single-cell RNA-seq data with Bioconductor.” F1000Res. (2016)

4. Wolf, F. Alexander, Philipp Angerer, and Fabian J. Theis. "SCANPY: large-scale single-cell gene expression data analysis." Genome biology. (2018)

5. Shao, Chunxuan, and Thomas Höfer. "Robust classification of single-cell transcriptome data by nonnegative matrix factorization." Bioinformatics 33.2 (2017): 235-242.

6. Ran, Xun, et al. "A differentially private nonnegative matrix factorization for recommender system." Information Sciences 592 (2022): 21-35.
