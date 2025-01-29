# :blueberries: BLUE TEAM: Evaluate synthetic data

Evaluation workflow:

1. [**Prerequisites**](#prerequisites): Preparations evaluation workflow setup.
2. [**Generate split-specific evaluation results**](#step-1-generate-split-specific-results): Run an evaluation script for each data split.
3. [**Combine evaluation results**](#step-2-combine-split-specific-results): Merge all split-specific results into a single file.
4. [**Run PCA analysis**](#step-3-run-pca-analysis): Perform PCA for each data split and save the visualizations.
5. [**Biological analysis**](#step-4-biological-analysis): Perform differential gene expression and gene co-expression analyses.
6. [**Evaluation metrics**](#evaluation): Evaluation metrics provided are explained. 
7. [**Baseline evaluation results**](#baseline-results): Performances of the baseline generator methods are provided for comparison. 

---

## Prerequisites

### **Activate the python environment**: 

```bash
micromamba activate <environment>
```

### **Ensure the following directories exist**:
Remember to update `home_dir`. The other configurations can stay depending on your preferences. 
   - `res_files`: Directory for storing evaluation metric scores in a CSV file.
   - `figures`: Directory for saving PCA plots.
   - `bio_files`: Directory to store image and csv files for biological evalutions. 
   - `mia_files`: Directory to store membership inference attack scores. Please refer to [Red Team homepage](/experiments/red_team/README.md) if you want to test this out. 
 

### **Configuration Variables**:
You need to modify `config.yaml` according to your need for each experiment. Define the following variables according to your setup in `config.yaml`:
   - `dataset_config`: Update the name of the dataset you want to evaluate. Total number of splits remains same as in your generation configuration (e.g., `5`).
   - `generator_config`: Update the name of the method and experiment name you want to generate evaluation scores. 

Please be reminded that you need to put `config.yaml` in the same directory you are running your experiment. 


### Example config for evaluating multivariate

e.g. In the Generation step, we set the `--experiment_name` argument as `noise_0.5`, therefore we will use the same `experiment_name` in `generator_config`.  

```bash
dataset_config:
  name: "TCGA-COMBINED" 
  num_splits: 5


generator_config:
  name: "multivariate"
  experiment_name: "noise_0.5"
```


## Step 1: Generate Split-Specific Results
Refer to [Evaluation Metrics](#evaluation-metrics) for the list of evaluation metrics generated. 

Make sure to replace `<src_dir>`  with your corresponding path. For each data split (from 1 to `split_num`), run the following command:

```bash
python {src_dir}/evaluation/evaluate.py run-evaluator {split_no}
```

For each split, evaluation metrics are computed and saved under: ``{home_dir}/{res_files}/{dataset_name}/{generator_name}/{experiment_name}/split_{split}.csv``


## Step 2: Combine Split-Specific Results

`combine-results` function  assumes all generated split results (split_{split_no}.csv) are saved under the following directory: `{home_dir}/{res_files}/{dataset_name}/{generator_name}/{experiment_name}`. 

Run the below line to combine the results from each split file into single CSV file and to compute the average of the folds. 

```bash
python {src_dir}/evaluation/evaluate.py combine-results 
```

The output is a CSV file in the same directory above named `evaluation_results.csv`.


## Step 3: Run PCA Analysis

For each split, generate a PCA visualization using the following command:

```bash
python {src_dir}/evaluation/evaluate.py plot-pca {split}
```

This command generates a PCA plot for each split in the following format:
``{home_dir}/{figures}/{dataset_name}/{generator_name}/{experiment_name}/pca_compare_split_{split}.png``


## Step 4: Biological Analysis

We adopt [the analysis](https://github.com/MarieOestreich/PRO-GENE-GEN/tree/main/eval/bio_eval) generated for [(Chen, Oestreich, & Afonja et al. 2024)](https://arxiv.org/abs/2402.04912) and utilize `hcocena` for gene co-expression and `scran` for differential gene expression analysis on R. 

### Install R libraries

The scripts are executed in `R/4.3.2` using the below R packages, make sure to install them before running the provided scripts.  

```R
library(hcocena)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(dplyr)
library(ggplot2)
library(scran)
```

### Differential expression

`{p_value_th}` is assigned as `0.05` for the significance threshold, and can be modified from `bio_params` key `config.yaml`.

```bash
    module load R/4.3.2
    Rscript {src_dir}/evaluation/bio/diffexpression.R {home_dir} {split_no} {dataset_name} {generator_name} {experiment_name} {p_value_th} 
```

e.g. 


### Co-expression

 `hcocena` package requires some reference files from the [hcocena repo](https://github.com/MarieOestreich/hCoCena). Please clone the repo first, and save it under `{hcocena_dir}` in your local.

```bash     
    module load R/4.3.2
    Rscript {src_dir}/evaluation/bio/coexpression.R {home_dir} {split_no} {dataset_name} {generator_name} {experiment_name} {hcocena_dir} 
```

# Evaluation 

## Metric definitions 
- Here we describe the list of evaluation metrics used in [evaluate.py](/src/evaluation/evaluate.py). **We strongly encourage the participants to also use other evaluation metrics, or even, propose their own.**

- The term "synthetic datasets" here refers to the datasets generated for each training set in each split. The test split, which is never used in the training or synthetic data generation process, is reserved solely for evaluation purposes, such as training on synthetic data and testing on real data, among other evaluations.

| Category | Method Name                | Method Details                       | Description                                         | Value (Better) |
|----------|----------------------------|--------------------------------------|-----------------------------------------------------|----------------|
| Utility  | accuracy_synthetic         | Accuracy    |  Train on Synthetic, Test on Real (for downstream task)                                                  | (High)         |
| Utility  | avg_pr_macro_synthetic     | AUPR        | Train on Synthetic, Test on Real (for downstream task)                                                    | (High)         |
| Utility  | accuracy_real         | Accuracy    |  Train on Real, Test on Real (for downstream task)                                                  | (High)         |
| Utility  | avg_pr_macro_real     | AUPR        | Train on Real Test on Real (for downstream task)                                                    | (High)         |
| Utility  | feature_overlap_count      | Number of Overlapping Important Features | 10 features * per class                                               | (High)         |
| Utility  | PCA Plot                   | Visualizing 2D clusters                                   | -                                                   |                |
| Fidelity | MMD_score                  | Maximum Mean Discrepancy             | Difference between synthetic and real datasets' probability distributions    | (Low)          |
| Fidelity | discriminative_score             | Discriminative score           | F1 score for distinguishing  synthetic and real dataset  | (Low)          |
| Privacy  | distance_to_closest        | Distance to the Closest Neighbor     | Average distance of synthetic dataset to the nearest real data point                  | (High)         |
| Privacy  | distance_to_closest_base        | Distance to the Closest Neighbor     | Average distance within real dataset to the nearest data point                  | (High)         |


## Baseline results

Default values in [config.yaml](/experiments/blue_team/2_generation/config.yaml) are used. The average scores are reported. 

### TCGA-BRCA

| Metric / Method            | Multivariate | CVAE      | DP-CVAE   | CTGAN     | DP-CTGAN  |
|----------------------------|--------------|-----------|-----------|-----------|-----------|
| accuracy_synthetic         | 0.8476       | 0.8348    | 0.6731    | 0.2112    | 0.1955    |
| accuracy_real              | 0.8650       | 0.8650    | 0.8650    | 0.8650    | 0.8632    |
| avg_pr_macro_synthetic     | 0.8415       | 0.8396    | 0.4529    | 0.2260    | 0.2090    |
| avg_pr_macro_real          | 0.8656       | 0.8656    | 0.8656    | 0.8656    | 0.8659    |
| feature_overlap_count      | 19.6000      | 20.8000   | 6.2000    | 5.4000    | 1.2000    |
| MMD_score                  | 0.0180       | 0.0177    | 0.1928    | 0.0935    | 1.1529    |
| discriminative_score       | 0.5496       | 0.7997    | 1.0000    | 0.9969    | 1.0000    |
| distance_to_closest        | 28.5348      | 16.0372   | 49.8122   | 19.6806   | 101.9678  |
| distance_to_closest_base   | 24.0435      | 24.0435   | 24.0435   | 24.0435   | 24.0435   |



### TCGA-COMBINED

| Metric                        | Multivariate | CVAE    | DP-CVAE | CTGAN   | DP-CTGAN |
|-------------------------------|--------------|---------|---------|---------|----------|
| accuracy_synthetic            | 0.9755       | 0.9695  | 0.79528 | 0.08584 | 0.10294  |
| accuracy_real                 | 0.97826      | 0.97826 | 0.97826 | 0.97826 | 0.97826  |
| avg_pr_macro_synthetic        | 0.9908       | 0.98684 | 0.72286 | 0.1112  | 0.10824  |
| avg_pr_macro_real             | 0.9919       | 0.9919  | 0.9919  | 0.9919  | 0.9919   |
| feature_overlap_count         | 65.6         | 61.8    | 23.6    | 13.4    | 11.6     |
| MMD_score                     | 0.00938      | 0.02117 | 0.11442 | 0.09132 | 0.91257  |
| discriminative_score          | 0.57832      | 0.81038 | 0.99989 | 0.99687 | 1.0      |
| distance_to_closest           | 28.87758     | 17.3058 | 65.4376 | 26.2715 | 104.4205 |
| distance_to_closest_base      | 23.27216     | 23.27216| 23.27216| 23.27216| 23.27216 |




