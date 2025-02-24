# :blueberries: Evaluate synthetic scRNA-seq data

Evaluation workflow:

1. [**Prerequisites**](#prerequisites): Preparations evaluation workflow setup.
2. [**Statistical methods**](#statistical-methods): Run statistical evaluation scripts.
3. [**Classification-based methods**](#classification-based-methods): Run non-statistical evaluation scripts. 
4. [**UMAP**](#umap): UMAP-based visualization. 
5. [**Evaluation metrics**](#evaluation-metrics): Evaluation metrics. 
6. [**Baseline evaluation results**](#baseline-results): Performance of the baseline generator method. 

---

## Prerequisites

### **Activate the python environment**: 

```bash
micromamba activate <environment>
```

<!--### **Ensure the following directories exist**:
The other configurations can stay depending on your preferences. 
   - `res_files`: Directory for storing evaluation metric scores in a CSV file.
   - `figures`: Directory for saving UMAP plots.
  - `mia_files`: Directory to store membership inference attack scores. Please refer to [Red Team homepage](/experiments/track_i/red_team/README.md) if you want to test this out. -->
 

### **Configuration Variables**:
Remember to update `home` directory. You need to modify `config.yaml` according to your need for each experiment. Define the following variables according to your setup in `config.yaml`:
   - `dataset_config`: Update the directories of the downloaded datasets. Always assume that the final path is joined with `home_dir`. 
   - `generator_config`: Update the name of the method and experiment name you want to generate evaluation scores. 

**NOTE**: Please be reminded that you need to put `config.yaml` in the same directory you are running your experiment. 

### **Evaluation variables**:

  - We use Celltypist as a part of our evaluation metrics. We used the *Immune_All_High* model from Celltypist as an example to infer, therefore please download it from here. Make sure the path to the saved model reflects your directory path in `celltypist_model` inside  `config.yaml`. 
  - Fell free  to experiment with other existing [Celltypist models](https://www.celltypist.org/models). 


### Example config for evaluating Poisson generated synthetic data

e.g. In the Generation step, we set the `--experiment_name` argument as `distr_Poisson`, therefore we will use the same `experiment_name` in `generator_config`.  

```bash
dataset_config:
  name: "onek1k"
  train_count_file: "data/processed/onek1k/onek1k_annotated_train.h5ad" 
  test_count_file: "data/processed/onek1k/onek1k_annotated_test.h5ad" 
  cell_type_col_name: "cell_type" 
  cell_label_col_name: "cell_label"
  celltypist_model: "data/meta/downloads/Immune_All_High.pkl"
  random_seed: 42

generator_config:
  name: "sc_dist"
  experiment_name: "distr_Poisson"
    

evaluator_config:   
  random_seed: 42
```

Feel free to update celltypist based evaluation with other existing models here, https://www.celltypist.org/models 

## Statistical methods
Refer to [Evaluation Metrics](#evaluation-metrics) for the list of evaluation metrics generated. 

Make sure to replace `<src_dir>`  with your corresponding path, then run the following command:

```bash
python {src_dir}/evaluation/sc_evaluate.py run-statistical-eval     
```

The results are saved to: ``{home_dir}/{res_files}/{dataset_name}/{generator_name}/{experiment_name}/statistics_evals.csv``


## Classification-based methods

```bash
python {src_dir}/evaluation/sc_evaluate.py run-classification-eval   
```

The results are saved under the following directory: `{home_dir}/{res_files}/{dataset_name}/{generator_name}/{experiment_name}/classification_evals.csv`. 


## UMAP

Generate a UMAP visualization using the following command, where both real and synthetic data points are mapped into 2D together. 

`experiment_name` is the location where the UMAP figure is saved, 
`n_hvgs` is the number of Highly Variable Genes (HVG)

```bash
python {src_dir}/evaluation/sc_evaluate.py run-umap-eval {n_hvgs}
```

This command generates a UMAP saved under the directory:
``{home_dir}/{figures}/{dataset_name}/{generator_name}/{experiment_name}/umap....HVG={n_hvgs}.png``



## Evaluation metrics 

To assess the similarity between synthetic and real cells, we evaluated the generated synthetic data using several metrics adopted from [`ScDiffusion`](https://github.com/EperLuo/scDiffusion/tree/main). 

For computational efficiency, we used **Highly Variable Genes (HVG)** for the analyses. We provide these metrics to help kickstart your analysis and **strongly encourage you to modify these according to your needs, and to explore additional or novel metrics that could offer deeper insights.** 

### Statistical Indicators
- **Spearman Correlation Coefficient (SCC)** measures how well the gene rankings correlate between real and synthetic datasets, focusing on the most highly variable genes (HVGs). 
  - **Higher SCC (1.0)** indicates high correlation between synthetic and real. 
- **Maximum Mean Discrepancy (MMD)** (Gretton et al., 2012) measures distributional similarity. 
  - **Lower MMD (~0.0)** indicates higher similarity between synthetic and real. 
  - Here we used subsampling for computational efficiency. You can modify the size of sampling or introduce a batch based computation for more reliable MMD score. 
- **Local Inverse Simpson’s Index (LISI)** (Korsunsky et al., 2019) measures how well real and synthetic cells mix together in a shared latent space. 
  - **Higher LISI** indicates good integration. 
- **Adjusted Rand Index (ARI)** measures how well real and synthetic cells cluster together. 
  - ARI (real vs. synthetic clusters) measures how well synthetic cells cluster similarly to real cells.
  - ARI (ground truth vs. combined clusters) checks whether synthetic data maintains biological structure.
  - **Higher ARI indicates (1.0)** synthetic and real cells form similar clusters. 


### Classification and visualization based metrics 
- **Uniform Manifold Approximation and Projection (UMAP)** (McInnes et al., 2018) visualizes the structure of the synthetic and real cells in 2D. 
- **CellTypist classification** (Dominguez Conde et al., 2022) measures whether cell type identity is retrained. 
  - **High ARI & Jaccard** indicates synthetic cells match real cells in terms of cell-type classification.
- **Random forest evaluation** measures whether synthetic and real cells can be distinguished. 
  - **Low AUC (close to 0.5)** indicates good quality of synthetic cells. 



## Baseline results

Default values in [config.yaml](/experiments/track_ii/1_generation/config.yaml) are used. The scores are reported for the synthetic dataset and test dataset. 


| Metric / Method            | Poisson      | 
|----------------------------|--------------|
| celltypist_ari             | 0.7196       | 
| celltypist_jaccard         | 0.2167       | 
| randomforest_roc           | 1.0          | 
| SCC                        | 0.0185       | 
| MMD                        | 0.0001       | 
| LISI                       | 0.6254       | 
| ARI_real_vs_syn            | 0.3949       | 
| ARI_gt_vs_comb             | 0.1012       | 






