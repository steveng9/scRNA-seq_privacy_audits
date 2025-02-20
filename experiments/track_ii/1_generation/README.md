# :blueberries: Generate synthetic single-cell RNA-seq data 

## Prerequisites

`config.yaml` needs to be modified according to your need for each experiment. Remember to update `home_dir`. The other configurations remain as they are depending on your preferences. 

### Activate the environment

```bash
micromamba activate <environment>
```

## OneK1K dataset

We re-distribute raw counts of OneK1K single-cell RNA-seq dataset (https://onek1k.org/), a cohort containing 1.26 million peripheral blood mononuclear cells (PBMCs) of 981 donors. 

After the following filtering, 
```python 
    sc.pp.filter_genes(adata, min_cells=3) 
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.total_counts > 10,:]
    adata = adata[adata.obs.total_counts < 40000,:]
```

The dataset is split into donor-based train and test sets of relatively equal numbers of cells, and similar cell-type distributions.

- Train dataset: < 633711 cells x 490 donors > 
- Test dataset:  < 634022 cells x 491 donors > 

We share these datasets in annData format with the following annotations: `individual`, `barcode`, `cell_type`, `cell_label`. 


## Generate synthetic sc-RNAseq data 
The below code generate synthetic data same size as `test` dataset using the statistics learned on `train`. We provide `Poisson` distribution as a baseline method.  `Negative Binomial` is also available as `NB`, however evaluation metrics, especially UMAP indicates that the synthetic cells don't integrate well with real data. 

In order to run these methods, you first need to modify inside the ``config.yaml``. Both distributions are part of the generator named `sc_dist`. In the provided logic, each generator model has their own configuration arguments. e.g. ``{generator_name}_config``. 

 ### Example: Poisson

```bash
generator_name: "sc_dist"
generate: True
train: True
load_from_checkpoint: False


sc_dist_config:
  noise_level: 0.5 # noise level exists, but not used in generation (provided as an example in case you want to use)
  random_seed: 42
  distribution: "Poisson" # NB is available, but do not perform well

```

In order to generate synthetic data with this setting, simply run the below script. 

```bash
 python {src_dir}/generators/blue_team.py run-singlecell-generator  --experiment_name {experiment_name}
```

`experiment_name` can be, e.g. `distr_Poisson`.  This will generate synthetic data  under data_splits directory,

e.g. ``data_splits/{onek1k}/synthetic/sc_dist/distr_Poisson/onek1k_annotated_synthetic.h5ad``

