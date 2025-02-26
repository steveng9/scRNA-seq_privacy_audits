# :blueberries: Generate synthetic single-cell RNA-seq data 

## Prerequisites

`config.yaml` needs to be modified according to your need for each experiment. Remember to update your `home` directory accordinly. The other configurations remain as they are depending on your preferences. 

Please make sure to run the scripts from the same directory as `config.yaml` is placed. 

**NOTE:** The generation method might require up to **~125GB memory space.** We're currently working on including **memory-efficient** alternatives to **generate** and **evaluate** functions on [sparse-impl branch](https://github.com/PMBio/Health-Privacy-Challenge/tree/sparse-impl). 

### Activate the environment

```bash
micromamba activate <environment>
```


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

