#  :blueberries: BLUE TEAM: Dataset Split Generation

You need to modify `config.yaml` according to your need for each experiment. Remember to update `home_dir`. 

Make sure to update `subtype_col_name` argument in the `config.yaml` based on the dataset: 
- For TCGA-BRCA: `subtype_col_name:` "Subtype"
- For TCGA-COMBINED:  `subtype_col_name:` "cancer_type"

e.g. 
```bash
dataset_config:
  name: "TCGA-BRCA" #"TCGA-COMBINED" # 
  count_file: "data/processed/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv" #"TCGA-COMBINED" # 
  annot_file: "data/meta/TCGA-BRCA_primary_tumor_subtypes.csv" #"TCGA-COMBINED" # 
  sample_col_name: "samplesID" 
  subtype_col_name: "Subtype" #"cancer_type"  "Subtype"
  num_splits: 5
  random_seed: 42
```



## Activate the environment

```bash
micromamba activate <environment>
```

## 1. Generate Split Indices
- We adopt a 5-fold cross-validation (CV) design, which involves dividing the data into 5 distinct splits. This is due to relatively small size of the datasets. 

- In each split, the data is partitioned into a training set and a test set, with the training set comprising the remaining 4 folds. The test set is kept separate from the training process and is used exclusively during evaluation. The goal is to generate synthetic datasets corresponding to each training split.

- You can use the provided ``split_indices/{dataset_name}_splits.yaml.`` for evaluating your models against the provided baselines which are created with `random_seed=42` inside the `config.yaml` file. 

- Please use `random_seed=42` **only** to compare your method against baseline results. 

- :exclamation:  You must assign a new value to the `random_seed` for your submission and provide us your final `config.yaml` with the updated `random_seed` consistent in `dataset_config` and `{generator_name}_config`.
    -  `{dataset_name}_splits.yaml` files musy be generated using this new value. 
    - **This means that each blue team has different real and synthetic datasets based on the split using this value.** 

- The below script will generate the indices and save them under  ``split_indices/{dataset_name}_splits.yaml.``. Replace `{src_dir}` with your corresponding path. 

```bash
python {src_dir}/generators/blue_team.py generate-split-indices
```


## 2. Generate Data Splits

Once the split indices are generated, you can generate the dataset splits using those. 

```bash
python <src_dir>/generators/blue_team.py generate-data-splits
```

This will create the data splits ``data_splits/{dataset_name}/real/X_train_real_split_{split_no}.csv`` based on the default configurations ``config.yaml``.
