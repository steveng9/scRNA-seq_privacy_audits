# :blueberries: BLUE TEAM 

Welcome to your home page! 

**Contents**
- [Tasks](#tasks)
- [Guideline for running and evaluating baseline methods](#thread-workflow-for-running-and-evaluating-baseline-methods)
- [Guideline for developing your own method](#bookmark_tabs-guideline-for-developing-your-own-method)
- [Submission checklist](#white_check_mark-submission-checklist)

## Tasks 
Blue teams participate in **Phase 1**, and work towards developing methods that improve the baseline methods and generating novel insights into privacy preservation in biological datasets.

 

## :thread: Guideline for running and evaluating baseline methods

Please follow the three main steps below:
1. **[Create cross-fold dataset splits](/experiments/blue_team/1_data/)**: Prepares the data in stratified 5 fold train-test splits. 
2. **[Generate synthetic data](/experiments/blue_team/2_generation/)**: Generates synthetic data using baseline methods on each data split generated in the previous step. 
3. **[Evaluate](/experiments/blue_team/3_evaluation/)**: Reports the evaluation metrics for each data splits (and in average).


## :bookmark_tabs: Guideline for developing your own method

We provide `BaseDataGenerator` class inside [generators/models/base.py](/src/generators/models/base.py) for generator models to inherit. This class handles the configuration arguments passed through `config.yaml`. 

It also has three abstract functions:  
- `train()`, 
- `load_from_checkpoint()`, 
- `generate()`, 
-  and `save_synthetic_data()`, a fixed function that implements saving the synthetic datasets in the desired format. 

We expect your submitted code to follow the logic in [blue_team.py](/src/generators/blue_team.py). Your generate() function should return synthetic data and synthetic labels.

```bash
@click.command()
@click.argument('split_no', type=int)
@click.option('--experiment_name', type=str, default="")
def run_generator(split_no: int, experiment_name: str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    generator_name = config.get('generator_name')
    GeneratorClass = get_generator_class(generator_name)

    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config, split_no=split_no)

    if not isinstance(generator, MultivariateDataGenerator):
        if not config.get("load_from_checkpoint", False):
            if config.get("train", False):
                generator.train()
        else:
            generator.load_from_checkpoint()

    if config.get("generate", True):
        syn_data, syn_lbl = generator.generate()
        generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)
```
Please test your code inside `run_generator` function to ensure its reproducibility. 


## :white_check_mark: Submission checklist
The following files are required for submission, for each dataset:

1. Modified [blue_team.py](/src/generators/blue_team.py) with your generator class included, and any additional `.py` file required to run your method. 
2.  Five sets of synthetic data and labels saved with `save_synthetic_data()` functionality
3.  Updated `config.yaml` file containing `{your_generator}_config` and `random_seed` with a new value inside `dataset_config` and `{your_generator}_config`. 
    - Please note that a new value for random seed means that: **each blue team has different real and synthetic datasets based on the split using this value.** 
4. `{dataset_name}_split.yaml` generated with the new `random_seed` value inside dataset_config.
5. `environment.yaml` that is required to run your method.


This means we expect two zip files from you in the following file name format,
1.  `blueteam_{yourteamname}_TCGA-BRCA.zip`
2.  `blueteam_{yourteamname}_TCGA-COMBINED.zip`


Please take look at the [example submission zip file](/experiments/blue_team/blueteam_example_TCGA-BRCA.zip) to double check correct formatting. 


**Best of luck!** :four_leaf_clover:





