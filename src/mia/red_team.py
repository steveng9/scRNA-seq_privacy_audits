import click
import yaml
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)


#from wyzx.gan_leaks import GANLeaksModel
from mia.models.baseline import DOMIASBaselineModels


# mapping model names to their respective classes
# be consistent with config.yaml about the use of names 
mia_classes = {
    'domias_baselines': DOMIASBaselineModels
}


@click.group()
def cli():
    pass


## the outputs of the attack are always saved under
## /{home_dir}/results/mia/{dataset_name}/{attacker_name}/{generator_model}/{experiment_name}
@click.command()
@click.argument('synthetic_file', type=click.Path(exists=True))
@click.argument('mmb_test_file', type=click.Path(exists=True))
@click.argument('mia_experiment_name', type=str, default="")
@click.option('--mmb_labels_file', type=click.Path(exists=True), default=None)
@click.option('--reference_file', type=click.Path(exists=True), default=None)
def run_mia(synthetic_file:str, 
            mmb_test_file:str, 
            mia_experiment_name:str = "",
            mmb_labels_file:str = None,
            reference_file:str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    attack_model = config.get('attack_model')
    MIAClass = mia_classes.get(attack_model)

    if not MIAClass:
        raise ValueError(f"Unknown MIA model name: {attack_model}")

    mia_model = MIAClass(config, 
                         synthetic_file,
                         mmb_test_file,
                         mmb_labels_file,
                         mia_experiment_name,
                         reference_file)
    
    predictions, y_test = mia_model.run_attack()
    mia_model.save_predictions(predictions)

    if y_test is not None:
        mia_model.evaluate_attack(predictions, 
                                 y_test, 
                                "evaluation_results.csv")
    


cli.add_command(run_mia)
if __name__ == '__main__':
    cli()
