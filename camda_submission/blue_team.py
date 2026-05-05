import click
import yaml
import os
import sys
import importlib

# Ensure the directory containing this file is on sys.path so that
# ppml_generators/ is importable regardless of the working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.utils.prepare_data import RealDataLoader
from generators.models.multivariate import MultivariateDataGenerator

# ---------------------------------------------------------------------------
# Generator registry
#
# Original CAMDA generators:
#   multivariate, cvae, dpcvae, ctgan, dpctgan, sc_dist, cvae_gmm,
#   wgan_gp, private_pgm
#
# PPML-Huskies submission generators (set generator_name in config.yaml to one of):
#   scdesign2_dp  — scDesign2 with Gaussian-mechanism DP (ε=100 by default)
#   scvi          — scVI VAE (requires secondary conda env: see environment_scvi.yaml)
#   zinbwave      — ZINBWave ZINB latent-factor model
# ---------------------------------------------------------------------------
generator_classes = {
    # --- original CAMDA generators ---
    'multivariate': ('models.multivariate', 'MultivariateDataGenerator'),
    'cvae':         ('models.cvae',         'CVAEDataGenerationPipeline'),
    'dpcvae':       ('models.cvae',         'CVAEDataGenerationPipeline'),
    'ctgan':        ('models.sdv_ctgan',    'CTGANDataGenerationPipeline'),
    'dpctgan':      ('models.dpctgan',      'DPCTGANDataGenerationPipeline'),
    'sc_dist':      ('models.sc_dist',      'ScDistributionDataGenerator'),
    'cvae_gmm':     ('models.cvae_gmm',     'CVAEGMMDataGenerator'),
    'wgan_gp':      ('models.wgan_gp',      'WGANGPDataGenerator'),
    'private_pgm':  ('models.private_pgm',  'PrivatePGMDataGenerator'),

    # --- PPML-Huskies submission generators ---
    'scdesign2_dp': ('ppml_generators.scdesign2_dp', 'ScDesign2DPGenerator'),
    'scvi':         ('ppml_generators.scvi_gen',      'ScVIGenerator'),
    'zinbwave':     ('ppml_generators.zinbwave_gen',  'ZINBWaveGenerator'),
}


def get_generator_class(generator_name):
    if generator_name in generator_classes:
        module_name, class_name = generator_classes[generator_name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        raise ValueError(f"Unknown generator name: {generator_name}")


@click.group()
def cli():
    pass


@click.command()
def generate_split_indices():
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))
    rdataloader = RealDataLoader(config)
    rdataloader.save_split_indices()


@click.command()
def generate_data_splits():
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))
    rdataloader = RealDataLoader(config)
    rdataloader.save_split_data()


@click.command()
@click.argument('split_no', type=int)
@click.option('--experiment_name', type=str, default="")
def run_generator(split_no: int, experiment_name: str = None):
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


@click.command()
@click.argument('split_no', type=int)
@click.argument('subtype', type=str)
@click.argument('num_samples', type=int)
@click.option('--experiment_name', type=str, default="")
def run_pretrained_generator_for_type(split_no: int, subtype: str, num_samples: int,
                                       experiment_name: str = None):
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    generator_name = config.get('generator_name')
    GeneratorClass = get_generator_class(generator_name)

    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config, split_no=split_no)

    if not isinstance(generator, MultivariateDataGenerator):
        generator.load_from_checkpoint()

    syn_data, syn_lbl = generator.generate_for_type(subtype, num_samples)
    generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)


@click.command()
@click.option('--experiment_name', type=str, default="")
@click.option('--cfg_file', type=str, default="config.yaml")
def run_singlecell_generator(experiment_name: str = None, cfg_file: str = "config.yaml"):
    """Run a single-cell generator (train + generate) from a YAML config file.

    This is the entry point used for the PPML-Huskies submission generators:
      scdesign2_dp, scvi, zinbwave.

    Usage:
      python blue_team.py run-singlecell-generator
      python blue_team.py run-singlecell-generator --cfg_file path/to/config.yaml
    """
    config = yaml.safe_load(open(cfg_file))

    generator_name = config.get('generator_name')
    GeneratorClass = get_generator_class(generator_name)

    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config)

    if not config.get("load_from_checkpoint", False):
        if config.get("train", False):
            generator.train()
    else:
        generator.load_from_checkpoint()

    if config.get("generate", False):
        syn_data = generator.generate()
        generator.save_synthetic_anndata(syn_data, experiment_name)


cli.add_command(generate_data_splits)
cli.add_command(generate_split_indices)
cli.add_command(run_generator)
cli.add_command(run_pretrained_generator_for_type)
cli.add_command(run_singlecell_generator)

if __name__ == '__main__':
    cli()
