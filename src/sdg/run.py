"""
SDG runner — instantiates and executes any registered Synthetic Data Generator.

Usage (from run_experiment.py or directly):
    from sdg.run import run_singlecell_sdg, format_cell_type_name
"""

import os
import sys
import importlib
import yaml
import click

# Ensure src/ is on path so sibling packages resolve
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------
# Map config name → (module_path, class_name) relative to src/
# Add new SDGs here.

GENERATOR_CLASSES = {
    "scdesign2": ("sdg.scdesign2.model", "ScDesign2"),
    # "scdesign3": ("sdg.scdesign3.model", "ScDesign3"),   # not yet implemented
    # "scvae":     ("sdg.scvae.model",     "ScVAE"),        # not yet implemented
}


def get_generator_class(generator_name: str):
    if generator_name not in GENERATOR_CLASSES:
        raise ValueError(
            f"Unknown generator: '{generator_name}'. "
            f"Available: {list(GENERATOR_CLASSES.keys())}"
        )
    module_name, class_name = GENERATOR_CLASSES[generator_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def format_cell_type_name(cell_name: str) -> str:
    """Sanitise a cell-type string for use as a filename."""
    return str(cell_name).replace(" ", "_")


# ---------------------------------------------------------------------------
# CLI commands (kept for standalone / legacy use)
# ---------------------------------------------------------------------------

@click.group()
def cli():
    pass


@click.command()
@click.option("--cfg_file", type=str, default="config.yaml")
@click.option("--experiment_name", type=str, default="")
def run_singlecell_sdg(cfg_file: str = "config.yaml", experiment_name: str = ""):
    """Train and/or generate synthetic scRNA-seq data for one config file."""
    config = yaml.safe_load(open(cfg_file))
    generator_name = config.get("generator_name")
    GeneratorClass = get_generator_class(generator_name)

    generator = GeneratorClass(config)

    if not config.get("load_from_checkpoint", False):
        if config.get("train", False):
            print("Training model...")
            generator.train()
        else:
            print("Skipping training.")
    else:
        print("Loading from checkpoint...")
        generator.load_from_checkpoint()

    if config.get("generate", False):
        syn_data = generator.generate()
        synthetic_path = os.path.join(
            config["dir_list"]["data"], "synthetic.h5ad"
        )
        generator.save_synthetic_anndata(
            syn_data, experiment_name, synthetic_path=synthetic_path
        )


# Alias kept for backward compatibility with any scripts that imported
# run_singlecell_generator from blue_team.
run_singlecell_generator = run_singlecell_sdg


@click.command()
def generate_split_indices():
    from sdg.utils.prepare_data import RealDataLoader
    config = yaml.safe_load(open("config.yaml"))
    RealDataLoader(config).save_split_indices()


@click.command()
def generate_data_splits():
    from sdg.utils.prepare_data import RealDataLoader
    config = yaml.safe_load(open("config.yaml"))
    RealDataLoader(config).save_split_data()


cli.add_command(run_singlecell_sdg)
cli.add_command(generate_split_indices)
cli.add_command(generate_data_splits)

if __name__ == "__main__":
    cli()
