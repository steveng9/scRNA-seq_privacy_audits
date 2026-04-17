#!/usr/bin/env python3

import os
import sys
import click
import yaml
import subprocess
import importlib

# Ensure parent src directory is on the import path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(src_dir)

from generators.utils.prepare_data import RealDataLoader
from generators.models.multivariate import MultivariateDataGenerator

# Map generator names to their modules and classes; special-case nmf_* pipelines
generator_classes = {
    "multivariate": ("models.multivariate", "MultivariateDataGenerator"),
    "cvae": ("models.cvae", "CVAEDataGenerationPipeline"),
    "dpcvae": ("models.cvae", "CVAEDataGenerationPipeline"),
    "ctgan": ("models.sdv_ctgan", "CTGANDataGenerationPipeline"),
    "dpctgan": ("models.dpctgan", "DPCTGANDataGenerationPipeline"),
    "sc_dist": ("models.sc_dist", "ScDistributionDataGenerator"),
    "nmf_cvae": (None, None),    # handled by external nmf_cvae.py
    "nmf_sampler": (None, None)  # handled by external nmf_sampler.py
}


def get_generator_class(name: str):
    """Dynamically import a generator class by name, except special-cased ones."""
    module_name, class_name = generator_classes.get(name, (None, None))
    if module_name and class_name:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    return None


@click.group()
def cli():
    pass


@click.command()
def generate_split_indices():
    """Create stratified split indices for real data."""
    config = yaml.safe_load(open("config.yaml"))
    rd = RealDataLoader(config)
    rd.save_split_indices()


@click.command()
def generate_data_splits():
    """Save train/test splits of the real data."""
    config = yaml.safe_load(open("config.yaml"))
    rd = RealDataLoader(config)
    rd.save_split_data()


@click.command()
@click.argument("split_no", type=int)
@click.option(
    "--experiment_name",
    default="",
    help="Optional experiment name"
)
def run_generator(split_no, experiment_name):
    """
    Train or load a generator for bulk/tabular data
    (multivariate, CTGAN, etc.) and produce synthetic samples.
    """
    config = yaml.safe_load(open("config.yaml"))
    gen_name = config.get("generator_name")

    # Handle nmf_cvae pipeline
    if gen_name == "nmf_cvae":
        home = config["dir_list"]["home"]
        train_adata = os.path.join(home, config["dataset_config"]["train_count_file"])
        test_adata = os.path.join(home, config["dataset_config"]["test_count_file"])
        output_h5 = experiment_name or config["nmf_cvae_config"]["output_h5ad"]
        script_pth = os.path.join(os.path.dirname(__file__), "models", "nmf_cvae.py")
        cmd = [
            sys.executable, script_pth,
            "--train-adata", train_adata,
            "--test-adata", test_adata,
            "--output-h5ad", output_h5
        ]
        click.echo(f"Running NMF+CVAE pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        click.echo("nmf_cvae completed.")
        return

    # Handle nmf_sampler pipeline
    if gen_name == "nmf_sampler":
        script_pth = os.path.join(os.path.dirname(__file__), "models", "nmf_sampler.py")
        click.echo(f"Running NMF sampler pipeline: {sys.executable} {script_pth}")
        subprocess.run([sys.executable, script_pth], check=True)
        click.echo(f"Running NMF sampler pipeline: {sys.executable} {script_pth} --dp {dp}")
        subprocess.run([sys.executable, script_pth, "--dp", dp], check=True)
        click.echo("nmf_sampler completed.")
        return
    # Otherwise, dynamic import of internal generator
    GeneratorClass = get_generator_class(gen_name)
    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {gen_name}")

    generator = GeneratorClass(config, split_no=split_no)

    # Train or load checkpoint
    if not config.get("load_from_checkpoint", False) and config.get("train", False):
        generator.train()
    else:
        generator.load_from_checkpoint()

    # Generate synthetic data
    if config.get("generate", True):
        syn_data, syn_lbl = generator.generate()
        generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)

    click.echo("run_generator completed.")


@click.command(name="run-singlecell-generator")
@click.option(
    "--experiment_name",
    default="",
    help="Optional experiment name"
)
@click.option(
    "--dp",
    type=click.Choice(["all", "nmf", "kmeans", "sampling", "none"]),
    default="all",
    help="Which DP steps to apply (for nmf_sampler)"
)
def run_singlecell_generator(experiment_name, dp):
    """
    Train or load a single-cell generator and save synthetic AnnData.
    Supports sc_dist, nmf_cvae, and nmf_sampler pipelines.
    """
    config = yaml.safe_load(open("config.yaml"))
    gen_name = config.get("generator_name")

    # nmf_cvae special case
    if gen_name == "nmf_cvae":
        home = config["dir_list"]["home"]
        train_adata = os.path.join(home, config["dataset_config"]["train_count_file"])
        test_adata = os.path.join(home, config["dataset_config"]["test_count_file"])
        output_h5 = experiment_name or config["nmf_cvae_config"]["output_h5ad"]
        script_pth = os.path.join(os.path.dirname(__file__), "models", "nmf_cvae.py")
        cmd = [
            sys.executable, script_pth,
            "--train-adata", train_adata,
            "--test-adata", test_adata,
            "--output-h5ad", output_h5
        ]
        click.echo(f"Running NMF+CVAE pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        click.echo("nmf_cvae completed.")
        return

    # Handle nmf_sampler pipeline
    if gen_name == "nmf_sampler":
        script_pth = os.path.join(
            os.path.dirname(__file__),
            "models",
            "nmf_sampler.py"
        )
        cmd = [
            sys.executable,
            script_pth,
            "--dp", dp,
            "--experiment_name", experiment_name
        ]
        click.echo(f"Running NMF sampler pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        click.echo("nmf_sampler completed.")
        return
    
    # All other single-cell generators
    GeneratorClass = get_generator_class(gen_name)
    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {gen_name}")

    generator = GeneratorClass(config)

    # Train or load checkpoint
    if not config.get("load_from_checkpoint", False) and config.get("train", False):
        generator.train()
    else:
        generator.load_from_checkpoint()

    # Generate and save AnnData
    if config.get("generate", True):
        syn_adata = generator.generate()
        generator.save_synthetic_anndata(syn_adata, experiment_name)

    click.echo("run-singlecell-generator completed.")


# Register commands
cli.add_command(generate_split_indices)
cli.add_command(generate_data_splits)
cli.add_command(run_generator)
cli.add_command(run_singlecell_generator)

if __name__ == "__main__":
    cli()
