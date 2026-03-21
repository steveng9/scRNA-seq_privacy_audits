#!/usr/bin/env bash
# Create a dedicated conda environment for scVI experiments.
#
# Usage:
#   bash experiments/scvi/setup_scvi_env.sh
#
# What this creates
# -----------------
#   conda env "scvi_" with:
#     - Python 3.10
#     - scvi-tools 1.1.x (PyTorch + JAX backend)
#     - anndata, scanpy, pandas, numpy
#
# After running this, all scVI training / scoring is dispatched
# into this env via  conda run -n scvi_  inside model.py.

set -euo pipefail

ENV_NAME="scvi_"

echo "==> Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.10

echo "==> Installing scvi-tools and dependencies"
conda run -n "${ENV_NAME}" pip install \
    "scvi-tools>=1.1,<1.3" \
    "anndata>=0.10,<0.11" \
    "scanpy>=1.9" \
    "torch>=2.0" \
    "lightning>=2.0" \
    "numpy<2" \
    "pandas>=1.5" \
    "scipy>=1.9" \
    "scikit-learn>=1.1" \
    "h5py>=3.0" \
    "tqdm"

echo "==> Verifying installation"
conda run -n "${ENV_NAME}" python -c "
import scvi, anndata, scanpy, torch
print(f'scvi-tools  : {scvi.__version__}')
print(f'anndata     : {anndata.__version__}')
print(f'scanpy      : {scanpy.__version__}')
print(f'torch       : {torch.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
print('OK')
"

echo "Done.  Use:  conda activate ${ENV_NAME}"
