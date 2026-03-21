#!/usr/bin/env bash
# Create the scdiff_ conda environment for scDiffusion (Luo et al. 2024).
#
# Usage: bash experiments/scdiffusion/setup_scdiffusion_env.sh
#
# Requires: /home/golobs/scDiffusion (already cloned from EperLuo/scDiffusion)

set -euo pipefail

ENV="scdiff_"
SCDIFF_ROOT="/home/golobs/scDiffusion"

echo "==> Creating conda env: ${ENV}"
conda create -y -n "${ENV}" python=3.9

echo "==> Installing dependencies"
# torch 1.13.1 + CUDA 11.7 (pinned by scDiffusion)
conda run -n "${ENV}" pip install \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

conda install -y -n "${ENV}" -c conda-forge mpi4py

conda run -n "${ENV}" pip install \
    "numpy==1.23.4" \
    "anndata>=0.9,<0.11" \
    "scanpy>=1.9,<1.10" \
    "scikit-learn==1.2.2" \
    "pandas==1.5.3" \
    "blobfile==2.0.0" \
    "scipy>=1.9" \
    "tqdm"

echo "==> Verifying"
conda run -n "${ENV}" python -c "
import torch, anndata, scanpy, blobfile, mpi4py
print(f'torch       : {torch.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
print(f'anndata     : {anndata.__version__}')
print(f'scanpy      : {scanpy.__version__}')
print('OK')
"

echo "Done."
