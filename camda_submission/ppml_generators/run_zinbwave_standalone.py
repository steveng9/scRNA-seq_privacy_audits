"""
ZINBWave standalone runner.

Fits a per-cell-type ZINBWave model (Risso et al. 2018) on training scRNA-seq
data and generates synthetic counts by bootstrapping from fitted ZINB parameters.

Algorithm
---------
1. Split training data by cell type; write per-cell-type h5ad files.
2. For each cell type, call Rscript zinbwave.r train ... to fit the model.
3. For each cell type, call Rscript zinbwave.r generate ... to sample counts.
4. Assemble all cell types into a single synthetic AnnData and write to output.

Model (ZINBWave, Risso et al. 2018)
-------------------------------------
  Y_ig ~ ZINB(μ_ig, φ_g, π_ig)
  log(μ) = X β_μ + W γ_μ + V α_μ + offset
  logit(π) = X β_π + W γ_π + V α_π

Generation is via bootstrap: resample training-cell fitted parameters (Mu, Pi,
Phi) and draw new ZINB counts.  The latent W factor is implicit — the fitted Mu
and Pi already encode the per-cell structure learned from W.

Defaults
--------
  K (n_latent)        = 10   latent dimensions
  max_cells_per_type  = 3000 cells used for fitting (subsampled if exceeded)
  n_workers           = 4    parallel R subprocesses

Usage
-----
  python run_zinbwave_standalone.py \\
      --train-h5ad    /path/to/train.h5ad \\
      --output-h5ad   /path/to/synthetic.h5ad \\
      --model-dir     /path/to/models \\
      --n-latent      10 \\
      --cell-type-col cell_type
"""

import argparse
import os
import sys
import subprocess
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

# Absolute path to the companion R script
_R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zinbwave.r")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ct_name(cell_type: str) -> str:
    """File-system-safe cell type name for use in RDS filenames."""
    return cell_type.replace(" ", "_").replace("/", "_").replace(":", "_")


def _run_r(cmd: str, label: str = "R") -> None:
    """Run an R command via shell.  Raises RuntimeError on failure."""
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"    {line}", flush=True)
    if result.returncode != 0:
        stderr_tail = result.stderr[-800:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"R command failed [{label}]:\n{stderr_tail}"
        )


# ---------------------------------------------------------------------------
# Per-cell-type helpers (module-level → picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _train_one(ct_h5ad: str, n_latent: int, model_dir: str, cell_type: str) -> str:
    safe = _safe_ct_name(cell_type)
    out_rds = os.path.join(model_dir, f"{safe}.rds")
    if os.path.exists(out_rds):
        print(f"  [SKIP] Model already exists for '{cell_type}'", flush=True)
        return cell_type
    cmd = f"Rscript {_R_SCRIPT} train {ct_h5ad} {n_latent} {out_rds}"
    _run_r(cmd, f"train:{cell_type}")
    return cell_type


def _generate_one(n_cells: int, model_dir: str, tmp_dir: str,
                  cell_type: str, seed: int) -> str:
    safe = _safe_ct_name(cell_type)
    model_rds = os.path.join(model_dir, f"{safe}.rds")
    out_rds   = os.path.join(tmp_dir,   f"synth_{safe}.rds")
    if not os.path.exists(model_rds):
        raise FileNotFoundError(f"Model RDS not found: {model_rds}")
    if os.path.exists(out_rds):
        return cell_type
    cmd = f"Rscript {_R_SCRIPT} generate {n_cells} {model_rds} {out_rds} {seed}"
    _run_r(cmd, f"gen:{cell_type}")
    return cell_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ZINBWave synthetic data generator (train + generate)"
    )
    ap.add_argument("--train-h5ad",        required=True,
                    help="HVG-filtered training AnnData (.h5ad)")
    ap.add_argument("--output-h5ad",       required=True,
                    help="Output path for synthetic AnnData (.h5ad)")
    ap.add_argument("--model-dir",         required=True,
                    help="Directory to save per-cell-type zinbwave model RDS files")
    ap.add_argument("--n-latent",          type=int, default=10,
                    help="Number of ZINBWave latent factors K (default: 10)")
    ap.add_argument("--max-cells-per-type", type=int, default=3000,
                    help="Cap on training cells per cell type (subsampled if exceeded; default: 3000)")
    ap.add_argument("--cell-type-col",     default="cell_type",
                    help="obs column with cell-type labels (default: cell_type)")
    ap.add_argument("--n-workers",         type=int, default=4,
                    help="Parallel R subprocesses (default: 4)")
    ap.add_argument("--seed",              type=int, default=42,
                    help="Random seed for generation (default: 42)")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(args.output_h5ad))
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.output_h5ad):
        print(f"[SKIP] synthetic.h5ad already exists: {args.output_h5ad}", flush=True)
        return

    # -----------------------------------------------------------------------
    # 1. Load training data
    # -----------------------------------------------------------------------
    print(f"Loading training data from {args.train_h5ad}", flush=True)
    adata = ad.read_h5ad(args.train_h5ad)
    n_obs, n_vars = adata.n_obs, adata.n_vars
    gene_names = adata.var_names.tolist()
    cell_types_arr = adata.obs[args.cell_type_col].values
    ct_counts = Counter(cell_types_arr)
    print(f"  {n_obs:,} cells × {n_vars:,} genes, {len(ct_counts)} cell types", flush=True)

    # -----------------------------------------------------------------------
    # 2. Write per-cell-type h5ads (temporary) for R subprocess
    # -----------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="zinbwave_tmp_",
                               dir=os.path.dirname(args.output_h5ad))
    ct_h5ad_paths: dict[str, str] = {}

    print("Writing per-cell-type h5ad files for R...", flush=True)
    for ct, n_ct in ct_counts.items():
        safe = _safe_ct_name(ct)
        ct_h5ad = os.path.join(tmp_dir, f"train_{safe}.h5ad")
        ct_h5ad_paths[ct] = ct_h5ad
        if os.path.exists(ct_h5ad):
            continue
        mask = cell_types_arr == ct
        sub = adata[mask].copy()
        # Subsample if over max_cells_per_type
        if sub.n_obs > args.max_cells_per_type:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(sub.n_obs, size=args.max_cells_per_type, replace=False)
            sub = sub[idx].copy()
            print(f"  Subsampled '{ct}': {n_ct} → {args.max_cells_per_type} cells", flush=True)
        sub.write_h5ad(ct_h5ad)

    del adata  # free memory before forking

    # -----------------------------------------------------------------------
    # 3. Train ZINBWave per cell type (parallel R subprocesses)
    # -----------------------------------------------------------------------
    print(f"\nTraining ZINBWave ({len(ct_counts)} cell types, {args.n_workers} workers)...",
          flush=True)
    failed_train = set()
    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futures = {
            exe.submit(
                _train_one,
                ct_h5ad_paths[ct], args.n_latent, args.model_dir, ct
            ): ct
            for ct in ct_counts
        }
        for f in as_completed(futures):
            ct = futures[f]
            try:
                f.result()
                print(f"  Trained: {ct}", flush=True)
            except Exception as e:
                print(f"  [WARN] Training failed for '{ct}': {e}", flush=True)
                failed_train.add(ct)

    # -----------------------------------------------------------------------
    # 4. Generate synthetic counts per cell type (parallel R subprocesses)
    # -----------------------------------------------------------------------
    print(f"\nGenerating synthetic counts ({len(ct_counts)} cell types)...", flush=True)
    failed_gen = set()
    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futures = {
            exe.submit(
                _generate_one,
                ct_counts[ct], args.model_dir, tmp_dir, ct, args.seed
            ): ct
            for ct in ct_counts if ct not in failed_train
        }
        for f in as_completed(futures):
            ct = futures[f]
            try:
                f.result()
                print(f"  Generated: {ct}", flush=True)
            except Exception as e:
                print(f"  [WARN] Generation failed for '{ct}': {e}", flush=True)
                failed_gen.add(ct)

    # -----------------------------------------------------------------------
    # 5. Assemble synthetic AnnData
    # -----------------------------------------------------------------------
    import pyreadr

    print("\nAssembling synthetic AnnData...", flush=True)
    all_counts = []
    all_obs_ct = []

    for ct in ct_counts:
        if ct in failed_train or ct in failed_gen:
            print(f"  [SKIP] '{ct}' had errors — omitting from output", flush=True)
            continue
        safe = _safe_ct_name(ct)
        out_rds = os.path.join(tmp_dir, f"synth_{safe}.rds")
        if not os.path.exists(out_rds):
            print(f"  [SKIP] Missing synth RDS for '{ct}'", flush=True)
            continue
        mat_dict = pyreadr.read_r(out_rds)
        mat = list(mat_dict.values())[0]
        # mat is a DataFrame: n_cells rows × modeled_genes cols
        # colnames = subset of gene_names (all-zero genes were dropped before fitting)
        counts_raw = mat.to_numpy() if hasattr(mat, "to_numpy") else np.array(mat)
        modeled_genes = list(mat.columns)
        n_new = counts_raw.shape[0]

        # Reconstruct full-gene matrix (zeros for dropped all-zero genes)
        full_row = np.zeros((n_new, n_vars), dtype=np.float32)
        gene_idx = [gene_names.index(g) for g in modeled_genes if g in gene_names]
        for col_i, g_i in enumerate(gene_idx):
            full_row[:, g_i] = counts_raw[:, col_i]

        all_counts.append(full_row)
        all_obs_ct.extend([ct] * n_new)

    if not all_counts:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("No synthetic data was generated — all cell types failed.")

    synth_X = np.vstack(all_counts)          # (n_total_cells, n_genes)
    synth_adata = ad.AnnData(
        X=sp.csr_matrix(synth_X),
        var=pd.DataFrame(index=gene_names),
    )
    synth_adata.obs[args.cell_type_col] = all_obs_ct
    synth_adata.obs_names = [f"cell_{i}" for i in range(synth_adata.n_obs)]

    synth_adata.write_h5ad(args.output_h5ad, compression="gzip")
    print(f"\nSynthetic data saved: {synth_adata.n_obs:,} cells × {synth_adata.n_vars:,} genes")
    print(f"  → {args.output_h5ad}", flush=True)

    # Clean up temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
