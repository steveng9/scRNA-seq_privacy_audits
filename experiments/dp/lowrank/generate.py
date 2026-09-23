"""
generate.py — generate synthetic data from lowrank copulas.

Three modes:
  --no-dp             Rank-r reduction only, no noise (project_lowrank_no_dp).
                       Answers "how much does rank truncation alone cost?"
                       Output: lowrank_r{R}_no_dp/{nd}d/{trial}/datasets/synthetic.h5ad
  --epsilon E [E ...]  apply_gaussian_dp_lowrank: rank-r reduction + Gaussian
                       noise calibrated to the r-dependent sensitivity bound.
                       Output: lowrank_r{R}_eps_{E}/{nd}d/{trial}/datasets/synthetic.h5ad

`--rank` is required in both modes. All (rank, epsilon) combinations read
from the SAME lowrank_no_dp/{nd}d/{trial}/models/ (trained once by train.py;
training does not depend on rank or epsilon).

Mirrors experiments/dp/v2/generate.py's structure closely (isolated,
independent copy per the v1/v2/lowrank isolation contract -- does not import
from or modify the v2 pipeline).
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sdg.dp.sensitivity import TRUE_CLIP_VALUE

DATA_ROOT   = "/home/golobs/data/scMAMAMIA"
R_SCRIPT_GEN = os.path.join(REPO_ROOT, "src", "sdg", "scdesign2", "scdesign2.r")  # gen mode unchanged from v1, same as v2/generate.py reuses

DELTA       = 1e-5
CLIP_VALUE  = TRUE_CLIP_VALUE
DONOR_COL     = "individual"
CELL_TYPE_COL = "cell_type"


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_corrected_rds(copula_rds, cell_type, corrected_cov_np, out_path):
    """Patch cov_mat on the in-memory R copula object and saveRDS to out_path."""
    from rpy2.robjects import r as R
    from rpy2.robjects.vectors import FloatVector
    G = corrected_cov_np.shape[0]
    flat = corrected_cov_np.flatten(order="F").tolist()
    R.assign("lr_copula_obj", copula_rds)
    R.assign("lr_noised_flat", FloatVector(flat))
    R(f'lr_copula_obj[["{cell_type}"]][["cov_mat"]] <- '
      f'matrix(lr_noised_flat, nrow={G}, ncol={G})')
    R(f'saveRDS(lr_copula_obj, file="{out_path}")')


def _run_r_gen(copula_path, n_cells, out_rds_path):
    cmd = ["Rscript", R_SCRIPT_GEN, "gen", str(int(n_cells)), copula_path, out_rds_path]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        out = e.output.decode("utf-8", errors="replace")
        print(f"    [WARN] R gen failed: {out[-300:]}")
        return False


def _get_k_max(full_obs, train_donors, cell_type):
    mask = full_obs[DONOR_COL].isin(train_donors) & (full_obs[CELL_TYPE_COL] == cell_type)
    counts = full_obs[mask].groupby(DONOR_COL).size()
    return int(counts.max()) if len(counts) > 0 else 1


def _get_n_cells_from_copula(copula_rds, cell_type):
    return int(copula_rds.rx2(str(cell_type)).rx2("n_cell")[0])


def generate_one(dataset, nd, trial, rank, epsilon, seed=42):
    """Generate lowrank synthetic for one (dataset, nd, trial, rank, epsilon|None) combo."""
    from rpy2.robjects import r as R
    from sdg.scdesign2.copula import parse_copula
    from sdg.dp.lowrank import apply_gaussian_dp_lowrank, project_lowrank_no_dp
    from sdg.dp.sensitivity import gaussian_noise_scale

    base = os.path.join(DATA_ROOT, dataset)
    full_h5ad   = os.path.join(base, "full_dataset_cleaned.h5ad")
    hvg_path    = os.path.join(base, "hvg.csv")
    splits_dir  = os.path.join(base, "splits", f"{nd}d", str(trial))
    train_npy   = os.path.join(splits_dir, "train.npy")

    models_dir  = os.path.join(base, "scdesign2", "lowrank_no_dp", f"{nd}d", str(trial), "models")

    is_dp = epsilon is not None
    variant_tag = f"lowrank_r{rank}_eps_{int(epsilon)}" if is_dp else f"lowrank_r{rank}_no_dp"
    out_root      = os.path.join(base, "scdesign2", variant_tag, f"{nd}d", str(trial))
    out_datasets  = os.path.join(out_root, "datasets")
    os.makedirs(out_datasets, exist_ok=True)
    synth_path    = os.path.join(out_datasets, "synthetic.h5ad")

    if os.path.exists(synth_path):
        print(f"[lowrank/gen] {variant_tag} {nd}d t{trial}: synthetic.h5ad already exists — skipping", flush=True)
        return synth_path

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"lowrank models dir missing: {models_dir} — run train.py first")

    train_donors = np.load(train_npy, allow_pickle=True)
    full = sc.read_h5ad(full_h5ad, backed="r")
    full_obs = full.obs[[DONOR_COL, CELL_TYPE_COL]].copy()
    all_var_names = full.var_names.copy()
    full.file.close()

    hvg_df = pd.read_csv(hvg_path, index_col=0)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)

    train_mask = full_obs[DONOR_COL].isin(train_donors)
    test_cell_type_arr = full_obs.loc[train_mask, CELL_TYPE_COL].values

    cell_types = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(models_dir)
        if f.endswith(".rds")
    ])
    cell_types = [
        ct for ct in cell_types
        if ct.isdigit() or (ct.startswith("-") and ct[1:].isdigit())
    ]

    rng = np.random.default_rng(
        seed + hash((nd, trial, rank, int(epsilon) if is_dp else -1)) % (2**31)
    )

    sigma_log = {}
    with tempfile.TemporaryDirectory(prefix=f"lowrank_gen_{variant_tag}_") as tmp_dir:
        for ct in cell_types:
            copula_path = os.path.join(models_dir, f"{ct}.rds")
            if not os.path.exists(copula_path):
                print(f"    [SKIP] {copula_path} missing")
                continue

            try:
                copula_rds = R["readRDS"](copula_path)
                ct_obj = copula_rds.rx2(str(ct))
                parsed = parse_copula(ct_obj)
            except Exception as e:
                print(f"    [WARN] parse failed for {ct}: {e}")
                continue

            if parsed.get("cov_matrix") is None:
                print(f"    [SKIP] {ct}: cov_mat is None (no group-1 genes)")
                continue

            # parse_copula() doesn't extract quantile_normal (it's specific
            # to scdesign2_lowrank.r's output) -- pull it directly.
            qn = ct_obj.rx2("quantile_normal")
            if qn is None or (hasattr(qn, "__len__") and len(qn) == 0):
                print(f"    [SKIP] {ct}: quantile_normal missing "
                      f"(re-fit with scdesign2_lowrank.r?)")
                continue
            parsed["quantile_normal"] = qn

            n_genes_ct = np.array(parsed["cov_matrix"], dtype=np.float64)
            n_genes_ct = n_genes_ct.size
            side = int(round(n_genes_ct ** 0.5))
            rank_ct = min(rank, side)  # don't exceed available genes for tiny cell types

            n_cells_ct = _get_n_cells_from_copula(copula_rds, ct)
            k_max = _get_k_max(full_obs, train_donors, ct)
            if n_cells_ct <= k_max:
                k_max = max(1, n_cells_ct - 1)

            if is_dp:
                noised = apply_gaussian_dp_lowrank(
                    copula_dict=parsed,
                    epsilon=epsilon,
                    delta=DELTA,
                    n_cells=n_cells_ct,
                    k_max=k_max,
                    rank=rank_ct,
                    clip_value=CLIP_VALUE,
                    rng=rng,
                    projection_seed=abs(hash((dataset, str(ct), trial, rank_ct))) % (2**31),
                )
                final_cov = noised["cov_matrix"]
                sigma = gaussian_noise_scale(
                    epsilon=epsilon, delta=DELTA, n_cells=n_cells_ct,
                    k_max=k_max, n_genes=rank_ct,
                    clip_value=CLIP_VALUE, dp_variant="v2",
                )
                sigma_log[str(ct)] = {
                    "sigma": float(sigma), "n_cells": int(n_cells_ct),
                    "k_max": int(k_max), "rank": int(rank_ct),
                }
            else:
                noised = project_lowrank_no_dp(
                    copula_dict=parsed,
                    rank=rank_ct,
                    clip_value=CLIP_VALUE,
                    projection_seed=abs(hash((dataset, str(ct), trial, rank_ct))) % (2**31),
                )
                final_cov = noised["cov_matrix"]

            patched_rds = os.path.join(tmp_dir, f"patched_{ct}.rds")
            out_rds     = os.path.join(tmp_dir, f"out{ct}.rds")
            n_to_gen    = int((test_cell_type_arr == ct).sum())
            if n_to_gen == 0:
                print(f"    [SKIP] {ct}: 0 test cells")
                continue
            try:
                _save_corrected_rds(copula_rds, ct, final_cov, patched_rds)
                ok = _run_r_gen(patched_rds, n_to_gen, out_rds)
                if not ok:
                    continue
                tag = f"σ={sigma_log[str(ct)]['sigma']:.3f}" if is_dp else "no-DP"
                print(f"    {ct}: n_cells={n_cells_ct}, k_max={k_max}, rank={rank_ct}, "
                      f"gen={n_to_gen}, {tag}", flush=True)
            except Exception as e:
                print(f"    [WARN] gen failed for {ct}: {e}")
                continue

        adata = _assemble(cell_types, test_cell_type_arr, tmp_dir, hvg_mask, all_var_names)
        adata.write(synth_path, compression="gzip")
        print(f"[lowrank/gen] saved: {synth_path}  shape={adata.shape}", flush=True)

    prov = {
        "stage":         "generate",
        "dp_variant":    "lowrank",
        "dataset":       dataset,
        "n_donors":      nd,
        "trial":         trial,
        "rank":          rank,
        "epsilon":       epsilon,
        "delta":         DELTA if is_dp else None,
        "clip_value":    CLIP_VALUE,
        "seed":          seed,
        "inputs": {
            "models_dir": models_dir,
            "full_h5ad":  full_h5ad,
            "splits_dir": splits_dir,
            "hvg_path":   hvg_path,
        },
        "per_cell_type_sigma": sigma_log if is_dp else None,
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(out_root, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    return synth_path


def _assemble(cell_types, test_cell_type_arr, tmp_dir, hvg_mask, all_var_names):
    import pyreadr
    hvg_indices = np.where(hvg_mask)[0]
    n_cells = len(test_cell_type_arr)
    n_genes = len(all_var_names)
    X = sp.lil_matrix((n_cells, n_genes), dtype=np.float64)

    for ct in cell_types:
        out_path = os.path.join(tmp_dir, f"out{ct}.rds")
        if not os.path.exists(out_path):
            continue
        try:
            r_mat = list(pyreadr.read_r(out_path).values())[0]
            counts_np = r_mat.to_numpy() if hasattr(r_mat, "to_numpy") else np.array(r_mat)
            cell_indices = np.where(test_cell_type_arr == ct)[0]
            n_assign = min(len(cell_indices), counts_np.shape[1])
            for i in range(n_assign):
                n_g = min(len(hvg_indices), counts_np.shape[0])
                X[cell_indices[i], hvg_indices[:n_g]] = counts_np[:n_g, i]
        except Exception as e:
            print(f"    [SKIP] {out_path}: {e}")

    a = ad.AnnData(X=X.tocsr())
    a.obs[CELL_TYPE_COL] = test_cell_type_arr
    a.var_names = all_var_names
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd",      type=int, required=True)
    ap.add_argument("--trial",   type=int, nargs="+", required=True)
    ap.add_argument("--rank",    type=int, required=True,
                    help="Target reduced dimension r (per cell type, capped at that cell type's gene count).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--no-dp",    action="store_true",
                   help="Rank reduction only, no DP noise.")
    g.add_argument("--epsilon",  type=float, nargs="+",
                   help="One or more ε values for DP runs.")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    eps_iter = [None] if args.no_dp else args.epsilon
    for t in args.trial:
        for e in eps_iter:
            generate_one(args.dataset, args.nd, t, args.rank, e, seed=args.seed)


if __name__ == "__main__":
    main()
