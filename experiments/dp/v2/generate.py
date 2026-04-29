"""
generate.py — generate synthetic data from v2 copulas.

Two modes:
  --no-dp           PSD-project + normalize-to-correlation, then sample.
                    Output:  v2_no_dp/{nd}d/{trial}/datasets/synthetic.h5ad
  --epsilon E       apply_gaussian_dp(..., dp_variant='v2'): adds Gaussian
                    noise calibrated to the v2 sensitivity bound, projects,
                    normalises, then samples.
                    Output:  v2_eps_{E}/{nd}d/{trial}/datasets/synthetic.h5ad

Each output directory gets a provenance.json describing the variant, the
R script SHA-256, the seed, and (for DP runs) ε / δ / σ / clip_value.

Cell-type loop is sequential by default (each Rscript gen call is fast and
running them serially keeps memory pressure low while the baselines sweep
is also running).  Override with --parallel-cell-types N if needed.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
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

DATA_ROOT     = "/home/golobs/data/scMAMAMIA"
R_SCRIPT_V1   = os.path.join(REPO_ROOT, "src", "sdg", "scdesign2", "scdesign2.r")
R_SCRIPT_V2   = os.path.join(REPO_ROOT, "src", "sdg", "scdesign2", "scdesign2_v2.r")

DELTA       = 1e-5
CLIP_VALUE  = 3.0
DONOR_COL     = "individual"
CELL_TYPE_COL = "cell_type"


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _project_and_normalize(M):
    """PSD-project + normalize to correlation. Mirrors dp_copula._project_to_psd
    + _normalise_to_correlation, used here as standalone post-processing in the
    no-DP path. (For DP runs apply_gaussian_dp does this step internally.)"""
    M_sym = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals, 1e-8)
    M_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    std = np.sqrt(np.diag(M_psd))
    std = np.where(std > 0, std, 1.0)
    corr = M_psd / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def _save_corrected_rds(copula_rds, cell_type, corrected_cov_np, out_path):
    """Patch cov_mat on the in-memory R copula object and saveRDS to out_path."""
    from rpy2.robjects import r as R
    from rpy2.robjects.vectors import FloatVector
    G = corrected_cov_np.shape[0]
    flat = corrected_cov_np.flatten(order="F").tolist()
    R.assign("v2_copula_obj", copula_rds)
    R.assign("v2_noised_flat", FloatVector(flat))
    R(f'v2_copula_obj[["{cell_type}"]][["cov_mat"]] <- '
      f'matrix(v2_noised_flat, nrow={G}, ncol={G})')
    R(f'saveRDS(v2_copula_obj, file="{out_path}")')


def _run_r_gen(copula_path, n_cells, out_rds_path):
    cmd = ["Rscript", R_SCRIPT_V1, "gen", str(int(n_cells)), copula_path, out_rds_path]
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


def generate_one(dataset, nd, trial, epsilon, seed=42):
    """Generate v2 synthetic for one (dataset, nd, trial, epsilon|None) combo."""
    from rpy2.robjects import r as R
    from sdg.scdesign2.copula import parse_copula
    from sdg.dp.dp_copula import apply_gaussian_dp
    from sdg.dp.sensitivity import gaussian_noise_scale

    base = os.path.join(DATA_ROOT, dataset)
    full_h5ad   = os.path.join(base, "full_dataset_cleaned.h5ad")
    hvg_path    = os.path.join(base, "hvg.csv")
    splits_dir  = os.path.join(base, "splits", f"{nd}d", str(trial))
    train_npy   = os.path.join(splits_dir, "train.npy")

    models_dir  = os.path.join(base, "scdesign2", "v2_no_dp", f"{nd}d", str(trial), "models")

    is_dp = epsilon is not None
    eps_tag = f"v2_eps_{int(epsilon)}" if is_dp else "v2_no_dp"
    out_root      = os.path.join(base, "scdesign2", eps_tag, f"{nd}d", str(trial))
    out_datasets  = os.path.join(out_root, "datasets")
    os.makedirs(out_datasets, exist_ok=True)
    synth_path    = os.path.join(out_datasets, "synthetic.h5ad")

    if os.path.exists(synth_path):
        print(f"[v2/gen] {eps_tag} {nd}d t{trial}: synthetic.h5ad already exists — skipping", flush=True)
        return synth_path

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"v2 models dir missing: {models_dir} — run train.py first")

    # Inputs
    train_donors = np.load(train_npy, allow_pickle=True)
    full = sc.read_h5ad(full_h5ad, backed="r")
    full_obs = full.obs[[DONOR_COL, CELL_TYPE_COL]].copy()
    all_var_names = full.var_names.copy()
    full.file.close()

    hvg_df = pd.read_csv(hvg_path, index_col=0)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)

    # The v2_no_dp variant generates as many cells as the train slice (as the
    # existing v1 DP pipeline does — gen size matches train cell-type counts).
    train_mask = full_obs[DONOR_COL].isin(train_donors)
    test_cell_type_arr = full_obs.loc[train_mask, CELL_TYPE_COL].values

    # Discover cell types from .rds files in the v2 models dir
    cell_types = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(models_dir)
        if f.endswith(".rds")
    ])
    cell_types = [
        ct for ct in cell_types
        if ct.isdigit() or (ct.startswith("-") and ct[1:].isdigit())
    ]

    rng = np.random.default_rng(seed + hash((nd, trial, int(epsilon) if is_dp else -1)) % (2**31))

    sigma_log = {}
    with tempfile.TemporaryDirectory(prefix=f"v2_gen_{eps_tag}_") as tmp_dir:
        for ct in cell_types:
            copula_path = os.path.join(models_dir, f"{ct}.rds")
            if not os.path.exists(copula_path):
                print(f"    [SKIP] {copula_path} missing")
                continue

            # Load + parse
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

            n_cells_ct = _get_n_cells_from_copula(copula_rds, ct)
            k_max = _get_k_max(full_obs, train_donors, ct)
            if n_cells_ct <= k_max:
                k_max = max(1, n_cells_ct - 1)

            # Convert v2 raw second-moment matrix to a usable correlation.
            #   no-DP path: PSD + normalize
            #   DP path:    apply_gaussian_dp adds noise then PSD + normalize
            if is_dp:
                noised = apply_gaussian_dp(
                    copula_dict=parsed,
                    epsilon=epsilon,
                    delta=DELTA,
                    n_cells=n_cells_ct,
                    k_max=k_max,
                    clip_value=CLIP_VALUE,
                    rng=rng,
                    dp_variant="v2",
                )
                final_cov = noised["cov_matrix"]
                sigma = gaussian_noise_scale(
                    epsilon=epsilon, delta=DELTA, n_cells=n_cells_ct,
                    k_max=k_max, n_genes=parsed["len_primary"],
                    clip_value=CLIP_VALUE, dp_variant="v2",
                )
                sigma_log[str(ct)] = {
                    "sigma": float(sigma), "n_cells": int(n_cells_ct),
                    "k_max": int(k_max), "n_genes": int(parsed["len_primary"]),
                }
            else:
                M = np.array(parsed["cov_matrix"], dtype=np.float64)
                if M.ndim == 1:
                    side = int(round(M.size ** 0.5))
                    M = M.reshape(side, side)
                final_cov = _project_and_normalize(M)

            # Save the corrected copula to a temp .rds, then run R gen
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
                print(f"    {ct}: n_cells={n_cells_ct}, k_max={k_max}, "
                      f"gen={n_to_gen}, {tag}", flush=True)
            except Exception as e:
                print(f"    [WARN] gen failed for {ct}: {e}")
                continue

        # Assemble — same logic as v1 pipeline
        adata = _assemble(cell_types, test_cell_type_arr, tmp_dir, hvg_mask, all_var_names)
        adata.write(synth_path, compression="gzip")
        print(f"[v2/gen] saved: {synth_path}  shape={adata.shape}", flush=True)

    # Provenance
    prov = {
        "stage":         "generate",
        "dp_variant":    "v2",
        "dataset":       dataset,
        "n_donors":      nd,
        "trial":         trial,
        "epsilon":       epsilon,
        "delta":         DELTA if is_dp else None,
        "clip_value":    CLIP_VALUE if is_dp else None,
        "seed":          seed,
        "r_script_path_for_train": R_SCRIPT_V2,
        "r_script_sha256_for_train": _file_sha256(R_SCRIPT_V2),
        "r_script_path_for_gen":   R_SCRIPT_V1,
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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--no-dp",    action="store_true",
                   help="No DP — PSD+normalize and sample directly.")
    g.add_argument("--epsilon",  type=float, nargs="+",
                   help="One or more ε values for DP runs.")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    eps_iter = [None] if args.no_dp else args.epsilon
    for t in args.trial:
        for e in eps_iter:
            generate_one(args.dataset, args.nd, t, e, seed=args.seed)


if __name__ == "__main__":
    main()
