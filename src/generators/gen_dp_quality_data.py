"""
gen_dp_quality_data.py
======================
Generate DP-noised synthetic data from already-trained scDesign2 Gaussian copulas
(OneK1K, 10d / 20d / 50d, all 6 trials) and save the results in a directory layout
that is directly consumable by src/run_quality_eval.py.

Output root: /home/golobs/data/ok_dp/eps_{epsilon}/{n}d/{trial}/datasets/synthetic.h5ad

A matching set of exp_cfgs YAML files is written to
/home/golobs/data/ok_dp/exp_cfgs/{n}d_{trial}_eps{eps}.yaml
so you can run:
    python src/run_quality_eval.py T /home/golobs/data/ok_dp/exp_cfgs/10d_1_eps1.yaml

Usage:
    python src/generators/gen_dp_quality_data.py
    python src/generators/gen_dp_quality_data.py --n_donors 10 --trial 1 --epsilon 1
"""

import argparse
import os
import sys
import subprocess
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

OK_DIR         = "/home/golobs/data/ok"
FULL_H5AD      = os.path.join(OK_DIR, "full_dataset_cleaned.h5ad")
HVG_CSV        = os.path.join(OK_DIR, "hvg.csv")
DP_ROOT        = "/home/golobs/data/ok_dp"
R_SCRIPT       = os.path.join(SRC_DIR, "sdg", "scdesign2", "scdesign2.r")

N_DONORS_LIST  = [10, 20, 50]
TRIALS         = [1, 2, 3, 4, 5, 6]
EPSILONS       = [1, 10, 100, 1000, 10_000]
DELTA          = 1e-5
CLIP_VALUE     = 3.0
CELL_TYPE_COL  = "cell_type"
DONOR_COL      = "individual"

# ---------------------------------------------------------------------------
# Imports from project code
# ---------------------------------------------------------------------------
from sdg.scdesign2.copula import parse_copula
from sdg.dp.dp_copula import apply_gaussian_dp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_k_max(full_obs: pd.DataFrame, donor_ids, cell_type: str) -> int:
    """Max cells any single donor contributes to a given cell type."""
    mask = full_obs[DONOR_COL].isin(donor_ids) & (full_obs[CELL_TYPE_COL] == cell_type)
    counts = full_obs[mask].groupby(DONOR_COL).size()
    return int(counts.max()) if len(counts) > 0 else 1


def load_r_copula(copula_path: str, cell_type: str):
    """Load an .rds copula file and return the rpy2 copula object for one cell type."""
    from rpy2.robjects import r as R
    copula_rds = R["readRDS"](copula_path)
    return copula_rds.rx2(str(cell_type)), copula_rds


def get_n_cells_from_copula(copula_path: str, cell_type: str) -> int:
    """Read n_cell from the stored copula metadata."""
    from rpy2.robjects import r as R
    copula_rds = R["readRDS"](copula_path)
    ct_obj = copula_rds.rx2(str(cell_type))
    return int(ct_obj.rx2("n_cell")[0])


def save_noised_rds(copula_rds, cell_type: str, noised_corr: np.ndarray,
                    out_path: str):
    """
    Patch cov_mat in the loaded rpy2 copula object with noised_corr, then
    saveRDS to out_path.
    """
    from rpy2.robjects import r as R
    from rpy2.robjects.vectors import FloatVector

    G = noised_corr.shape[0]
    # Flatten column-major (R's default for matrix())
    flat = noised_corr.flatten(order="F").tolist()

    R.assign("dp_copula_obj", copula_rds)
    R.assign("dp_noised_flat", FloatVector(flat))
    R(f'dp_copula_obj[["{cell_type}"]][["cov_mat"]] <- '
      f'matrix(dp_noised_flat, nrow={G}, ncol={G})')
    R(f'saveRDS(dp_copula_obj, file="{out_path}")')


def run_r_gen(copula_path: str, n_cells: int, out_rds_path: str):
    """Call Rscript scdesign2.r gen to sample synthetic counts."""
    cmd = f"Rscript {R_SCRIPT} gen {n_cells} {copula_path} {out_rds_path}"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output.decode("utf-8", errors="replace")
        print(f"    [WARN] R gen failed: {out[:300]}")


def assemble_synthetic(cell_types, test_cell_type_arr, tmp_dir: str,
                       hvg_mask: np.ndarray, all_var_names) -> ad.AnnData:
    """
    Read per-cell-type .rds outputs and assemble into one AnnData aligned to the
    test-set cell order. Only HVG columns are filled (the rest stay 0).
    """
    import pyreadr

    hvg_indices = np.where(hvg_mask)[0]
    n_cells  = len(test_cell_type_arr)
    n_genes  = len(all_var_names)
    synthetic = sp.lil_matrix((n_cells, n_genes), dtype=np.float64)

    for ct in cell_types:
        out_path = os.path.join(tmp_dir, f"out{ct}.rds")
        if not os.path.exists(out_path):
            print(f"    [SKIP] {out_path} not found — cell type {ct} skipped")
            continue
        try:
            r_mat = list(pyreadr.read_r(out_path).values())[0]
            counts_np = r_mat.to_numpy() if hasattr(r_mat, "to_numpy") else np.array(r_mat)
            cell_indices = np.where(test_cell_type_arr == ct)[0]
            n_assign = min(len(cell_indices), counts_np.shape[1])
            for i in range(n_assign):
                n_g = min(len(hvg_indices), counts_np.shape[0])
                synthetic[cell_indices[i], hvg_indices[:n_g]] = counts_np[:n_g, i]
        except Exception as e:
            print(f"    [SKIP] Could not read {out_path}: {e}")

    adata = ad.AnnData(X=synthetic.tocsr())
    adata.obs[CELL_TYPE_COL] = test_cell_type_arr
    adata.var_names = all_var_names
    return adata


def setup_dp_dir(n_donors: int, trial: int, epsilon) -> str:
    """
    Create the output datasets directory and return its path.
    Also creates a top-level tracking.csv if it doesn't exist yet.
    """
    eps_tag  = f"eps_{int(epsilon)}" if epsilon == int(epsilon) else f"eps_{epsilon}"
    out_dir  = os.path.join(DP_ROOT, eps_tag, f"{n_donors}d", str(trial), "datasets")
    os.makedirs(out_dir, exist_ok=True)

    # tracking.csv in the split dir
    split_dir    = os.path.join(DP_ROOT, eps_tag, f"{n_donors}d")
    tracking_csv = os.path.join(split_dir, "tracking.csv")
    if not os.path.exists(tracking_csv):
        rows = [{"trial": t, "quality": 0} for t in TRIALS]
        pd.DataFrame(rows).to_csv(tracking_csv, index=False)

    # train.npy: copy from original trial (so run_quality_eval can find it)
    src_npy = os.path.join(OK_DIR, f"{n_donors}d", str(trial), "datasets", "train.npy")
    dst_npy = os.path.join(out_dir, "train.npy")
    if os.path.exists(src_npy) and not os.path.exists(dst_npy):
        shutil.copy2(src_npy, dst_npy)

    return out_dir


def write_quality_cfg(n_donors: int, trial: int, epsilon) -> str:
    """Write a YAML config for run_quality_eval.py and return its path."""
    eps_tag  = f"eps_{int(epsilon)}" if epsilon == int(epsilon) else f"eps_{epsilon}"
    cfg_dir  = os.path.join(DP_ROOT, "exp_cfgs")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{n_donors}d_{trial}_{eps_tag}.yaml")

    content = f"""\
dir_list:
  local:
    home: /home/golobs/scRNA-seq_privacy_audits
    data: /home/golobs/data
  server:
    home: /home/golobs/scRNA-seq_privacy_audits
    data: /home/golobs/data
dataset_name: ok_dp/{eps_tag}
plot_results: false
parallelize: false
min_aux_donors: 10

mamamia_params:
  IMPORTANCE_OF_CLASS_B_FPs: .17
  epsilon: .0001
  mahalanobis: true
  uniform_remapping_fn: zinb_cdf
  lin_alg_inverse_fn: pinv_gpu
  closeness_to_correlation_fn: closeness_to_correlation_1
  class_b_gene_set: secondary
  class_b_scoring: llr
  class_b_gamma: auto

mia_setting:
  sample_donors_strategy_fn: sample_donors_strategy_2
  num_donors: {n_donors}
  white_box: true
  use_wb_hvgs: true
  use_aux: true
"""
    with open(cfg_path, "w") as f:
        f.write(content)
    return cfg_path


# ---------------------------------------------------------------------------
# Core per-(n_donors, trial, epsilon) function
# ---------------------------------------------------------------------------

def generate_dp_synthetic(n_donors: int, trial: int, epsilon: float,
                           full_obs: pd.DataFrame,
                           hvg_mask: np.ndarray,
                           all_var_names,
                           rng_seed: int = 42):
    """
    Generate DP-noised synthetic data for one (n_donors, trial, epsilon) combo
    and save to the DP output directory.
    """
    eps_tag   = f"eps_{int(epsilon)}" if epsilon == int(epsilon) else f"eps_{epsilon}"
    out_dir   = setup_dp_dir(n_donors, trial, epsilon)
    synth_out = os.path.join(out_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] {synth_out} already exists — skipping")
        return

    models_dir = os.path.join(OK_DIR, f"{n_donors}d", str(trial), "models")
    train_npy  = os.path.join(OK_DIR, f"{n_donors}d", str(trial), "datasets", "train.npy")

    if not os.path.isdir(models_dir):
        print(f"  [SKIP] No models dir at {models_dir}")
        return
    if not os.path.exists(train_npy):
        print(f"  [SKIP] No train.npy at {train_npy}")
        return

    train_donors = np.load(train_npy, allow_pickle=True).tolist()
    rng = np.random.default_rng(rng_seed + hash((n_donors, trial, epsilon)) % (2**31))

    # Discover cell types from .rds files
    cell_types = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(models_dir)
        if f.endswith(".rds")
    ])
    # Remove non-numeric entries (e.g. mean_expr.csv generates no .rds)
    cell_types = [ct for ct in cell_types if ct.isdigit() or
                  (ct.startswith("-") and ct[1:].isdigit())]

    # Build test cell type array (same as training — scDesign2 generates matching size)
    train_mask = full_obs[DONOR_COL].isin(train_donors)
    test_cell_type_arr = full_obs.loc[train_mask, CELL_TYPE_COL].values

    with tempfile.TemporaryDirectory(prefix="dp_gen_") as tmp_dir:
        for ct in cell_types:
            copula_path = os.path.join(models_dir, f"{ct}.rds")
            if not os.path.exists(copula_path):
                print(f"    [SKIP] {copula_path} missing")
                continue

            # --- n_cells from copula; k_max from obs ---
            try:
                n_cells = get_n_cells_from_copula(copula_path, ct)
            except Exception as e:
                print(f"    [WARN] Could not read n_cell for {ct}: {e} — estimating from obs")
                n_cells = int((full_obs[DONOR_COL].isin(train_donors) &
                               (full_obs[CELL_TYPE_COL] == ct)).sum())

            k_max = get_k_max(full_obs, train_donors, ct)
            if n_cells <= k_max:
                k_max = max(1, n_cells - 1)

            # --- Parse copula and apply DP ---
            try:
                from rpy2.robjects import r as R
                copula_rds = R["readRDS"](copula_path)
                ct_obj = copula_rds.rx2(str(ct))
                parsed = parse_copula(ct_obj)
            except Exception as e:
                print(f"    [WARN] Could not parse copula for ct={ct}: {e}")
                continue

            if parsed.get("cov_matrix") is None:
                print(f"    [SKIP] ct={ct}: cov_matrix is None (vine / no group-1 genes)")
                continue

            try:
                noised = apply_gaussian_dp(
                    copula_dict=parsed,
                    epsilon=epsilon,
                    delta=DELTA,
                    n_cells=n_cells,
                    k_max=k_max,
                    clip_value=CLIP_VALUE,
                    rng=rng,
                )
            except Exception as e:
                print(f"    [WARN] DP noise failed for ct={ct}: {e}")
                continue

            # --- Save noised copula to tmp, then run R gen ---
            noised_rds = os.path.join(tmp_dir, f"noised_{ct}.rds")
            out_rds    = os.path.join(tmp_dir, f"out{ct}.rds")
            n_to_gen   = int((test_cell_type_arr == ct).sum())

            if n_to_gen == 0:
                print(f"    [SKIP] ct={ct}: 0 test cells — skipping generation")
                continue

            try:
                save_noised_rds(copula_rds, ct, noised["cov_matrix"], noised_rds)
                run_r_gen(noised_rds, n_to_gen, out_rds)
                from sdg.dp.sensitivity import gaussian_noise_scale
                sigma = gaussian_noise_scale(epsilon, DELTA, n_cells, k_max,
                                             len(parsed["primary_genes"]), CLIP_VALUE)
                print(f"    ct={ct}: n_cells={n_cells}, k_max={k_max}, "
                      f"sigma={sigma:.3f}, gen={n_to_gen}")
            except Exception as e:
                print(f"    [WARN] Gen failed for ct={ct}: {e}")
                continue

        # --- Assemble and save ---
        adata = assemble_synthetic(cell_types, test_cell_type_arr,
                                   tmp_dir, hvg_mask, all_var_names)
        adata.write(synth_out, compression="gzip")
        print(f"  Saved: {synth_out}  shape={adata.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_donors", type=int, nargs="+", default=N_DONORS_LIST)
    parser.add_argument("--trial",    type=int, nargs="+", default=TRIALS)
    parser.add_argument("--epsilon",  type=float, nargs="+", default=EPSILONS)
    args = parser.parse_args()

    print("Loading full dataset obs (no count matrix) ...")
    # backed='r' avoids loading the count matrix into RAM
    full_adata = sc.read_h5ad(FULL_H5AD, backed="r")
    full_obs   = full_adata.obs[[DONOR_COL, CELL_TYPE_COL]].copy()
    all_var_names = full_adata.var_names.copy()
    full_adata.file.close()

    print(f"Full dataset: {len(full_obs)} cells")

    print("Loading HVG mask ...")
    hvg_df   = pd.read_csv(HVG_CSV)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)
    print(f"  {hvg_mask.sum()} HVGs")

    # Write quality-eval configs
    for nd in args.n_donors:
        for eps in args.epsilon:
            for trial in args.trial:
                write_quality_cfg(nd, trial, eps)

    # Symlink shared files once per epsilon root
    for eps in args.epsilon:
        eps_tag = f"eps_{int(eps)}" if eps == int(eps) else f"eps_{eps}"
        eps_root = os.path.join(DP_ROOT, eps_tag)
        os.makedirs(eps_root, exist_ok=True)
        # full_dataset_cleaned.h5ad and Immune_All_High.pkl if present
        for fname in ["full_dataset_cleaned.h5ad", "Immune_All_High.pkl"]:
            src = os.path.join(OK_DIR, fname)
            dst = os.path.join(eps_root, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    total = len(args.n_donors) * len(args.trial) * len(args.epsilon)
    done  = 0
    for nd in args.n_donors:
        for trial in args.trial:
            for eps in args.epsilon:
                done += 1
                eps_tag = f"eps_{int(eps)}" if eps == int(eps) else f"eps_{eps}"
                print(f"\n[{done}/{total}] n_donors={nd}, trial={trial}, epsilon={eps}")
                try:
                    generate_dp_synthetic(nd, trial, eps, full_obs,
                                          hvg_mask, all_var_names,
                                          rng_seed=42)
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
