"""
sanity_check_no_clip.py
=======================
Sanity-check for the DP copula: verify that quality degradation at very high ε
(essentially zero noise) is caused by the post-noise clipping steps, NOT by the
noise itself.

Hypothesis:
  - DP with ε=1e9 AND clip=True  → copula is nearly identical to original, but
    PSD-projection and [-1,1] value-clipping may still alter it slightly.
  - DP with ε=1e9 AND clip=False → copula is bit-for-bit identical to original,
    so generated data quality should exactly match non-DP scDesign2.

This script:
  1. Generates synthetic OneK1K data from existing 50d copulas using:
       (a) ε=1e9, clip=True   → ok_dp/eps_noclip_cliptrue/50d/{trial}/
       (b) ε=1e9, clip=False  → ok_dp/eps_noclip/50d/{trial}/
  2. Evaluates LISI, ARI, MMD for both conditions.
  3. Loads existing non-DP and eps=10000 quality results.
  4. Prints a LaTeX table for comparison.

Usage
-----
    conda run -n tabddpm_ python experiments/dp/sanity_check_no_clip.py \\
        [--trials 1 2 3] [--n-donors 50] [--skip-gen]

    --skip-gen   Skip generation if synthetic.h5ad already exists.
"""

import argparse
import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")
for _d in [_REPO_ROOT, _SRC_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

OK_DIR    = "/home/golobs/data/ok"
DP_ROOT   = "/home/golobs/data/ok_dp"
FULL_H5AD = os.path.join(OK_DIR, "full_dataset_cleaned.h5ad")
HVG_CSV   = os.path.join(OK_DIR, "hvg.csv")
R_SCRIPT  = os.path.join(_SRC_DIR, "sdg", "scdesign2", "scdesign2.r")

CELL_TYPE_COL = "cell_type"
DONOR_COL     = "individual"
DELTA         = 1e-5
CLIP_VALUE    = 3.0

EPSILON_SANITY = 1e9   # effectively zero noise

# ---------------------------------------------------------------------------
# Imports from project code
# ---------------------------------------------------------------------------
from sdg.scdesign2.copula import parse_copula
from sdg.dp.dp_copula import apply_gaussian_dp


# ---------------------------------------------------------------------------
# Generation helpers (mirrors gen_dp_quality_data.py)
# ---------------------------------------------------------------------------

def _get_k_max(full_obs: pd.DataFrame, donor_ids, cell_type: str) -> int:
    mask = full_obs[DONOR_COL].isin(donor_ids) & (full_obs[CELL_TYPE_COL] == cell_type)
    counts = full_obs[mask].groupby(DONOR_COL).size()
    return int(counts.max()) if len(counts) > 0 else 1


def _get_n_cells_from_copula(copula_path: str, cell_type: str) -> int:
    from rpy2.robjects import r as R
    ct_obj = R["readRDS"](copula_path).rx2(str(cell_type))
    return int(ct_obj.rx2("n_cell")[0])


def _save_noised_rds(copula_rds, cell_type: str, noised_corr: np.ndarray,
                     out_path: str):
    """Patch cov_mat in the rpy2 copula object and saveRDS to out_path."""
    from rpy2.robjects import r as R
    from rpy2.robjects.vectors import FloatVector

    G = noised_corr.shape[0]
    flat = noised_corr.flatten(order="F").tolist()
    R.assign("sanity_copula_obj", copula_rds)
    R.assign("sanity_noised_flat", FloatVector(flat))
    R(f'sanity_copula_obj[["{cell_type}"]][["cov_mat"]] <- '
      f'matrix(sanity_noised_flat, nrow={G}, ncol={G})')
    R(f'saveRDS(sanity_copula_obj, file="{out_path}")')


def _run_r_gen(copula_path: str, n_cells: int, out_rds_path: str):
    import subprocess
    cmd = f"Rscript {R_SCRIPT} gen {n_cells} {copula_path} {out_rds_path}"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output.decode("utf-8", errors="replace")
        print(f"    [WARN] R gen failed: {out[:300]}")


def _assemble_synthetic(cell_types, test_cell_type_arr, tmp_dir: str,
                        hvg_mask: np.ndarray, all_var_names) -> ad.AnnData:
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


def generate_one_trial(n_donors: int, trial: int, epsilon: float, clip: bool,
                       full_obs: pd.DataFrame, hvg_mask: np.ndarray,
                       all_var_names, skip_if_exists: bool = True) -> str:
    """
    Generate DP-noised synthetic data for one trial.

    Returns the path to synthetic.h5ad.
    """
    tag = "eps_noclip" if not clip else "eps_noclip_cliptrue"
    out_dir   = os.path.join(DP_ROOT, tag, f"{n_donors}d", str(trial), "datasets")
    synth_out = os.path.join(out_dir, "synthetic.h5ad")

    if skip_if_exists and os.path.exists(synth_out):
        print(f"  [EXISTS] {synth_out} — skipping generation")
        return synth_out

    os.makedirs(out_dir, exist_ok=True)

    # Copy train.npy for quality eval (the evaluator needs it to load real data)
    src_npy = os.path.join(OK_DIR, f"{n_donors}d", str(trial), "datasets", "train.npy")
    dst_npy = os.path.join(out_dir, "train.npy")
    if os.path.exists(src_npy) and not os.path.exists(dst_npy):
        shutil.copy2(src_npy, dst_npy)

    models_dir = os.path.join(OK_DIR, f"{n_donors}d", str(trial), "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No models dir at {models_dir}")

    train_npy    = os.path.join(OK_DIR, f"{n_donors}d", str(trial), "datasets", "train.npy")
    train_donors = np.load(train_npy, allow_pickle=True).tolist()

    rng = np.random.default_rng(42)

    cell_types = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(models_dir)
        if f.endswith(".rds") and os.path.splitext(f)[0].lstrip("-").isdigit()
    ])

    train_mask = full_obs[DONOR_COL].isin(train_donors)
    test_cell_type_arr = full_obs.loc[train_mask, CELL_TYPE_COL].values

    print(f"  Generating {len(cell_types)} cell types, "
          f"{len(test_cell_type_arr)} cells, ε={epsilon:.0e}, clip={clip}")

    with tempfile.TemporaryDirectory(prefix="dp_sanity_") as tmp_dir:
        from rpy2.robjects import r as R

        for ct in cell_types:
            copula_path = os.path.join(models_dir, f"{ct}.rds")
            if not os.path.exists(copula_path):
                continue

            try:
                n_cells = _get_n_cells_from_copula(copula_path, ct)
            except Exception:
                n_cells = int((full_obs[DONOR_COL].isin(train_donors) &
                               (full_obs[CELL_TYPE_COL] == ct)).sum())

            k_max = _get_k_max(full_obs, train_donors, ct)
            if n_cells <= k_max:
                k_max = max(1, n_cells - 1)

            try:
                copula_rds = R["readRDS"](copula_path)
                ct_obj     = copula_rds.rx2(str(ct))
                parsed     = parse_copula(ct_obj)
            except Exception as e:
                print(f"    [WARN] Could not parse copula for ct={ct}: {e}")
                continue

            if parsed.get("cov_matrix") is None:
                print(f"    [SKIP] ct={ct}: no cov_matrix (vine/no group-1 genes)")
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
                    clip=clip,
                )
            except Exception as e:
                print(f"    [WARN] DP noise failed for ct={ct}: {e}")
                continue

            noised_rds = os.path.join(tmp_dir, f"noised_{ct}.rds")
            out_rds    = os.path.join(tmp_dir, f"out{ct}.rds")
            n_to_gen   = int((test_cell_type_arr == ct).sum())

            if n_to_gen == 0:
                continue

            try:
                _save_noised_rds(copula_rds, ct, noised["cov_matrix"], noised_rds)
                _run_r_gen(noised_rds, n_to_gen, out_rds)
                print(f"    ct={ct}: generated {n_to_gen} cells", flush=True)
            except Exception as e:
                print(f"    [WARN] Gen failed for ct={ct}: {e}")

        adata = _assemble_synthetic(cell_types, test_cell_type_arr,
                                    tmp_dir, hvg_mask, all_var_names)

    adata.write(synth_out, compression="gzip")
    print(f"  Saved: {synth_out}  shape={adata.shape}")
    return synth_out


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def eval_quality(synth_path: str, train_npy: str, out_csv: str) -> dict:
    """Run SingleCellEvaluator on synth_path, save to out_csv, return metrics."""
    from evaluation.sc_evaluate import SingleCellEvaluator

    out_dir = os.path.dirname(out_csv)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(out_dir), "figures"), exist_ok=True)

    cfg = {
        "dir_list": {
            "home": os.path.dirname(os.path.dirname(os.path.dirname(out_csv))),
            "figures": "figures",
            "res_files": "results",
        },
        "full_data_path": FULL_H5AD,
        "synthetic_file": synth_path,
        "dataset_config": {
            "name": "ok",
            "test_count_file": train_npy,
            "cell_type_col_name": CELL_TYPE_COL,
            "cell_label_col_name": "cell_label",
            "celltypist_model": "",
        },
        "evaluator_config": {"random_seed": 1},
    }

    evaluator = SingleCellEvaluator(config=cfg)
    results = evaluator.get_statistical_evals()
    evaluator.save_results_to_csv(results, out_csv)
    return results


# ---------------------------------------------------------------------------
# Load existing quality results
# ---------------------------------------------------------------------------

def load_existing_results(n_donors: int, trials, source: str) -> pd.DataFrame:
    """
    Load statistics_evals.csv for a given source ('nodp' or 'eps_XXXX').

    source='nodp'     → /home/golobs/data/ok/{n_donors}d/{trial}/results/...
    source='eps_XXXX' → /home/golobs/data/ok_dp/eps_XXXX/{n_donors}d/{trial}/results/...
    """
    rows = []
    for t in trials:
        if source == "nodp":
            csv = os.path.join(
                OK_DIR, f"{n_donors}d", str(t),
                "results", "quality_eval_results", "results", "statistics_evals.csv"
            )
        else:
            csv = os.path.join(
                DP_ROOT, source, f"{n_donors}d", str(t),
                "results", "quality_eval_results", "results", "statistics_evals.csv"
            )
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            rows.append(df.iloc[0] if len(df) > 0 else None)
        else:
            print(f"  [WARN] No CSV at {csv}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _fmt(val, is_mmd=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    if is_mmd:
        return f"{val:.2e}"
    return f"{val:.4f}"


def _mean_std(df, col):
    """Return (mean, std) for a column, handling missing gracefully."""
    if df.empty or col not in df.columns:
        return float("nan"), float("nan")
    vals = df[col].dropna()
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=0))


def print_latex_table(rows_data):
    """
    rows_data : list of dicts with keys:
        label, lisi_mean, lisi_std, ari_mean, ari_std, mmd_mean, mmd_std
    """
    header = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{lcccccc}" "\n"
        r"\hline" "\n"
        r"Method & LISI (mean) & LISI (std) & ARI (mean) & ARI (std) & MMD (mean) & MMD (std) \\" "\n"
        r"\hline"
    )
    print(header)
    for r in rows_data:
        lisi_m = _fmt(r.get("lisi_mean"))
        lisi_s = _fmt(r.get("lisi_std"))
        ari_m  = _fmt(r.get("ari_mean"))
        ari_s  = _fmt(r.get("ari_std"))
        mmd_m  = _fmt(r.get("mmd_mean"), is_mmd=True)
        mmd_s  = _fmt(r.get("mmd_std"),  is_mmd=True)
        label  = r.get("label", "?").replace("_", r"\_")
        print(f"{label} & {lisi_m} & {lisi_s} & {ari_m} & {ari_s} & {mmd_m} & {mmd_s} \\\\")
    footer = (
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\caption{Quality comparison: scDesign2 variants (OneK1K, 50 donors). "
        r"LISI $\uparrow$ better; ARI $\uparrow$ better; MMD $\downarrow$ better.}" "\n"
        r"\label{tab:clip_sanity}" "\n"
        r"\end{table}"
    )
    print(footer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: high-ε DP with clip=False should match non-DP quality."
    )
    parser.add_argument("--n-donors", type=int, default=50)
    parser.add_argument("--trials",   type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epsilon",  type=float, default=EPSILON_SANITY,
                        help="DP epsilon for sanity check (default: 1e9 ≈ no noise)")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip generation if synthetic.h5ad already exists")
    args = parser.parse_args()

    nd = args.n_donors
    trials = args.trials
    eps = args.epsilon

    print("Loading full dataset obs ...")
    full_adata = sc.read_h5ad(FULL_H5AD, backed="r")
    full_obs   = full_adata.obs[[DONOR_COL, CELL_TYPE_COL]].copy()
    all_var_names = full_adata.var_names.copy()
    full_adata.file.close()

    print("Loading HVG mask ...")
    hvg_df   = pd.read_csv(HVG_CSV)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)
    print(f"  {hvg_mask.sum()} HVGs, {len(full_obs)} total cells\n")

    # -----------------------------------------------------------------------
    # Part 1: Generate sanity-check data
    # -----------------------------------------------------------------------
    results_noclip   = {}
    results_cliptrue = {}

    for trial in trials:
        print(f"\n=== Trial {trial} / {nd} donors ===")

        # clip=False (the key sanity check)
        print(f"\n[clip=False, ε={eps:.0e}]")
        synth_noclip = generate_one_trial(
            nd, trial, eps, clip=False,
            full_obs=full_obs, hvg_mask=hvg_mask,
            all_var_names=all_var_names,
            skip_if_exists=args.skip_gen,
        )

        # clip=True (control: high-ε with clipping still active)
        print(f"\n[clip=True, ε={eps:.0e}]")
        synth_cliptrue = generate_one_trial(
            nd, trial, eps, clip=True,
            full_obs=full_obs, hvg_mask=hvg_mask,
            all_var_names=all_var_names,
            skip_if_exists=args.skip_gen,
        )

        # -----------------------------------------------------------------------
        # Part 2: Evaluate quality
        # -----------------------------------------------------------------------
        train_npy_orig = os.path.join(
            OK_DIR, f"{nd}d", str(trial), "datasets", "train.npy"
        )

        # clip=False eval
        out_csv_noclip = os.path.join(
            DP_ROOT, "eps_noclip", f"{nd}d", str(trial),
            "results", "quality_eval_results", "results", "statistics_evals.csv"
        )
        if os.path.exists(out_csv_noclip) and args.skip_gen:
            print(f"\n[SKIP EVAL] {out_csv_noclip} already exists")
            results_noclip[trial] = pd.read_csv(out_csv_noclip).iloc[0].to_dict()
        else:
            print(f"\nEvaluating clip=False quality ...")
            results_noclip[trial] = eval_quality(synth_noclip, train_npy_orig, out_csv_noclip)
            m = results_noclip[trial]
            print(f"  lisi={m.get('lisi','?'):.4f}  ari={m.get('ari_real_vs_syn','?'):.4f}  "
                  f"mmd={m.get('mmd','?'):.2e}")

        # clip=True eval
        out_csv_cliptrue = os.path.join(
            DP_ROOT, "eps_noclip_cliptrue", f"{nd}d", str(trial),
            "results", "quality_eval_results", "results", "statistics_evals.csv"
        )
        if os.path.exists(out_csv_cliptrue) and args.skip_gen:
            print(f"\n[SKIP EVAL] {out_csv_cliptrue} already exists")
            results_cliptrue[trial] = pd.read_csv(out_csv_cliptrue).iloc[0].to_dict()
        else:
            print(f"\nEvaluating clip=True quality ...")
            results_cliptrue[trial] = eval_quality(synth_cliptrue, train_npy_orig, out_csv_cliptrue)
            m = results_cliptrue[trial]
            print(f"  lisi={m.get('lisi','?'):.4f}  ari={m.get('ari_real_vs_syn','?'):.4f}  "
                  f"mmd={m.get('mmd','?'):.2e}")

    # -----------------------------------------------------------------------
    # Part 3: Load existing results
    # -----------------------------------------------------------------------
    print("\nLoading existing results ...")

    nodp_df    = load_existing_results(nd, trials, "nodp")
    eps10k_df  = load_existing_results(nd, list(range(1, 6)), "eps_10000")

    noclip_df   = pd.DataFrame(list(results_noclip.values()))
    cliptrue_df = pd.DataFrame(list(results_cliptrue.values()))

    # -----------------------------------------------------------------------
    # Part 4: Print table
    # -----------------------------------------------------------------------
    def _row(label, df, lisi_col="lisi", ari_col="ari_real_vs_syn", mmd_col="mmd"):
        lisi_m, lisi_s = _mean_std(df, lisi_col)
        ari_m,  ari_s  = _mean_std(df, ari_col)
        mmd_m,  mmd_s  = _mean_std(df, mmd_col)
        return dict(label=label,
                    lisi_mean=lisi_m, lisi_std=lisi_s,
                    ari_mean=ari_m,   ari_std=ari_s,
                    mmd_mean=mmd_m,   mmd_std=mmd_s)

    rows = [
        _row(f"scDesign2 (no DP), n={len(nodp_df)} trials",       nodp_df),
        _row(f"DP $\\varepsilon$=10000, clip=True, n={len(eps10k_df)} trials",  eps10k_df),
        _row(f"DP $\\varepsilon$={eps:.0e}, clip=True,  n={len(trials)} trials", cliptrue_df),
        _row(f"DP $\\varepsilon$={eps:.0e}, clip=False, n={len(trials)} trials", noclip_df),
    ]

    print("\n" + "="*70)
    print("RESULTS SUMMARY (50 donors, OneK1K)")
    print("="*70)
    print(f"{'Method':<50} {'LISI':>8} {'ARI':>8} {'MMD':>12}")
    print("-"*80)
    for r in rows:
        lisi_s = f"{r['lisi_mean']:.4f} ± {r['lisi_std']:.4f}" if not np.isnan(r['lisi_mean']) else "--"
        ari_s  = f"{r['ari_mean']:.4f} ± {r['ari_std']:.4f}"   if not np.isnan(r['ari_mean'])  else "--"
        mmd_s  = f"{r['mmd_mean']:.2e} ± {r['mmd_std']:.2e}"  if not np.isnan(r['mmd_mean'])  else "--"
        print(f"{r['label']:<50} {lisi_s:>20} {ari_s:>20} {mmd_s:>24}")

    print("\n\n--- LaTeX table ---\n")
    print_latex_table(rows)


if __name__ == "__main__":
    main()
