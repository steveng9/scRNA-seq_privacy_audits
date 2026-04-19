"""
DP sweep experiment for the scMAMA-MIA paper revision (Experiment 1).

For a given experimental trial (copulas already fitted by run_experiment.py),
this script re-runs the Mahalanobis attack for a range of ε values by injecting
calibrated Gaussian noise into the training copula before the attack.

Usage
-----
    python experiments/dp/run_dp_sweep.py <trial_dir> [options]

    --epsilons   space-separated ε values  (default: 0.1 0.5 1.0 2.0 5.0 10.0 100.0)
    --delta      δ for (ε,δ)-DP            (default: 1e-5)
    --clip       quantile-normal clip c    (default: 3.0)
    --seeds      per-epsilon random seeds  (default: 0 1 2 3 4 for 5 replicates)
    --no-aux     use the no-aux attack variant

Example
-------
    python experiments/dp/run_dp_sweep.py \
        /path/to/data/aida/100d/1 \
        --epsilons 0.1 0.5 1.0 2.0 5.0 10.0 \
        --delta 1e-5

What "trial_dir" must contain
------------------------------
    trial_dir/models/           — white-box copula (.rds per cell type)
    trial_dir/artifacts/aux/    — auxiliary shadow copula
    trial_dir/datasets/train.h5ad, holdout.h5ad
    trial_dir/datasets/train.npy, holdout.npy
    (the same structure produced by run_experiment.py)

Output
------
    trial_dir/results/dp_sweep_results.csv   — AUC + sensitivity stats per (ε, seed)
    trial_dir/results/dp_sensitivity_report.json — per-cell-type sensitivity budget
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import roc_auc_score

warnings.simplefilter(action="ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")
for _d in [_REPO_ROOT, _SRC_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from rpy2.robjects import r as R

from sdg.dp.sensitivity import sensitivity_report, frobenius_sensitivity, gaussian_noise_scale
from sdg.scdesign2.copula import parse_copula
from attacks.scmamamia.attack_dp import attack_mahalanobis_dp, attack_mahalanobis_no_aux_dp
from attacks.scmamamia.scoring import aggregate_scores_by_donor, merge_cell_type_results

import torch
from numpy.linalg import pinv
from data.cdf_utils import zinb_cdf, zinb_cdf_DT, zinb_uniform_transform
from box import Box

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pinv_gpu(a):
    if DEVICE.type == "cpu":
        return pinv(a)
    a = a.astype(np.float32)
    t = torch.from_numpy(a).cuda()
    return torch.linalg.pinv(t).cpu().numpy()


# ---------------------------------------------------------------------------
# Per-cell-type donor cell-count statistics
# ---------------------------------------------------------------------------

def compute_donor_cell_counts(train_h5ad_path: str) -> dict[str, dict]:
    """
    Return per-cell-type {n_cells, k_max, k_mean} computed from the training H5AD.

    These are the sensitivity parameters: n_cells = total cells of that cell type,
    k_max = max cells contributed by a single donor.
    """
    adata = ad.read_h5ad(train_h5ad_path, backed="r")
    obs   = adata.obs[["cell_type", "individual"]].copy()
    adata.file.close()

    stats = {}
    for ct, group in obs.groupby("cell_type"):
        donor_counts = group["individual"].value_counts()
        stats[str(ct)] = {
            "n_cells": int(len(group)),
            "k_max":   int(donor_counts.max()),
            "k_mean":  float(donor_counts.mean()),
        }
    return stats


# ---------------------------------------------------------------------------
# Minimal config object (mirrors what run_experiment.py creates)
# ---------------------------------------------------------------------------

def _ensure_h5ad_splits(trial_dir: str) -> None:
    """
    If datasets/train.h5ad or datasets/holdout.h5ad are missing, generate them from
    the full dataset filtered to the donor IDs in train.npy / holdout.npy, applying
    the HVG mask.  This is a one-time cost; the files are cached afterward.
    """
    datasets_dir = os.path.join(trial_dir, "datasets")
    train_h5  = os.path.join(datasets_dir, "train.h5ad")
    holdout_h5 = os.path.join(datasets_dir, "holdout.h5ad")

    if os.path.exists(train_h5) and os.path.exists(holdout_h5):
        return

    print("datasets/train.h5ad or holdout.h5ad missing — generating from full dataset...",
          flush=True)

    # Donor splits live in the shared splits/ dir; find it by walking up to the
    # dataset root (the ancestor dir that contains full_dataset_cleaned.h5ad)
    nd_tag = os.path.basename(os.path.dirname(trial_dir))   # e.g. "50d"
    trial  = os.path.basename(trial_dir)                     # e.g. "1"
    dataset_root = _find_dataset_root(trial_dir)
    splits_dir   = os.path.join(dataset_root, "splits", nd_tag, trial)
    train_npy    = os.path.join(splits_dir, "train.npy")
    holdout_npy  = os.path.join(splits_dir, "holdout.npy")
    if not os.path.exists(train_npy) or not os.path.exists(holdout_npy):
        raise FileNotFoundError(
            f"Cannot create H5AD splits: missing {train_npy} or {holdout_npy}"
        )

    train_donors   = np.load(train_npy,   allow_pickle=True)
    holdout_donors = np.load(holdout_npy, allow_pickle=True)

    full_path = None
    for rel in ["../../full_dataset_cleaned.h5ad", "../../../full_dataset_cleaned.h5ad",
                "../../../../full_dataset_cleaned.h5ad", "../../../../../full_dataset_cleaned.h5ad"]:
        candidate = os.path.normpath(os.path.join(trial_dir, rel))
        if os.path.exists(candidate):
            full_path = candidate
            break
    if full_path is None:
        raise FileNotFoundError(
            f"Could not find full_dataset_cleaned.h5ad relative to {trial_dir}"
        )

    # Apply HVG filter
    hvg_path = _find_hvg_path(trial_dir)
    hvg_df   = pd.read_csv(hvg_path)
    hvgs     = hvg_df[hvg_df["highly_variable"]].iloc[:, 0].values

    print(f"  Loading {full_path} ...", flush=True)
    full = ad.read_h5ad(full_path, backed="r")

    hvg_mask = full.var_names.isin(hvgs)

    if not os.path.exists(train_h5):
        print(f"  Writing {train_h5} ...", flush=True)
        subset = full[full.obs["individual"].isin(train_donors)].to_memory()
        subset[:, hvg_mask].write_h5ad(train_h5)

    if not os.path.exists(holdout_h5):
        print(f"  Writing {holdout_h5} ...", flush=True)
        subset = full[full.obs["individual"].isin(holdout_donors)].to_memory()
        subset[:, hvg_mask].write_h5ad(holdout_h5)

    full.file.close()
    print("  Done.", flush=True)


def make_minimal_cfg(trial_dir: str, use_aux: bool) -> Box:
    cfg = Box()
    cfg.trial_dir            = trial_dir
    cfg.models_path          = os.path.join(trial_dir, "models")
    cfg.aux_artifacts_path   = os.path.join(trial_dir, "artifacts", "aux")
    cfg.train_path           = os.path.join(trial_dir, "datasets", "train.h5ad")
    cfg.holdout_path         = os.path.join(trial_dir, "datasets", "holdout.h5ad")

    cfg.mamamia_params                    = Box()
    cfg.mamamia_params.epsilon            = 1e-4   # smoothing ε (not DP ε)
    cfg.mamamia_params.class_b_gene_set   = "secondary"
    cfg.mamamia_params.class_b_scoring    = "llr"
    cfg.mamamia_params.class_b_gamma      = "auto"

    cfg.lin_alg_inverse_fn    = pinv_gpu
    cfg.uniform_remapping_fn  = zinb_cdf

    # Threat model flags — WB+aux by default for the DP sweep
    cfg.mia_setting           = Box()
    cfg.mia_setting.white_box  = True
    cfg.mia_setting.use_wb_hvgs = True
    cfg.mia_setting.use_aux    = use_aux

    return cfg


def _find_dataset_root(trial_dir: str) -> str:
    """Walk up from trial_dir until we find full_dataset_cleaned.h5ad."""
    d = os.path.abspath(trial_dir)
    for _ in range(8):
        d = os.path.dirname(d)
        if os.path.exists(os.path.join(d, "full_dataset_cleaned.h5ad")):
            return d
    raise FileNotFoundError(
        f"Could not find dataset root (full_dataset_cleaned.h5ad) above {trial_dir}"
    )


def _find_hvg_path(trial_dir: str) -> str:
    """
    Locate the HVG CSV for this trial.  Tries several locations to handle
    both the current trial structure (trial_dir/models/hvg.csv) and the
    legacy flat structure.
    """
    candidates = [
        os.path.join(trial_dir, "models", "hvg.csv"),          # legacy flat
        os.path.join(trial_dir, "artifacts", "hvg.csv"),        # newer layout
        os.path.join(trial_dir, "..", "..", "hvg.csv"),         # dataset-level
        os.path.join(trial_dir, "..", "hvg.csv"),
    ]
    for p in candidates:
        resolved = os.path.normpath(p)
        if os.path.exists(resolved):
            return resolved
    raise FileNotFoundError(
        f"Could not find hvg.csv for trial at {trial_dir}. Tried: {candidates}"
    )


# ---------------------------------------------------------------------------
# Single epsilon × seed run
# ---------------------------------------------------------------------------

def _build_targets(cell_type: str, hvgs, train_h5, holdout_h5) -> pd.DataFrame:
    train   = train_h5[train_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    holdout = holdout_h5[holdout_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    train["member"]   = True
    holdout["member"] = False
    train["individual"]   = train_h5.obs["individual"]
    holdout["individual"] = holdout_h5.obs["individual"]
    for col in ["ethnicity", "age", "sex"]:
        train[col]   = train_h5.obs[col]   if col in train_h5.obs.columns   else None
        holdout[col] = holdout_h5.obs[col] if col in holdout_h5.obs.columns else None
    return pd.concat([train, holdout])


def run_one_epsilon(
    cfg: Box,
    cell_types: list[str],
    donor_stats: dict,
    epsilon: float,
    delta: float,
    clip_value: float,
    seed: int,
    use_aux: bool,
) -> tuple[float, dict]:
    """
    Attack all cell types with a fixed (ε, δ, seed) and return (overall_auc, cell_type_aucs).
    """
    rng = np.random.default_rng(seed)

    hvg_path = _find_hvg_path(cfg.trial_dir)
    hvg_df   = pd.read_csv(hvg_path)
    hvgs   = hvg_df[hvg_df["highly_variable"]].iloc[:, 0].values

    train_adata   = ad.read_h5ad(cfg.train_path)
    holdout_adata = ad.read_h5ad(cfg.holdout_path)

    attack_fn = attack_mahalanobis_dp if use_aux else attack_mahalanobis_no_aux_dp

    results = []
    for ct in cell_types:
        synth_rds = os.path.join(cfg.models_path, f"{ct}.rds")
        aux_rds   = os.path.join(cfg.aux_artifacts_path, f"{ct}.rds")

        if not os.path.exists(synth_rds):
            print(f"  [SKIP] no synth copula for {ct}", flush=True)
            continue
        if use_aux and not os.path.exists(aux_rds):
            print(f"  [SKIP] no aux copula for {ct}", flush=True)
            continue

        copula_synth_r = R["readRDS"](synth_rds).rx2(str(ct))
        copula_aux_r   = R["readRDS"](aux_rds).rx2(str(ct)) if use_aux else None

        ct_stats = donor_stats.get(ct, {})
        n_cells  = ct_stats.get("n_cells", 1000)
        k_max    = ct_stats.get("k_max", 10)

        dp_params = {
            "epsilon":    epsilon,
            "delta":      delta,
            "n_cells":    n_cells,
            "k_max":      k_max,
            "clip_value": clip_value,
            "rng":        rng,
        }

        targets = _build_targets(ct, hvgs, train_adata, holdout_adata)

        t0 = time.process_time()
        try:
            result_df = attack_fn(cfg, targets, ct,
                                   copula_synth_r, copula_aux_r=copula_aux_r,
                                   dp_params=dp_params)
            runtime = time.process_time() - t0
            results.append((ct, result_df, runtime))
        except Exception as exc:
            print(f"  [ERROR] {ct}: {exc}", flush=True)
            results.append((ct, None, None))

    train_adata.file.close() if hasattr(train_adata, 'file') else None
    holdout_adata.file.close() if hasattr(holdout_adata, 'file') else None

    all_scores_df, _ = merge_cell_type_results(cfg, results)
    true_labels, predictions = aggregate_scores_by_donor(cfg, all_scores_df)
    overall_auc = roc_auc_score(true_labels, predictions)

    # Per-cell-type AUC for diagnostics
    from attacks.scmamamia.scoring import _threat_model_code
    tm = _threat_model_code(cfg)
    score_col = "score:" + tm
    ct_aucs = {}
    for ct, df, _ in results:
        if df is not None and len(df) > 0 and df["membership"].nunique() == 2:
            ct_df    = df.groupby("donor id")[score_col].mean()
            label_df = df.groupby("donor id")["membership"].first()
            try:
                ct_aucs[ct] = float(roc_auc_score(label_df, ct_df))
            except Exception:
                ct_aucs[ct] = float("nan")

    return overall_auc, ct_aucs


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    trial_dir: str,
    epsilons: list[float],
    delta: float,
    clip_value: float,
    seeds: list[int],
    use_aux: bool,
):
    print(f"DP sweep — trial: {trial_dir}")
    print(f"  epsilons:  {epsilons}")
    print(f"  delta:     {delta}")
    print(f"  clip:      {clip_value}")
    print(f"  seeds:     {seeds}")
    print(f"  use_aux:   {use_aux}")
    print(f"  device:    {DEVICE}\n", flush=True)

    cfg = make_minimal_cfg(trial_dir, use_aux)
    _train_h5_preexisted   = os.path.exists(cfg.train_path)
    _holdout_h5_preexisted = os.path.exists(cfg.holdout_path)
    _ensure_h5ad_splits(trial_dir)

    # Cell types
    _meta = ad.read_h5ad(cfg.train_path, backed="r")
    cell_types = list(_meta.obs["cell_type"].unique())
    _meta.file.close()
    print(f"Cell types ({len(cell_types)}): {cell_types}", flush=True)

    # Donor cell-count stats for sensitivity computation
    donor_stats = compute_donor_cell_counts(cfg.train_path)

    # Sensitivity report (printed once, saved to JSON)
    sens_report = {}
    for ct, st in donor_stats.items():
        # Use a representative gene count (approximate; actual G varies per cell type)
        n_genes_approx = 400
        sens_report[ct] = sensitivity_report(
            n_cells=st["n_cells"],
            k_max=st["k_max"],
            n_genes=n_genes_approx,
            clip_value=clip_value,
            epsilons=epsilons,
            delta=delta,
        )
    sens_path = os.path.join(trial_dir, "results", "dp_sensitivity_report.json")
    os.makedirs(os.path.join(trial_dir, "results"), exist_ok=True)
    with open(sens_path, "w") as f:
        json.dump(sens_report, f, indent=2)
    print(f"Sensitivity report saved to {sens_path}", flush=True)

    # Baseline (ε → ∞, i.e. no noise) — run once with seed=seeds[0]
    print("\n[Baseline: no DP noise]", flush=True)
    baseline_auc, _ = run_one_epsilon(
        cfg, cell_types, donor_stats,
        epsilon=1e9, delta=delta, clip_value=clip_value,
        seed=seeds[0], use_aux=use_aux,
    )
    print(f"  Baseline AUC: {baseline_auc:.4f}", flush=True)

    # Sweep
    rows = []
    for eps in epsilons:
        for seed in seeds:
            print(f"\n[ε={eps}, seed={seed}]", flush=True)
            auc, ct_aucs = run_one_epsilon(
                cfg, cell_types, donor_stats,
                epsilon=eps, delta=delta, clip_value=clip_value,
                seed=seed, use_aux=use_aux,
            )
            print(f"  AUC: {auc:.4f}", flush=True)
            row = {
                "epsilon":      eps,
                "delta":        delta,
                "clip_value":   clip_value,
                "seed":         seed,
                "auc":          auc,
                "baseline_auc": baseline_auc,
                "use_aux":      use_aux,
            }
            row.update({f"auc_{ct}": v for ct, v in ct_aucs.items()})
            rows.append(row)

    results_df = pd.DataFrame(rows)
    out_path = os.path.join(trial_dir, "results", "dp_sweep_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSweep complete.  Results saved to {out_path}")

    # Summary table
    summary = (results_df.groupby("epsilon")["auc"]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "auc_mean", "std": "auc_std", "count": "n_seeds"}))
    summary["auc_vs_baseline"] = summary["auc_mean"] - baseline_auc
    summary["privacy_reduction_pct"] = (
        (baseline_auc - summary["auc_mean"]) / (baseline_auc - 0.5) * 100
    ).clip(0, 100)

    n_donors = os.path.basename(os.path.dirname(trial_dir)).replace("d", "")
    trial_id = os.path.basename(trial_dir)
    print(f"\n{'='*65}")
    print(f"  RESULTS — {n_donors} donors, trial {trial_id}, c={clip_value}, δ={delta}")
    print(f"  Baseline AUC (no DP): {baseline_auc:.4f}   (random chance = 0.5000)")
    print(f"{'='*65}")
    print(f"  {'epsilon':>10}  {'AUC mean':>9}  {'± std':>7}  {'vs base':>8}  {'attack weakened':>15}")
    print(f"  {'-'*60}")
    for eps, row in summary.iterrows():
        print(f"  {eps:>10.0f}  {row['auc_mean']:>9.4f}  "
              f"±{row['auc_std']:>6.4f}  {row['auc_vs_baseline']:>+8.4f}  "
              f"{row['privacy_reduction_pct']:>13.1f}%")
    print(f"{'='*65}", flush=True)

    # Clean up H5AD splits we generated (donor IDs are preserved in .npy files)
    for path, preexisted in [
        (cfg.train_path,   _train_h5_preexisted),
        (cfg.holdout_path, _holdout_h5_preexisted),
    ]:
        if not preexisted and os.path.exists(path):
            os.remove(path)
            print(f"Removed temporary {path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DP sweep for scMAMA-MIA")
    p.add_argument("trial_dir",
                   help="Path to a completed trial directory (produced by run_experiment.py)")
    p.add_argument("--epsilons", nargs="+", type=float,
                   default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    p.add_argument("--delta",    type=float, default=1e-5)
    p.add_argument("--clip",     type=float, default=3.0,
                   help="Quantile-normal clipping bound c (default 3.0)")
    p.add_argument("--seeds",    nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--no-aux",   action="store_true",
                   help="Use the no-auxiliary-data attack variant")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        trial_dir=args.trial_dir,
        epsilons=args.epsilons,
        delta=args.delta,
        clip_value=args.clip,
        seeds=args.seeds,
        use_aux=not args.no_aux,
    )
