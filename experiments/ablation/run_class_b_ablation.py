"""
Ablation study for the Class B (secondary gene LLR) extension to scMAMA-MIA.

Goals
-----
1. Identify the best combination of (gene_set, scoring_method, gamma) for the
   secondary gene Class B extension.
2. Confirm the improvement holds on true scDesign2 targets (ok, aida, cg).
3. Verify no regression on proxy-model targets (ok_scvi, ok_sd3g, ok_sd3v, ok_scdiff).

Key design feature: the ablation re-uses existing artefacts (copula .rds files and
donor .npy arrays) without regenerating synthetic data.  Per-cell-type gene LLR
vectors are computed ONCE per trial and then combined with each gamma variant in O(1),
so the ablation is fast even with dozens of variants.

Results are saved to experiments/ablation/results/<dataset>/<nd>d_trial<t>.csv and
a summary CSV at experiments/ablation/results/summary.csv.

Usage
-----
  # Full ablation (may take ~1-2 hours)
  python experiments/ablation/run_class_b_ablation.py

  # Only SD2 target datasets, 10d only
  python experiments/ablation/run_class_b_ablation.py --dataset ok aida cg --nd 10

  # Dry run: just print what would be evaluated
  python experiments/ablation/run_class_b_ablation.py --dry-run

  # Quick smoke test (one trial per dataset)
  python experiments/ablation/run_class_b_ablation.py --max-trials 1
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import roc_auc_score

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC  = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sdg.scdesign2.copula import parse_copula, build_shared_covariance_matrix, get_shared_genes
from data.cdf_utils import zinb_log_pmf, activate_from_logits
from attacks.scmamamia.attack_b import _mahalanobis_distances


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

DATA_DIR   = "/home/golobs/data/scMAMAMIA"
RESULT_DIR = os.path.join(_REPO, "experiments", "ablation", "results")
N_TRIALS   = 5

# Ablation variants: (name, gene_set, scoring_method, gamma)
# gamma=0.0 → pure Mahalanobis baseline
ABLATION_VARIANTS = [
    # --- Baseline (pure Mahalanobis, no Class B) ---
    ("baseline",            "secondary",  "llr",        0.0),
    # --- LLR, secondary genes, fixed gamma sweep ---
    ("llr_sec_g0.01",       "secondary",  "llr",        0.01),
    ("llr_sec_g0.05",       "secondary",  "llr",        0.05),
    ("llr_sec_g0.1",        "secondary",  "llr",        0.1),
    ("llr_sec_g0.2",        "secondary",  "llr",        0.2),
    ("llr_sec_g0.3",        "secondary",  "llr",        0.3),
    ("llr_sec_g0.5",        "secondary",  "llr",        0.5),
    ("llr_sec_g1.0",        "secondary",  "llr",        1.0),
    ("llr_sec_g2.0",        "secondary",  "llr",        2.0),
    # --- LLR, secondary genes, auto-normalised gamma ---
    ("llr_sec_auto",        "secondary",  "llr",        "auto"),
    # --- LLR, all genes (primary+secondary), selected gammas ---
    ("llr_all_g0.1",        "all",        "llr",        0.1),
    ("llr_all_g0.3",        "all",        "llr",        0.3),
    ("llr_all_auto",        "all",        "llr",        "auto"),
    # --- Diagonal Mahalanobis alternative scoring ---
    ("diag_sec_g0.1",       "secondary",  "diag_mahal", 0.1),
    ("diag_sec_g0.3",       "secondary",  "diag_mahal", 0.3),
    ("diag_sec_auto",       "secondary",  "diag_mahal", "auto"),
    ("diag_all_g0.1",       "all",        "diag_mahal", 0.1),
    ("diag_all_auto",       "all",        "diag_mahal", "auto"),
]

# Datasets for primary ablation (true scDesign2 targets)
SD2_TARGETS = [
    # (dataset_name, nd, hvg_path_rel_to_base_dataset)
    ("ok/scdesign2/no_dp",   10, "hvg.csv"),
    ("ok/scdesign2/no_dp",   50, "hvg.csv"),
    ("aida/scdesign2/no_dp", 10, "hvg.csv"),
    ("aida/scdesign2/no_dp", 50, "hvg.csv"),
    ("cg/scdesign2/no_dp",   10, "hvg.csv"),
]

# Datasets for regression sanity check (proxy-model targets)
# Use only BB+aux for regression (tm:100), first 3 trials only
NON_SD2_REGRESSION = [
    ("ok/scvi/no_dp",         10, "ok"),
    ("ok/scdesign3/gaussian", 10, "ok"),
    ("ok/scdesign3/vine",     10, "ok"),
    ("ok/scdiffusion/no_dp",  10, "ok"),
]

EPS = 1e-4   # matches existing mamamia_params.epsilon


# ---------------------------------------------------------------------------
# Core ablation: one cell type in one trial
# ---------------------------------------------------------------------------

def _load_copula(rds_path, cell_type_name):
    from rpy2.robjects import r as R
    obj = R["readRDS"](rds_path).rx2(str(cell_type_name))
    return parse_copula(obj)


def _build_targets(train_donors, holdout_donors, full_h5ad_path, hvg_path, cell_type=None):
    """
    Reconstruct the targets DataFrame for a trial from the full dataset and
    the pre-saved donor arrays.

    Loads ALL genes (not just HVGs) so that secondary copula genes outside the
    strict HVG list are also available for the Class B attack.

    Returns a DataFrame with 'individual', 'member', 'cell_type' columns plus
    one column per gene.
    """
    train_set  = set(train_donors)
    target_set = train_set | set(holdout_donors)

    adata = ad.read_h5ad(full_h5ad_path, backed="r")
    obs   = adata.obs

    if cell_type is not None:
        mask = obs["individual"].isin(target_set) & (obs["cell_type"] == cell_type)
    else:
        mask = obs["individual"].isin(target_set)

    subset = adata[mask].to_memory()
    adata.file.close()

    df = subset.to_df()
    df["individual"] = subset.obs["individual"].values
    df["cell_type"]  = subset.obs["cell_type"].values
    df["member"]     = subset.obs["individual"].isin(train_set).values
    return df


# ---------------------------------------------------------------------------
# Pre-compute per-gene LLR and diag-Mahal scores for all genes in both copulas
# ---------------------------------------------------------------------------

def _precompute_gene_scores(targets_ct, cs, ca, gene_list, remap_fn):
    """
    For each gene in gene_list, compute:
      - llr[gene]: log(p_synth) - log(p_aux) per cell   (N,)
      - diag_s[gene]: squared deviation from 0.5 in synth uniform space
      - diag_a[gene]: squared deviation from 0.5 in aux uniform space

    Returns dicts keyed by gene name.
    """
    llr    = {}
    diag_s = {}
    diag_a = {}

    for gene in gene_list:
        if gene not in targets_ct.columns:
            continue
        x_g = targets_ct[gene].values.astype(int)
        pi_s, theta_s, mu_s = cs["get_gene_params"](gene)
        pi_a, theta_a, mu_a = ca["get_gene_params"](gene)

        llr[gene] = (zinb_log_pmf(x_g, pi_s, theta_s, mu_s)
                     - zinb_log_pmf(x_g, pi_a, theta_a, mu_a))

        u_s = remap_fn(x_g, pi_s, theta_s, mu_s) - 0.5
        u_a = remap_fn(x_g, pi_a, theta_a, mu_a) - 0.5
        diag_s[gene] = u_s ** 2
        diag_a[gene] = u_a ** 2

    return llr, diag_s, diag_a


def _precompute_gene_logprob_noaux(targets_ct, cs, gene_list):
    """Per-gene log p_synth for no-aux variant."""
    log_p = {}
    for gene in gene_list:
        if gene not in targets_ct.columns:
            continue
        x_g = targets_ct[gene].values.astype(int)
        pi_s, theta_s, mu_s = cs["get_gene_params"](gene)
        log_p[gene] = zinb_log_pmf(x_g, pi_s, theta_s, mu_s)
    return log_p


# ---------------------------------------------------------------------------
# Evaluate one variant for one cell type
# ---------------------------------------------------------------------------

def _evaluate_variant_ct(
    variant_name, gene_set, scoring, gamma,
    d_s, d_a,
    targets_ct,
    covariate_genes, all_genes,
    llr, diag_s, diag_a,
):
    """
    Return (variant_name, per_cell_scores_aux, per_cell_scores_noaux) for one cell type.
    Raises if copula has no covariance matrix (caller must handle vine fallback).
    """
    if gene_set == "secondary":
        covariate_set = set(covariate_genes)
        b_genes = [g for g in all_genes if g not in covariate_set and g in llr]
    else:
        b_genes = [g for g in all_genes if g in llr]

    n_b = len(b_genes)

    # Baseline: pure Mahalanobis
    log_primary_aux   = np.log(d_a + EPS) - np.log(d_s + EPS)
    log_primary_noaux = -np.log(d_s + EPS)

    if gamma == 0.0 or n_b == 0:
        return log_primary_aux, log_primary_noaux

    gamma_eff = 1.0 / np.sqrt(n_b) if gamma == "auto" else float(gamma)

    if scoring == "llr":
        log_b = sum(llr[g] for g in b_genes)
        # For no-aux, use only the synth log-prob (no ratio possible)
        log_b_noaux = sum(zinb_log_pmf(targets_ct[g].values.astype(int),
                                       *[v for v in [llr[g]]][0:0],   # placeholder
                                       ) for g in b_genes)
        # Actually recompute from llr: llr[g] = log_p_s - log_p_a; we need just log_p_s
        # That requires knowing log_p_s separately — use precomputed diag_s as proxy? No.
        # We'll pass log_b_noaux via the precomputed log_p dict (done in outer caller).
        log_b_noaux = None   # sentinel: outer caller supplies this

    elif scoring == "diag_mahal":
        d_sq_s_b = sum(diag_s[g] for g in b_genes)
        d_sq_a_b = sum(diag_a[g] for g in b_genes)
        d_b_s = np.sqrt(d_sq_s_b * 12.0)
        d_b_a = np.sqrt(d_sq_a_b * 12.0)
        log_b = np.log(d_b_a + EPS) - np.log(d_b_s + EPS)
        log_b_noaux = -np.log(d_b_s + EPS)   # no-aux: just how far from synth

    else:
        raise ValueError(scoring)

    logit_aux   = log_primary_aux   + gamma_eff * log_b
    if log_b_noaux is not None:
        logit_noaux = log_primary_noaux + gamma_eff * log_b_noaux
    else:
        logit_noaux = None   # caller will fill in

    return logit_aux, logit_noaux


# ---------------------------------------------------------------------------
# Run ablation for one full trial
# ---------------------------------------------------------------------------

def run_one_trial(dataset_name, nd, trial, hvg_relpath, base_dataset=None, remap_fn=None):
    """
    Run all ablation variants for one (dataset, nd, trial) combination.

    Returns a dict: variant_name -> {"auc_aux": float, "auc_noaux": float}
    Returns None if artefacts are missing.
    """
    from scipy.stats import zscore as _zscore
    from data.cdf_utils import zinb_cdf

    if remap_fn is None:
        remap_fn = zinb_cdf

    # base_dataset_name: first path component (e.g. 'ok' from 'ok/scdesign2/no_dp')
    base_dataset_name = (base_dataset or dataset_name).split('/')[0]
    data_dir      = os.path.join(DATA_DIR, *dataset_name.split("/"))
    base_data_dir = os.path.join(DATA_DIR, base_dataset_name)
    trial_dir     = os.path.join(data_dir, f"{nd}d", str(trial))
    splits_dir    = os.path.join(base_data_dir, "splits", f"{nd}d", str(trial))
    synth_rds_dir = os.path.join(trial_dir, "artifacts", "synth")
    aux_rds_dir   = os.path.join(trial_dir, "artifacts", "aux")

    train_npy   = os.path.join(splits_dir, "train.npy")
    holdout_npy = os.path.join(splits_dir, "holdout.npy")
    full_h5ad   = os.path.join(base_data_dir, "full_dataset_cleaned.h5ad")
    hvg_path    = os.path.join(base_data_dir, hvg_relpath)

    for path in [train_npy, holdout_npy, full_h5ad, hvg_path, synth_rds_dir, aux_rds_dir]:
        if not os.path.exists(path):
            print(f"  [SKIP] missing: {path}")
            return None

    train_donors   = np.load(train_npy,   allow_pickle=True)
    holdout_donors = np.load(holdout_npy, allow_pickle=True)

    cell_types = [
        os.path.splitext(f)[0]
        for f in os.listdir(synth_rds_dir) if f.endswith(".rds")
    ]
    if not cell_types:
        return None

    # Accumulate per-cell logits across cell types, then compute donor AUC
    # Shape: dict[variant_name] -> (logits_aux, logits_noaux, donor_ids, membership)
    accum_aux   = {v[0]: [] for v in ABLATION_VARIANTS}
    accum_noaux = {v[0]: [] for v in ABLATION_VARIANTS}
    accum_donor = {v[0]: [] for v in ABLATION_VARIANTS}
    accum_memb  = {v[0]: [] for v in ABLATION_VARIANTS}

    # A simple config-like namespace that attack helpers need
    class _Cfg:
        class mamamia_params:
            epsilon = EPS
        uniform_remapping_fn = staticmethod(remap_fn)
        lin_alg_inverse_fn   = staticmethod(__import__("numpy.linalg", fromlist=["pinv"]).pinv)

    fake_cfg = _Cfg()

    for ct in cell_types:
        synth_rds = os.path.join(synth_rds_dir, f"{ct}.rds")
        aux_rds   = os.path.join(aux_rds_dir,   f"{ct}.rds")
        if not (os.path.exists(synth_rds) and os.path.exists(aux_rds)):
            continue

        try:
            cs = _load_copula(synth_rds, ct)
            ca = _load_copula(aux_rds,   ct)
        except Exception as e:
            print(f"  [WARN] copula load failed for {ct}: {e}")
            continue

        if cs.get("copula_type") == "vine" or cs.get("cov_matrix") is None:
            continue   # skip vine copulas in ablation

        # Build targets DataFrame for this cell type
        try:
            targets_full = _build_targets(
                train_donors, holdout_donors, full_h5ad, hvg_path, cell_type=ct
            )
        except Exception as e:
            print(f"  [WARN] target build failed for {ct}: {e}")
            continue

        if len(targets_full) < 4:
            continue

        # Mahalanobis distances (shared across all variants)
        _, d_s, d_a = _mahalanobis_distances(fake_cfg, targets_full, cs, ca)
        if d_s is None:
            continue

        # Shared gene sets
        covariate_genes, all_genes = get_shared_genes(
            cs["primary_genes"], cs["secondary_genes"],
            ca["primary_genes"], ca["secondary_genes"],
        )

        # Pre-compute per-gene LLR and diag scores (one pass, reused by all variants)
        llr, diag_s_g, diag_a_g = _precompute_gene_scores(
            targets_full, cs, ca, all_genes, remap_fn
        )
        # Also pre-compute log_p_s for no-aux LLR
        log_p_s_g = _precompute_gene_logprob_noaux(targets_full, cs, all_genes)

        donor_ids  = targets_full["individual"].values
        membership = targets_full["member"].values.astype(int)

        for vname, gene_set, scoring, gamma in ABLATION_VARIANTS:
            if gene_set == "secondary":
                covariate_set = set(covariate_genes)
                b_genes = [g for g in all_genes if g not in covariate_set and g in llr]
            else:
                b_genes = [g for g in all_genes if g in llr]

            n_b = len(b_genes)
            gamma_eff = 1.0 / np.sqrt(n_b) if (gamma == "auto" and n_b > 0) else (
                float(gamma) if gamma != "auto" else 0.0
            )

            log_primary_aux   = np.log(d_a + EPS) - np.log(d_s + EPS)
            log_primary_noaux = -np.log(d_s + EPS)

            if gamma_eff == 0.0 or n_b == 0:
                logit_aux   = log_primary_aux
                logit_noaux = log_primary_noaux
            elif scoring == "llr":
                log_b_aux   = sum(llr[g]       for g in b_genes)
                log_b_noaux = sum(log_p_s_g[g] for g in b_genes)
                logit_aux   = log_primary_aux   + gamma_eff * log_b_aux
                logit_noaux = log_primary_noaux + gamma_eff * log_b_noaux
            elif scoring == "diag_mahal":
                d_sq_s_b = sum(diag_s_g[g] for g in b_genes)
                d_sq_a_b = sum(diag_a_g[g] for g in b_genes)
                d_b_s = np.sqrt(d_sq_s_b * 12.0)
                d_b_a = np.sqrt(d_sq_a_b * 12.0)
                log_b_aux   = np.log(d_b_a + EPS) - np.log(d_b_s + EPS)
                log_b_noaux = -np.log(d_b_s + EPS)
                logit_aux   = log_primary_aux   + gamma_eff * log_b_aux
                logit_noaux = log_primary_noaux + gamma_eff * log_b_noaux

            # Replace non-finite values with 0 (neutral)
            for arr in [logit_aux, logit_noaux]:
                bad = ~np.isfinite(arr)
                if bad.any():
                    arr[bad] = 0.0

            accum_aux[vname].extend(logit_aux.tolist())
            accum_noaux[vname].extend(logit_noaux.tolist())
            accum_donor[vname].extend(donor_ids.tolist())
            accum_memb[vname].extend(membership.tolist())

    # Donor-level AUC for each variant
    results = {}
    for vname, _, _, _ in ABLATION_VARIANTS:
        logits_aux   = np.array(accum_aux[vname])
        logits_noaux = np.array(accum_noaux[vname])
        donors       = np.array(accum_donor[vname])
        memb         = np.array(accum_memb[vname])

        if len(logits_aux) == 0:
            results[vname] = {"auc_aux": float("nan"), "auc_noaux": float("nan"),
                              "n_cells": 0}
            continue

        auc_aux   = _donor_auc(activate_from_logits(logits_aux),   donors, memb)
        auc_noaux = _donor_auc(activate_from_logits(logits_noaux), donors, memb)
        results[vname] = {"auc_aux": auc_aux, "auc_noaux": auc_noaux,
                          "n_cells": len(logits_aux)}

    return results


def _donor_auc(scores, donor_ids, membership):
    """Aggregate cell scores to donor level and compute ROC AUC."""
    df = pd.DataFrame({"donor": donor_ids, "score": scores, "member": membership})
    donor_df = df.groupby("donor", observed=True).agg(
        score=("score", "mean"),
        member=("member", "mean"),
    )
    donor_df = donor_df[donor_df["member"].isin([0.0, 1.0])]
    if donor_df["member"].nunique() < 2 or len(donor_df) < 4:
        return float("nan")
    try:
        return roc_auc_score(donor_df["member"].astype(int), donor_df["score"])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",     nargs="*", default=None,
                        help="Dataset names to include (default: all SD2 + regression targets)")
    parser.add_argument("--nd",          nargs="*", type=int, default=None,
                        help="Donor counts to evaluate (default: all defined per dataset)")
    parser.add_argument("--max-trials",  type=int, default=0,
                        help="Max trials per (dataset, nd) — 0 = unlimited")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print jobs without running them")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    all_rows = []

    def _run_target(dataset_name, nd, trial_max, hvg_relpath, base_dataset=None, label="SD2"):
        tag = f"[{label}] {dataset_name} {nd}d"
        data_dir  = os.path.join(DATA_DIR, *dataset_name.split("/"))
        nd_dir    = os.path.join(data_dir, f"{nd}d")
        if not os.path.isdir(nd_dir):
            return

        n_trials = min(trial_max if trial_max > 0 else N_TRIALS, N_TRIALS)
        for trial in range(1, n_trials + 1):
            if args.dry_run:
                print(f"  [DRY-RUN] {tag} trial {trial}")
                return

            print(f"\n{tag} trial {trial} ...", flush=True)
            t0 = time.time()
            res = run_one_trial(dataset_name, nd, trial, hvg_relpath,
                                base_dataset=base_dataset)
            if res is None:
                print(f"  [SKIP]")
                continue
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.0f}s  baseline AUC(BB+aux)="
                  f"{res.get('baseline', {}).get('auc_aux', float('nan')):.3f}")

            for vname, row in res.items():
                all_rows.append({
                    "dataset":  dataset_name,
                    "nd":       nd,
                    "trial":    trial,
                    "variant":  vname,
                    "auc_aux":  row["auc_aux"],
                    "auc_noaux": row["auc_noaux"],
                    "n_cells":  row["n_cells"],
                    "label":    label,
                })

            # Save incrementally
            _save_summary(all_rows)

    # SD2 primary targets
    for dataset_name, nd, hvg_relpath in SD2_TARGETS:
        if args.dataset and dataset_name not in args.dataset:
            continue
        if args.nd and nd not in args.nd:
            continue
        _run_target(dataset_name, nd, args.max_trials, hvg_relpath, label="SD2")

    # Non-SD2 regression checks (max 3 trials)
    for dataset_name, nd, base_ds in NON_SD2_REGRESSION:
        if args.dataset and dataset_name not in args.dataset:
            continue
        if args.nd and nd not in args.nd:
            continue
        _run_target(dataset_name, nd, min(args.max_trials or 3, 3), "hvg.csv",
                    base_dataset=base_ds, label="non-SD2")

    if args.dry_run:
        return

    _save_summary(all_rows)
    print(f"\nAblation complete. Results at {RESULT_DIR}/summary.csv")


def _save_summary(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = os.path.join(RESULT_DIR, "summary.csv")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
