#!/usr/bin/env python3
"""
experiments/playground/run_playground.py
═════════════════════════════════════════

End-to-end mini-demo of the full scMAMA-MIA pipeline, refreshing how all the
pieces fit together after the repo refactor.

What it does
------------
1. Loads the OneK1K dataset and carves out a tiny slice:
       N_TRAIN  donors  → training data for the SDG
       N_HOLDOUT donors → held-out non-members (attack targets)
       N_AUX    donors  → auxiliary dataset (separate pool, for BB+aux attack)
   Cells are capped at MAX_CELLS_PER_DONOR_CT per (donor, cell type) for speed.
   Only N_GENES HVGs are kept.

2. Runs the full BB+aux threat-model pipeline for BOTH scDesign2 and scDesign3
   (Gaussian copula):
       a. Train target SDG on train data, generate synthetic data
       b. Train synth shadow model on synthetic data   (focal-point for BB)
       c. Train aux  shadow model on auxiliary data    (focal-point for aux)

3. Runs every implemented scMAMA-MIA attack variant on the output:
       • Mahalanobis BB+aux   — primary attack (d_aux / (d_synth + d_aux))
       • Mahalanobis BB-aux   — no auxiliary data (1 / (d_synth + ε))

4. Aggregates cell-level scores to donor-level (mean), computes ROC AUC,
   and prints a summary table.

Usage (from repo root):
    conda run -n camda_conda python experiments/playground/run_playground.py

Expected runtime: 8–15 minutes on a laptop (dominated by R subprocess calls).
Note: AUC values on this tiny dataset are not meaningful — the point is to
      verify the pipeline runs end-to-end without error.
"""

import os
import sys
import shutil
import time

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import roc_auc_score
from numpy.linalg import pinv

# ── make src/ importable regardless of CWD ────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from sdg.scdesign2.model  import ScDesign2
from sdg.scdesign3.model  import ScDesign3
from sdg.scdesign2.copula import (
    parse_copula,
    build_shared_covariance_matrix,
    get_shared_genes,
)
from sdg.scdesign3.copula import load_copula_sd3
from data.cdf_utils       import zinb_cdf


# ═══════════════════════════════════════════════════════════════════════════
# Settings — adjust these to trade off speed vs. richness
# ═══════════════════════════════════════════════════════════════════════════

FULL_DATA_PATH = (
    "/Users/stevengolob/Documents/school/PhD/"
    "Ghent_project_mia_scRNAseq/data/ok/full_dataset_cleaned.h5ad"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

N_TRAIN               = 10   # donors whose cells train the target SDG
N_HOLDOUT             = 10   # donors held out as non-members
N_AUX                 = 10   # donors for the auxiliary shadow model
N_CELL_TYPES          = 2   # use the N_CELL_TYPES most populous cell types
N_GENES               = 1000 # HVGs (more → richer copula, slower training)
MAX_CELLS_PER_DONOR_CT = 100 # cap cells per (donor, cell type) for speed
SEED = 9


# ═══════════════════════════════════════════════════════════════════════════
# Minimal config-dict factories for each generator class
# ═══════════════════════════════════════════════════════════════════════════

def _sd2_config(home_dir, data_dir, model_rel_path, hvg_path, train_file, test_file):
    """Minimal config dict accepted by ScDesign2."""
    return {
        "generator_name": "scdesign2",
        "dir_list":       {"home": home_dir, "data": data_dir},
        "scdesign2_config": {
            "out_model_path": model_rel_path,
            "hvg_path":       hvg_path,
        },
        "dataset_config": {
            "name":                "playground",
            "train_count_file":    train_file,
            "test_count_file":     test_file,
            "cell_type_col_name":  "cell_type",
            "cell_label_col_name": "cell_type",
            "random_seed":         SEED,
        },
    }


def _sd3_config(home_dir, data_dir, model_rel_path, hvg_path, train_file, test_file,
                copula_type="gaussian"):
    """Minimal config dict accepted by ScDesign3."""
    return {
        "generator_name": "scdesign3",
        "dir_list":       {"home": home_dir, "data": data_dir},
        "scdesign3_config": {
            "out_model_path": model_rel_path,
            "hvg_path":       hvg_path,
            "copula_type":    copula_type,
            "family_use":     "nb",
        },
        "dataset_config": {
            "name":                "playground",
            "train_count_file":    train_file,
            "test_count_file":     test_file,
            "cell_type_col_name":  "cell_type",
            "cell_label_col_name": "cell_type",
            "random_seed":         SEED,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Copula loaders — return a unified parsed dict for either generator
# ═══════════════════════════════════════════════════════════════════════════

def _load_sd2_copula(rds_path: str, cell_type: str) -> dict:
    """Load and parse a scDesign2 per-cell-type .rds."""
    from rpy2.robjects import r
    model_r = r["readRDS"](rds_path).rx2(str(cell_type))
    return parse_copula(model_r)


def _load_sd3_copula(rds_path: str, cell_type: str) -> dict:
    """Load and parse a scDesign3 per-cell-type .rds."""
    return load_copula_sd3(rds_path, str(cell_type))


# ═══════════════════════════════════════════════════════════════════════════
# Attack functions — work with parsed copula dicts from either generator
#
# Both scDesign2 (rpy2 objects) and scDesign3 (numpy arrays) produce dicts
# where cov_matrix and primary_marginals are either rpy2 or numpy objects.
# build_shared_covariance_matrix calls np.array() on both, so it handles
# either transparently.
# ═══════════════════════════════════════════════════════════════════════════

def _mahalanobis_bb_aux(synth_cop: dict, aux_cop: dict, targets: pd.DataFrame):
    """
    Mahalanobis BB+aux attack.  Score per cell: λ = d_aux / (d_synth + d_aux).
    Higher λ → cell is closer to synth copula → more likely a training member.
    Returns a 1-D numpy array of cell scores (same row order as `targets`),
    or None if the copulas share too few genes.
    """
    shared, _ = get_shared_genes(
        synth_cop["primary_genes"], synth_cop["secondary_genes"],
        aux_cop["primary_genes"],   aux_cop["secondary_genes"],
    )
    if len(shared) < 2:
        return None

    cov_s, marg_s = build_shared_covariance_matrix(
        shared, synth_cop["primary_genes"],
        synth_cop["cov_matrix"], synth_cop["primary_marginals"],
    )
    cov_a, marg_a = build_shared_covariance_matrix(
        shared, aux_cop["primary_genes"],
        aux_cop["cov_matrix"], aux_cop["primary_marginals"],
    )

    # Ensure all shared genes are in the target DataFrame
    if not all(g in targets.columns for g in shared):
        return None

    remap = np.vectorize(zinb_cdf)
    inv_s = pinv(cov_s)
    inv_a = pinv(cov_a)

    scores = []
    for _, row in targets[shared].iterrows():
        x    = row.values
        u_s  = remap(x,           *np.moveaxis(marg_s, 1, 0))
        u_a  = remap(x,           *np.moveaxis(marg_a, 1, 0))
        mu_s = remap(marg_s[:, 2], *np.moveaxis(marg_s, 1, 0))
        mu_a = remap(marg_a[:, 2], *np.moveaxis(marg_a, 1, 0))
        d_s  = float(np.sqrt((u_s - mu_s) @ inv_s @ (u_s - mu_s)))
        d_a  = float(np.sqrt((u_a - mu_a) @ inv_a @ (u_a - mu_a)))
        lam  = d_a / (d_s + d_a + 1e-10)
        scores.append(lam if not np.isnan(lam) else 0.5)

    return np.array(scores)


def _mahalanobis_bb_no_aux(synth_cop: dict, targets: pd.DataFrame):
    """
    Mahalanobis BB-aux attack (no auxiliary data).
    Score per cell: 1 / (d_synth + ε).  Uses the synth shadow copula as the
    only reference point.
    """
    genes = synth_cop["primary_genes"]
    if len(genes) < 2 or not all(g in targets.columns for g in genes):
        return None

    cov_s, marg_s = build_shared_covariance_matrix(
        genes, genes, synth_cop["cov_matrix"], synth_cop["primary_marginals"],
    )

    remap = np.vectorize(zinb_cdf)
    inv_s = pinv(cov_s)

    scores = []
    for _, row in targets[genes].iterrows():
        x    = row.values
        u_s  = remap(x,           *np.moveaxis(marg_s, 1, 0))
        mu_s = remap(marg_s[:, 2], *np.moveaxis(marg_s, 1, 0))
        d_s  = float(np.sqrt((u_s - mu_s) @ inv_s @ (u_s - mu_s)))
        scores.append(1.0 / (d_s + 1e-4))

    return np.array(scores)


# ═══════════════════════════════════════════════════════════════════════════
# Donor-level AUC
# ═══════════════════════════════════════════════════════════════════════════

def _donor_auc(cell_scores: np.ndarray, individual: pd.Series,
               membership: pd.Series) -> float:
    """Average cell scores per donor, compute ROC AUC at the donor level."""
    df = pd.DataFrame({
        "individual": individual.values,
        "membership": membership.values,
        "score":      cell_scores,
    })
    donor_df = (
        df.groupby("individual")
          .agg(score=("score", "mean"), membership=("membership", "first"))
          .reset_index()
    )
    if donor_df["membership"].nunique() < 2:
        return float("nan")
    return roc_auc_score(donor_df["membership"], donor_df["score"])


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

def _banner(msg: str):
    print(f"\n{'═'*62}")
    print(f"  {msg}")
    print(f"{'═'*62}")


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.0f}s"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)
    t_wall = time.time()

    # ── Clean and recreate output dir ─────────────────────────────────────
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    # ═════════════════════════════════════════════════════════════════════
    _banner("STEP 1 — Data preparation")
    # ═════════════════════════════════════════════════════════════════════

    if not os.path.exists(FULL_DATA_PATH):
        sys.exit(
            f"\nDataset not found:\n  {FULL_DATA_PATH}\n"
            "Update FULL_DATA_PATH at the top of this script."
        )

    print(f"Reading {FULL_DATA_PATH} …")
    full = sc.read_h5ad(FULL_DATA_PATH)
    print(
        f"Full dataset: {full.n_obs:,} cells × {full.n_vars:,} genes, "
        f"{full.obs['individual'].nunique()} donors"
    )

    # Pick the two most populous cell types
    ct_counts  = full.obs["cell_type"].value_counts()
    cell_types = ct_counts.index[:N_CELL_TYPES].tolist()
    print(f"Cell types selected: {cell_types}")
    full = full[full.obs["cell_type"].isin(cell_types)].copy()

    # Sample N_TRAIN + N_HOLDOUT + N_AUX donors (disjoint pools for train/holdout, then aux)
    donors   = full.obs["individual"].unique().tolist()
    n_needed = N_TRAIN + N_HOLDOUT + N_AUX
    if len(donors) < n_needed:
        sys.exit(f"Need ≥ {n_needed} donors; dataset has only {len(donors)}.")

    chosen         = rng.choice(donors, size=n_needed, replace=False)
    train_donors   = chosen[:N_TRAIN].tolist()
    holdout_donors = chosen[N_TRAIN : N_TRAIN + N_HOLDOUT].tolist()
    aux_donors     = chosen[N_TRAIN + N_HOLDOUT :].tolist()

    print(f"Train donors   ({N_TRAIN}):   {train_donors}")
    print(f"Holdout donors ({N_HOLDOUT}): {holdout_donors}")
    print(f"Aux donors     ({N_AUX}):     {aux_donors}")

    mini = full[full.obs["individual"].isin(chosen)].copy()

    # Cap cells per (donor, cell_type) for speed
    keep = []
    for _, grp in mini.obs.groupby(["individual", "cell_type"], observed=True):
        idx = grp.index.tolist()
        keep.extend(
            rng.choice(idx, min(MAX_CELLS_PER_DONOR_CT, len(idx)), replace=False).tolist()
        )
    mini = mini[keep].copy()

    # HVG selection on normalised data; restore raw counts afterwards
    mini.layers["raw"] = mini.X.copy()
    sc.pp.normalize_total(mini, target_sum=1e4)
    sc.pp.log1p(mini)
    sc.pp.highly_variable_genes(mini, min_mean=0.0125, max_mean=3, min_disp=0.5)

    n_hvg = mini.var["highly_variable"].sum()
    top_n = min(N_GENES, n_hvg)
    hvg_genes = (
        mini.var[mini.var["highly_variable"]]
        .nlargest(top_n, "dispersions_norm")
        .index.tolist()
    )
    print(f"HVGs selected: {len(hvg_genes)}")

    # Restore raw counts, restrict to HVGs
    mini.X = mini.layers["raw"]
    mini   = mini[:, hvg_genes].copy()
    del mini.layers["raw"]
    print(f"Mini dataset:  {mini.n_obs:,} cells × {mini.n_vars} genes")

    # Split into train / holdout / aux AnnData objects
    train_h5   = mini[mini.obs["individual"].isin(train_donors)].copy()
    holdout_h5 = mini[mini.obs["individual"].isin(holdout_donors)].copy()
    aux_h5     = mini[mini.obs["individual"].isin(aux_donors)].copy()
    print(
        f"Split sizes — train: {train_h5.n_obs}  "
        f"holdout: {holdout_h5.n_obs}  aux: {aux_h5.n_obs}"
    )

    # Save h5ads to a shared data dir (both generators will read from here)
    data_dir = os.path.join(OUT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    train_h5.write_h5ad(os.path.join(data_dir, "train.h5ad"))
    holdout_h5.write_h5ad(os.path.join(data_dir, "holdout.h5ad"))
    aux_h5.write_h5ad(os.path.join(data_dir, "aux.h5ad"))

    # Build the joint target DataFrame (train members + holdout non-members)
    # used by the attack functions.
    target_df = pd.concat([train_h5.to_df(), holdout_h5.to_df()])
    target_df["individual"] = pd.concat(
        [train_h5.obs["individual"], holdout_h5.obs["individual"]]
    ).values
    target_df["cell_type"] = pd.concat(
        [train_h5.obs["cell_type"], holdout_h5.obs["cell_type"]]
    ).values
    target_df["membership"] = (
        [1] * train_h5.n_obs + [0] * holdout_h5.n_obs
    )

    # ═════════════════════════════════════════════════════════════════════
    # Loop over generators
    # ═════════════════════════════════════════════════════════════════════

    all_results = {}  # { generator_name: { attack_name: auc } }

    generators = [
        ("scDesign2", ScDesign2, _sd2_config, _load_sd2_copula),
        ("scDesign3", ScDesign3, _sd3_config, _load_sd3_copula),
    ]

    for gen_name, GenClass, make_cfg, load_copula_fn in generators:

        _banner(f"STEPS 2–4 — {gen_name}: train / generate / shadow models")

        gen_dir = os.path.join(OUT_DIR, gen_name.lower())
        os.makedirs(gen_dir, exist_ok=True)

        # Copy data files into gen_dir so generator classes find them via data_dir
        for fname in ("train.h5ad", "holdout.h5ad", "aux.h5ad"):
            shutil.copy(
                os.path.join(data_dir, fname),
                os.path.join(gen_dir, fname),
            )

        # Shared HVG mask path (written by the first train() call, reused by shadows)
        hvg_path = os.path.join(gen_dir, "models", "hvg.csv")
        os.makedirs(os.path.join(gen_dir, "models"), exist_ok=True)

        # ── 2a. Target model: train on train.h5ad, generate from holdout template
        print(f"\n[{gen_name}] Training TARGET model …")
        t0 = time.time()
        target = GenClass(make_cfg(
            home_dir=gen_dir, data_dir=gen_dir,
            model_rel_path="models",
            hvg_path=hvg_path,
            train_file="train.h5ad", test_file="holdout.h5ad",
        ))
        target.train()
        print(f"  done in {_elapsed(t0)}")

        print(f"[{gen_name}] Generating synthetic data …")
        t0 = time.time()
        synth_adata = target.generate()
        synth_path  = os.path.join(gen_dir, "synthetic.h5ad")
        synth_adata.write_h5ad(synth_path)
        print(f"  {synth_adata.n_obs} cells generated in {_elapsed(t0)}")

        # ── 2b. Synth shadow model (BB focal point): trained on synthetic.h5ad
        os.makedirs(os.path.join(gen_dir, "artifacts", "synth"), exist_ok=True)
        os.makedirs(os.path.join(gen_dir, "artifacts", "aux"),   exist_ok=True)
        print(f"\n[{gen_name}] Training SYNTH shadow model on synthetic data …")
        t0 = time.time()
        synth_shadow = GenClass(make_cfg(
            home_dir=gen_dir, data_dir=gen_dir,
            model_rel_path="artifacts/synth",
            hvg_path=hvg_path,
            train_file="synthetic.h5ad", test_file="synthetic.h5ad",
        ))
        synth_shadow.train()
        print(f"  done in {_elapsed(t0)}")

        # ── 2c. Aux shadow model: trained on aux.h5ad
        print(f"\n[{gen_name}] Training AUX shadow model on aux data …")
        t0 = time.time()
        aux_shadow = GenClass(make_cfg(
            home_dir=gen_dir, data_dir=gen_dir,
            model_rel_path="artifacts/aux",
            hvg_path=hvg_path,
            train_file="aux.h5ad", test_file="aux.h5ad",
        ))
        aux_shadow.train()
        print(f"  done in {_elapsed(t0)}")

        # ═══════════════════════════════════════════════════════════════════
        _banner(f"STEP 5 — scMAMA-MIA attack ({gen_name})")
        # ═══════════════════════════════════════════════════════════════════

        synth_model_dir = os.path.join(gen_dir, "artifacts", "synth")
        aux_model_dir   = os.path.join(gen_dir, "artifacts", "aux")

        # Collect per-cell scores across all cell types
        rows_aux   = []   # (individual, membership, score) for BB+aux
        rows_noaux = []   # (individual, membership, score) for BB-aux

        for ct in cell_types:
            synth_rds = os.path.join(synth_model_dir, f"{ct}.rds")
            aux_rds   = os.path.join(aux_model_dir,   f"{ct}.rds")

            if not (os.path.exists(synth_rds) and os.path.exists(aux_rds)):
                print(f"  [SKIP] {ct}: .rds model file not found")
                continue

            print(f"  Attacking cell type: {ct}")
            try:
                synth_cop = load_copula_fn(synth_rds, ct)
                aux_cop   = load_copula_fn(aux_rds,   ct)
            except Exception as e:
                print(f"  [SKIP] {ct}: copula load failed — {e}")
                continue

            # Subset target DataFrame to this cell type
            ct_mask  = target_df["cell_type"] == ct
            ct_df    = target_df[ct_mask]
            ct_indiv = ct_df["individual"]
            ct_mem   = ct_df["membership"]

            # BB+aux Mahalanobis
            scores_aux = _mahalanobis_bb_aux(synth_cop, aux_cop, ct_df)
            if scores_aux is not None:
                rows_aux.append(pd.DataFrame({
                    "individual": ct_indiv.values,
                    "membership": ct_mem.values,
                    "score":      scores_aux,
                }))
                n_shared = len(get_shared_genes(
                    synth_cop["primary_genes"], synth_cop["secondary_genes"],
                    aux_cop["primary_genes"],   aux_cop["secondary_genes"],
                )[0])
                print(f"    BB+aux: {n_shared} shared copula genes")
            else:
                print(f"    BB+aux: skipped (too few shared genes)")

            # BB-aux Mahalanobis
            scores_noaux = _mahalanobis_bb_no_aux(synth_cop, ct_df)
            if scores_noaux is not None:
                rows_noaux.append(pd.DataFrame({
                    "individual": ct_indiv.values,
                    "membership": ct_mem.values,
                    "score":      scores_noaux,
                }))

        gen_results = {}

        if rows_aux:
            all_aux   = pd.concat(rows_aux, ignore_index=True)
            auc_aux   = _donor_auc(all_aux["score"].values,
                                   all_aux["individual"],
                                   all_aux["membership"])
            gen_results["Mahalanobis BB+aux"] = auc_aux
            print(f"\n  AUC  Mahalanobis BB+aux  : {auc_aux:.4f}")

        if rows_noaux:
            all_noaux  = pd.concat(rows_noaux, ignore_index=True)
            auc_noaux  = _donor_auc(all_noaux["score"].values,
                                    all_noaux["individual"],
                                    all_noaux["membership"])
            gen_results["Mahalanobis BB-aux"] = auc_noaux
            print(f"  AUC  Mahalanobis BB-aux  : {auc_noaux:.4f}")

        all_results[gen_name] = gen_results

    # ═════════════════════════════════════════════════════════════════════
    _banner("STEP 6 — Results summary")
    # ═════════════════════════════════════════════════════════════════════

    total_min = (time.time() - t_wall) / 60
    print(f"Total runtime: {total_min:.1f} min\n")

    # Print table
    attack_names = sorted({a for r in all_results.values() for a in r})
    gen_names    = list(all_results.keys())

    col_w = 22
    header = f"{'Attack':<28}" + "".join(f"{g:<{col_w}}" for g in gen_names)
    print(header)
    print("─" * len(header))
    for attack in attack_names:
        row = f"{attack:<28}"
        for g in gen_names:
            auc = all_results[g].get(attack, float("nan"))
            row += f"{auc:<{col_w}.4f}"
        print(row)

    print()
    print("Note: AUC > 0.5 = attack beats random; AUC ~ 0.5 = random.")
    print(
        "Values on this tiny dataset (~50 cells/donor) are noisy and not "
        "scientifically meaningful.\nRun full experiments for real results."
    )
    print(f"\nOutput written to: {OUT_DIR}")


if __name__ == "__main__":
    main()