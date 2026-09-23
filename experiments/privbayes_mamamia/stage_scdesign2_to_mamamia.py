#!/usr/bin/env python3
"""
stage_scdesign2_to_mamamia.py — bridge scDesign2 scRNA-seq data into MAMA-MIA's
discretised-tabular contract (Experiment #4; see docs/privbayes_mamamia_plan.md).

Produces three discretised frames (rows = cells, columns = binned genes) plus a
domain file, all sharing ONE binning (quantile bins fit on `aux`):

  aux.parquet      real held-out-AUX-donor cells  + HHID(=donor)   [FP extraction + candidate sampling]
  targets.parquet  real TRAIN(member)+HOLDOUT(non-member) cells + HHID(=donor)
  synth.parquet    scDesign2 synthetic cells (NO HHID — it is the released synth)
  meta.json        {gene_col: n_bins}   (categorical domain for PrivBayes)
  membership.csv   donor -> {0,1}        (1 = member = train donor)
  manifest.json    provenance + shapes + params

MAMA-MIA set-membership mode uses HHID as the set id; here HHID = donor, so the
attack is donor-level with no extra aggregator (see plan doc).

Gene set (default `topk` by aux variance) and binning are the cost/comparability
knobs flagged for confirmation in the plan doc; `copula2` (group-2 copula genes)
is a documented TODO extension.

Usage
-----
    conda run -n tabddpm_ python experiments/privbayes_mamamia/stage_scdesign2_to_mamamia.py \
        --dataset ok --sdg scdesign2/no_dp --nd 50 --trial 1 \
        --gene-set topk --n-genes 50 --n-bins 10 \
        --out-dir experiments/privbayes_mamamia/_staged/ok_no_dp_50d_t1

    # cheap shape check without writing frames:
    ... --dry-run
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

DATA = "/home/golobs/data/scMAMAMIA"
DONOR_COL = "individual"
CELL_TYPE_COL = "cell_type"


def _load_split(dataset, nd, trial, name):
    p = os.path.join(DATA, dataset, "splits", f"{nd}d", str(trial), f"{name}.npy")
    return np.load(p, allow_pickle=True)


def _copula_genes(dataset, nd, trial):
    """Union of the scDesign2 COPULA-COVARIANCE genes (primary_genes / gene_sel1)
    across all cell-type models — i.e. the Mahalanobis focal genes scMAMA-MIA
    exploits (NOT the secondary/marginal-only genes). Requires the no_dp models."""
    import glob
    from rpy2.robjects import r as R
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from sdg.scdesign2.copula import parse_copula
    mdir = os.path.join(DATA, dataset, "scdesign2", "no_dp", f"{nd}d", str(trial), "models")
    rds = sorted(glob.glob(os.path.join(mdir, "*.rds")))
    if not rds:
        raise FileNotFoundError(f"no scDesign2 models for copula gene set: {mdir}")
    genes = set()
    for p in rds:
        ct = os.path.splitext(os.path.basename(p))[0]
        try:
            parsed = parse_copula(R["readRDS"](p).rx2(str(ct)))
        except Exception:
            continue
        if parsed.get("cov_matrix") is not None:
            genes.update(parsed["primary_genes"])
    return genes


def _select_genes(gene_set, n_genes, hvg_mask, var_names, aux_X, dataset, nd, trial):
    """Return a boolean gene mask over var_names."""
    if gene_set == "hvg":
        return hvg_mask.copy()
    if gene_set in ("topk", "copula"):
        hvg_idx = np.where(hvg_mask)[0]
        if gene_set == "copula":
            cop = _copula_genes(dataset, nd, trial)
            name_to_idx = {g: i for i, g in enumerate(var_names)}
            cand = np.array([name_to_idx[g] for g in cop if g in name_to_idx], dtype=int)
            print(f"  copula (primary) genes across cell types: {len(cop)} "
                  f"({len(cand)} in var_names)")
        else:
            cand = hvg_idx
        # rank candidates by variance across AUX cells, take top-n_genes.
        sub = aux_X[:, cand]
        v = np.asarray(sub.power(2).mean(axis=0)).ravel() - np.asarray(sub.mean(axis=0)).ravel() ** 2
        top = cand[np.argsort(v)[::-1][:n_genes]]
        mask = np.zeros(len(var_names), dtype=bool)
        mask[top] = True
        return mask
    raise ValueError(f"unknown gene_set {gene_set!r}")


def _fit_bins(aux_mat, n_bins):
    """Zero-AWARE per-gene binner fit on aux: bin 0 = exact zero; bins 1..k =
    quantile bins of the NON-zero values. scRNA-seq counts are zero-inflated, so a
    plain quantile grid collapses (median ~3 distinct edges); separating the zero
    mass first restores resolution on the informative non-zero tail.
    Returns per-gene interior edges of the non-zero quantile grid (np.array)."""
    binners = []
    for g in range(aux_mat.shape[1]):
        col = aux_mat[:, g]
        nz = col[col > 0]
        if nz.size == 0:
            binners.append(np.array([]))                 # all-zero -> single bin
        else:
            # n_bins non-zero bins -> up to n_bins+1 quantile points (deduped).
            binners.append(np.unique(np.quantile(nz, np.linspace(0, 1, n_bins + 1))))
    return binners


def _apply_bins(mat, binners):
    """Apply zero-aware binner: 0 -> bin 0; non-zero -> 1..k via digitize on the
    interior non-zero quantile edges."""
    out = np.zeros(mat.shape, dtype=np.int16)
    for g, q in enumerate(binners):
        col = mat[:, g]
        if q.size <= 1:                                  # only a zero bin (+nonzero -> 1)
            out[:, g] = (col > 0).astype(np.int16)
            continue
        idx = np.digitize(col, q[1:-1], right=False) + 1  # non-zero -> 1..len(q)-1
        out[:, g] = np.where(col > 0, np.clip(idx, 1, len(q) - 1), 0).astype(np.int16)
    return out


def _domain_from(*mats):
    """Per-column categorical cardinality = max bin index across ALL frames + 1,
    so aux/targets/synth share one domain and no value is out-of-range."""
    stacked = np.vstack(mats)
    return (stacked.max(axis=0).astype(int) + 1).tolist()


def _dense(adata, gene_mask, donors=None):
    """Return (dense_matrix, donor_array_or_None) for cells of `donors` (or all)."""
    obs = adata.obs
    if donors is not None:
        cell_mask = obs[DONOR_COL].isin(donors).values
    else:
        cell_mask = np.ones(adata.n_obs, dtype=bool)
    X = adata[cell_mask, :].X
    X = X[:, gene_mask]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    d = obs.loc[cell_mask, DONOR_COL].values if donors is not None else None
    return X.astype(np.float32), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--sdg", default="scdesign2/no_dp")
    ap.add_argument("--nd", type=int, default=50)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--gene-set", choices=["copula", "topk", "hvg"], default="copula")
    ap.add_argument("--n-genes", type=int, default=50)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--max-cells-per-donor", type=int, default=0,
                    help="subsample cells per donor (0 = keep all)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print shapes; write nothing")
    a = ap.parse_args()
    if not a.dry_run and not a.out_dir:
        ap.error("--out-dir is required unless --dry-run")

    base = os.path.join(DATA, a.dataset)
    full_p = os.path.join(base, "full_dataset_cleaned.h5ad")
    synth_p = os.path.join(base, *a.sdg.split("/"), f"{a.nd}d", str(a.trial),
                           "datasets", "synthetic.h5ad")
    hvg_p = os.path.join(base, "hvg.csv")
    for p in (full_p, synth_p, hvg_p):
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing input: {p}")

    train = _load_split(a.dataset, a.nd, a.trial, "train")
    holdout = _load_split(a.dataset, a.nd, a.trial, "holdout")
    aux = _load_split(a.dataset, a.nd, a.trial, "auxiliary")
    print(f"donors: train={len(train)} holdout={len(holdout)} aux={len(aux)}")

    print("loading real data + synth ...", flush=True)
    full = sc.read_h5ad(full_p)
    synth = sc.read_h5ad(synth_p)
    var_names = np.array(full.var_names)
    hvg_mask = pd.read_csv(hvg_p, index_col=0)["highly_variable"].values.astype(bool)

    # gene selection uses aux real cells
    aux_X_all, aux_d = _dense(full, np.ones(len(var_names), bool), donors=aux)
    gene_mask = _select_genes(a.gene_set, a.n_genes, hvg_mask, var_names,
                              _sparse(aux_X_all), a.dataset, a.nd, a.trial)
    genes = var_names[gene_mask]
    print(f"selected {gene_mask.sum()} genes ({a.gene_set})")

    # dense frames on selected genes
    aux_X = aux_X_all[:, gene_mask]
    tgt_X, tgt_d = _dense(full, gene_mask, donors=np.concatenate([train, holdout]))
    syn_X, _ = _dense(synth, gene_mask, donors=None)
    print(f"cells: aux={aux_X.shape[0]} targets={tgt_X.shape[0]} synth={syn_X.shape[0]}")

    binners = _fit_bins(aux_X, a.n_bins)
    aux_b = _apply_bins(aux_X, binners)
    tgt_b = _apply_bins(tgt_X, binners)
    syn_b = _apply_bins(syn_X, binners)
    domain = _domain_from(aux_b, tgt_b, syn_b)          # per-gene cardinality

    member = set(map(str, train.tolist()))
    membership = {str(d): (1 if str(d) in member else 0)
                  for d in np.unique(np.concatenate([train, holdout]))}

    if a.dry_run:
        print(f"[dry-run] genes={list(genes[:8])}...  domain(median)="
              f"{int(np.median(domain))} min/max={min(domain)}/{max(domain)}  "
              f"members={sum(membership.values())}/{len(membership)}")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    cols = list(genes)

    def _frame(binned, donors=None):
        df = pd.DataFrame(binned, columns=cols)
        if donors is not None:
            df.insert(0, "HHID", [str(x) for x in donors])
        return df

    _frame(aux_b, aux_d).to_parquet(os.path.join(a.out_dir, "aux.parquet"))
    _frame(tgt_b, tgt_d).to_parquet(os.path.join(a.out_dir, "targets.parquet"))
    _frame(syn_b, None).to_parquet(os.path.join(a.out_dir, "synth.parquet"))
    pd.DataFrame({"HHID": list(membership), "member": list(membership.values())}) \
        .to_csv(os.path.join(a.out_dir, "membership.csv"), index=False)
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump({g: int(d) for g, d in zip(cols, domain)}, f, indent=2)
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump({
            "dataset": a.dataset, "sdg": a.sdg, "nd": a.nd, "trial": a.trial,
            "gene_set": a.gene_set, "n_genes": int(gene_mask.sum()), "n_bins": a.n_bins,
            "n_aux_cells": int(aux_b.shape[0]), "n_target_cells": int(tgt_b.shape[0]),
            "n_synth_cells": int(syn_b.shape[0]),
            "n_member_donors": int(sum(membership.values())),
            "n_nonmember_donors": int(len(membership) - sum(membership.values())),
            "genes": cols,
        }, f, indent=2)
    print(f"[ok] staged -> {a.out_dir}")


def _sparse(x):
    import scipy.sparse as sp
    return sp.csr_matrix(x)


if __name__ == "__main__":
    main()
