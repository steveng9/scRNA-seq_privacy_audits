"""
make_nmf_umaps.py — 3-panel UMAPs for NMF synthetic data.

For each figure:
  Panel 1: Real data, coloured by cell type
  Panel 2: NMF synthetic data, coloured by cell type
  Panel 3: Both real and synthetic together, coloured by source
            (real = blue, synthetic = orange)

Figures are saved to figures/umaps/nmf/.

Coverage:
  OK1K : 10d / 20d / 50d / 100d  — trial 1 each; plus 50d combined (all trials)
  AIDA :  10d / 20d / 50d         — trial 1 each; plus 20d combined (all trials)

Usage:
    python experiments/sdg_comparison/make_nmf_umaps.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_ROOT = Path("/home/golobs/data/scMAMAMIA")
OUT_DIR   = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps/nmf")

DATASET_CFG = {
    "ok": {
        "real":        DATA_ROOT / "ok" / "full_dataset_cleaned.h5ad",
        "hvg":         DATA_ROOT / "ok" / "hvg_full.csv",
        "nmf_root":    DATA_ROOT / "ok" / "nmf" / "no_dp",
        "ct_col":      "cell_type",
        "backed_real": False,
        "label":       "OK1K",
    },
    "aida": {
        "real":        DATA_ROOT / "aida" / "full_dataset_cleaned.h5ad",
        "hvg":         DATA_ROOT / "aida" / "hvg_full.csv",
        "nmf_root":    DATA_ROOT / "aida" / "nmf" / "no_dp",
        "ct_col":      "cell_type",
        "backed_real": True,
        "label":       "AIDA",
    },
}

N_CELLS_REAL  = 2000
N_CELLS_SYNTH = 2000
N_CELLS_COMB  = 1000   # per trial in pooled figures

SOURCE_COLORS = {"Real": "#1f77b4", "NMF": "#ff7f0e"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hvg_genes(hvg_path):
    hvg = pd.read_csv(hvg_path, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def subsample(adata, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, min(n, adata.n_obs), replace=False)
    return adata[idx].copy()


def load_adata(path, label, hvg_genes, ct_col, n_cells, seed=42, backed=False):
    if not Path(path).exists():
        return None
    if backed:
        tmp = anndata.read_h5ad(path, backed="r")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(tmp.n_obs, min(n_cells, tmp.n_obs), replace=False))
        common = [g for g in hvg_genes if g in tmp.var_names]
        adata = tmp[idx, common].to_memory()
        tmp.file.close()
    else:
        adata = anndata.read_h5ad(path)
        common = [g for g in hvg_genes if g in adata.var_names]
        adata = adata[:, common].copy()
        adata = subsample(adata, n_cells, seed=seed)
    adata.obs["_source"] = label
    if ct_col in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[ct_col].astype(str)
    else:
        adata.obs["_cell_type"] = "unknown"
    return adata


def compute_joint_umap(adatas):
    combined = anndata.concat(adatas, merge="same")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.scale(combined, max_value=10)
    sc.pp.pca(combined, n_comps=30, random_state=0)
    sc.pp.neighbors(combined, n_pcs=30, random_state=0)
    sc.tl.umap(combined, random_state=0)
    return combined


def build_color_map(cell_types):
    n = len(cell_types)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    return {ct: cmap(i) for i, ct in enumerate(sorted(cell_types))}


def plot_three_panels(combined, ct_color_map, title, out_path):
    """3-panel figure: Real (by CT) | NMF (by CT) | Combined (by source)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    coords    = combined.obsm["X_umap"]
    src_vals  = combined.obs["_source"].values
    ct_vals   = combined.obs["_cell_type"].values
    known_cts = sorted(ct_color_map.keys())

    for ax, src in zip(axes[:2], ["Real", "NMF"]):
        mask = src_vals == src
        c    = coords[mask]
        cts  = ct_vals[mask]
        for ct in known_cts:
            m2 = cts == ct
            if m2.any():
                ax.scatter(c[m2, 0], c[m2, 1],
                           c=[ct_color_map[ct]], s=2, alpha=0.5,
                           linewidths=0, rasterized=True)
        ax.set_title(src, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Panel 3: source colour
    ax3 = axes[2]
    for src, col in SOURCE_COLORS.items():
        mask = src_vals == src
        ax3.scatter(coords[mask, 0], coords[mask, 1],
                    c=col, s=2, alpha=0.5, label=src,
                    linewidths=0, rasterized=True)
    ax3.set_title("Real vs NMF", fontsize=11, fontweight="bold")
    ax3.set_xlabel("UMAP 1", fontsize=8)
    ax3.set_ylabel("UMAP 2", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.legend(markerscale=4, fontsize=8, loc="best")

    # Cell-type legend at the bottom
    patches = [mpatches.Patch(color=ct_color_map[ct], label=ct) for ct in known_cts]
    if patches:
        fig.legend(handles=patches, title="Cell type",
                   loc="lower center", ncol=min(len(patches), 7),
                   fontsize=7, title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(title, fontsize=12, y=1.02)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def make_single_trial(dataset_name, nd, trial):
    cfg      = DATASET_CFG[dataset_name]
    hvg_genes = load_hvg_genes(cfg["hvg"])
    ct_col   = cfg["ct_col"]

    synth_path = cfg["nmf_root"] / f"{nd}d" / str(trial) / "datasets" / "synthetic.h5ad"
    if not synth_path.exists():
        print(f"  SKIP {dataset_name} NMF {nd}d t{trial} (missing)", flush=True)
        return

    real = load_adata(cfg["real"], "Real", hvg_genes, ct_col,
                      N_CELLS_REAL, seed=42, backed=cfg["backed_real"])
    if real is None:
        print(f"  ERROR: real data missing for {dataset_name}")
        return
    synth = load_adata(synth_path, "NMF", hvg_genes, ct_col, N_CELLS_SYNTH, seed=trial * 100)
    if synth is None:
        return

    print(f"  Computing UMAP: {dataset_name} NMF {nd}d t{trial}…", flush=True)
    combined = compute_joint_umap([real, synth])

    real_cts  = set(combined.obs.loc[combined.obs["_source"] == "Real", "_cell_type"])
    real_cts.discard("unknown")
    ct_color  = build_color_map(real_cts)

    title    = f"{cfg['label']} NMF | {nd} donors | trial {trial}"
    out_path = OUT_DIR / f"umap_nmf_{dataset_name}_{nd}d_t{trial}.png"
    plot_three_panels(combined, ct_color, title, out_path)


def make_combined_trials(dataset_name, nd, trials):
    cfg       = DATASET_CFG[dataset_name]
    hvg_genes = load_hvg_genes(cfg["hvg"])
    ct_col    = cfg["ct_col"]
    n_each    = max(1, N_CELLS_COMB // len(trials))

    real = load_adata(cfg["real"], "Real", hvg_genes, ct_col,
                      N_CELLS_REAL, seed=0, backed=cfg["backed_real"])
    if real is None:
        print(f"  ERROR: real data missing for {dataset_name}")
        return

    synth_parts = []
    for t in trials:
        p = cfg["nmf_root"] / f"{nd}d" / str(t) / "datasets" / "synthetic.h5ad"
        a = load_adata(p, "NMF", hvg_genes, ct_col, n_each, seed=t * 10)
        if a is not None:
            synth_parts.append(a)
    if not synth_parts:
        print(f"  SKIP {dataset_name} NMF {nd}d combined (no trials)", flush=True)
        return

    synth_pooled = anndata.concat(synth_parts, merge="same")
    synth_pooled.obs["_source"] = "NMF"

    print(f"  Computing UMAP: {dataset_name} NMF {nd}d combined ({len(synth_parts)} trials)…", flush=True)
    combined = compute_joint_umap([real, synth_pooled])

    real_cts = set(combined.obs.loc[combined.obs["_source"] == "Real", "_cell_type"])
    real_cts.discard("unknown")
    ct_color = build_color_map(real_cts)

    title    = (f"{cfg['label']} NMF | {nd} donors | "
                f"{len(synth_parts)} trials combined (~{N_CELLS_COMB} cells/trial)")
    out_path = OUT_DIR / f"umap_nmf_{dataset_name}_{nd}d_combined.png"
    plot_three_panels(combined, ct_color, title, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # OK1K — all donor sizes, trial 1
    for nd in [10, 20, 50, 100]:
        print(f"\n[OK1K] NMF {nd}d — trial 1…", flush=True)
        make_single_trial("ok", nd, 1)

    # OK1K — 50d combined
    print("\n[OK1K] NMF 50d — combined (all 5 trials)…", flush=True)
    make_combined_trials("ok", 50, [1, 2, 3, 4, 5])

    # AIDA — all donor sizes, trial 1
    for nd in [10, 20, 50]:
        print(f"\n[AIDA] NMF {nd}d — trial 1…", flush=True)
        make_single_trial("aida", nd, 1)

    # AIDA — 20d combined
    print("\n[AIDA] NMF 20d — combined (all 5 trials)…", flush=True)
    make_combined_trials("aida", 20, [1, 2, 3, 4, 5])

    print(f"\nAll done. Figures in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
