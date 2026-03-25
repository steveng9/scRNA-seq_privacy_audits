"""
Generate UMAP panel comparing real vs. scDesign2 synthetic (no-DP) vs.
DP-scDesign2 synthetic at various epsilon levels.

Usage: python make_dp_umaps.py [--donor_count 10] [--trial 1] [--n_cells 2000]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
DATA_REAL   = Path("/home/golobs/data/ok/full_dataset_cleaned.h5ad")
HVG_CSV     = Path("/home/golobs/data/ok/hvg.csv")
DATA_NON_DP = Path("/home/golobs/data/ok")
DATA_DP     = Path("/home/golobs/data/ok_dp")
EPSILONS    = [1, 10, 100, 1000, 10000]
OUT_DIR     = Path("/home/golobs/scRNA-seq_privacy_audits/figures")

# cell-type colours — 14 types (0-15 minus 7,11 which seem absent)
CELL_TYPES = ['0','1','2','3','4','5','6','8','9','10','12','13','14','15']
CMAP = plt.cm.get_cmap("tab20", len(CELL_TYPES))
COLOR_MAP = {ct: CMAP(i) for i, ct in enumerate(CELL_TYPES)}


def load_hvg_genes():
    hvg = pd.read_csv(HVG_CSV, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def subsample_adata(adata, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, min(n, adata.n_obs), replace=False)
    return adata[idx].copy()


def load_real(hvg_genes, n_cells):
    adata = anndata.read_h5ad(DATA_REAL)
    adata = adata[:, hvg_genes].copy()
    adata = subsample_adata(adata, n_cells)
    adata.obs["source"] = "Real"
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    return adata


def load_synthetic(path, label, hvg_genes, n_cells):
    adata = anndata.read_h5ad(path)
    # subset to HVG columns that exist in the file
    common = [g for g in hvg_genes if g in adata.var_names]
    adata = adata[:, common].copy()
    adata = subsample_adata(adata, n_cells)
    adata.obs["source"] = label
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    return adata


def compute_joint_umap(adatas):
    combined = anndata.concat(adatas, label="dataset_idx",
                               keys=list(range(len(adatas))),
                               merge="same")
    # log-normalise (scDesign2 output is raw counts)
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.scale(combined, max_value=10)
    sc.pp.pca(combined, n_comps=30, random_state=0)
    sc.pp.neighbors(combined, n_pcs=30, random_state=0)
    sc.tl.umap(combined, random_state=0)
    return combined


def plot_panel(combined, donor_count, trial, out_dir):
    sources = combined.obs["source"].unique().tolist()
    # order: Real first, then no-DP, then epsilons ascending
    def sort_key(s):
        if s == "Real": return (-1, 0)
        if s == "No-DP": return (0, 0)
        return (1, int(s.replace("ε=", "").replace(",", "")))
    sources = sorted(sources, key=sort_key)

    n_panels = len(sources)
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    umap_coords = combined.obsm["X_umap"]
    all_ct = combined.obs["cell_type"].values

    for ax, src in zip(axes, sources):
        mask = combined.obs["source"] == src
        coords = umap_coords[mask]
        ct_vals = all_ct[mask]

        for ct in CELL_TYPES:
            ct_mask = ct_vals == ct
            if ct_mask.sum() == 0:
                continue
            ax.scatter(coords[ct_mask, 0], coords[ct_mask, 1],
                       c=[COLOR_MAP[ct]], s=1.5, alpha=0.4, linewidths=0,
                       rasterized=True)

        title = src
        if src == "Real":
            title = "Real data"
        elif src == "No-DP":
            title = "Synthetic (no DP)"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=7)

    # hide unused axes
    for ax in axes[len(sources):]:
        ax.set_visible(False)

    # shared cell-type legend
    patches = [mpatches.Patch(color=COLOR_MAP[ct], label=f"Type {ct}")
               for ct in CELL_TYPES if ct in all_ct]
    fig.legend(handles=patches, title="Cell type",
               loc="lower center", ncol=len(patches),
               fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"UMAP: Real vs. scDesign2 synthetic (no-DP) vs. DP-scDesign2\n"
        f"OneK1K | {donor_count} donors | trial {trial}",
        fontsize=13, y=1.01
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"umap_dp_comparison_{donor_count}d_trial{trial}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {png_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor_count", type=int, default=10,
                        help="Number of donors (must exist in both ok/ and ok_dp/)")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--n_cells", type=int, default=2000,
                        help="Cells to subsample per condition")
    args = parser.parse_args()

    dc = args.donor_count
    trial = args.trial
    n = args.n_cells

    print(f"Loading HVG genes...")
    hvg_genes = load_hvg_genes()
    print(f"  {len(hvg_genes)} HVGs")

    print(f"Loading real data (~{n} cells)...")
    real = load_real(hvg_genes, n)

    print(f"Loading no-DP synthetic ({dc}d, trial {trial})...")
    nodp_path = DATA_NON_DP / f"{dc}d" / str(trial) / "datasets" / "synthetic.h5ad"
    nodp = load_synthetic(nodp_path, "No-DP", hvg_genes, n)

    adatas = [real, nodp]

    for eps in EPSILONS:
        dp_path = DATA_DP / f"eps_{eps}" / f"{dc}d" / str(trial) / "datasets" / "synthetic.h5ad"
        if not dp_path.exists():
            print(f"  WARNING: missing {dp_path}, skipping ε={eps}")
            continue
        label = f"ε={eps:,}"
        print(f"Loading DP synthetic ε={eps}...")
        adatas.append(load_synthetic(dp_path, label, hvg_genes, n))

    print(f"Computing joint UMAP on {len(adatas)} conditions × {n} cells each...")
    combined = compute_joint_umap(adatas)

    print("Plotting...")
    plot_panel(combined, dc, trial, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
