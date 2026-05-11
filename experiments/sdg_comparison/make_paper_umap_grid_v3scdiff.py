"""
make_paper_umap_grid_v3scdiff.py — 5×11 UMAP grid for OK1K 50d.

Rows   = trials 1–5
Columns = Real (train donors only) | scDesign2 | scD2 ε=1 | scD2 ε=10² |
          scD2 ε=10⁵ | scDesign3-V | NMF | NMF ε=2.8 | scVI |
          scDiffusion-v3 (v1_celltypist) | ZINBWave

Changes from make_paper_umap_grid.py:
  - "Real" column uses only the training donors for each trial (not the full
    dataset), so each row's real panel shows ~50 donors' worth of cells.
  - No subsampling — all cells used from every source.
  - scDiffusion column replaced with scdiffusion_v3/v1_celltypist.

Each row's UMAP is computed jointly across all sources in that trial so all
panels in a row share the same embedding coordinates.

Output: figures/umaps/umap_paper_ok_50d_grid_v3scdiff.png

Usage:
    python experiments/sdg_comparison/make_paper_umap_grid_v3scdiff.py [--dpi N]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT   = Path("/home/golobs/data/scMAMAMIA")
OUT_DIR     = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps")
OUT_PATH    = OUT_DIR / "umap_paper_ok_50d_grid_v3scdiff.png"

DATASET     = "ok"
ND          = 50
TRIALS      = [1, 2, 3, 4, 5]
DPI         = 150

REAL_PATH   = DATA_ROOT / DATASET / "full_dataset_cleaned.h5ad"
HVG_PATH    = DATA_ROOT / DATASET / "hvg_full.csv"
SPLITS_ROOT = DATA_ROOT / DATASET / "splits"
DONOR_COL   = "individual"
CT_COL      = "cell_type"
CT_LABEL_COL = "cell_label"

# (column_label, sdg_path_relative_to_dataset_root)
SDG_METHODS = [
    ("scDesign2",        "scdesign2/no_dp"),
    ("scD2 ε=1",         "scdesign2/eps_1"),
    ("scD2 ε=10²",       "scdesign2/eps_100"),
    ("scD2 ε=10⁵",       "scdesign2/eps_100000"),
    ("scDesign3-V",      "scdesign3/vine"),
    ("NMF",              "nmf/no_dp"),
    ("NMF ε=2.8",        "nmf/eps_2.8"),
    ("scVI",             "scvi/no_dp"),
    ("scDiff-v3",        "scdiffusion_v3/v1_celltypist"),
    ("ZINBWave",         "zinbwave/no_dp"),
]

COL_LABELS = ["Real"] + [label for label, _ in SDG_METHODS]
N_COLS = len(COL_LABELS)   # 11
N_ROWS = len(TRIALS)       # 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hvg_genes():
    hvg = pd.read_csv(HVG_PATH, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def build_ct_name_map():
    tmp = anndata.read_h5ad(REAL_PATH, backed="r")
    obs = tmp.obs[[CT_COL, CT_LABEL_COL]].drop_duplicates()
    mapping = dict(zip(obs[CT_COL].astype(str), obs[CT_LABEL_COL].astype(str)))
    tmp.file.close()
    return mapping


def attach_cell_type(adata, ct_name_map, use_label_col=False):
    if use_label_col and CT_LABEL_COL in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[CT_LABEL_COL].astype(str)
    elif ct_name_map and CT_COL in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[CT_COL].astype(str).map(ct_name_map).fillna("unknown")
    elif CT_COL in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[CT_COL].astype(str)
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
    sorted_cts = sorted(cell_types)
    n = len(sorted_cts)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    return {ct: cmap(i) for i, ct in enumerate(sorted_cts)}


# ---------------------------------------------------------------------------
# Per-row data loading + UMAP
# ---------------------------------------------------------------------------

def load_row(trial, real_full, hvg_genes, ct_name_map):
    """Load real (train donors only) + all SDG sources for one trial.

    Returns (combined_adata, list_of_source_labels_per_column).
    present[i] is the _source tag string if that column is present, else None.
    """
    # Real: only the training donors for this trial
    train_donors = set(
        np.load(SPLITS_ROOT / f"{ND}d" / str(trial) / "train.npy",
                allow_pickle=True).tolist()
    )
    mask = real_full.obs[DONOR_COL].isin(train_donors)
    real = real_full[mask].copy()
    real.obs["_source"] = "Real"
    real = attach_cell_type(real, ct_name_map, use_label_col=True)
    print(f"  Trial {trial}: {mask.sum():,} real cells from {len(train_donors)} training donors")

    adatas  = [real]
    present = ["Real"]

    for label, sdg_rel in SDG_METHODS:
        path = DATA_ROOT / DATASET / sdg_rel / f"{ND}d" / str(trial) / "datasets" / "synthetic.h5ad"
        if not path.exists():
            print(f"  SKIP t{trial} {label} (missing: {path})")
            adatas.append(None)
            present.append(None)
            continue

        a = anndata.read_h5ad(path)
        common = [g for g in hvg_genes if g in a.var_names]
        a = a[:, common].copy()
        a.obs["_source"] = label
        a = attach_cell_type(a, ct_name_map, use_label_col=False)
        adatas.append(a)
        present.append(label)

    valid = [a for a in adatas if a is not None]
    total = sum(a.n_obs for a in valid)
    print(f"  → {total:,} total cells across {len(valid)} sources; computing UMAP…", flush=True)
    combined = compute_joint_umap(valid)
    return combined, present


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(rows_data, color_map):
    panel_w  = 2.6
    panel_h  = 2.4
    legend_w = 1.8

    fig_w = panel_w * N_COLS + legend_w
    fig_h = panel_h * N_ROWS

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(fig_w, fig_h),
        sharey="row",
        sharex="row",
        gridspec_kw={"wspace": 0.04, "hspace": 0.12},
    )

    known_cts = sorted(color_map.keys())

    for row_idx, (trial, (combined, present)) in enumerate(zip(TRIALS, rows_data)):
        coords   = combined.obsm["X_umap"]
        src_vals = combined.obs["_source"].values
        ct_vals  = combined.obs["_cell_type"].values

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        pad_x = (x_max - x_min) * 0.04
        pad_y = (y_max - y_min) * 0.04

        for col_idx, (col_label, src) in enumerate(zip(COL_LABELS, present)):
            ax = axes[row_idx, col_idx]

            if src is None:
                ax.set_visible(False)
                continue

            mask = src_vals == src
            c    = coords[mask]
            cts  = ct_vals[mask]

            for ct in known_cts:
                m2 = cts == ct
                if not m2.any():
                    continue
                ax.scatter(c[m2, 0], c[m2, 1],
                           c=[color_map[ct]], s=1.5, alpha=0.5,
                           linewidths=0, rasterized=True)

            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(col_label, fontsize=8, fontweight="bold", pad=3)
            if row_idx == N_ROWS - 1:
                ax.set_xlabel("UMAP 1", fontsize=7, labelpad=2)
            if col_idx == 0:
                ax.set_ylabel(f"Trial {trial}\nUMAP 2", fontsize=7, labelpad=3)

    ordered_cts = sorted(color_map.keys(), key=lambda ct: (len(ct), ct))
    patches = [mpatches.Patch(facecolor=color_map[ct], label=ct) for ct in ordered_cts]
    fig.legend(
        handles=patches,
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.0 - legend_w / fig_w + 0.005, 0.5),
        ncol=1,
        fontsize=7,
        title_fontsize=8,
        frameon=False,
        labelspacing=0.3,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DPI

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=DPI, help="Output DPI (default: 150)")
    args = ap.parse_args()
    DPI = args.dpi

    print(f"Grid: {DATASET} {ND}d, {N_ROWS} trials × {N_COLS} columns, "
          f"no subsampling, real from train splits, dpi={DPI}")
    print("Loading HVGs and cell-type name map…")
    hvg_genes   = load_hvg_genes()
    ct_name_map = build_ct_name_map()

    all_cts   = set(ct_name_map.values())
    color_map = build_color_map(all_cts)

    print("Pre-loading full real dataset…")
    real_full = anndata.read_h5ad(REAL_PATH)
    common_hvg = [g for g in hvg_genes if g in real_full.var_names]
    real_full = real_full[:, common_hvg].copy()

    rows_data = []
    for trial in TRIALS:
        print(f"\nTrial {trial}: loading data…", flush=True)
        combined, present = load_row(trial, real_full, hvg_genes, ct_name_map)
        rows_data.append((combined, present))

    print(f"\nPlotting {N_ROWS}×{N_COLS} grid…")
    plot_grid(rows_data, color_map)
    print("Done.")


if __name__ == "__main__":
    main()
