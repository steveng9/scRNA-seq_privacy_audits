"""
make_cg_umap_grid.py — 5×2 UMAP grid for HFRA (cg) dataset.

Rows   = trials 1–5
Columns = Real | scDesign2

Each row's UMAP is computed jointly so both panels share the same embedding.

Output: figures/umaps/umap_cg_{nd}d_grid.png

Usage:
    python experiments/sdg_comparison/make_cg_umap_grid.py
    python experiments/sdg_comparison/make_cg_umap_grid.py --nd 20 --dpi 200
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/golobs/data/scMAMAMIA")
OUT_DIR   = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps")

DATASET   = "cg"
ND        = 10
TRIALS    = [1, 2, 3, 4, 5]
DPI       = 150

REAL_PATH = DATA_ROOT / DATASET / "full_dataset_cleaned.h5ad"
HVG_PATH  = DATA_ROOT / DATASET / "hvg_full.csv"
CT_COL    = "cell_type"

COL_LABELS = ["Real", "scDesign2"]
N_COLS     = 2
N_ROWS     = len(TRIALS)

N_REAL  = 1_000_000   # effectively all
N_SYNTH = 1_000_000   # effectively all

# OUT_PATH set after args parsed
OUT_PATH = None


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


def load_adata(path, label, hvg_genes, ct_col, n_cells, seed=42):
    if not Path(path).exists():
        return None
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
    sorted_cts = sorted(cell_types)
    n = len(sorted_cts)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    return {ct: cmap(i) for i, ct in enumerate(sorted_cts)}


# ---------------------------------------------------------------------------
# Per-row data loading + UMAP
# ---------------------------------------------------------------------------

def load_row(trial, hvg_genes):
    seed_real  = trial * 7
    seed_synth = trial * 100

    real = load_adata(REAL_PATH, "Real", hvg_genes, CT_COL, N_REAL, seed=seed_real)
    if real is None:
        raise FileNotFoundError(f"Real data missing: {REAL_PATH}")

    synth_path = DATA_ROOT / DATASET / "scdesign2" / "no_dp" / f"{ND}d" / str(trial) / "datasets" / "synthetic.h5ad"
    synth = load_adata(synth_path, "scDesign2", hvg_genes, CT_COL, N_SYNTH, seed=seed_synth)

    adatas = [a for a in [real, synth] if a is not None]
    present = ["Real", "scDesign2" if synth is not None else None]

    combined = compute_joint_umap(adatas)
    return combined, present


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(rows_data, color_map):
    panel_w  = 3.0
    panel_h  = 2.8
    legend_w = 2.0

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
                ax.set_title(col_label, fontsize=9, fontweight="bold", pad=3)

            if row_idx == N_ROWS - 1:
                ax.set_xlabel("UMAP 1", fontsize=7, labelpad=2)

            if col_idx == 0:
                ax.set_ylabel(f"Trial {trial}\nUMAP 2", fontsize=7, labelpad=3)

    # Legend to the right
    ordered_cts = sorted(color_map.keys(), key=lambda ct: (len(ct), ct))
    patches = [mpatches.Patch(facecolor=color_map[ct], label=ct) for ct in ordered_cts]
    fig.legend(
        handles=patches,
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.0 - legend_w / fig_w + 0.01, 0.5),
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
    global ND, DPI, OUT_PATH, N_REAL, N_SYNTH

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nd",      type=int, default=ND,      help="Donor count (default: 10)")
    ap.add_argument("--dpi",     type=int, default=DPI,     help="Output DPI (default: 150)")
    ap.add_argument("--n-real",  type=int, default=N_REAL,  help="Max real cells per row (default: all)")
    ap.add_argument("--n-synth", type=int, default=N_SYNTH, help="Max synth cells per row (default: all)")
    args = ap.parse_args()

    ND      = args.nd
    DPI     = args.dpi
    N_REAL  = args.n_real
    N_SYNTH = args.n_synth
    OUT_PATH = OUT_DIR / f"umap_cg_{ND}d_grid.png"

    n_real_str  = "all" if N_REAL  >= 1_000_000 else str(N_REAL)
    n_synth_str = "all" if N_SYNTH >= 1_000_000 else str(N_SYNTH)
    print(f"Grid: HFRA (cg) {ND}d, {N_ROWS} trials × {N_COLS} columns, "
          f"real={n_real_str} synth={n_synth_str} dpi={DPI}")

    print("Loading HVGs…")
    hvg_genes = load_hvg_genes(HVG_PATH)
    print(f"  {len(hvg_genes)} HVGs")

    # Build color map from all cell types in real data
    tmp = anndata.read_h5ad(REAL_PATH, backed="r")
    all_cts   = set(tmp.obs[CT_COL].astype(str).unique())
    color_map = build_color_map(all_cts)
    tmp.file.close()

    rows_data = []
    for trial in TRIALS:
        print(f"Trial {trial}: loading data…", flush=True)
        combined, present = load_row(trial, hvg_genes)
        print(f"  → {combined.n_obs:,} total cells", flush=True)
        rows_data.append((combined, present))

    print(f"Plotting {N_ROWS}×{N_COLS} grid…")
    plot_grid(rows_data, color_map)
    print("Done.")


if __name__ == "__main__":
    main()
