"""
make_paper_umap.py — side-by-side UMAP: real data vs. scDesign2 synthetic data,
coloured by cell type.  Legend entries ordered by cell-type name length.

Quick test target: OK1K 10d trial 1 (small, fast).

Usage:
    python experiments/sdg_comparison/make_paper_umap.py
"""

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

DATASET   = "ok"
ND        = 10
TRIAL     = 1
SDG       = "scdesign2/no_dp"

REAL_PATH  = DATA_ROOT / DATASET / "full_dataset_cleaned.h5ad"
SYNTH_PATH = DATA_ROOT / DATASET / SDG / f"{ND}d" / str(TRIAL) / "datasets" / "synthetic.h5ad"
HVG_PATH   = DATA_ROOT / DATASET / "hvg_full.csv"

# OK1K: numeric cell_type + named cell_label; synthetic only has cell_type (numeric).
# We resolve names via a mapping built from real data.
CT_COL      = "cell_type"    # numeric code present in both real + synthetic
CT_LABEL_COL = "cell_label"  # human-readable names, real data only

N_REAL  = 2000
N_SYNTH = 2000

# Primary output lives inside the data dir (tracked by print_unified_status.py).
# A copy also goes to the figures/ tree for quick review.
DATA_UMAP_PATH = DATA_ROOT / DATASET / SDG / f"{ND}d" / str(TRIAL) / "umaps" / "paper_umap.png"
OUT_PATH       = OUT_DIR / f"umap_paper_{DATASET}_{ND}d_t{TRIAL}.png"

# ---------------------------------------------------------------------------
# Helpers (shared with existing UMAP scripts)
# ---------------------------------------------------------------------------

def load_hvg_genes(hvg_path):
    hvg = pd.read_csv(hvg_path, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def subsample(adata, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, min(n, adata.n_obs), replace=False)
    return adata[idx].copy()


def load_adata(path, label, hvg_genes, ct_col, n_cells, seed=42,
               ct_label_col=None, ct_name_map=None):
    """Load h5ad, subset to HVGs, subsample, and resolve cell-type names.

    ct_label_col: column with human-readable names (real data).
    ct_name_map:  dict {numeric_code -> name} to apply when ct_label_col absent (synthetic).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing: {path}")
    adata = anndata.read_h5ad(path)
    common = [g for g in hvg_genes if g in adata.var_names]
    adata = adata[:, common].copy()
    adata = subsample(adata, n_cells, seed=seed)
    adata.obs["_source"] = label

    if ct_label_col and ct_label_col in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[ct_label_col].astype(str)
    elif ct_name_map and ct_col in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[ct_col].astype(str).map(ct_name_map).fillna("unknown")
    elif ct_col in adata.obs.columns:
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
    """Assign tab20 colours to cell types (sorted alphabetically for stability)."""
    sorted_cts = sorted(cell_types)
    n = len(sorted_cts)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    return {ct: cmap(i) for i, ct in enumerate(sorted_cts)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_side_by_side(combined, color_map, out_path, data_umap_path=None):
    """Two panels (Real | Synthetic) sharing axes, legend in a single column on the right."""
    coords   = combined.obsm["X_umap"]
    src_vals = combined.obs["_source"].values
    ct_vals  = combined.obs["_cell_type"].values

    # Shared axis limits from full data range
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = (x_max - x_min) * 0.04
    pad_y = (y_max - y_min) * 0.04
    xlim  = (x_min - pad_x, x_max + pad_x)
    ylim  = (y_min - pad_y, y_max + pad_y)

    # Alphabetical order for consistent layer stacking
    known_cts = sorted(color_map.keys())

    # Width ratios: two equal UMAP panels + narrow legend column
    fig, axes = plt.subplots(
        1, 2,
        figsize=(9, 4.5),
        sharey=True,
        gridspec_kw={"wspace": 0.04},
    )

    panel_labels = [("Real data", "Real"), ("scDesign2 synthetic", "Synthetic")]
    for i, (ax, (title, src)) in enumerate(zip(axes, panel_labels)):
        mask = src_vals == src
        c    = coords[mask]
        cts  = ct_vals[mask]
        for ct in known_cts:
            m2 = cts == ct
            if not m2.any():
                continue
            ax.scatter(c[m2, 0], c[m2, 1],
                       c=[color_map[ct]], s=2, alpha=0.5,
                       linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        # Only the left panel gets a y-axis label
        if i == 0:
            ax.set_ylabel("UMAP 2", fontsize=9)
        else:
            ax.set_ylabel("")

    # Legend: one tall column to the right of the figure, no box
    ordered_cts = sorted(color_map.keys(), key=lambda ct: (len(ct), ct))
    patches = [
        mpatches.Patch(facecolor=color_map[ct], label=ct)
        for ct in ordered_cts
    ]
    fig.legend(
        handles=patches,
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        labelspacing=0.35,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if data_umap_path is not None:
        Path(data_umap_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(data_umap_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {data_umap_path}")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_ct_name_map(real_path, ct_col, ct_label_col):
    """Read real data obs to build {numeric_code -> label} mapping."""
    tmp = anndata.read_h5ad(real_path, backed="r")
    obs = tmp.obs[[ct_col, ct_label_col]].drop_duplicates()
    mapping = dict(zip(obs[ct_col].astype(str), obs[ct_label_col].astype(str)))
    tmp.file.close()
    return mapping


def main():
    print(f"Loading HVGs from {HVG_PATH}…")
    hvg_genes = load_hvg_genes(HVG_PATH)

    print("Building cell-type name mapping from real data…")
    ct_name_map = build_ct_name_map(REAL_PATH, CT_COL, CT_LABEL_COL)
    print(f"  {len(ct_name_map)} cell types: {ct_name_map}")

    print(f"Loading real data ({N_REAL} cells)…")
    real = load_adata(REAL_PATH, "Real", hvg_genes, CT_COL, N_REAL, seed=42,
                      ct_label_col=CT_LABEL_COL)

    print(f"Loading synthetic data ({N_SYNTH} cells)…")
    synth = load_adata(SYNTH_PATH, "Synthetic", hvg_genes, CT_COL, N_SYNTH, seed=7,
                       ct_name_map=ct_name_map)

    print("Computing joint UMAP…")
    combined = compute_joint_umap([real, synth])

    real_cts  = set(combined.obs.loc[combined.obs["_source"] == "Real", "_cell_type"])
    real_cts.discard("unknown")
    color_map = build_color_map(real_cts)

    print("Plotting…")
    plot_side_by_side(combined, color_map, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
