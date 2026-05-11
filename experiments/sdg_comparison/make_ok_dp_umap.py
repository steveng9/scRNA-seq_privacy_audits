"""
make_ok_dp_umap.py — 2×5 UMAP grid for OneK1K DP comparison.

Uses the 10-source joint UMAP embedding (row 1 cache):
  Real + SD2/no_dp + SD2/eps_1 + SD2/eps_100 + SD2/eps_10000
  + scVI + ZINBWave + NMF/eps_2.8 + scDiff_v3 + sd3v

Displayed as a 2×5 panel grid:
  Row 0: Real | scDesign2 | Noisy-SD2 ε=10⁴ | Noisy-SD2 ε=100 | Noisy-SD2 ε=1
  Row 1: scVI | scDesign3 (vine) | ZINB-WaVE | scDiffusion | NMF

OneK1K 50d trial 1.

UMAPs are cached per-row under figures/umaps/umap_cache/ok_dp_row/.
Use --force to recompute the embedding (row 1 only).

Output: figures/umaps/umap_ok_dp.png

Usage:
    python experiments/sdg_comparison/make_ok_dp_umap.py
    python experiments/sdg_comparison/make_ok_dp_umap.py --force
    python experiments/sdg_comparison/make_ok_dp_umap.py --dpi 200
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import pickle
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT  = Path("/home/golobs/data/scMAMAMIA")
OUT_DIR    = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps")
CACHE_DIR  = OUT_DIR / "umap_cache" / "ok_dp_row"
OUT_PATH   = OUT_DIR / "umap_ok_dp.png"

DATASET      = "ok"
ND           = 50
TRIAL        = 1
DONOR_COL    = "individual"
CT_COL       = "cell_type"
CT_LABEL_COL = "cell_label"

# ---------------------------------------------------------------------------
# All SDG methods (used for embedding computation)
# (display_label_in_cache, sdg_rel_path)
# ---------------------------------------------------------------------------

ALL_SDG_METHODS = [
    ("SD2\nno DP",   "scdesign2/no_dp"),
    ("SD2\nε=1",     "scdesign2/eps_1"),
    ("SD2\nε=100",   "scdesign2/eps_100"),
    ("SD2\nε=10⁴",   "scdesign2/eps_10000"),
    ("scVI",          "scvi/no_dp"),
    ("ZINBWave",      "zinbwave/no_dp"),
    ("NMF\nε=2.8",   "nmf/eps_2.8"),
    ("scDiff",        "scdiffusion_v3/v1_celltypist"),
    ("SD3v",          "scdesign3/vine"),
]

# ---------------------------------------------------------------------------
# 2×5 panel layout: (source_key_in_cache, display_title)
# source keys = display_label.replace("\n", " ") set during loading
# ---------------------------------------------------------------------------

PANEL_GRID = [
    [
        ("Real",       "Real"),
        ("SD2 no DP",  "scDesign2"),
        ("SD2 ε=10⁴",  "Noisy-scDesign2\nε=10⁴"),
        ("SD2 ε=100",  "Noisy-scDesign2\nε=100"),
        ("SD2 ε=1",    "Noisy-scDesign2\nε=1"),
    ],
    [
        ("scVI",       "scVI"),
        ("SD3v",       "scDesign3 (vine)"),
        ("ZINBWave",   "ZINB-WaVE"),
        ("scDiff",     "scDiffusion"),
        ("NMF ε=2.8",  "NMF"),
    ],
]

N_DISPLAY_ROWS = len(PANEL_GRID)       # 2
N_DISPLAY_COLS = len(PANEL_GRID[0])    # 5

# 1-row square variant
PANEL_GRID_1ROW = [
    [
        ("Real",       "Real"),
        ("SD2 no DP",  "scDesign2"),
        ("SD2 ε=100",  "Noisy-scDesign2\nε=100"),
        ("scVI",       "scVI"),
        ("ZINBWave",   "ZINB-WaVE"),
    ],
]
OUT_PATH_1ROW = OUT_DIR / "umap_ok_dp_1row.png"

DPI      = 150
PANEL_W  = 2.3
PANEL_H  = 2.4
LEGEND_W = 2.6

# ---------------------------------------------------------------------------
# Color palette — AIDA sunset (pinks, purples, salmons, oranges)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#F4875B",   # salmon
    "#9B72CF",   # purple
    "#D890A8",   # dusty pink
    "#F5B942",   # warm orange
    "#C47EC0",   # orchid
    "#F0956A",   # peach
    "#BF6890",   # muted rose
    "#A868C8",   # medium purple
    "#F8C888",   # light amber
    "#C88090",   # dusty mauve
]


def build_color_map(cell_types, counts=None):
    if counts is not None:
        sorted_cts = sorted(cell_types, key=lambda ct: -counts.get(ct, 0))
    else:
        sorted_cts = sorted(cell_types)

    def _soften(rgba, blend):
        r, g, b, a = rgba
        return (r + (1-r)*blend, g + (1-g)*blend, b + (1-b)*blend, a)

    t20b = plt.cm.get_cmap("tab20b", 20)
    t20c = plt.cm.get_cmap("tab20c", 20)
    palette = (
        [to_rgba(c) for c in _PALETTE] +
        [_soften(t20b(i), 0.22) for i in range(20)] +
        [_soften(t20c(i), 0.22) for i in range(20)]
    )
    return {ct: palette[i] for i, ct in enumerate(sorted_cts)}


# ---------------------------------------------------------------------------
# Data loading & UMAP computation (row 1: all 9 SDGs + Real)
# ---------------------------------------------------------------------------

def _cache_path():
    return CACHE_DIR / "row1.pkl"


def _load_hvg_genes():
    hvg_path = DATA_ROOT / DATASET / "hvg_full.csv"
    hvg = pd.read_csv(hvg_path, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def _build_ct_name_map(real_path):
    tmp = anndata.read_h5ad(real_path, backed="r")
    obs = tmp.obs[[CT_COL, CT_LABEL_COL]].drop_duplicates()
    mapping = dict(zip(obs[CT_COL].astype(str), obs[CT_LABEL_COL].astype(str)))
    tmp.file.close()
    return mapping


def _attach_cell_type(adata, ct_name_map):
    if CT_LABEL_COL and CT_LABEL_COL in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[CT_LABEL_COL].astype(str)
    elif ct_name_map and CT_COL in adata.obs.columns:
        adata.obs["_cell_type"] = (
            adata.obs[CT_COL].astype(str).map(ct_name_map).fillna("unknown")
        )
    elif CT_COL in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[CT_COL].astype(str)
    else:
        adata.obs["_cell_type"] = "unknown"
    return adata


def _compute_joint_umap(adatas):
    combined = anndata.concat(adatas, merge="same")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.scale(combined, max_value=10)
    sc.pp.pca(combined, n_comps=30, random_state=0)
    sc.pp.neighbors(combined, n_pcs=30, random_state=0)
    sc.tl.umap(combined, random_state=0)
    return combined


def compute_embedding(verbose=True):
    real_path   = DATA_ROOT / DATASET / "full_dataset_cleaned.h5ad"
    splits_root = DATA_ROOT / DATASET / "splits"

    if verbose:
        print("  Loading HVGs and cell-type name map…")
    hvg_genes   = _load_hvg_genes()
    ct_name_map = _build_ct_name_map(real_path)

    if verbose:
        print("  Loading real dataset…")
    real_full = anndata.read_h5ad(real_path)
    common    = [g for g in hvg_genes if g in real_full.var_names]
    real_full = real_full[:, common].copy()

    train_donors = set(
        np.load(splits_root / f"{ND}d" / str(TRIAL) / "train.npy",
                allow_pickle=True).tolist()
    )
    mask = real_full.obs[DONOR_COL].isin(train_donors)
    real = real_full[mask].copy()
    del real_full
    real.obs["_source"] = "Real"
    real = _attach_cell_type(real, ct_name_map)
    if verbose:
        print(f"  Real: {mask.sum():,} cells from {len(train_donors)} training donors")

    adatas = [real]
    for col_label, sdg_rel in ALL_SDG_METHODS:
        path = (DATA_ROOT / DATASET / sdg_rel /
                f"{ND}d" / str(TRIAL) / "datasets" / "synthetic.h5ad")
        if not path.exists():
            if verbose:
                print(f"  SKIP {col_label!r} — missing: {path}")
            continue
        a = anndata.read_h5ad(path)
        common_g = [g for g in hvg_genes if g in a.var_names]
        a = a[:, common_g].copy()
        src_key = col_label.replace("\n", " ")
        a.obs["_source"] = src_key
        a = _attach_cell_type(a, ct_name_map)
        adatas.append(a)
        if verbose:
            print(f"  Loaded {src_key!r}: {a.n_obs:,} cells")

    total = sum(a.n_obs for a in adatas)
    if verbose:
        print(f"  → {total:,} cells across {len(adatas)} sources; computing UMAP…",
              flush=True)

    combined = _compute_joint_umap(adatas)
    cache = {
        "coords":    combined.obsm["X_umap"].astype(np.float32),
        "source":    combined.obs["_source"].values,
        "cell_type": combined.obs["_cell_type"].values,
    }
    return cache


def load_or_compute(force=False, verbose=True):
    cp = _cache_path()
    if not force and cp.exists():
        if verbose:
            print(f"  Loading cached embedding: {cp}")
        with open(cp, "rb") as f:
            return pickle.load(f)

    if verbose:
        print(f"  Computing joint UMAP for {DATASET} {ND}d trial {TRIAL}…")
    cache = compute_embedding(verbose=verbose)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    if verbose:
        print(f"  Saved cache: {cp}")
    return cache


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(cache, panel_grid, panel_w, panel_h, legend_w, ncol_legend, out_path):
    coords   = cache["coords"]
    src_vals = cache["source"]
    ct_vals  = cache["cell_type"]

    n_rows = len(panel_grid)
    n_cols = len(panel_grid[0])

    # Build color map from Real panel cell-type frequencies
    real_mask = src_vals == "Real"
    counts: dict = {}
    for ct in ct_vals[real_mask]:
        counts[ct] = counts.get(ct, 0) + 1
    all_cts = set(ct_vals)
    cmap = build_color_map(all_cts, counts=counts)
    known_cts = sorted(cmap.keys())

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = -(x_max - x_min) * 0.025
    pad_y = -(y_max - y_min) * 0.025

    fig_w = panel_w * n_cols + legend_w
    fig_h = panel_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h))
    width_ratios = [panel_w] * n_cols + [legend_w]
    hspace = 0.14 if n_rows > 1 else 0.0
    gs = GridSpec(
        n_rows, n_cols + 1,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.04,
        hspace=hspace,
        left=0.06,
        right=0.995,
        top=0.93,
        bottom=0.07,
    )

    legend_ax = fig.add_subplot(gs[:, n_cols])
    legend_ax.axis("off")

    for grid_r, row_panels in enumerate(panel_grid):
        for grid_c, (src_key, display_title) in enumerate(row_panels):
            ax = fig.add_subplot(gs[grid_r, grid_c])

            mask = src_vals == src_key
            if not mask.any():
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="#888888")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                c_   = coords[mask]
                cts  = ct_vals[mask]
                for ct in known_cts:
                    m2 = cts == ct
                    if not m2.any():
                        continue
                    ax.scatter(c_[m2, 0], c_[m2, 1],
                               c=[cmap[ct]], s=1.0, alpha=0.5,
                               linewidths=0, rasterized=True)
                ax.set_xlim(x_min - pad_x, x_max + pad_x)
                ax.set_ylim(y_min - pad_y, y_max + pad_y)

            ax.set_xticks([])
            ax.set_yticks([])

            if src_key == "Real":
                for spine in ax.spines.values():
                    spine.set_linewidth(3.0)
                    spine.set_edgecolor("black")
                    spine.set_visible(True)

            ax.set_title(display_title, fontsize=13, fontweight="bold", pad=4)

            if grid_c == 0:
                ax.set_ylabel("UMAP 2", fontsize=11, labelpad=3)
            if grid_r == n_rows - 1:
                ax.set_xlabel("UMAP 1", fontsize=11, labelpad=2)

    patches = [mpatches.Patch(facecolor=cmap[ct], label=ct) for ct in known_cts]
    legend_ax.legend(
        handles=patches,
        title="Cell type",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=legend_ax.transAxes,
        ncol=ncol_legend,
        fontsize=13,
        title_fontsize=14,
        frameon=False,
        labelspacing=0.30,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=0.8,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DPI

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--force", action="store_true",
                    help="Recompute the UMAP embedding (ignores cache)")
    ap.add_argument("--compute-only", action="store_true",
                    help="Compute/cache embedding but skip plotting")
    ap.add_argument("--square-1row", action="store_true",
                    help="Produce the 1-row square-panel variant")
    args = ap.parse_args()
    DPI = args.dpi

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_or_compute(force=args.force)
    n_src = len(set(cache["source"]))
    print(f"  Embedding: {len(cache['coords']):,} cells, {n_src} sources.")

    if args.compute_only:
        print("--compute-only: skipping plot generation.")
        return

    if args.square_1row:
        n_cols = len(PANEL_GRID_1ROW[0])
        print(f"Plotting 1×{n_cols} square grid…")
        plot_grid(cache,
                  panel_grid=PANEL_GRID_1ROW,
                  panel_w=PANEL_H,        # square: width = height
                  panel_h=PANEL_H,
                  legend_w=3.6,
                  ncol_legend=2,
                  out_path=OUT_PATH_1ROW)
    else:
        print(f"Plotting {N_DISPLAY_ROWS}×{N_DISPLAY_COLS} grid…")
        plot_grid(cache,
                  panel_grid=PANEL_GRID,
                  panel_w=PANEL_W,
                  panel_h=PANEL_H,
                  legend_w=LEGEND_W,
                  ncol_legend=1,
                  out_path=OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
