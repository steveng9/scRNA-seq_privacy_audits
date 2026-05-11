"""
make_neurips_3row_umap.py — 3×5 UMAP grid for NeurIPS paper.

Rows   = OneK1K 50d t1 | AIDA 50d t1 | HFRA 10d t1
Columns = Real | scDesign2 | scVI | ZINBWave | scDiffusion

UMAP embeddings are cached per row under figures/umaps/umap_cache/.
Re-runs load cached rows and skip recomputation. Use --force-row N
(0=OneK1K, 1=AIDA, 2=HFRA) to recompute a specific row, or --force
to recompute all rows.

Output: figures/umaps/umap_neurips_3row.png

Usage:
    python experiments/sdg_comparison/make_neurips_3row_umap.py
    python experiments/sdg_comparison/make_neurips_3row_umap.py --force-row 2
    python experiments/sdg_comparison/make_neurips_3row_umap.py --force --dpi 200
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
# Layout config
# ---------------------------------------------------------------------------

DATA_ROOT  = Path("/home/golobs/data/scMAMAMIA")
OUT_DIR    = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps")
CACHE_DIR  = OUT_DIR / "umap_cache" / "neurips3row"
OUT_PATH   = OUT_DIR / "umap_neurips_3row.png"

# Per-row dataset configuration
ROWS = [
    {
        "dataset":      "ok",
        "nd":           50,
        "trial":        1,
        "row_label":    "OneK1K\n50 donors",
        "donor_col":    "individual",
        "ct_col":       "cell_type",
        "ct_label_col": "cell_label",   # maps numeric cell_type → readable name
        "sdiff_variant": "faithful",
    },
    {
        "dataset":      "aida",
        "nd":           50,
        "trial":        1,
        "row_label":    "AIDA\n50 donors",
        "donor_col":    "individual",
        "ct_col":       "cell_type",
        "ct_label_col": None,
        "sdiff_variant": "v1_celltypist",
    },
    {
        "dataset":      "cg",
        "nd":           10,
        "trial":        1,
        "row_label":    "HFRA\n10 donors",
        "donor_col":    "individual",
        "ct_col":       "cell_type",
        "ct_label_col": None,
        "sdiff_variant": "faithful",
    },
]

# Columns: (display_label, path_template_relative_to_dataset_root)
# {variant} in path is substituted per row from row["sdiff_variant"]
SDG_METHODS = [
    ("scDesign2",   "scdesign2/no_dp"),
    ("scVI",        "scvi/no_dp"),
    ("ZINBWave",    "zinbwave/no_dp"),
    ("scDiffusion", "scdiffusion_v3/{variant}"),
]

COL_LABELS = ["Real"] + [lbl for lbl, _ in SDG_METHODS]
N_COLS = len(COL_LABELS)   # 9
N_ROWS = len(ROWS)         # 3

DPI         = 150

# ---------------------------------------------------------------------------
# Cell-type abbreviations — applied at plot time (caches store raw names)
# ---------------------------------------------------------------------------

CT_ABBREV = {
    # AIDA (CellOntology long names → compact immunology abbreviations)
    "B_cell":                                                   "B cell",
    "CD14-low__CD16-positive_monocyte":                         "CD14⁻CD16⁺ Mono",
    "CD14-positive_monocyte":                                   "CD14⁺ Mono",
    "CD141-positive_myeloid_dendritic_cell":                    "CD141⁺ mDC",
    "CD16-negative__CD56-bright_natural_killer_cell__human":    "NK CD56ᵇʳⁱ",
    "CD16-positive__CD56-dim_natural_killer_cell__human":       "NK CD56ᵈⁱᵐ",
    "CD1c-positive_myeloid_dendritic_cell":                     "CD1c⁺ mDC",
    "CD4-positive__alpha-beta_T_cell":                          "CD4⁺ αβ T",
    "CD4-positive__alpha-beta_cytotoxic_T_cell":                "CD4⁺ CTL",
    "CD8-positive__alpha-beta_T_cell":                          "CD8⁺ αβ T",
    "CD8-positive__alpha-beta_cytotoxic_T_cell":                "CD8⁺ CTL",
    "CD8-positive__alpha-beta_memory_T_cell":                   "CD8⁺ Mem T",
    "T_cell":                                                   "T cell",
    "central_memory_CD4-positive__alpha-beta_T_cell":           "CD4⁺ TCM",
    "conventional_dendritic_cell":                              "cDC",
    "dendritic_cell":                                           "DC",
    "double_negative_T_regulatory_cell":                        "DN Treg",
    "effector_memory_CD4-positive__alpha-beta_T_cell":          "CD4⁺ TEM",
    "erythrocyte":                                              "Erythrocyte",
    "gamma-delta_T_cell":                                       "γδ T",
    "innate_lymphoid_cell":                                     "ILC",
    "mature_B_cell":                                            "Mature B",
    "memory_B_cell":                                            "Mem B",
    "monocyte":                                                 "Monocyte",
    "mucosal_invariant_T_cell":                                 "MAIT",
    "naive_B_cell":                                             "Naive B",
    "naive_thymus-derived_CD4-positive__alpha-beta_T_cell":     "CD4⁺ Naive T",
    "naive_thymus-derived_CD8-positive__alpha-beta_T_cell":     "CD8⁺ Naive T",
    "natural_killer_cell":                                      "NK",
    "plasma_cell":                                              "Plasma",
    "plasmacytoid_dendritic_cell":                              "pDC",
    "platelet":                                                 "Platelet",
    "regulatory_T_cell":                                        "Treg",
    # HFRA / cg (retinal cell types)
    "Mueller_cell":                                             "Müller",
    "amacrine_cell":                                            "Amacrine",
    "microglial_cell":                                          "Microglia",
    "retina_horizontal_cell":                                   "Horizontal",
    "retinal_astrocyte":                                        "Astrocyte",
    "retinal_cone_cell":                                        "Cone",
    "retinal_ganglion_cell":                                    "RGC",
    "retinal_progenitor_cell":                                   "Progenitor",
    "retinal_rod_cell":                                         "Rod",
}
PANEL_W     = 2.1   # inches per data panel
PANEL_H     = 2.5   # taller to accommodate larger legend fonts
LEGEND_W    = 3.0   # inches for the legend column (wide enough for 2-col aida legend)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cache_path(row_cfg):
    ds = row_cfg["dataset"]
    nd = row_cfg["nd"]
    t  = row_cfg["trial"]
    return CACHE_DIR / f"{ds}_{nd}d_t{t}.pkl"


def load_hvg_genes(dataset):
    hvg_path = DATA_ROOT / dataset / "hvg_full.csv"
    hvg = pd.read_csv(hvg_path, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def build_ct_name_map(real_path, ct_col, ct_label_col):
    """Return dict mapping ct_col → display name. If no label col, identity map."""
    if ct_label_col is None:
        return {}
    tmp = anndata.read_h5ad(real_path, backed="r")
    obs = tmp.obs[[ct_col, ct_label_col]].drop_duplicates()
    mapping = dict(zip(obs[ct_col].astype(str), obs[ct_label_col].astype(str)))
    tmp.file.close()
    return mapping


def attach_cell_type(adata, ct_name_map, ct_col, use_label_col=False, ct_label_col=None):
    if use_label_col and ct_label_col and ct_label_col in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[ct_label_col].astype(str)
    elif ct_name_map and ct_col in adata.obs.columns:
        adata.obs["_cell_type"] = (
            adata.obs[ct_col].astype(str).map(ct_name_map).fillna("unknown")
        )
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


# Per-row curated palettes for the 10 most prominent cell types.
# Each row has a distinct hue family; rarer types fall back to softened tab20b/tab20c.
_ROW_PALETTES = [
    # Row 0 (OneK1K) — ocean: blues and teals, with deliberate lightness range
    [
        "#2878A8",  # deep navy blue      (dark anchor)
        "#5B9BD5",  # steel blue          (medium)
        "#A0CBE8",  # light sky blue      (light)
        "#226878",  # deep teal           (dark anchor)
        "#4DBBCC",  # ocean teal          (medium)
        "#A8DDE8",  # pale teal           (light)
        "#6870B8",  # periwinkle          (purple accent for contrast)
        "#3EA090",  # teal-green
        "#C0D8EC",  # very pale blue
        "#4058C0",  # royal blue          (saturated anchor)
    ],
    # Row 1 (AIDA) — sunset: pinks muted, purples, salmons, oranges
    [
        "#F4875B",  # salmon
        "#9B72CF",  # purple
        "#D890A8",  # dusty pink          (was #F07BAD — less bright)
        "#F5B942",  # warm orange
        "#C47EC0",  # orchid
        "#F0956A",  # peach
        "#BF6890",  # muted rose          (was #D870A8 — less bright)
        "#A868C8",  # medium purple
        "#F8C888",  # light amber
        "#C88090",  # dusty mauve         (was #DC8890 — harmonised)
    ],
    # Row 2 (HFRA) — earth: greens and ambers with wider lightness spread
    [
        "#3A7848",  # dark forest green   (dark anchor)
        "#E0A040",  # amber
        "#8CC070",  # leaf green          (medium-bright)
        "#C87850",  # terracotta
        "#C8D870",  # yellow-green        (light, more contrast)
        "#C09040",  # golden brown
        "#A0D890",  # pale mint           (light anchor)
        "#E8B068",  # warm gold
        "#587840",  # olive               (darker green contrast)
        "#F0D090",  # light straw         (very pale warm)
    ],
]


def build_color_map(cell_types, counts=None, row_palette=None):
    """Build a color map for up to 50 cell types.
    Most-frequent types get the row-specific curated palette; rarer types get softened tab20b/tab20c."""
    if counts is not None:
        sorted_cts = sorted(cell_types, key=lambda ct: -counts.get(ct, 0))
    else:
        sorted_cts = sorted(cell_types)

    def _soften(rgba, blend):
        r, g, b, a = rgba
        return (r + (1 - r) * blend, g + (1 - g) * blend, b + (1 - b) * blend, a)

    prominent = row_palette if row_palette is not None else _ROW_PALETTES[0]
    t20b = plt.cm.get_cmap("tab20b", 20)
    t20c = plt.cm.get_cmap("tab20c", 20)
    palette = (
        [to_rgba(c) for c in prominent] +                # top 10: curated hues
        [_soften(t20b(i), 0.22) for i in range(20)] +    # rarer: lightly softened
        [_soften(t20c(i), 0.22) for i in range(20)]
    )
    return {ct: palette[i] for i, ct in enumerate(sorted_cts)}


# ---------------------------------------------------------------------------
# Per-row: load data and compute UMAP
# ---------------------------------------------------------------------------

def compute_row(row_cfg, verbose=True):
    """Load data, compute joint UMAP for one row. Returns cache dict."""
    ds  = row_cfg["dataset"]
    nd  = row_cfg["nd"]
    t   = row_cfg["trial"]
    vc  = row_cfg["sdiff_variant"]

    real_path   = DATA_ROOT / ds / "full_dataset_cleaned.h5ad"
    splits_root = DATA_ROOT / ds / "splits"
    ct_col      = row_cfg["ct_col"]
    ct_label_col = row_cfg["ct_label_col"]
    donor_col   = row_cfg["donor_col"]

    if verbose:
        print(f"  Loading HVGs and cell-type name map for {ds}…")
    hvg_genes   = load_hvg_genes(ds)
    ct_name_map = build_ct_name_map(real_path, ct_col, ct_label_col)

    # Load real dataset restricted to HVGs (backed to save RAM)
    if verbose:
        print(f"  Loading real dataset: {real_path}")
    real_full = anndata.read_h5ad(real_path)
    common = [g for g in hvg_genes if g in real_full.var_names]
    real_full = real_full[:, common].copy()

    # Subset to training donors only
    train_donors = set(
        np.load(splits_root / f"{nd}d" / str(t) / "train.npy",
                allow_pickle=True).tolist()
    )
    mask = real_full.obs[donor_col].isin(train_donors)
    real = real_full[mask].copy()
    del real_full
    real.obs["_source"] = "Real"
    real = attach_cell_type(real, ct_name_map, ct_col,
                            use_label_col=True, ct_label_col=ct_label_col)
    if verbose:
        print(f"  Real: {mask.sum():,} cells from {len(train_donors)} training donors")

    # Load each SDG method
    adatas  = [real]
    present = ["Real"]
    path_overrides = row_cfg.get("path_overrides", {})

    for method_idx, (col_label, path_tpl) in enumerate(SDG_METHODS):
        if method_idx in path_overrides:
            sdg_rel = path_overrides[method_idx]
        else:
            sdg_rel = path_tpl.replace("{variant}", vc)
        path = DATA_ROOT / ds / sdg_rel / f"{nd}d" / str(t) / "datasets" / "synthetic.h5ad"
        if not path.exists():
            if verbose:
                print(f"  SKIP {col_label} — missing: {path}")
            adatas.append(None)
            present.append(None)
            continue

        a = anndata.read_h5ad(path)
        common_g = [g for g in hvg_genes if g in a.var_names]
        a = a[:, common_g].copy()
        a.obs["_source"] = col_label
        a = attach_cell_type(a, ct_name_map, ct_col)
        adatas.append(a)
        present.append(col_label)
        if verbose:
            print(f"  Loaded {col_label}: {a.n_obs:,} cells")

    valid = [a for a in adatas if a is not None]
    total = sum(a.n_obs for a in valid)
    if verbose:
        print(f"  → {total:,} cells across {len(valid)} sources; computing UMAP…",
              flush=True)

    combined = compute_joint_umap(valid)

    cache = {
        "coords":    combined.obsm["X_umap"].astype(np.float32),
        "source":    combined.obs["_source"].values,
        "cell_type": combined.obs["_cell_type"].values,
        "present":   present,       # list[str|None], len = N_COLS
        "row_cfg":   row_cfg,
    }
    return cache


def load_or_compute_row(row_cfg, force=False, verbose=True):
    cp = cache_path(row_cfg)
    if not force and cp.exists():
        if verbose:
            print(f"  Loading cached UMAP: {cp}")
        with open(cp, "rb") as f:
            return pickle.load(f)

    if verbose:
        ds = row_cfg["dataset"]
        nd = row_cfg["nd"]
        t  = row_cfg["trial"]
        print(f"  Computing UMAP for {ds} {nd}d trial {t}…")

    cache = compute_row(row_cfg, verbose=verbose)

    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    if verbose:
        print(f"  Saved UMAP cache: {cp}")
    return cache


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(rows_cache):
    fig_w = PANEL_W * N_COLS + LEGEND_W
    fig_h = PANEL_H * N_ROWS

    # Build per-row color maps; frequency-sort so prominent types get best colors
    color_maps = []
    for row_idx, cache in enumerate(rows_cache):
        ct_abbrev  = np.array([CT_ABBREV.get(ct, ct) for ct in cache["cell_type"]])
        real_mask  = cache["source"] == "Real"
        counts: dict = {}
        for ct in ct_abbrev[real_mask]:
            counts[ct] = counts.get(ct, 0) + 1
        all_cts = set(ct_abbrev)
        color_maps.append(build_color_map(all_cts, counts=counts,
                                          row_palette=_ROW_PALETTES[row_idx]))

    # GridSpec: N_COLS data columns + 1 legend column
    fig = plt.figure(figsize=(fig_w, fig_h))
    width_ratios = [PANEL_W] * N_COLS + [LEGEND_W]
    gs = GridSpec(
        N_ROWS, N_COLS + 1,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.04,
        hspace=0.05,
        left=0.045,
        right=0.995,
        top=0.94,
        bottom=0.04,
    )

    # Create data axes with per-row shared x/y
    axes = []
    for r in range(N_ROWS):
        row_axes = []
        for c in range(N_COLS):
            if c == 0:
                ax = fig.add_subplot(gs[r, c])
            else:
                ax = fig.add_subplot(gs[r, c],
                                     sharex=row_axes[0],
                                     sharey=row_axes[0])
            row_axes.append(ax)
        axes.append(row_axes)

    # Legend axes (one per row, rightmost column)
    legend_axes = [fig.add_subplot(gs[r, N_COLS]) for r in range(N_ROWS)]
    for ax in legend_axes:
        ax.axis("off")

    # Draw panels
    ct_abbrev_rows = [
        np.array([CT_ABBREV.get(ct, ct) for ct in cache["cell_type"]])
        for cache in rows_cache
    ]
    for row_idx, (row_cfg, cache) in enumerate(zip(ROWS, rows_cache)):
        coords   = cache["coords"]
        src_vals = cache["source"]
        ct_vals  = ct_abbrev_rows[row_idx]
        present  = cache["present"]
        cmap     = color_maps[row_idx]
        known_cts = sorted(cmap.keys())

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        pad_x = -(x_max - x_min) * 0.025   # crop 2.5% each side → ~5% zoom
        pad_y = -(y_max - y_min) * 0.025

        present_set = set(p for p in present if p is not None)
        for col_idx, col_label in enumerate(COL_LABELS):
            src = col_label if col_label in present_set else None
            ax = axes[row_idx][col_idx]

            if src is None:
                # No data for this panel — leave blank with a note
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="#888888")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                mask = src_vals == src
                c    = coords[mask]
                cts  = ct_vals[mask]

                for ct in known_cts:
                    m2 = cts == ct
                    if not m2.any():
                        continue
                    ax.scatter(c[m2, 0], c[m2, 1],
                               c=[cmap[ct]], s=1.0, alpha=0.5,
                               linewidths=0, rasterized=True)

                ax.set_xlim(x_min - pad_x, x_max + pad_x)
                ax.set_ylim(y_min - pad_y, y_max + pad_y)

            ax.set_xticks([])
            ax.set_yticks([])

            # Column header (top row only)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=13, fontweight="bold", pad=3)

            # Row label (first column only)
            if col_idx == 0:
                ax.set_ylabel(row_cfg["row_label"] + "\nUMAP 2",
                              fontsize=13, labelpad=3)

            # x-axis label (bottom row only)
            if row_idx == N_ROWS - 1:
                ax.set_xlabel("UMAP 1", fontsize=11.5, labelpad=2)

        # Per-row legend
        n_cts = len(known_cts)
        ordered_cts = sorted(cmap.keys())
        patches = [mpatches.Patch(facecolor=cmap[ct], label=ct)
                   for ct in ordered_cts]
        ncol        = 3 if n_cts > 25 else 2
        lbl_spacing = 0.13 if n_cts > 25 else 0.25

        # For the 3-col AIDA legend, put the widest label alone on the last row.
        # Matplotlib fills column-major (top→bottom, then next col), so we must
        # explicitly arrange: col0 = rest[:nrows-1] + lone, col1 = rest[…] + blank,
        # col2 = rest[…]  — which puts lone at bottom of col0, alone on that row.
        if n_cts > 25:
            lone_lbl = "CD14⁻CD16⁺ Mono"
            lone = [p for p in patches if p.get_label() == lone_lbl]
            if lone:
                rest  = [p for p in patches if p.get_label() != lone_lbl]
                blank = mpatches.Patch(color="none", label="")
                nrows = -(-(len(rest) + 2) // ncol)   # ceiling division
                col0  = rest[:nrows - 1] + lone
                col1  = rest[nrows - 1 : 2 * (nrows - 1)] + [blank]
                col2  = rest[2 * (nrows - 1):]
                patches = col0 + col1 + col2

        legend_y = 0.62 if row_idx == N_ROWS - 1 else 0.98
        legend_axes[row_idx].legend(
            handles=patches,
            title="Cell type",
            loc="upper left",
            bbox_to_anchor=(0.02, legend_y),
            bbox_transform=legend_axes[row_idx].transAxes,
            ncol=ncol,
            fontsize=11,
            title_fontsize=12,
            frameon=False,
            labelspacing=lbl_spacing,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=0.8,
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

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dpi", type=int, default=DPI,
                    help="Output DPI (default: 150)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute UMAP embeddings for all rows")
    ap.add_argument("--force-row", type=int, metavar="N", action="append",
                    dest="force_rows", default=[],
                    help="Recompute UMAP for row N (0=OneK1K, 1=AIDA, 2=HFRA); "
                         "may be repeated")
    ap.add_argument("--compute-only", action="store_true",
                    help="Compute and cache UMAPs but do not produce plot")
    args = ap.parse_args()
    DPI = args.dpi

    print(f"NeurIPS 3-row UMAP grid: {N_ROWS} rows × {N_COLS} cols, dpi={DPI}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rows_cache = []
    for row_idx, row_cfg in enumerate(ROWS):
        force_this = args.force or (row_idx in args.force_rows)
        ds = row_cfg["dataset"]
        nd = row_cfg["nd"]
        t  = row_cfg["trial"]
        print(f"\nRow {row_idx}: {ds} {nd}d trial {t} "
              f"({'recomputing' if force_this else 'checking cache'})…")
        cache = load_or_compute_row(row_cfg, force=force_this)
        rows_cache.append(cache)
        # Report what's present
        present_set = set(p for p in cache["present"] if p is not None)
        missing = [lbl for lbl in COL_LABELS if lbl not in present_set]
        if missing:
            print(f"  Missing panels: {missing}")
        else:
            print(f"  All {N_COLS} panels present.")

    if args.compute_only:
        print("\n--compute-only: skipping plot generation.")
        return

    print(f"\nPlotting {N_ROWS}×{N_COLS} grid…")
    plot_grid(rows_cache)
    print("Done.")


if __name__ == "__main__":
    main()
