"""
Generate side-by-side UMAP panels comparing real data vs. synthetic data from
multiple SDG methods (scDesign2, scDesign3-G, scDesign3-V, scVI, scDiffusion).

Figures produced (all saved to figures/umaps/):
  umap_ok_10d_combined.png   – OK1K 10d, all 5 trials pooled per method
  umap_ok_10d_t1.png         – OK1K 10d, trial 1
  umap_ok_10d_t3.png         – OK1K 10d, trial 3
  umap_ok_50d_t1.png         – OK1K 50d, trial 1 (methods with data)
  umap_ok_50d_t2.png         – OK1K 50d, trial 2 (methods with data)
  umap_aida_20d_t1.png       – AIDA 20d, trial 1
  umap_aida_20d_t2.png       – AIDA 20d, trial 2

Usage:
    python experiments/sdg_comparison/make_sdg_umaps.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/golobs/data")
OUT_DIR   = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps")

DATASET_CFG = {
    "ok": {
        "real":         DATA_ROOT / "ok" / "full_dataset_cleaned.h5ad",
        "hvg":          DATA_ROOT / "ok" / "hvg_full.csv",
        "ct_col":       "cell_type",
        "backed_real":  False,   # OK1K ~2 GB — fine to load fully
        "methods": {
            "scDesign2":   DATA_ROOT / "ok",
            "scDesign3-G": DATA_ROOT / "ok_sd3g",
            "scDesign3-V": DATA_ROOT / "ok_sd3v",
            "scVI":        DATA_ROOT / "ok_scvi",
            "scDiffusion": DATA_ROOT / "ok_scdiff",
        },
    },
    "aida": {
        "real":         DATA_ROOT / "aida" / "full_dataset_cleaned.h5ad",
        "hvg":          DATA_ROOT / "aida" / "hvg_full.csv",
        "ct_col":       "cell_type",
        "backed_real":  True,    # AIDA 57 GB — must subsample in backed mode
        "methods": {
            "scVI":        DATA_ROOT / "aida_scvi",
            "scDiffusion": DATA_ROOT / "aida_scdiff",
        },
    },
}

# Cells to subsample per source per panel
N_CELLS_SINGLE  = 1500   # for single-trial panels
N_CELLS_COMBINED = 1500  # per method (drawn from pooled trials) for combined panel
N_CELLS_REAL    = 2000   # real data subsample


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


def load_adata(path, label, hvg_genes, ct_col, n_cells, seed=42,
               backed=False):
    """Load one h5ad, subset to HVGs, subsample, attach source label.

    backed=True uses backed='r' mode to subsample rows before loading
    everything into RAM — important for very large real-data files.
    """
    if not Path(path).exists():
        return None

    if backed:
        # Subsample obs indices first, then load only those rows
        tmp = anndata.read_h5ad(path, backed="r")
        n_total = tmp.n_obs
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, min(n_cells, n_total), replace=False))
        # Read selected rows and relevant columns into memory
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
    """Build a color map for a list of cell types."""
    n = len(cell_types)
    if n <= 20:
        cmap = plt.cm.get_cmap("tab20", n)
    else:
        # For >20 types use hsv-like palette
        cmap = plt.cm.get_cmap("gist_rainbow", n)
    return {ct: cmap(i) for i, ct in enumerate(sorted(cell_types))}


def plot_panel(combined, sources, color_map, title, out_path, n_cols=None):
    """
    Draw one UMAP panel per source side by side.
    Sources without cell_type info are plotted in a neutral grey.
    """
    if n_cols is None:
        n_cols = min(len(sources), 4)
    n_rows = (len(sources) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.2 * n_rows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    coords   = combined.obsm["X_umap"]
    src_vals = combined.obs["_source"].values
    ct_vals  = combined.obs["_cell_type"].values

    known_cts = sorted(color_map.keys())

    for ax, src in zip(axes, sources):
        mask = src_vals == src
        c    = coords[mask]
        cts  = ct_vals[mask]

        if set(cts) == {"unknown"}:
            ax.scatter(c[:, 0], c[:, 1],
                       color="grey", s=1.5, alpha=0.4,
                       linewidths=0, rasterized=True)
        else:
            for ct in known_cts:
                m2 = cts == ct
                if m2.sum() == 0:
                    continue
                ax.scatter(c[m2, 0], c[m2, 1],
                           c=[color_map[ct]], s=1.5, alpha=0.4,
                           linewidths=0, rasterized=True)

        ax.set_title(src, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(sources):]:
        ax.set_visible(False)

    # Legend (skip "unknown")
    patches = [mpatches.Patch(color=color_map[ct], label=ct)
               for ct in known_cts]
    if patches:
        ncol_leg = min(len(patches), 7)
        fig.legend(handles=patches, title="Cell type",
                   loc="lower center", ncol=ncol_leg,
                   fontsize=7, title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(title, fontsize=12, y=1.01)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def make_single_trial_figure(dataset_name, nd, trial, method_names=None,
                              suffix=None, n_cols=None):
    """One panel per method for a single trial."""
    cfg      = DATASET_CFG[dataset_name]
    hvg_genes = load_hvg_genes(cfg["hvg"])
    ct_col   = cfg["ct_col"]
    methods  = cfg["methods"]
    if method_names is not None:
        methods = {k: v for k, v in methods.items() if k in method_names}

    adatas = []

    # Real data first
    real = load_adata(cfg["real"], "Real data", hvg_genes, ct_col,
                      N_CELLS_REAL, seed=42,
                      backed=cfg.get("backed_real", False))
    if real is None:
        print(f"  ERROR: real data missing for {dataset_name}")
        return
    adatas.append(real)

    present_methods = []
    for label, root in methods.items():
        path = root / f"{nd}d" / str(trial) / "datasets" / "synthetic.h5ad"
        a = load_adata(path, label, hvg_genes, ct_col, N_CELLS_SINGLE,
                       seed=trial * 100)
        if a is None:
            print(f"  SKIP {label} {nd}d t{trial} (missing)")
            continue
        adatas.append(a)
        present_methods.append(label)

    if not present_methods:
        print(f"  No synthetic data found — skipping figure.")
        return

    sources = ["Real data"] + present_methods

    ds_label   = "OK1K" if dataset_name == "ok" else "AIDA"
    trial_label = f"trial {trial}"
    tag        = suffix or f"{dataset_name}_{nd}d_t{trial}"
    title      = f"UMAP: {ds_label} | {nd} donors | {trial_label}"

    print(f"  Computing joint UMAP ({len(adatas)} sources × ~{N_CELLS_SINGLE} cells)…")
    combined = compute_joint_umap(adatas)

    # Build color map from real data cell types
    real_cts   = set(combined.obs.loc[combined.obs["_source"] == "Real data", "_cell_type"])
    real_cts.discard("unknown")
    color_map  = build_color_map(real_cts)

    out_path = OUT_DIR / f"umap_{tag}.png"
    plot_panel(combined, sources, color_map, title, out_path, n_cols=n_cols)


def make_combined_trials_figure(dataset_name, nd, trials, method_names=None):
    """Pool multiple trials per method, subsample, then compute joint UMAP."""
    cfg       = DATASET_CFG[dataset_name]
    hvg_genes = load_hvg_genes(cfg["hvg"])
    ct_col    = cfg["ct_col"]
    methods   = cfg["methods"]
    if method_names is not None:
        methods = {k: v for k, v in methods.items() if k in method_names}

    n_per_trial = max(1, N_CELLS_COMBINED // len(trials))

    adatas = []

    # Real data
    real = load_adata(cfg["real"], "Real data", hvg_genes, ct_col,
                      N_CELLS_REAL, seed=0,
                      backed=cfg.get("backed_real", False))
    if real is None:
        print(f"  ERROR: real data missing for {dataset_name}")
        return
    adatas.append(real)

    present_methods = []
    for label, root in methods.items():
        trial_adatas = []
        for t in trials:
            path = root / f"{nd}d" / str(t) / "datasets" / "synthetic.h5ad"
            a = load_adata(path, label, hvg_genes, ct_col,
                           n_per_trial, seed=t * 10)
            if a is not None:
                trial_adatas.append(a)
        if not trial_adatas:
            print(f"  SKIP {label} {nd}d combined (no trials found)")
            continue
        pooled = anndata.concat(trial_adatas, merge="same")
        pooled.obs["_source"] = label
        adatas.append(pooled)
        present_methods.append(label)

    if not present_methods:
        print(f"  No synthetic data found — skipping figure.")
        return

    sources = ["Real data"] + present_methods

    ds_label = "OK1K" if dataset_name == "ok" else "AIDA"
    title    = (f"UMAP: {ds_label} | {nd} donors | "
                f"all {len(trials)} trials combined (~{N_CELLS_COMBINED} cells/method)")

    print(f"  Computing joint UMAP ({len(adatas)} sources, pooled trials)…")
    combined = compute_joint_umap(adatas)

    real_cts  = set(combined.obs.loc[combined.obs["_source"] == "Real data", "_cell_type"])
    real_cts.discard("unknown")
    color_map = build_color_map(real_cts)

    out_path  = OUT_DIR / f"umap_{dataset_name}_{nd}d_combined.png"
    plot_panel(combined, sources, color_map, title, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── OK1K 10d: all methods ─────────────────────────────────────────────
    OK_10D_METHODS = ["scDesign2", "scDesign3-G", "scDesign3-V", "scVI", "scDiffusion"]

    print("\n[1/7] OK1K 10d — combined (all 5 trials)…")
    make_combined_trials_figure("ok", 10, list(range(1, 6)),
                                method_names=OK_10D_METHODS)

    print("\n[2/7] OK1K 10d — trial 1…")
    make_single_trial_figure("ok", 10, 1, method_names=OK_10D_METHODS)

    print("\n[3/7] OK1K 10d — trial 3…")
    make_single_trial_figure("ok", 10, 3, method_names=OK_10D_METHODS)

    # ── OK1K 50d: methods with data ───────────────────────────────────────
    # trial 1: sd3g + scvi + scdiff (sd3v missing)
    print("\n[4/7] OK1K 50d — trial 1…")
    make_single_trial_figure("ok", 50, 1,
                              method_names=["scDesign2", "scDesign3-G",
                                            "scVI", "scDiffusion"])

    # trial 2: all four (sd3v t2 exists)
    print("\n[5/7] OK1K 50d — trial 2…")
    make_single_trial_figure("ok", 50, 2,
                              method_names=["scDesign2", "scDesign3-G",
                                            "scDesign3-V", "scVI", "scDiffusion"])

    # ── AIDA 20d: scVI + scDiffusion ──────────────────────────────────────
    print("\n[6/7] AIDA 20d — trial 1…")
    make_single_trial_figure("aida", 20, 1)

    print("\n[7/7] AIDA 20d — trial 2…")
    make_single_trial_figure("aida", 20, 2)

    print("\nAll done. Figures in:", OUT_DIR)


if __name__ == "__main__":
    main()
