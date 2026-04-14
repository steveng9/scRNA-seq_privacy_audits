"""
Generate 3-panel UMAP figures for each (donor_count, SDG method):
  Panel 1 — Real data, colored by cell type
  Panel 2 — Synthetic data, colored by cell type
  Panel 3 — Real (orange) + Synthetic (blue) overlaid, to check mixing

Figures saved to figures/umaps/mixing/
  umap_mixing_sd3v_10d.png
  umap_mixing_sd3v_20d.png
  umap_mixing_sd3g_10d.png
  ...

Usage:
    python experiments/sdg_comparison/make_mixing_umaps.py
    python experiments/sdg_comparison/make_mixing_umaps.py --nd 10
    python experiments/sdg_comparison/make_mixing_umaps.py --method sd3v
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
DATA_ROOT = Path("/home/golobs/data")
OUT_DIR   = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps/mixing")

REAL_PATH = DATA_ROOT / "ok" / "full_dataset_cleaned.h5ad"
HVG_PATH  = DATA_ROOT / "ok" / "hvg_full.csv"
CT_COL    = "cell_type"

METHODS = {
    "sd3v":   DATA_ROOT / "ok_sd3v",
    "sd3g":   DATA_ROOT / "ok_sd3g",
    "scvi":   DATA_ROOT / "ok_scvi",
    "scdiff": DATA_ROOT / "ok_scdiff",
}

# Donor counts and trial to use
DONOR_COUNTS = [10, 20]
TRIAL        = 1         # use trial 1 throughout

N_REAL  = 3000   # real cells to subsample
N_SYNTH = 2000   # synth cells to subsample

# Cell-type colour palette (tab20)
TAB20 = plt.cm.get_cmap("tab20")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hvg_genes():
    hvg = pd.read_csv(HVG_PATH, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def subsample(adata, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, min(n, adata.n_obs), replace=False)
    return adata[idx].copy()


def load_real(hvg_genes, n, seed=42):
    """Load and subsample real OneK1K data to HVGs."""
    adata = anndata.read_h5ad(REAL_PATH)
    common = [g for g in hvg_genes if g in adata.var_names]
    adata = adata[:, common].copy()
    adata = subsample(adata, n, seed=seed)
    adata.obs["_source"] = "Real"
    adata.obs["_cell_type"] = adata.obs[CT_COL].astype(str)
    return adata


def load_synth(method_root, nd, trial, hvg_genes, n, seed=42):
    """Load synthetic data for a given method / nd / trial."""
    path = method_root / f"{nd}d" / str(trial) / "datasets" / "synthetic.h5ad"
    if not path.exists():
        return None
    adata = anndata.read_h5ad(path)
    common = [g for g in hvg_genes if g in adata.var_names]
    adata = adata[:, common].copy()
    adata = subsample(adata, n, seed=seed)
    adata.obs["_source"] = "Synth"
    if CT_COL in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[CT_COL].astype(str)
    else:
        adata.obs["_cell_type"] = "unknown"
    return adata


def check_synth_quality(synth):
    """Print basic quality stats."""
    X = synth.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    nz = (X != 0).mean()
    print(f"    Synth: {synth.n_obs} cells x {synth.n_vars} genes, {nz:.1%} nonzero")


def compute_joint_umap(real, synth):
    combined = anndata.concat([real, synth], merge="same")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.scale(combined, max_value=10)
    sc.pp.pca(combined, n_comps=30, random_state=0)
    sc.pp.neighbors(combined, n_pcs=30, random_state=0)
    sc.tl.umap(combined, random_state=0)
    return combined


def build_ct_colormap(cell_types):
    cts = sorted(cell_types)
    n = len(cts)
    cmap = TAB20 if n <= 20 else plt.cm.get_cmap("gist_rainbow", n)
    return {ct: cmap(i / max(n - 1, 1)) for i, ct in enumerate(cts)}


def plot_3panel(combined, method_label, nd, out_path):
    """3-panel UMAP: Real CT | Synth CT | Real+Synth mixing."""
    coords   = combined.obsm["X_umap"]
    sources  = combined.obs["_source"].values
    cts      = combined.obs["_cell_type"].values

    real_mask  = sources == "Real"
    synth_mask = sources == "Synth"

    # Build cell-type colours from real data
    real_ct_set = set(cts[real_mask]) - {"unknown"}
    ct_cmap     = build_ct_colormap(real_ct_set)
    sorted_cts  = sorted(real_ct_set)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # ── Panel 1: Real, coloured by cell type ────────────────────────────────
    ax = axes[0]
    for ct in sorted_cts:
        m = real_mask & (cts == ct)
        if m.sum() == 0:
            continue
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=[ct_cmap[ct]], s=2, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title("Real — cell type", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── Panel 2: Synthetic, coloured by cell type ────────────────────────────
    ax = axes[1]
    synth_ct_set = set(cts[synth_mask]) - {"unknown"}
    if synth_ct_set:
        for ct in sorted_cts:
            m = synth_mask & (cts == ct)
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1],
                       c=[ct_cmap[ct]], s=2, alpha=0.5, linewidths=0, rasterized=True)
    else:
        ax.scatter(coords[synth_mask, 0], coords[synth_mask, 1],
                   color="grey", s=2, alpha=0.4, linewidths=0, rasterized=True)
    ax.set_title(f"Synthetic ({method_label}) — cell type", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── Panel 3: Real (orange) vs Synth (blue) mixing ───────────────────────
    ax = axes[2]
    # Plot synth first so real sits on top for readability
    ax.scatter(coords[synth_mask, 0], coords[synth_mask, 1],
               color="#2196F3", s=2, alpha=0.35, linewidths=0,
               label="Synthetic", rasterized=True)
    ax.scatter(coords[real_mask, 0], coords[real_mask, 1],
               color="#FF6B00", s=2, alpha=0.35, linewidths=0,
               label="Real", rasterized=True)
    ax.set_title("Mixing: Real (orange) vs Synth (blue)", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(markerscale=5, fontsize=8, loc="upper right")

    # Legend for cell types (shared across panels 1 & 2)
    patches = [mpatches.Patch(color=ct_cmap[ct], label=ct) for ct in sorted_cts]
    fig.legend(handles=patches, title="Cell type",
               loc="lower center", ncol=min(len(patches), 7),
               fontsize=7, title_fontsize=8,
               bbox_to_anchor=(0.5, -0.08))

    title = f"OneK1K | {nd} donors | trial {TRIAL} | {method_label}"
    fig.suptitle(title, fontsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(methods_to_run, donor_counts):
    hvg_genes = load_hvg_genes()
    print(f"HVG genes: {len(hvg_genes)}")

    for nd in donor_counts:
        # Load real data once per nd (same subsample for all methods)
        print(f"\nLoading real data for {nd}d…")
        real = load_real(hvg_genes, N_REAL, seed=nd * 7)
        print(f"  Real: {real.n_obs} cells x {real.n_vars} genes")

        for method_key in methods_to_run:
            method_root  = METHODS[method_key]
            method_label = {
                "sd3v":   "scDesign3-V",
                "sd3g":   "scDesign3-G",
                "scvi":   "scVI",
                "scdiff": "scDiffusion",
            }[method_key]

            print(f"\n  [{method_label} {nd}d] Loading synthetic…")
            synth = load_synth(method_root, nd, TRIAL, hvg_genes, N_SYNTH,
                               seed=TRIAL * 100 + nd)
            if synth is None:
                print(f"  SKIP — no data found")
                continue

            check_synth_quality(synth)

            # Subset real to genes also in synth (they may differ for sd3v)
            common_genes = [g for g in real.var_names if g in synth.var_names]
            real_sub  = real[:, common_genes].copy()
            synth_sub = synth[:, common_genes].copy()
            print(f"    Common genes: {len(common_genes)}")

            print(f"    Computing joint UMAP…")
            combined = compute_joint_umap(real_sub, synth_sub)

            out_path = OUT_DIR / f"umap_mixing_{method_key}_{nd}d.png"
            plot_3panel(combined, method_label, nd, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", nargs="+", choices=list(METHODS.keys()),
                        default=list(METHODS.keys()),
                        help="Which SDG methods to plot (default: all)")
    parser.add_argument("--nd", nargs="+", type=int, default=DONOR_COUNTS,
                        help="Donor counts to plot (default: 10 20)")
    args = parser.parse_args()

    print(f"Methods: {args.method}")
    print(f"Donor counts: {args.nd}")
    run(args.method, args.nd)
    print(f"\nAll done. Figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
