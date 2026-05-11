"""
5 rows × 4 columns UMAP grid comparing scDiffusion variants across 50d trials.

Rows: trials 1–5
Columns:
  1. Real data
  2. scDiffusion (old, no_dp)
  3. scDiffusion-v3 / v1_celltypist
  4. scDiffusion-v3 / faithful  (trial 1 only; blank for rows 2–5)

No subsampling — all cells used.

Output: figures/umaps/umap_scdiff_v3_grid_50d.png
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
OUT_DIR   = Path("/home/golobs/scRNA-seq_privacy_audits/figures/umaps")

REAL_PATH     = DATA_ROOT / "ok" / "full_dataset_cleaned.h5ad"
HVG_PATH      = DATA_ROOT / "ok" / "hvg_full.csv"
SPLITS_ROOT   = DATA_ROOT / "ok" / "splits"
OLD_ROOT      = DATA_ROOT / "ok" / "scdiffusion"  / "no_dp"
COMP_ROOT     = DATA_ROOT / "ok" / "scdiffusion_v3" / "v1_celltypist"
FAITH_ROOT    = DATA_ROOT / "ok" / "scdiffusion_v3" / "faithful"

DONOR_COL = "individual"

TRIALS   = [1, 2, 3, 4, 5]
ND       = 50
CT_COL   = "cell_type"

COL_LABELS = [
    "Real data",
    "scDiffusion (old)",
    "scDiff-v3 / v1_celltypist",
    "scDiff-v3 / faithful",
]


def load_hvg_genes():
    hvg = pd.read_csv(HVG_PATH, index_col=0)
    return hvg[hvg["highly_variable"]].index.tolist()


def load_adata(path, label, hvg_genes):
    """Load full h5ad (no subsampling), subset to HVGs, attach labels."""
    if not Path(path).exists():
        return None
    adata = anndata.read_h5ad(path)
    common = [g for g in hvg_genes if g in adata.var_names]
    adata = adata[:, common].copy()
    adata.obs["_source"] = label
    if CT_COL in adata.obs.columns:
        adata.obs["_cell_type"] = adata.obs[CT_COL].astype(str)
    else:
        adata.obs["_cell_type"] = "unknown"
    return adata


def build_color_map(cell_types):
    n = len(cell_types)
    cmap = plt.cm.get_cmap("tab20", n) if n <= 20 else plt.cm.get_cmap("gist_rainbow", n)
    return {ct: cmap(i) for i, ct in enumerate(sorted(cell_types))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading real data…")
    real_full = anndata.read_h5ad(REAL_PATH)
    hvg_genes = load_hvg_genes()
    common_hvg = [g for g in hvg_genes if g in real_full.var_names]
    real_full = real_full[:, common_hvg].copy()
    if CT_COL in real_full.obs.columns:
        real_full.obs["_cell_type"] = real_full.obs[CT_COL].astype(str)
    else:
        real_full.obs["_cell_type"] = "unknown"

    all_adatas = []
    panel_specs = []  # (row, col, _source_tag)

    for trial in TRIALS:
        # Use only the 50 training donors for this trial's real data panel
        split_path = SPLITS_ROOT / f"{ND}d" / str(trial) / "train.npy"
        train_donors = set(np.load(split_path, allow_pickle=True).tolist())
        mask = real_full.obs[DONOR_COL].isin(train_donors)
        r = real_full[mask].copy()
        t_tag_real = f"Real_t{trial}"
        r.obs["_source"] = t_tag_real
        r.obs["_trial"]  = str(trial)
        all_adatas.append(r)
        panel_specs.append((trial, 0, t_tag_real))
        print(f"  Trial {trial}: {mask.sum():,} real cells from {len(train_donors)} training donors")

        # Old scDiffusion
        path_old = OLD_ROOT / f"{ND}d" / str(trial) / "datasets" / "synthetic.h5ad"
        tag_old  = f"scDiffusion_old_t{trial}"
        a = load_adata(path_old, tag_old, hvg_genes)
        if a is not None:
            a.obs["_trial"] = str(trial)
            all_adatas.append(a)
        panel_specs.append((trial, 1, tag_old if a is not None else None))

        # v1_celltypist
        path_comp = COMP_ROOT / f"{ND}d" / str(trial) / "datasets" / "synthetic.h5ad"
        tag_comp  = f"scDiff_v3_comp_t{trial}"
        b = load_adata(path_comp, tag_comp, hvg_genes)
        if b is not None:
            b.obs["_trial"] = str(trial)
            all_adatas.append(b)
        panel_specs.append((trial, 2, tag_comp if b is not None else None))

        # faithful — trial 1 only, shown in every row for reference? No — user said
        # "just the first trial (top row)". So only row 1 gets faithful; others blank.
        if trial == 1:
            path_faith = FAITH_ROOT / f"{ND}d" / "1" / "datasets" / "synthetic.h5ad"
            tag_faith  = "scDiff_v3_faithful_t1"
            c = load_adata(path_faith, tag_faith, hvg_genes)
            if c is not None:
                c.obs["_trial"] = "1"
                all_adatas.append(c)
            panel_specs.append((trial, 3, tag_faith if c is not None else None))
        else:
            panel_specs.append((trial, 3, None))

    print(f"Computing joint UMAP over {len(all_adatas)} source×trial slices "
          f"({sum(a.n_obs for a in all_adatas):,} total cells)…")
    combined = anndata.concat(all_adatas, merge="same")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.scale(combined, max_value=10)
    sc.pp.pca(combined, n_comps=30, random_state=0)
    sc.pp.neighbors(combined, n_pcs=30, random_state=0)
    sc.tl.umap(combined, random_state=0)

    # Build color map from real data cell types
    real_cts  = set(combined.obs.loc[combined.obs["_source"].str.startswith("Real_"), "_cell_type"])
    real_cts.discard("unknown")
    color_map = build_color_map(real_cts)
    known_cts = sorted(color_map.keys())

    coords   = combined.obsm["X_umap"]
    src_vals = combined.obs["_source"].values
    ct_vals  = combined.obs["_cell_type"].values

    n_rows = len(TRIALS)
    n_cols = len(COL_LABELS)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.0 * n_rows),
                              constrained_layout=True)

    for (trial, col, src_tag) in panel_specs:
        row = trial - 1
        ax  = axes[row][col]

        if src_tag is None:
            ax.set_visible(False)
            continue

        mask = src_vals == src_tag
        c    = coords[mask]
        cts  = ct_vals[mask]

        if set(cts) == {"unknown"}:
            ax.scatter(c[:, 0], c[:, 1], color="grey", s=0.5, alpha=0.3,
                       linewidths=0, rasterized=True)
        else:
            for ct in known_cts:
                m2 = cts == ct
                if m2.sum() == 0:
                    continue
                ax.scatter(c[m2, 0], c[m2, 1], c=[color_map[ct]], s=0.5, alpha=0.3,
                           linewidths=0, rasterized=True)

        ax.tick_params(labelsize=6)
        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)

        # Column headers on top row, trial labels on left column
        if row == 0:
            ax.set_title(COL_LABELS[col], fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"Trial {trial}\nUMAP 2", fontsize=8)

    patches = [mpatches.Patch(color=color_map[ct], label=ct) for ct in known_cts]
    if patches:
        fig.legend(handles=patches, title="Cell type",
                   loc="lower center", ncol=min(len(patches), 7),
                   fontsize=7, title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"UMAP: OK1K | {ND} donors | scDiffusion variant comparison",
                 fontsize=12, y=1.01)

    out_path = OUT_DIR / "umap_scdiff_v3_grid_50d.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
