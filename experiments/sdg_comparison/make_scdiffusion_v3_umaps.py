"""
Compare old scDiffusion vs. new scDiffusion-v3 (faithful) UMAPs.

Generates 2 figures:
  figures/umaps/umap_scdiff_v3_10d_t1.png
  figures/umaps/umap_scdiff_v3_50d_t1.png

Each figure has 3 panels: Real data | scDiffusion (old) | scDiffusion-v3 faithful
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import experiments.sdg_comparison.make_sdg_umaps as umap_mod
from experiments.sdg_comparison.make_sdg_umaps import (
    DATASET_CFG, make_single_trial_figure, OUT_DIR, DATA_ROOT
)
from pathlib import Path

# Use full data — no subsampling
umap_mod.N_CELLS_SINGLE  = 10_000_000
umap_mod.N_CELLS_COMBINED = 10_000_000
umap_mod.N_CELLS_REAL    = 10_000_000

# Inject v3 faithful into the ok methods dict
DATASET_CFG["ok"]["methods"]["scDiff-v3 faithful"] = (
    DATA_ROOT / "ok" / "scdiffusion_v3" / "faithful"
)

METHODS = ["scDesign2", "scDiffusion", "scDiff-v3 faithful"]

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] OK1K 10d — trial 1 (scDiffusion comparison)…")
    make_single_trial_figure("ok", 10, 1,
                              method_names=METHODS,
                              suffix="scdiff_v3_10d_t1",
                              n_cols=3)

    print("\n[2/2] OK1K 50d — trial 1 (scDiffusion comparison)…")
    make_single_trial_figure("ok", 50, 1,
                              method_names=METHODS,
                              suffix="scdiff_v3_50d_t1",
                              n_cols=3)

    print("\nDone. Figures in:", OUT_DIR)
