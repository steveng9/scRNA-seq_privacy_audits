"""
Onboard the Human Lung Cell Atlas (HLCA) core as the 4th dataset ("hlca").

Source: CZ CELLxGENE — "The integrated Human Lung Cell Atlas", core object
    dataset_id 066943a2-fdac-4b29-b348-40cede398e4e
    (584,944 cells, 107 donors, 50 cell types, 5 protocols, all `normal`)
    downloaded to  scMAMAMIA/hlca/full_dataset.h5ad

Produces, matching the conventions of ok/aida/cg:
  - full_dataset_cleaned.h5ad : X = raw integer counts (moved from .raw.X),
      var_names = gene symbols (feature_name, made unique),
      obs["individual"] (donor), obs["cell_type"] (formatted), + metadata.
  - hvg.csv / hvg_full.csv    : gene-indexed `highly_variable` bool mask,
      computed with the canonical recipe (compute_hvgs.py):
      normalize_total(1e4) -> log1p -> highly_variable_genes(0.0125, 3, 0.5),
      on a 200k-cell subsample (seed 42).

Run:
    conda run --no-capture-output -n tabddpm_ \
        python experiments/hlca_onboard/onboard_hlca.py
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

sys.path.insert(0, "/home/golobs/scRNA-seq_privacy_audits/src")
from data.clean_data import format_ct_name, extract_age  # noqa: E402

BASE = "/home/golobs/data/scMAMAMIA/hlca"
RAW = os.path.join(BASE, "full_dataset.h5ad")
CLEAN = os.path.join(BASE, "full_dataset_cleaned.h5ad")
HVG_PARAMS = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)
MAX_CELLS = 200_000


def clean():
    print("[clean] reading raw h5ad ...", flush=True)
    adata = ad.read_h5ad(RAW)
    print(f"[clean] full shape {adata.shape}; raw shape "
          f"{adata.raw.shape if adata.raw is not None else None}", flush=True)

    # Raw integer counts -> primary AnnData (X = counts, as ok/aida/cg store it).
    rad = adata.raw.to_adata()          # X=raw.X, var=raw.var, obs=parent obs
    del adata

    chk = rad.X[:200].toarray() if sp.issparse(rad.X) else np.asarray(rad.X[:200])
    print(f"[clean] counts frac_integer={np.mean(np.isclose(chk, np.round(chk))):.3f} "
          f"max={chk.max():.1f}", flush=True)

    # var_names = gene symbols (feature_name), keep Ensembl id as a column.
    rad.var["ensembl_id"] = rad.var_names.astype(str)
    rad.var_names = rad.var["feature_name"].astype(str).values
    rad.var_names_make_unique()

    # obs renames (mirrors clean_data.clean_dataset) + formatting.
    ren = {}
    for src, dst in [("donor_id", "individual"),
                     ("self_reported_ethnicity", "ethnicity"),
                     ("development_stage", "age")]:
        if src in rad.obs.columns:
            ren[src] = dst
    rad.obs.rename(columns=ren, inplace=True)
    if "age" in rad.obs.columns:
        rad.obs["age"] = rad.obs["age"].apply(extract_age)
    rad.obs["cell_type"] = rad.obs["cell_type"].apply(format_ct_name).astype(str)

    # Coerce every non-numeric obs column to a plain string dtype so the h5ad
    # write never hits a mixed-type categorical (e.g. `age` = int + str after
    # extract_age).  The pipeline only reads `individual` and `cell_type`.
    for col in rad.obs.columns:
        dt = rad.obs[col].dtype
        if str(dt) == "category" or dt == object:
            rad.obs[col] = rad.obs[col].astype(str)

    # Slim: counts + obs + var only (pipeline recomputes normalization itself).
    if not sp.issparse(rad.X):
        rad.X = sp.csr_matrix(rad.X)
    rad.X = rad.X.tocsr().astype(np.float32)
    for attr in (rad.obsm, rad.varm, rad.obsp, rad.varp, rad.uns):
        attr.clear()
    rad.raw = None

    print(f"[clean] writing {CLEAN} ({rad.shape}) ...", flush=True)
    rad.write_h5ad(CLEAN)

    n_don = rad.obs["individual"].nunique()
    n_ct = rad.obs["cell_type"].nunique()
    cpd = rad.obs.groupby("individual").size()
    print(f"[clean] DONE. donors={n_don} cell_types={n_ct} "
          f"cells/donor min={cpd.min()} median={int(cpd.median())} max={cpd.max()}",
          flush=True)
    return rad


def compute_hvg(adata):
    print("[hvg] computing canonical HVG mask ...", flush=True)
    if adata.n_obs > MAX_CELLS:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(adata.n_obs, size=MAX_CELLS, replace=False))
        sub = adata[idx].copy()
        print(f"[hvg] subsampled to {MAX_CELLS} cells (seed 42)", flush=True)
    else:
        sub = adata.copy()
    sub.layers["counts"] = sub.X.copy()
    sc.pp.normalize_total(sub, layer="counts", target_sum=1e4)
    sc.pp.log1p(sub, layer="counts")
    sc.pp.highly_variable_genes(sub, layer="counts", **HVG_PARAMS)
    hvg = sub.var[["highly_variable"]].copy()
    hvg.to_csv(os.path.join(BASE, "hvg_full.csv"))
    hvg.to_csv(os.path.join(BASE, "hvg.csv"))
    print(f"[hvg] DONE. {int(hvg['highly_variable'].sum())} HVGs of {len(hvg)} genes "
          f"-> hvg.csv + hvg_full.csv", flush=True)


if __name__ == "__main__":
    adata = clean()
    compute_hvg(adata)
    print("[onboard] ALL DONE.", flush=True)
