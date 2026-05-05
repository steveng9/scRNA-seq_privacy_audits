"""
scDesign2 + Gaussian-mechanism DP generator (CAMDA Track II submission).

Fits a per-cell-type Gaussian copula via scDesign2, then adds calibrated
Gaussian noise to each covariance matrix before generating synthetic data,
implementing (ε,δ)-differential privacy at the donor level.

Pipeline:
  train()    — run scDesign2 R training; inject DP noise; overwrite .rds copulas
  generate() — run scDesign2 R generation from noised copulas; return AnnData

Config keys (under scdesign2_dp_config):
  epsilon        : DP privacy budget ε  (default: 100.0)
  delta          : DP failure prob  δ  (default: 1e-5)
  dp_variant     : "v2" (recommended, 4× less noise) or "v1"  (default: "v2")
  clip_value     : quantile-normal clip c  (default: 3.0)
  out_model_path : directory where .rds copula files are saved  (required)
  hvg_path       : path to HVG mask CSV (gene × highly_variable bool); computed
                   from training data if the file does not yet exist
  donor_col      : obs column for donor ID (default: "individual")
  n_workers      : parallel training processes (default: 4)
"""

import os
import sys
import subprocess
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from dp_copula import apply_gaussian_dp

from models.sc_base import BaseSingleCellDataGenerator

_R_SCRIPT = os.path.join(_PKG_DIR, "scdesign2.r")


# ---------------------------------------------------------------------------
# Module-level subprocess helpers (picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _run_train_ct(hvg_h5ad, cell_type, copula_path):
    os.makedirs(os.path.dirname(copula_path), exist_ok=True)
    cmd = f"Rscript {_R_SCRIPT} train {hvg_h5ad} {cell_type!r} {copula_path}"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return cell_type, None
    except subprocess.CalledProcessError as e:
        return cell_type, e.output[:300].decode("utf-8", errors="replace")


def _run_gen_ct(n_cells, copula_path, out_rds):
    cmd = f"Rscript {_R_SCRIPT} gen {int(n_cells)} {copula_path} {out_rds}"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ScDesign2DPGenerator(BaseSingleCellDataGenerator):
    """
    scDesign2 with donor-level Gaussian-mechanism DP.

    Set generator_name: scdesign2_dp in config.yaml.
    """

    def __init__(self, config):
        super().__init__(config)
        gcfg = self.generator_config

        self.epsilon     = float(gcfg.get("epsilon",    100.0))
        self.delta       = float(gcfg.get("delta",      1e-5))
        self.dp_variant  = gcfg.get("dp_variant",       "v2")
        self.clip_value  = float(gcfg.get("clip_value", 3.0))
        self.donor_col   = gcfg.get("donor_col",        "individual")
        self.n_workers   = int(gcfg.get("n_workers",    4))
        self.cell_type_col = self.dataset_config["cell_type_col_name"]

        self.hvg_path  = gcfg.get("hvg_path", None)
        self.model_dir = os.path.join(self.home_dir, gcfg["out_model_path"])

    # ------------------------------------------------------------------

    def train(self):
        """Train scDesign2 per cell type, inject DP noise into covariances."""
        import multiprocessing

        os.makedirs(self.model_dir, exist_ok=True)
        train_adata = self.load_train_anndata()
        hvg_mask    = self._get_or_compute_hvg_mask(train_adata)
        ct_labels   = train_adata.obs[self.cell_type_col].values
        ct_counts   = Counter(ct_labels)

        donor_labels = (train_adata.obs[self.donor_col].values
                        if self.donor_col in train_adata.obs.columns else None)
        if donor_labels is None:
            print(f"  [WARN] donor_col '{self.donor_col}' not found — using k_max=1",
                  flush=True)

        tmp_dir  = tempfile.mkdtemp(dir=self.home_dir, prefix="sd2dp_train_")
        hvg_h5ad = os.path.join(tmp_dir, "hvg_train.h5ad")
        try:
            train_adata[:, hvg_mask].copy().write_h5ad(hvg_h5ad)
            del train_adata

            # Step 1: parallel scDesign2 R training
            print(f"scDesign2 training ({len(ct_counts)} cell types)...", flush=True)
            ctx = multiprocessing.get_context("spawn")
            trained = set()
            with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as exe:
                futures = {
                    exe.submit(_run_train_ct, hvg_h5ad, ct,
                               os.path.join(self.model_dir, f"{ct}.rds")): ct
                    for ct in ct_counts
                }
                for fut in as_completed(futures):
                    ct, err = fut.result()
                    if err:
                        print(f"  [WARN] train failed '{ct}': {err}", flush=True)
                    else:
                        trained.add(ct)
                        print(f"  Trained: {ct}", flush=True)

            # Step 2: inject DP noise into each copula
            print(f"Injecting DP noise (ε={self.epsilon}, variant={self.dp_variant})...",
                  flush=True)
            from rpy2.robjects import r as R
            from rpy2.robjects.vectors import FloatVector

            for ct in trained:
                copula_path = os.path.join(self.model_dir, f"{ct}.rds")
                if not os.path.exists(copula_path):
                    continue

                copula_rds = R["readRDS"](copula_path)
                copula_ct  = copula_rds.rx2(str(ct))
                cov_mat_r  = copula_ct.rx2("cov_mat")

                # Check for NULL / vine copula
                from rpy2.rinterface_lib.sexp import NULLType as _NullType
                if isinstance(cov_mat_r, _NullType):
                    print(f"  [SKIP] '{ct}' has no cov_mat — skipping DP", flush=True)
                    continue

                primary_genes = list(copula_ct.rx2("gene_sel1").names)
                primary_marg  = copula_ct.rx2("marginal_param1")

                minimal_copula = {
                    "cov_matrix":        cov_mat_r,
                    "primary_genes":     primary_genes,
                    "primary_marginals": primary_marg,
                }

                n_cells_ct = int((ct_labels == ct).sum())
                if donor_labels is not None:
                    per_donor = Counter(donor_labels[ct_labels == ct])
                    k_max = max(per_donor.values()) if per_donor else 1
                else:
                    k_max = 1

                if n_cells_ct <= k_max:
                    print(f"  [SKIP] '{ct}' n_cells={n_cells_ct} ≤ k_max={k_max}",
                          flush=True)
                    continue

                noised = apply_gaussian_dp(
                    minimal_copula,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    n_cells=n_cells_ct,
                    k_max=k_max,
                    clip_value=self.clip_value,
                    dp_variant=self.dp_variant,
                )

                # Patch cov_mat in the R object and re-save
                G    = noised["cov_matrix"].shape[0]
                flat = noised["cov_matrix"].flatten(order="F").tolist()
                R.assign("ppml_copula",   copula_rds)
                R.assign("ppml_noised",   FloatVector(flat))
                ct_safe = ct.replace('"', '\\"')
                R(f'ppml_copula[["{ct_safe}"]][["cov_mat"]] '
                  f'<- matrix(ppml_noised, nrow={G}, ncol={G})')
                R(f'saveRDS(ppml_copula, file="{copula_path}")')
                print(f"  DP noised: {ct}", flush=True)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------

    def generate(self):
        """Generate from the (noised) .rds copulas; return AnnData."""
        import pyreadr

        train_adata = self.load_train_anndata()
        hvg_mask    = self._get_or_compute_hvg_mask(train_adata)
        ct_labels   = train_adata.obs[self.cell_type_col].values
        ct_counts   = Counter(ct_labels)
        gene_names  = train_adata.var_names.tolist()
        hvg_idx     = np.where(hvg_mask)[0]
        n_vars_full = train_adata.n_vars
        del train_adata

        tmp_dir = tempfile.mkdtemp(dir=self.home_dir, prefix="sd2dp_gen_")
        try:
            print(f"Generating synthetic data ({len(ct_counts)} cell types)...", flush=True)
            with ProcessPoolExecutor(max_workers=self.n_workers) as exe:
                futures = {}
                for ct, n in ct_counts.items():
                    copula_path = os.path.join(self.model_dir, f"{ct}.rds")
                    out_rds     = os.path.join(tmp_dir, f"synth_{ct}.rds")
                    if os.path.exists(copula_path):
                        futures[exe.submit(_run_gen_ct, n, copula_path, out_rds)] = ct
                for fut in as_completed(futures):
                    ct = futures[fut]
                    ok = fut.result()
                    status = "Generated" if ok else "[WARN] failed"
                    print(f"  {status}: {ct}", flush=True)

            # Assemble full-gene AnnData
            all_counts = []
            all_ct_obs = []

            for ct in ct_counts:
                out_rds = os.path.join(tmp_dir, f"synth_{ct}.rds")
                if not os.path.exists(out_rds):
                    continue
                mat_dict  = pyreadr.read_r(out_rds)
                counts_np = list(mat_dict.values())[0].to_numpy()
                # scDesign2 outputs genes×cells; transpose → cells×genes
                n_ct_cells = counts_np.shape[1]
                all_counts.append(counts_np.T)
                all_ct_obs.extend([ct] * n_ct_cells)

            if not all_counts:
                raise RuntimeError("All cell-type generation steps failed.")

            synth_hvg = np.vstack(all_counts)                          # (n_cells, n_hvg)
            full_X    = sp.lil_matrix((len(all_ct_obs), n_vars_full))
            for col_i, g_i in enumerate(hvg_idx):
                full_X[:, g_i] = synth_hvg[:, col_i]

            synth = ad.AnnData(X=full_X.tocsr().astype(np.float64))
            synth.obs[self.cell_type_col] = all_ct_obs
            synth.var_names = gene_names
            print(f"  → {synth.n_obs:,} cells × {synth.n_vars} genes", flush=True)
            return synth

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------

    def load_from_checkpoint(self):
        pass

    # ------------------------------------------------------------------

    def _get_or_compute_hvg_mask(self, adata):
        if self.hvg_path and os.path.exists(self.hvg_path):
            hvg_df = pd.read_csv(self.hvg_path, index_col=0)
            if len(hvg_df) != len(adata.var_names):
                hvg_df = hvg_df.reindex(adata.var_names).fillna(False)
            return hvg_df["highly_variable"].values.astype(bool)

        tmp = adata.copy()
        tmp.layers["counts"] = tmp.X.copy()
        sc.pp.normalize_total(tmp, layer="counts", target_sum=1e4)
        sc.pp.log1p(tmp, layer="counts")
        sc.pp.highly_variable_genes(tmp, layer="counts",
                                     min_mean=0.0125, max_mean=3, min_disp=0.5)
        mask = tmp.var["highly_variable"].values.astype(bool)
        print(f"  Computed {mask.sum()} HVGs from training data", flush=True)
        if self.hvg_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.hvg_path)), exist_ok=True)
            pd.Series(mask, index=adata.var_names,
                      name="highly_variable").to_csv(self.hvg_path)
        return mask
