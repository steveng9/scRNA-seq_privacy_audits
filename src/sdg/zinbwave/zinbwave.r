#!/usr/bin/env Rscript
# zinbwave.r — ZINBWave synthetic data generator (R backend)
#
# Usage:
#   Rscript zinbwave.r train  <h5ad_path> <n_latent> <out_rds>
#       Fit a zinbwave model on the data in h5ad_path (already subsetted to
#       one cell type by the Python wrapper). Saves fitted Mu, Pi, Phi,
#       and gene names to out_rds.
#
#   Rscript zinbwave.r generate  <n_cells> <model_rds> <out_rds> <seed>
#       Load a fitted model, bootstrap-resample n_cells from training cells,
#       and generate ZINB counts. Saves a (n_cells × n_genes) integer matrix
#       to out_rds (cells in rows, genes in columns — Python/AnnData convention).
#
# Model: Y_ig ~ ZINB(μ_ig, φ_g, π_ig)  (zero-inflated negative binomial)
#   getMu(model) : N×G matrix of fitted means
#   getPi(model) : N×G matrix of fitted dropout probabilities
#   getPhi(model): G-vector of per-gene dispersion (common across cells)
#
# Generation: bootstrap-sample training cell parameters → sample ZINB counts.

suppressPackageStartupMessages({
  library(zinbwave)
  library(SingleCellExperiment)
  library(BiocParallel)
  library(Matrix)
  library(zellkonverter)
})


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

train_zinbwave <- function(h5ad_path, n_latent, out_rds) {
  cat(sprintf("Reading %s\n", h5ad_path))
  sce <- readH5AD(h5ad_path, use_hdf5 = FALSE)

  # zellkonverter puts counts in the "X" assay; ensure it's accessible
  if (!("counts" %in% assayNames(sce))) {
    assay(sce, "counts") <- assay(sce, "X")
  }

  # zinbwave requires integer counts
  cnt <- assay(sce, "counts")
  if (!is.integer(cnt)) {
    cnt <- round(cnt)
    storage.mode(cnt) <- "integer"
    assay(sce, "counts") <- cnt
  }

  n_cells <- ncol(sce)
  n_genes <- nrow(sce)
  K <- min(as.integer(n_latent), n_cells - 1L)

  cat(sprintf("Fitting zinbwave: %d cells × %d genes, K=%d\n",
              n_cells, n_genes, K))

  BPPARAM <- SerialParam()  # one R process per cell-type already

  model <- zinbFit(
    sce,
    K           = K,
    verbose     = FALSE,
    BPPARAM     = BPPARAM
  )

  mu    <- getMu(model)   # N × G
  pi_m  <- getPi(model)   # N × G
  phi   <- getPhi(model)  # G-vector (common dispersion per gene)

  model_data <- list(
    mu         = mu,
    pi         = pi_m,
    phi        = phi,
    n_cells    = n_cells,
    n_genes    = n_genes,
    gene_names = rownames(sce)
  )

  saveRDS(model_data, file = out_rds)
  cat(sprintf("Model saved to %s\n", out_rds))
}


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

generate_zinbwave <- function(n_cells_new, model_rds, out_rds, seed = 42L) {
  set.seed(seed)

  cat(sprintf("Loading model from %s\n", model_rds))
  md <- readRDS(model_rds)

  mu    <- md$mu     # N_train × G
  pi_m  <- md$pi     # N_train × G
  phi   <- md$phi    # G-vector
  n_train <- nrow(mu)
  G       <- ncol(mu)
  n_new   <- as.integer(n_cells_new)

  cat(sprintf("Generating %d cells from %d training cells × %d genes\n",
              n_new, n_train, G))

  # Bootstrap-sample training cell indices
  idx <- sample(n_train, n_new, replace = TRUE)

  mu_new  <- mu[idx,  , drop = FALSE]   # n_new × G
  pi_new  <- pi_m[idx, , drop = FALSE]  # n_new × G
  # getPhi() returns exp(zeta) = 1/theta where theta is the NB size parameter.
  # For rnbinom(n, mu, size), size = theta = 1/getPhi().
  # Clamp getPhi to avoid 1/0 (very large getPhi → very small size → near-zero samples;
  # clamp getPhi at 1e-10 so size is capped at 1e10 ≈ Poisson limit).
  phi_size <- 1.0 / pmax(phi, 1e-10)          # G-vector of NB size parameters

  # Expand per-gene size to n_new × G matrix
  size_mat <- matrix(rep(phi_size, n_new), nrow = n_new, byrow = TRUE)

  # Sample ZINB: dropout mask then NB
  n_total  <- as.integer(n_new) * as.integer(G)
  dropout  <- matrix(rbinom(n_total, 1L, as.vector(pi_new)),  nrow = n_new, ncol = G)
  nb_draw  <- matrix(rnbinom(n_total, mu = as.vector(mu_new), size = as.vector(size_mat)),
                     nrow = n_new, ncol = G)
  synth    <- nb_draw * (1L - dropout)  # n_new × G

  rownames(synth) <- NULL
  colnames(synth) <- md$gene_names

  # Save as n_cells × n_genes (AnnData/Python convention: cells in rows)
  saveRDS(synth, file = out_rds)
  cat(sprintf("Saved synthetic matrix (%d × %d) to %s\n", n_new, G, out_rds))
}


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: zinbwave.r <train|generate> [args...]")
}

if (args[1] == "train") {
  # train <h5ad_path> <n_latent> <out_rds>
  train_zinbwave(
    h5ad_path = args[2],
    n_latent  = as.integer(args[3]),
    out_rds   = args[4]
  )
} else if (args[1] == "generate") {
  # generate <n_cells> <model_rds> <out_rds> <seed>
  generate_zinbwave(
    n_cells_new = as.integer(args[2]),
    model_rds   = args[3],
    out_rds     = args[4],
    seed        = as.integer(args[5])
  )
} else {
  stop(sprintf("Unknown mode: %s", args[1]))
}
