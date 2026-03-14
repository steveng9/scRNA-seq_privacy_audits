# scDesign3 R driver script
#
# Subcommands:
#   train   <h5ad_path> <cell_type> <out_rds_path> <family_use> <copula_type> [trunc_lvl]
#   gen     <n_cells>   <model_rds_path>            <out_rds_path>
#
# <family_use>  : one of "nb", "zinb", "poisson", "gaussian"   (default "nb")
# <copula_type> : one of "gaussian", "vine"                     (default "gaussian")
# <trunc_lvl>   : vine truncation level integer or "Inf"        (default "Inf")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cloud.r-project.org")

if (!require("scDesign3", quietly = TRUE))
    BiocManager::install("scDesign3", ask = FALSE)

if (!require("zellkonverter", quietly = TRUE))
    BiocManager::install("zellkonverter", ask = FALSE)

suppressPackageStartupMessages({
    library(scDesign3)
    library(SingleCellExperiment)
    library(zellkonverter)
    library(Matrix)
})


# ---------------------------------------------------------------------------
# train: fit scDesign3 for one cell type and save the model
# ---------------------------------------------------------------------------

train_cell_type <- function(h5ad_path, cell_type, out_rds_path,
                             family_use = "nb", copula_type = "gaussian",
                             trunc_lvl = Inf) {
    message(sprintf("[scDesign3] Reading %s", h5ad_path))
    sce_full <- readH5AD(h5ad_path, use_hdf5 = FALSE)

    # Coerce cell_type column to character to avoid integer-index issues in simu_new
    sce_full$cell_type <- as.character(sce_full$cell_type)

    # Filter to the target cell type
    ct_mask <- sce_full$cell_type == cell_type
    if (sum(ct_mask) == 0) stop(sprintf("No cells found for cell type '%s'", cell_type))
    sce_ct <- sce_full[, ct_mask]
    message(sprintf("[scDesign3] Cell type '%s': %d cells, %d genes",
                    cell_type, ncol(sce_ct), nrow(sce_ct)))

    # Ensure counts are an ordinary (non-HDF5) matrix
    cnt <- assay(sce_ct, "X")
    if (!is.matrix(cnt)) cnt <- as.matrix(cnt)
    assay(sce_ct, "counts") <- cnt

    RNGkind("L'Ecuyer-CMRG")
    set.seed(1)

    # 1. Construct data object (covariates, correlation groups)
    message("[scDesign3] construct_data ...")
    data_obj <- construct_data(
        sce          = sce_ct,
        assay_use    = "counts",
        celltype     = "cell_type",
        pseudotime   = NULL,
        spatial      = NULL,
        other_covariates = NULL,
        corr_by      = "cell_type"
    )

    # 2. Fit marginal distributions per gene
    message(sprintf("[scDesign3] fit_marginal (family=%s) ...", family_use))
    marginal_list <- fit_marginal(
        data         = data_obj,
        predictor    = "gene",
        mu_formula   = "1",
        sigma_formula = "1",
        family_use   = family_use,
        n_cores      = 1,
        usebam       = FALSE
    )

    # 3. Fit copula (Gaussian or vine)
    # Note: trunc_lvl is not exposed by scDesign3::fit_copula 1.6.0 (no ... passthrough).
    # It is stored in the model for future use when scDesign3 is upgraded.
    if (!is.infinite(trunc_lvl))
        message(sprintf("[scDesign3] NOTE: trunc_lvl=%s requested but not supported by this scDesign3 version — ignored.", trunc_lvl))
    message(sprintf("[scDesign3] fit_copula (copula=%s) ...", copula_type))
    copula_result <- fit_copula(
        sce              = sce_ct,
        assay_use        = "counts",
        input_data       = data_obj$dat,
        marginal_list    = marginal_list,
        family_use       = family_use,
        copula           = copula_type,
        DT               = TRUE,
        n_cores          = 1
    )

    # 4. Extract per-cell marginal parameters
    message("[scDesign3] extract_para ...")
    para <- extract_para(
        sce           = sce_ct,
        marginal_list = marginal_list,
        n_cores       = 1,
        family_use    = family_use,
        new_covariate = data_obj$newCovariate,
        data          = data_obj$dat
    )

    # Save one representative row from each parameter matrix
    # (intercept-only formula → all rows for same cell type are identical)
    mean_rep  <- para$mean_mat[1, , drop = FALSE]
    sigma_rep <- para$sigma_mat[1, , drop = FALSE]
    zero_rep  <- as.matrix(para$zero_mat[1, , drop = FALSE])

    model <- list(
        cell_type         = cell_type,
        all_gene_names    = rownames(sce_ct),
        filtered_gene     = data_obj$filtered_gene,
        copula_list       = copula_result$copula_list,
        important_feature = copula_result$important_feature,
        mean_mat_rep      = mean_rep,
        sigma_mat_rep     = sigma_rep,
        zero_mat_rep      = zero_rep,
        family_use        = family_use,
        copula_type       = copula_type,
        trunc_lvl         = trunc_lvl,
        n_training_cells  = ncol(sce_ct),
        input_dat         = data_obj$dat
    )

    message(sprintf("[scDesign3] Saving model to %s", out_rds_path))
    dir.create(dirname(out_rds_path), showWarnings = FALSE, recursive = TRUE)
    saveRDS(model, file = out_rds_path)
    message("[scDesign3] Training complete.")
}


# ---------------------------------------------------------------------------
# gen: sample synthetic cells from a saved model
# ---------------------------------------------------------------------------

gen_cell_type <- function(n_new, model_rds_path, out_rds_path) {
    message(sprintf("[scDesign3] Loading model from %s", model_rds_path))
    m <- readRDS(model_rds_path)

    cell_type <- m$cell_type
    n_new     <- as.integer(n_new)
    message(sprintf("[scDesign3] Generating %d cells for '%s'", n_new, cell_type))

    # Build a minimal dummy SCE with correct gene names
    n_genes     <- length(m$all_gene_names)
    dummy_cnts  <- matrix(0L, nrow = n_genes, ncol = 1)
    rownames(dummy_cnts) <- m$all_gene_names
    sce_dummy   <- SingleCellExperiment(assays = list(counts = dummy_cnts))

    # Expand representative parameter rows to n_new cells
    expand_rep <- function(row_rep, n) {
        mat <- row_rep[rep(1, n), , drop = FALSE]
        rownames(mat) <- as.character(seq_len(n))
        mat
    }
    mean_mat  <- expand_rep(m$mean_mat_rep,  n_new)
    sigma_mat <- expand_rep(m$sigma_mat_rep, n_new)
    zero_mat  <- expand_rep(m$zero_mat_rep,  n_new)

    # New covariate: n_new cells all belonging to this cell type
    new_cov <- data.frame(
        corr_group = rep(cell_type, n_new),
        row.names  = as.character(seq_len(n_new))
    )

    RNGkind("L'Ecuyer-CMRG")
    set.seed(1)

    message("[scDesign3] simu_new ...")
    new_counts <- simu_new(
        sce               = sce_dummy,
        assay_use         = "counts",
        mean_mat          = mean_mat,
        sigma_mat         = sigma_mat,
        zero_mat          = zero_mat,
        copula_list       = m$copula_list,
        n_cores           = 1,
        fastmvn           = FALSE,
        family_use        = m$family_use,
        nonnegative       = TRUE,
        nonzerovar        = FALSE,
        input_data        = m$input_dat,
        new_covariate     = new_cov,
        important_feature = m$important_feature,
        parallelization   = "mcmapply",
        BPPARAM           = NULL,
        filtered_gene     = m$filtered_gene
    )

    message(sprintf("[scDesign3] Generated matrix: %d genes × %d cells",
                    nrow(new_counts), ncol(new_counts)))
    dir.create(dirname(out_rds_path), showWarnings = FALSE, recursive = TRUE)
    saveRDS(new_counts, file = out_rds_path)
    message(sprintf("[scDesign3] Saved to %s", out_rds_path))
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
    stop("Usage: scdesign3.r <train|gen> [args...]")
}

cmd <- args[1]

if (cmd == "train") {
    if (length(args) < 4) stop("train requires: h5ad_path cell_type out_rds_path [family_use] [copula_type] [trunc_lvl]")
    h5ad_path   <- args[2]
    cell_type   <- args[3]
    out_rds     <- args[4]
    family_use  <- if (length(args) >= 5) args[5] else "nb"
    copula_type <- if (length(args) >= 6) args[6] else "gaussian"
    trunc_lvl   <- if (length(args) >= 7) {
        v <- suppressWarnings(as.numeric(args[7]))
        if (is.na(v)) Inf else v
    } else Inf
    train_cell_type(h5ad_path, cell_type, out_rds, family_use, copula_type, trunc_lvl)

} else if (cmd == "gen") {
    if (length(args) < 4) stop("gen requires: n_cells model_rds_path out_rds_path")
    n_new         <- as.integer(args[2])
    model_rds     <- args[3]
    out_rds       <- args[4]
    gen_cell_type(n_new, model_rds, out_rds)

} else {
    stop(sprintf("Unknown subcommand '%s'. Use 'train' or 'gen'.", cmd))
}
