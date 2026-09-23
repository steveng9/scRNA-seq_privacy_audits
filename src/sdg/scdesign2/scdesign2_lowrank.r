# scdesign2_lowrank.r — DP-proof "lowrank" variant, forked from scdesign2_v2.r
#
# DIFFERENCE FROM scdesign2_v2.r
# -------------------------------
# Identical to v2 in every respect EXCEPT that `fit_Gaussian_copula` now ALSO
# saves the per-cell normal-quantile matrix `quantile_normal` (genes x cells)
# in the returned list, instead of discarding it after computing `cov_mat`.
#
# Why: the low-rank DP mechanism (src/sdg/dp/lowrank.py) needs to project each
# CELL's quantile-normal vector through a fixed public projection P and
# re-clip it BEFORE aggregating to a covariance matrix — this is what makes
# the resulting sensitivity bound scale with the reduced rank r instead of the
# full gene count G (see src/sdg/dp/lowrank.py module docstring for the full
# argument, including why the naive "project the already-fitted G x G matrix"
# shortcut does NOT give a valid r-dependent bound).  That per-cell access is
# not available from the already-fitted v2 .rds files (v2 keeps only the
# aggregate `cov_mat`), so this fork exists purely to expose it.  cov_mat's
# computation, and everything else in the fitting pipeline, is byte-for-byte
# identical to scdesign2_v2.r — this is an additive change, not a behavior
# change, and does not affect existing v1/v2 outputs (this is a separate file).
#
# dp_variant is tagged 'lowrank' (vs. v2's 'v2') purely for provenance;
# downstream consumption of cov_mat/marginals is identical to v2.

if(!require(devtools, quietly=TRUE)) install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(SingleCellExperiment, quietly=TRUE)) install.packages("SingleCellExperiment", repos = "http://cran.us.r-project.org")

if(!require(scDesign2)) devtools::install_github("JSB-UCLA/scDesign2")
library(scDesign2)
library(parallel)
library(MASS)


if(!require("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = "http://cran.us.r-project.org")
if(!require(zellkonverter, quietly=TRUE)) BiocManager::install("zellkonverter", ask=FALSE)

suppressPackageStartupMessages(library(zellkonverter))
suppressPackageStartupMessages(library(devtools))
suppressPackageStartupMessages(library(SingleCellExperiment))


fit_marginals <- function(x, marginal = c('auto_choose', 'zinb', 'nb', 'poisson'),
                          pval_cutoff = 0.05, epsilon = 1e-5,
                          jitter = TRUE, DT = TRUE){
  p <- nrow(x)
  n <- ncol(x)

  marginal <- match.arg(marginal)
  if(marginal == 'auto_choose'){
    params <- t(apply(x, 1, function(gene){
      m <- mean(gene)
      v <- var(gene)
      if(m >= v){
        mle_Poisson <- glm(gene ~ 1, family = poisson)
        tryCatch({
          mle_ZIP <- zeroinfl(gene ~ 1|1, dist = 'poisson')
          chisq_val <- 2 * (logLik(mle_ZIP) - logLik(mle_Poisson))
          pvalue <- as.numeric(1 - pchisq(chisq_val, 1))
          if(pvalue < pval_cutoff)
            c(plogis(mle_ZIP$coefficients$zero), Inf, exp(mle_ZIP$coefficients$count))
          else
            c(0.0, Inf, m)
        },
        error = function(cond){
          c(0.0, Inf, m)})
      }else{
        mle_NB <- glm.nb(gene ~ 1)
        if(min(gene) > 0)
          c(0.0, mle_NB$theta, exp(mle_NB$coefficients))
        else
          tryCatch({
            mle_ZINB <- zeroinfl(gene ~ 1|1, dist = 'negbin')
            chisq_val <- 2 * (logLik(mle_ZINB) - logLik(mle_NB))
            pvalue <- as.numeric(1 - pchisq(chisq_val, 1))
            if(pvalue < pval_cutoff)
              c(plogis(mle_ZINB$coefficients$zero), mle_ZINB$theta, exp(mle_ZINB$coefficients$count))
            else
              c(0.0, mle_NB$theta, exp(mle_NB$coefficients))
          },
          error = function(cond){
            c(0.0, mle_NB$theta, exp(mle_NB$coefficients))
          })
      }
    }))
  }else if(marginal == 'zinb'){
    params <- t(apply(x, 1, function(gene){
      m <- mean(gene)
      v <- var(gene)
      if(m >= v)
      {
        mle_Poisson <- glm(gene ~ 1, family = poisson)
        tryCatch({
          mle_ZIP <- zeroinfl(gene ~ 1|1, dist = 'poisson')
          chisq_val <- 2 * (logLik(mle_ZIP) - logLik(mle_Poisson))
          pvalue <- as.numeric(1 - pchisq(chisq_val, 1))
          if(pvalue < pval_cutoff)
            c(plogis(mle_ZIP$coefficients$zero), Inf, exp(mle_ZIP$coefficients$count))
          else
            c(0.0, Inf, m)
        },
        error = function(cond){
          c(0.0, Inf, m)})
      }
      else
      {
        if(min(gene) > 0)
        {
          mle_NB <- glm.nb(gene ~ 1)
          c(0.0, mle_NB$theta, exp(mle_NB$coefficients))
        }
        else
          tryCatch({
            mle_ZINB <- zeroinfl(gene ~ 1|1, dist = 'negbin')
            c(plogis(mle_ZINB$coefficients$zero), mle_ZINB$theta, exp(mle_ZINB$coefficients$count))
          },
          error = function(cond){
            mle_NB <- glm.nb(gene ~ 1)
            c(0.0, mle_NB$theta, exp(mle_NB$coefficients))
          })
      }
    }))
  }else if(marginal == 'nb'){
    params <- t(apply(x, 1, function(gene){
      m <- mean(gene)
      v <- var(gene)
      if(m >= v){
        c(0.0, Inf, m)
      }else{
        mle_NB <- glm.nb(gene ~ 1)
        c(0.0, mle_NB$theta, exp(mle_NB$coefficients))
      }
    }))
  }else if(marginal == 'poisson'){
    params <- t(apply(x, 1, function(gene){
      c(0.0, Inf, mean(gene))
    }))
  }

  if(DT){
    u <- t(sapply(1:p, function(iter){
      param <- params[iter, ]
      gene <- unlist(x[iter,])
      prob0 <- param[1]
      u1 <- prob0 + (1 - prob0) * pnbinom(gene, size = param[2], mu = param[3])
      u2 <- (prob0 + (1 - prob0) * pnbinom(gene - 1, size = param[2], mu = param[3])) *
        as.integer(gene > 0)
      if(jitter)
        v <- runif(n)
      else
        v <- rep(0.5, n)
      r <- u1 * v + u2 * (1 - v)
      idx_adjust <- which(1-r < epsilon)
      r[idx_adjust] <- r[idx_adjust] - epsilon
      idx_adjust <- which(r < epsilon)
      r[idx_adjust] <- r[idx_adjust] + epsilon

      r
    }))
  }else{
    u <- NULL
  }

  return(list(params = params, u = u))
}


fit_Gaussian_copula <- function(x, marginal = c('auto_choose', 'zinb', 'nb', 'poisson'),
                                jitter = TRUE, zp_cutoff = 0.8,
                                min_nonzero_num = 2){
  marginal <- match.arg(marginal)
  n <- ncol(x)
  p <- nrow(x)

  gene_zero_prop <- apply(x, 1, function(y){
    sum(y < 1e-5) / n
  })

  gene_sel1 <- which(gene_zero_prop < zp_cutoff)
  gene_sel2 <- which(gene_zero_prop < 1.0 - min_nonzero_num/n &
                       gene_zero_prop >= zp_cutoff)
  gene_sel3 <- (1:p)[-c(gene_sel1, gene_sel2)]

  if(length(gene_sel1) > 0){
    marginal_result1 <- fit_marginals(x[gene_sel1, , drop = FALSE], marginal, jitter = jitter, DT = TRUE)
    quantile_normal <- qnorm(marginal_result1$u)
    # ============================================================
    # v2 CHANGE (inherited): uncentered second moment instead of cor()
    # ============================================================
    # v1 was: cov_mat <- cor(t(quantile_normal))
    # Note: PSD-projection and normalization to correlation are deferred
    # to the Python side (post-processing under DP).  The cov_mat saved
    # here is therefore the RAW uncentered second moment matrix, NOT a
    # valid correlation matrix; it must be post-processed before use.
    n_cells_for_norm <- ncol(quantile_normal)
    cov_mat <- (quantile_normal %*% t(quantile_normal)) / max(n_cells_for_norm - 1, 1)
  }else{
    cov_mat = NULL
    marginal_result1 = NULL
    quantile_normal = NULL
  }

  if(length(gene_sel2) > 0){
    marginal_result2 <- fit_marginals(x[gene_sel2, , drop = FALSE], marginal, DT = FALSE)
  }else{
    marginal_result2 = NULL
  }
  return(list(cov_mat = cov_mat, marginal_param1 = marginal_result1$params,
              marginal_param2 = marginal_result2$params,
              gene_sel1 = gene_sel1, gene_sel2 = gene_sel2, gene_sel3 = gene_sel3,
              zp_cutoff = zp_cutoff, min_nonzero_num = min_nonzero_num,
              sim_method = 'copula', n_cell = n, n_read = sum(x),
              # LOWRANK CHANGE: keep the per-cell normal-quantile matrix
              # (genes x cells) instead of discarding it.  Everything above
              # this line is identical to scdesign2_v2.r.
              quantile_normal = quantile_normal,
              dp_variant = 'lowrank'))
}


fit_wo_copula <- function(x, marginal = c('auto_choose', 'zinb', 'nb', 'poisson'),
                          jitter = TRUE, min_nonzero_num = 2){
  marginal <- match.arg(marginal)
  n <- ncol(x)
  p <- nrow(x)

  gene_zero_prop <- apply(x, 1, function(y){
    sum(y < 1e-5) / n
  })

  gene_sel1 <- which(gene_zero_prop < 1.0 - min_nonzero_num/n)
  gene_sel2 <- (1:p)[-gene_sel1]

  if(length(gene_sel1) > 0){
    marginal_result1 <- fit_marginals(x[gene_sel1, ], marginal, jitter = jitter, DT = FALSE)
  }else{
    marginal_result1 = NULL
  }

  return(list(marginal_param1 = marginal_result1$params,
              gene_sel1 = gene_sel1, gene_sel2 = gene_sel2,
              min_nonzero_num = min_nonzero_num, sim_method = 'ind',
              n_cell = n, n_read = sum(x)))
}

fit_model_scDesignX <- function(data_mat, cell_type_sel, col_data, sim_method = c('copula', 'ind'),
                                marginal = c('auto_choose', 'zinb', 'nb', 'poisson'),
                                jitter = TRUE, zp_cutoff = 0.8,
                                min_nonzero_num = 2, ncores = 1){
  sim_method <- match.arg(sim_method)
  marginal <- match.arg(marginal)

  if(sum(abs(data_mat - round(data_mat))) > 1e-5){
    warning('The entries in the input matrix are not integers. Rounding is performed.')
    data_mat <- round(data_mat)
  }

  if(sim_method == 'copula'){
    param <- mclapply(1:length(cell_type_sel), function(iter){
      cell_type_indices <- col_data$cell_type == cell_type_sel[iter]
      subset_data <- data_mat[, cell_type_indices]
      fit_Gaussian_copula(subset_data, marginal,
                          jitter = jitter, zp_cutoff = zp_cutoff,
                          min_nonzero_num = min_nonzero_num)
    }, mc.cores = ncores)
  } else if(sim_method == 'ind'){
    param <- mclapply(1:length(cell_type_sel), function(iter){
      cell_type_indices <- col_data$cell_type == cell_type_sel[iter]
      subset_data <- data_mat[, cell_type_indices]
      fit_wo_copula(subset_data, marginal,
                    jitter = jitter,
                    min_nonzero_num = min_nonzero_num)
    }, mc.cores = ncores)
  }

  names(param) <- cell_type_sel
  param
}

train_copula <- function(train_h5ad_path, type, out_copula_path) {
    sprintf("Reading %s", train_h5ad_path)
    data <- readH5AD(train_h5ad_path)
    print("Succeeded")

    col_data <- colData(data)
    train_cnts <- assay(data, "X")

    print(class(train_cnts))
    colnames(train_cnts) <- col_data[colnames(train_cnts), "cell_type"]

    cell_type_sel <- unique(col_data$cell_type)

    RNGkind("L'Ecuyer-CMRG")
    set.seed(1)
    sprintf("Writing to %s", out_copula_path)
    copula_result <- fit_model_scDesignX(train_cnts, type, col_data, sim_method = 'copula', ncores=1)
    print(copula_result)
    saveRDS(copula_result, file = out_copula_path)
}

# Generation is unchanged from v1/v2: it samples from the copula's cov_mat,
# which by the time it gets here has been PSD-projected and normalized to a
# correlation matrix on the Python side.  So no lowrank-specific gen function
# -- experiments/dp/lowrank/generate.py reuses scdesign2.r's gen mode, exactly
# as experiments/dp/v2/generate.py does for v2.
gen_synth_data <- function(n_cell_new, copula_path, out_rds_path) {
    copula_result <- readRDS(copula_path)
    print(copula_path)
    print(copula_result)
    print(dim(copula_result))
    sim_count_copula <- simulate_count_scDesign2(copula_result, n_cell_new, sim_method = 'copula')
    print(dim(sim_count_copula))
    saveRDS(sim_count_copula, file = out_rds_path)
}

args <- commandArgs(trailingOnly = TRUE)
if(args[1] == "train") {
    train_h5ad <- args[2]
    type <- args[3]
    out_path <- args[4]
    train_copula(train_h5ad, type, out_path)
} else if(args[1] == "gen") {
    num <- as.numeric(args[2])
    copula_path <- args[3]
    out_path <- args[4]
    gen_synth_data(num, copula_path, out_path)
}
