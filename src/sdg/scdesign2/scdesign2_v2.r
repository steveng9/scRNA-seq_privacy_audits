# scdesign2_v2.r — DP-proof variant v2
#
# DIFFERENCE FROM scdesign2.r
# ---------------------------
# In `fit_Gaussian_copula`, the copula's `cov_mat` is the UNCENTERED SECOND
# MOMENT of the normal-quantile matrix, not the centered correlation:
#
#   v1 (scdesign2.r:163)
#       cov_mat <- cor(t(quantile_normal))                   # centers + normalizes
#
#   v2 (this file)
#       cov_mat <- (quantile_normal %*% t(quantile_normal)) / (n - 1)
#                                                            # uncentered, un-normalized
#
# Rationale: the v2 sensitivity bound (sdg.dp.sensitivity, dp_variant="v2")
# bounds the donor-level Frobenius sensitivity of the uncentered second
# moment, dropping the leading factor of 4 vs. the centered covariance.  For
# this bound to apply honestly, the implementation must noise the uncentered
# matrix.  PSD-projection and normalization-to-correlation are deferred to
# Python (post-processing under DP, free).
#
# The `cov_mat` produced by this script is therefore NOT directly usable as
# a correlation matrix — sampling/Gen requires a follow-up PSD+normalize step
# on the Python side (see experiments/dp/v2/generate.py).
#
# Otherwise this file is line-for-line identical to scdesign2.r.

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
    # v2 CHANGE: uncentered second moment instead of cor()
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
              dp_variant = 'v2'))
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

# Generation is unchanged from v1: it samples from the copula's cov_mat,
# which by the time it gets here has been PSD-projected and normalized to a
# correlation matrix on the Python side.  So no v2-specific gen function.
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
