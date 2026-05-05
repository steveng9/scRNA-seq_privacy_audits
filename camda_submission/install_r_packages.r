#!/usr/bin/env Rscript
# Install R packages required for the PPML-Huskies CAMDA Track II submission.
#
# Run this AFTER creating and activating the ppml-camda-env conda environment:
#   conda activate ppml-camda-env
#   Rscript install_r_packages.r
#
# Expected installation time: 10–30 minutes (downloads from Bioconductor).

cat("Installing required R packages for the PPML-Huskies CAMDA Track II submission...\n")
cat("This may take 10–30 minutes.\n\n")

# CRAN packages
cran_pkgs <- c("Matrix", "MASS", "mvtnorm", "doParallel", "foreach",
               "BiocManager", "remotes")

cat("Installing CRAN packages...\n")
install.packages(cran_pkgs,
                 repos = "https://cran.rstudio.com/",
                 dependencies = TRUE,
                 quiet = FALSE)

# Bioconductor packages
cat("\nInstalling Bioconductor packages...\n")
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cran.rstudio.com/")

BiocManager::install(c(
    "scDesign2",   # required for scDesign2+DP generator and shadow model
    "zinbwave",    # required for ZINBWave generator
    "SingleCellExperiment",
    "BiocParallel"
), ask = FALSE, update = FALSE)

# Verify installations
pkgs_to_check <- c("scDesign2", "zinbwave", "Matrix", "MASS")
cat("\n--- Verification ---\n")
for (pkg in pkgs_to_check) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("  ✓  %s\n", pkg))
    } else {
        cat(sprintf("  ✗  %s  (FAILED — check errors above)\n", pkg))
    }
}
cat("\nDone.\n")
