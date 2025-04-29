args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript muscat.R [train_dataset.h5ad] [test_dataset].h5ad [NO_SAMPLES] [OUTPATH]")
}
train_path <- args[1]
test_path <- args[2]
no_samples <- as.numeric(args[3])
out_path <- args[4]

options(error = function() {
  traceback(2)
  quit(status = 1)
})

suppressPackageStartupMessages({
    library(muscat)
    library(zellkonverter)
    library(SingleCellExperiment)
})

sce <- readH5AD(train_path)
test_sce <- readH5AD(test_path)
cat("Genes:", nrow(sce), "\nCells:", ncol(sce), "\n")

#colData(sce)$sample_id <- colData(sce)$cell_label
#colData(sce)$sample_id <- colData(sce)$individual
colData(sce)$sample_id <- 0
colData(sce)$cluster_id <- colData(sce)$cell_type
#colData(sce)$group_id <- colData(sce)$cell_type
colData(sce)$group_id <- "all"
if ("X" %in% assayNames(sce)) {
  assay(sce, "counts") <- assay(sce, "X")
  assayNames(sce) <- setdiff(assayNames(sce), "X")
}

ref <- prepSim(sce, verbose = TRUE, min_size=NULL)

cat("Prep sim finished\n")

#sim <- simData(ref, 
#    nc = 2e3, ng = 1e3, force = TRUE,
#    p_dd = diag(6)[1, ], nk = 3, ns = 3
#)

# Compute proportions of cell types in test dataset
tab <- table(colData(test_sce)$cell_type)
cell_ids <- as.integer(names(tab))
ord <- order(cell_ids)
tab <- tab[ord]
props <- as.numeric(tab) / sum(tab)
cells <- floor(1.05 * no_samples)
sim <- simData(ref, 
    nc = cells, ng = nrow(sce), force = TRUE,
    p_dd = diag(6)[1, ], nk = length(tab), ns = 1,
    probs=list(props, NULL, NULL)
)
#saveRDS(sim, file = "sim_sce.rds")

counts_dense <- as.matrix(assay(sim, "counts"))
sim_dense <- SingleCellExperiment(
  assays = list(
    counts = counts_dense
  ),
  colData = as.data.frame(colData(sim)),
  rowData = as.data.frame(rowData(sim))
)
metadata(sim_dense) <- list()
writeH5AD(sim_dense, out_path)
