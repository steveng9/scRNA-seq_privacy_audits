# Setup dependencies
if(!require(devtools, quietly=TRUE)) install.packages("devtools")
if(!require(SingleCellExperiment, quietly=TRUE)) install.packages("SingleCellExperiment")

if(!require(scDesign2)) devtools::install_github("JSB-UCLA/scDesign2")
library(scDesign2)

if(!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if(!require(zellkonverter, quietly=TRUE)) BiocManager::install("zellkonverter", ask=FALSE)

suppressPackageStartupMessages(library(zellkonverter))
suppressPackageStartupMessages(library(devtools))
suppressPackageStartupMessages(library(SingleCellExperiment))


train_copula <- function(train_h5ad_path, type, out_copula_path) {
	# Convert data into required matrix format:
	# cels are cols, genes are rows, col names are
	# cell types (and you can have multiple cols w/
	# same name).
	sprintf("Reading %s", train_h5ad_path)
	data <- readH5AD(train_h5ad_path)
	print("Succeeded")
	col_data <- colData(data)
	train_cnts <- assay(data, "X")
	
	# Truncate to avoid RAM errors. Extract first 500 for
	# cell type.
	#colnames(counts_matrix) <- lapply(colnames(counts_matrix), function(name) col_data[name, "cell_type"])
	#correct_cell_type <- counts_matrix[, colnames(counts_matrix) == type]
	#train_cnts <- correct_cell_type[,1:min(1000, ncol(correct_cell_type))]
	#train_cnts <- counts_matrix[,1:1000]
	print(dim(train_cnts))
	colnames(train_cnts) <- lapply(colnames(train_cnts), function(name) col_data[name, "cell_type"])

	cell_type_sel <- unique(col_data[, "cell_type"])
	
	RNGkind("L'Ecuyer-CMRG")
	set.seed(1)
	sprintf("Writing to %s", out_copula_path)
	# We could use more than one core for this if we had more memory
	copula_result <- fit_model_scDesign2(train_cnts, type, sim_method = 'copula',
		ncores=1)
	print(copula_result)
	saveRDS(copula_result, file = out_copula_path)
}

train_copula_all_cell_types <- function(train_h5ad_path, type, out_copula_path) {
	# Convert data into required matrix format:
	# cels are cols, genes are rows, col names are
	# cell types (and you can have multiple cols w/
	# same name).
	sprintf("Reading %s", train_h5ad_path)
	data <- readH5AD(train_h5ad_path)
	print("Succeeded")
	col_data <- colData(data)
	train_cnts <- assay(data, "X")
	
	# Truncate to avoid RAM errors. Extract first 500 for
	# cell type.
	#colnames(counts_matrix) <- lapply(colnames(counts_matrix), function(name) col_data[name, "cell_type"])
	#correct_cell_type <- counts_matrix[, colnames(counts_matrix) == type]
	#train_cnts <- correct_cell_type[,1:min(1000, ncol(correct_cell_type))]
	#train_cnts <- counts_matrix[,1:1000]
	print(dim(train_cnts))
	colnames(train_cnts) <- lapply(colnames(train_cnts), function(name) col_data[name, "cell_type"])

	cell_type_sel <- unique(col_data[, "cell_type"])
	
	RNGkind("L'Ecuyer-CMRG")
	set.seed(1)
	sprintf("Writing to %s", out_copula_path)
	# We could use more than one core for this if we had more memory
	copula_result <- fit_model_scDesign2(train_cnts, type, sim_method = 'copula',
		ncores=1)
	print(copula_result)
	saveRDS(copula_result, file = out_copula_path)
}

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
} else if(args[1] == "train_all") {
	train_h5ad <- args[2]
	out_path <- args[3]
    train_copula_all_celltypes(train_h5ad, type, out_path)
} else if(args[1] == "gen") {
	num <- as.numeric(args[2])
	copula_path <- args[3]
	out_path <- args[4]
	gen_synth_data(num, copula_path, out_path)
}
