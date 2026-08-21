# MetaSpatial — R interface (thin reticulate wrapper around the Python model).
#
# The prediction is done by the SAME pickled Python model, so R and Python outputs are
# byte-identical. Requires: reticulate + a Python env with `metaspatial` installed
# (see README install options). Point reticulate at that env before sourcing, e.g.
#   library(reticulate); use_condaenv("metaspatial", required = TRUE)
#
#   source("R/metaspatial.R")
#   model <- ms_load_model("metaspatial_model.pkl")
#   seu   <- ms_run_rds("your_section.RDS", model)     # adds a 'metaspatial' assay
#
# Notes:
#  * gene symbols are matched by the Python side; query-absent genes are mean-imputed.
#  * multi-section objects: pass section_col= (the Seurat meta.data column naming each
#    tissue section) so the transport graph is built within each section.

suppressMessages({
  library(reticulate)
})

.ms_np <- NULL
.ms_ad <- NULL
.ms_mod <- NULL

.ms_init <- function() {
  if (is.null(.ms_np))  .ms_np  <<- import("numpy", delay_load = TRUE)
  if (is.null(.ms_ad))  .ms_ad  <<- import("anndata", delay_load = TRUE)
  if (is.null(.ms_mod)) .ms_mod <<- import("metaspatial", delay_load = TRUE)
}

#' Load a pickled MetaSpatial model (handles the shipped {model,...} bundle).
ms_load_model <- function(path = "metaspatial_model.pkl") {
  .ms_init()
  .ms_mod$MetaSpatial$load(path)
}

#' Build an AnnData (genes x spots -> spots x genes) from a Seurat spatial object.
#' @param seu       a Seurat object with a spatial assay and image coordinates
#' @param assay     expression assay to use (default "Spatial")
#' @param section_col optional meta.data column naming each tissue section
.ms_seurat_to_anndata <- function(seu, assay = "Spatial", section_col = NULL) {
  .ms_init()
  if (!requireNamespace("Seurat", quietly = TRUE)) stop("Seurat is required for ms_run_rds().")
  # log1p(CP10k) expression, spots x genes
  X <- Seurat::GetAssayData(seu, assay = assay, slot = "data")   # genes x spots (log-normalized)
  X <- Matrix::t(X)                                              # spots x genes
  genes <- colnames(X)
  coords <- Seurat::GetTissueCoordinates(seu)
  coords <- as.matrix(coords[, 1:2])
  ad <- .ms_ad$AnnData(
    X = r_to_py(as.matrix(X)),
    obs = r_to_py(data.frame(row.names = rownames(X)))
  )
  ad$var_names <- as.list(genes)
  ad$layers["log1p"] <- ad$X
  ad$obsm["spatial"] <- .ms_np$asarray(coords, dtype = "float64")
  if (!is.null(section_col) && section_col %in% colnames(seu@meta.data)) {
    ad$obs[section_col] <- as.character(seu@meta.data[[section_col]])
  }
  ad
}

#' Predict the spatial metabolome for a Seurat .RDS and add a 'metaspatial' assay.
#' @param rds_path path to a saved Seurat object (.RDS)
#' @param model    a model returned by ms_load_model()
#' @param assay    expression assay to read (default "Spatial")
#' @param section_col optional meta.data column naming each section (multi-section objects)
#' @return the Seurat object with a new 'metaspatial' assay (metabolites x spots) and,
#'         in misc, the per-ion reliability table and conformal widths.
ms_run_rds <- function(rds_path, model, assay = "Spatial", section_col = NULL,
                       out_csv = "predicted_metabolites.csv") {
  .ms_init()
  seu <- readRDS(rds_path)
  ad  <- .ms_seurat_to_anndata(seu, assay = assay, section_col = section_col)
  # IMPORTANT: the Python API argument is `section_key` (NOT `groups`).
  pred <- model$predict_metabolome(ad, section_key = section_col)
  pred <- py_to_r(pred)                                   # spots x metabolites
  mz   <- py_to_r(model$mz_)
  colnames(pred) <- paste0("mz_", round(as.numeric(mz), 4))
  rownames(pred) <- rownames(ad$obs)
  # add as an assay (features x cells)
  seu[["metaspatial"]] <- Seurat::CreateAssayObject(data = t(pred))
  overlap <- tryCatch(py_to_r(model$last_gene_overlap_), error = function(e) NA)
  if (is.finite(overlap) && overlap < 0.5)
    warning(sprintf("Low gene-panel overlap (%.2f) with the training model; predictions are extrapolative.", overlap))
  # per-ion reliability table (recommended output)
  rel <- tryCatch(py_to_r(model$reliability_table(ad)), error = function(e) NULL)
  if (!is.null(rel)) {
    utils::write.csv(rel, out_csv, row.names = FALSE)
    seu@misc$metaspatial_reliability <- rel
  }
  seu@misc$metaspatial_gene_overlap <- overlap
  seu
}
