# =====================================================================================
# predict_on_your_data.R — apply a PRE-TRAINED MetaSpatial model to your own spatial
# transcriptomics, from a Seurat .RDS. No MSI, no paired data, no training needed.
#
#   Rscript examples/predict_on_your_data.R  <model.pkl>  <query.RDS>  [out.RDS]
#
# The query only needs gene expression (gene SYMBOLS) + spatial coordinates — exactly what a
# Visium/Xenium/etc. Seurat object already has. Predictions land in a new 'metaspatial' assay.
# =====================================================================================
args  <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript predict_on_your_data.R <model.pkl> <query.RDS> [out.RDS]")
model_pkl <- args[[1]]; query_rds <- args[[2]]
out_rds   <- if (length(args) >= 3) args[[3]] else sub("\\.RDS$", "_with_metaspatial.RDS", query_rds, ignore.case = TRUE)

## point this at your MetaSpatial checkout (folder that CONTAINS metaspatial/) before running,
## or edit REPO_DIR inside R/metaspatial.R and source that copy instead.
REPO_DIR <- Sys.getenv("METASPATIAL_REPO", dirname(dirname(normalizePath(model_pkl))))
source(file.path(REPO_DIR, "R", "metaspatial.R"))

model <- ms_load_model(model_pkl)                 # pre-trained; ST-only prediction
obj   <- ms_run_rds(query_rds, model, out_rds = out_rds)   # section-aware; adds 'metaspatial' assay

cat("\nDone. Predicted metabolome saved to:\n  ", out_rds, "\n")
cat("Address metabolites by common name, e.g.:\n")
cat("  DefaultAssay(obj) <- 'metaspatial'\n")
cat("  FeaturePlot(obj, ms_feature(obj, 'Glutathione'))\n")
cat("  ms_spatial_plot(obj, 'Glutathione')      # if the object has section labels (sample.ident)\n")
cat("  head(ms_annotation_table(obj), 20)       # every named ion: feature | m/z | metabolite\n")
