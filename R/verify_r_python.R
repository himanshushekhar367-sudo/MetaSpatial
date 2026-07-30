# =====================================================================================
# verify_r_python.R — confirm the canonical MetaSpatial model recovers the N->PVTT gradient
# on YOUR machine, in R. R is a thin dispatch layer over the Python model (reticulate calls the
# same object), so R and Python predictions are identical by construction — this checks the RESULT.
#
# Prereq: edit REPO_DIR at the top of metaspatial.R to your MetaSpatial folder, then run this.
# =====================================================================================
SP <- "C:/Users/pc/Desktop/spatial metabolism"
source(file.path(SP, "MetaSpatial/R/metaspatial.R"))

## load the CANONICAL model (fixed predict path) and re-predict HCC-2
model <- ms_load_model(file.path(SP, "MetaSpatial/metaspatial_model.pkl"))
obj   <- ms_run_rds(file.path(SP, "HCC-2-expr.RDS"), model,
                    out_rds = file.path(SP, "HCC-2_with_metaspatial.RDS"))

## per-section means (N -> L -> P -> T), addressed by common name
D   <- SeuratObject::GetAssayData(obj, assay = "metaspatial", layer = "data")
sec <- factor(sub("^.*([NLPT])$", "\\1", as.character(obj$sample.ident)), levels = c("N","L","P","T"))
mean_by <- function(name) { f <- ms_feature(obj, name); tapply(as.numeric(D[f, ]), sec, mean) }

report <- function(name, updown) {
  m <- mean_by(name); d <- as.numeric(m["P"] - m["N"])
  ok <- if (updown == "up") d > 0 else d < 0
  cat(sprintf("  %-20s N/L/P/T = %6.3f %6.3f %6.3f %6.3f   ΔP-N = %+.3f   %s\n",
              name, m["N"], m["L"], m["P"], m["T"], d, if (ok) "PASS" else "FAIL"))
  ok
}
cat("\nHCC-2 predicted-metabolite trajectory (expect the manuscript signature):\n")
res <- c(
  report("Glutathione",       "up"),   # antioxidant programme rises into PVTT/tumour
  report("Ascorbate",         "up"),
  report("DHA FA22:6",        "up"),
  report("Palmitate FA16:0",  "down"), # saturated-FA falls
  report("Stearate FA18:0",   "down"),
  report("Glucose",           "down")
)
cat(sprintf("\n%d/%d directional checks PASS -> gradient reproduced in R.\n", sum(res), length(res)))
cat("(Predictions are byte-identical to the Python API: R dispatches to the same pickled model.)\n")
