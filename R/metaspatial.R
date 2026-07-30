# =====================================================================================
# metaspatial.R — use the Python MetaSpatial tool from R, on Seurat / .RDS objects.
# Uses the ACTIVE reticulate env (no env-switching) and imports Python modules with
# convert=FALSE so objects stay Python-side (avoids the "$ on atomic vector" errors).
#
# QUICK START:
#   1) once:  reticulate::py_install(c("numpy","pandas","scipy","scikit-learn","anndata","scanpy"))
#   2) edit REPO_DIR below, then:  source(".../MetaSpatial/R/metaspatial.R")
#   3) model <- ms_load_model("<full path>/metaspatial_model.pkl"); ms_run_rds(...)
#   If you see "another version of Python has already been initialized", restart R and re-source.
# ------------------------------------------------------------------------------------
REPO_DIR <- "C:/Users/pc/Desktop/spatial metabolism/MetaSpatial"  # folder that CONTAINS metaspatial/
PY_ENV   <- ""     # OPTIONAL dedicated env name; "" = use reticulate's default (recommended)
PYTHON   <- ""     # OPTIONAL full path to python.exe; "" = use active/default env
USE_CONDA <- FALSE # only used if PY_ENV set AND Python not yet initialized
REQUIRED_PY <- c("numpy","pandas","scipy","scikit-learn","anndata","scanpy")
# =====================================================================================

## ---- 1. R packages -----------------------------------------------------------------
.need_r <- c("reticulate", "Matrix", "Seurat", "SeuratObject")
.miss_r <- .need_r[!vapply(.need_r, requireNamespace, logical(1), quietly = TRUE)]
if (length(.miss_r)) install.packages(.miss_r, repos = "https://cloud.r-project.org")
suppressMessages({ library(reticulate); library(Seurat); library(Matrix) })

.as_r <- function(x) if (inherits(x, "python.builtin.object")) reticulate::py_to_r(x) else x

## ---- curated metabolite <-> m/z annotation (neutral monoisotopic mass; negative-mode adducts) ----
.MS_DB <- c(
  Lactate=90.0317, Pyruvate=88.0160, Alanine=89.0477, Serine=105.0426, Glycine=75.0320,
  Succinate=118.0266, Fumarate=116.0110, Malate=134.0215, Aspartate=133.0375, Glutamate=147.0532,
  Glutamine=146.0691, `alpha-Ketoglutarate`=146.0215, Citrate=192.0270, Glucose=180.0634,
  `Glucose-6-phosphate`=260.0297, Taurine=125.0147, Creatine=131.0695, Creatinine=113.0589,
  Hypoxanthine=136.0385, Xanthine=152.0334, Inosine=268.0808, AMP=347.0631, ADP=427.0294, ATP=506.9957,
  Glutathione=307.0838, GSSG=612.1519, Ascorbate=176.0321, Urate=168.0283,
  `Palmitate FA16:0`=256.2402, `Stearate FA18:0`=284.2715, `Oleate FA18:1`=282.2559,
  `Linoleate FA18:2`=280.2402, `Arachidonate FA20:4`=304.2402, `EPA FA20:5`=302.2246, `DHA FA22:6`=328.2402,
  Sphingosine=299.2824, `Cholesterol-sulfate`=466.3111, Inositol=180.0634, `N-acetylaspartate`=175.0481,
  Phosphocreatine=211.0358, `UDP-GlcNAc`=607.0817, Spermidine=145.1579, Carnitine=161.1052,
  Acetylcarnitine=203.1158, Kynurenine=208.0848, Adenosine=267.0968)
.MS_ADD <- c(`[M-H]-` = -1.0073, `[M+Cl]-` = 34.9694, `[M+HCOO]-` = 44.9982)

## annotate a numeric vector of m/z with metabolite name(s) (exact mass +- 0.01 Da)
ms_annotate_mz <- function(mzvals, tol = 0.01) {
  out <- rep("", length(mzvals))
  for (nm in names(.MS_DB)) for (a in names(.MS_ADD)) {
    hit <- which(abs(mzvals - (.MS_DB[[nm]] + .MS_ADD[[a]])) <= tol)
    if (length(hit)) out[hit] <- ifelse(out[hit] == "", paste0(nm, " ", a), paste0(out[hit], "; ", nm, " ", a))
  }
  out
}
## table of every annotated feature in the metaspatial assay: feature | m/z | metabolite
ms_annotation_table <- function(obj, assay = "metaspatial") {
  f <- rownames(obj[[assay]]); v <- suppressWarnings(as.numeric(sub("^mz[-_]", "", f)))
  ann <- ms_annotate_mz(v); keep <- ann != ""
  data.frame(feature = f[keep], mz = v[keep], metabolite = ann[keep], stringsAsFactors = FALSE)
}
## list the metabolite common names you can pass to ms_feature()/ms_spatial_plot()
ms_list_metabolites <- function() sort(names(.MS_DB))
## resolve a metabolite common name to its detected m/z (neutral mass + adduct; [M-H]- by default)
.ms_name_to_mz <- function(name, adduct = "[M-H]-") {
  nm  <- names(.MS_DB)
  hit <- which(tolower(nm) == tolower(name))                    # exact (case-insensitive) first
  if (!length(hit)) hit <- grep(name, nm, ignore.case = TRUE, fixed = FALSE)  # partial ("DHA" -> "DHA FA22:6")
  if (!length(hit)) stop("Unknown metabolite '", name, "'. Run ms_list_metabolites() to see valid names.")
  if (length(hit) > 1)
    message("'", name, "' matched ", length(hit), " entries (", paste(nm[hit], collapse = ", "),
            "); using ", nm[hit[1]], ". Pass the full name to pick another.")
  unname(.MS_DB[[hit[1]]] + .MS_ADD[[adduct]])
}

## ---- 2. Python env + deps ----------------------------------------------------------
setup_metaspatial_python <- function() {
  initialized <- isTRUE(try(py_available(initialize = FALSE), silent = TRUE))
  if (!initialized) {
    if (nzchar(PYTHON)) {
      use_python(PYTHON, required = TRUE)
    } else if (nzchar(PY_ENV)) {
      if (isTRUE(USE_CONDA) && !is.null(tryCatch(conda_binary(), error = function(e) NULL))) {
        if (!condaenv_exists(PY_ENV)) conda_create(PY_ENV, python_version = "3.11")
        use_condaenv(PY_ENV, required = TRUE)
      } else {
        if (!virtualenv_exists(PY_ENV)) virtualenv_create(PY_ENV)
        use_virtualenv(PY_ENV, required = TRUE)
      }
    }
  } else if (nzchar(PYTHON) || nzchar(PY_ENV)) {
    message("NOTE: Python already initialised (", py_config()$python,
            "); using it. To force another env, restart R and source this file first.")
  }
  ## ensure deps importable in the ACTIVE env; install any missing
  mods <- c(numpy = "numpy", pandas = "pandas", scipy = "scipy",
            `scikit-learn` = "sklearn", anndata = "anndata", scanpy = "scanpy")
  miss <- names(mods)[!vapply(unname(mods), py_module_available, logical(1))]
  if (length(miss)) { message("Installing into active env: ", paste(miss, collapse = ", ")); py_install(miss, pip = TRUE) }
  ## add repo to sys.path so `import metaspatial` works WITHOUT `pip install -e .`
  ## (import sys with convert=FALSE so sys$path is a Python list, not an R vector)
  sys <- import("sys", convert = FALSE)
  if (!(REPO_DIR %in% py_to_r(sys$path))) sys$path$insert(0L, REPO_DIR)
  message("MetaSpatial Python ready: ", py_config()$python)
  invisible(TRUE)
}

## ---- 3. lazy handles (convert=FALSE keeps objects Python-side) ----------------------
.MS <- new.env(parent = emptyenv())
ms_init <- function(force = FALSE) {
  if (!force && isTRUE(.MS$ready)) return(invisible(TRUE))
  setup_metaspatial_python()
  .MS$ms     <- import("metaspatial", convert = FALSE)
  .MS$sc     <- import("scanpy",      convert = FALSE)
  .MS$ad     <- import("anndata",     convert = FALSE)
  .MS$np     <- import("numpy",       convert = FALSE)
  .MS$pickle <- import("pickle",      convert = FALSE)
  .MS$bi     <- import_builtins(convert = FALSE)
  .MS$ready  <- TRUE
  invisible(TRUE)
}

## ---- 4. get a trained model ---------------------------------------------------------
ms_load_model <- function(pkl_path) {
  ms_init(); con <- .MS$bi$open(normalizePath(pkl_path), "rb"); on.exit(con$close())
  obj <- .MS$pickle$load(con)
  ## some bundles store the fitted MetaSpatial inside a dict under key 'model' — unwrap it
  if (!isTRUE(py_has_attr(obj, "predict_metabolome"))) {
    inner <- tryCatch(obj[["model"]], error = function(e) NULL)
    if (!is.null(inner) && isTRUE(py_has_attr(inner, "predict_metabolome"))) obj <- inner
  }
  if (!isTRUE(py_has_attr(obj, "predict_metabolome")))
    stop("Loaded object has no predict_metabolome(); retrain instead with ms_fit(<.h5ad paths>).")
  obj
}
ms_fit <- function(train_h5ad_paths, use_kegg = FALSE, kegg_gmt = NULL) {
  ms_init(); train <- lapply(train_h5ad_paths, function(p) .MS$ad$read_h5ad(normalizePath(p)))
  model <- if (isTRUE(use_kegg)) .MS$ms$MetaSpatial(use_kegg = TRUE, kegg_gmt = kegg_gmt)
           else                  .MS$ms$MetaSpatial(use_kegg = FALSE)
  model$fit(train); model
}

## ---- 5. Seurat -> AnnData (log1p CP10k; genes matched by symbol) --------------------
seurat_to_anndata <- function(obj, assay = "RNA") {
  ms_init()
  getlayer <- function(l) tryCatch(SeuratObject::GetAssayData(obj, assay = assay, layer = l),
                          error = function(e) tryCatch(SeuratObject::GetAssayData(obj, assay = assay, slot = l),
                                              error = function(e2) NULL))
  counts <- getlayer("counts"); prenorm <- FALSE
  if (is.null(counts) || nrow(counts) == 0) {                # no raw counts -> fall back to 'data'
    counts <- getlayer("data"); prenorm <- TRUE
    if (is.null(counts) || nrow(counts) == 0) stop("Could not find a 'counts' or 'data' layer in assay '", assay, "'.")
    message("No 'counts' layer found; using 'data' layer.")
  }
  genes  <- rownames(counts)
  adata  <- .MS$ad$AnnData(X = Matrix::t(counts))            # cells x genes (sparse -> scipy)
  adata$var_names <- genes; adata$obs_names <- colnames(counts)
  if (!prenorm) { .MS$sc$pp$normalize_total(adata, target_sum = 1e4); .MS$sc$pp$log1p(adata) }  # raw -> CP10k, log1p
  else if (max(counts) > 50) { .MS$sc$pp$log1p(adata) }      # 'data' looked un-logged -> log1p
  adata$layers["log1p"] <- adata$X
  xy <- tryCatch(Seurat::GetTissueCoordinates(obj), error = function(e) NULL)
  if (is.null(xy)) {
    md <- obj@meta.data
    cx <- grep("imagecol|pxl_col|^col$|^x$", colnames(md), ignore.case = TRUE, value = TRUE)[1]
    cy <- grep("imagerow|pxl_row|^row$|^y$", colnames(md), ignore.case = TRUE, value = TRUE)[1]
    xy <- md[, c(cx, cy)]
  }
  adata$obsm["spatial"] <- as.matrix(xy[, 1:2])
  adata
}

## ---- 6. predict + attach as an assay -----------------------------------------------
ms_predict_seurat <- function(obj, model, assay = "RNA", add_assay = "metaspatial", section_col = NULL) {
  adata <- seurat_to_anndata(obj, assay = assay)
  ## Section labels -> block-diagonal transport graph so neighbours never cross tissue sections.
  ## Critical for multi-section objects (e.g. HCC-2 N/L/P/T each have their own coordinate frame):
  ## without this the kNN graph bleeds across sections and smears the gradient.
  md <- obj@meta.data
  if (is.null(section_col)) {
    hit <- intersect(c("sample.ident", "section", "library", "batch", "orig.ident"), colnames(md))
    section_col <- if (length(hit)) hit[1] else NA
  }
  grp <- if (!is.na(section_col) && section_col %in% colnames(md)) as.character(md[[section_col]]) else NULL
  ## The updated Python accepts section labels (groups=). If an OLD metaspatial module is still cached
  ## in this session's Python (reticulate keeps Python alive across source()), that arg won't exist.
  .predict <- function() {
    if (is.null(grp)) return(model$predict_metabolome(adata))
    res <- tryCatch(model$predict_metabolome(adata, groups = grp), error = function(e) e)
    if (inherits(res, "error")) {
      if (grepl("groups|unused argument|unexpected keyword|positional", conditionMessage(res)))
        stop("This R session has an OLD 'metaspatial' Python module in memory (no section-aware predict).\n",
             "  FIX: restart R (Session > Restart R, or Ctrl+Shift+F10), then re-run:\n",
             "    source('", file.path(REPO_DIR, "R/metaspatial.R"), "')\n",
             "    model <- ms_load_model('", file.path(REPO_DIR, "metaspatial_model.pkl"), "')\n",
             "    obj   <- ms_run_rds('.../HCC-2-expr.RDS', model, out_rds='.../HCC-2_with_metaspatial.RDS')\n",
             "  The corrected code is already on disk; a live Python session must restart to load it.",
             call. = FALSE)
      stop(res)
    }
    res
  }
  pred <- as.matrix(.as_r(.predict()))
  if (!is.null(grp)) message("Section-aware graph via '", section_col, "' (", length(unique(grp)), " sections).")
  else message("No section column found (looked for sample.ident/section/library/batch/orig.ident); using one graph.")
  mz <- tryCatch(as.numeric(.as_r(model$mz_)), error = function(e) seq_len(ncol(pred)))
  ## Seurat forbids "_" in feature names (it silently converts to "-"), so use "-" directly
  colnames(pred) <- paste0("mz-", format(round(mz, 4), trim = TRUE)); rownames(pred) <- colnames(obj)
  attr(pred, "mz") <- mz
  if (!is.null(add_assay)) obj[[add_assay]] <- Seurat::CreateAssayObject(data = t(pred))
  attr(obj, "metaspatial_pred") <- pred
  ov <- tryCatch(round(as.numeric(.as_r(model$last_gene_overlap_)), 3), error = function(e) NA)
  message(sprintf("Added assay '%s': %d predicted metabolites x %d spots (range %.2f..%.2f)%s. Address features by name, e.g. ms_feature(obj, \"Glutathione\").",
                  add_assay, ncol(pred), nrow(pred), min(pred), max(pred),
                  if (!is.na(ov)) sprintf("; gene overlap %.1f%%", 100*ov) else ""))
  obj
}

## find the metaspatial feature nearest a target (so you never guess the exact string). `target` may be:
##   - a number        : an m/z value                         ms_feature(obj, 306.0765)
##   - a common name    : resolved via .MS_DB (+[M-H]-)         ms_feature(obj, "Glutathione")
##   - an "mz-..." str  : an exact feature name (kept if it exists)
##   FeaturePlot(obj, ms_feature(obj, "Glutathione"))
ms_feature <- function(obj, target, assay = "metaspatial", adduct = "[M-H]-") {
  f <- rownames(obj[[assay]]); v <- suppressWarnings(as.numeric(sub("^mz[-_]", "", f)))
  if (is.character(target)) {
    if (grepl("^mz[-_]", target)) {                 # already a feature string
      if (target %in% f) return(target)
      target <- as.numeric(sub("^mz[-_]", "", target))
    } else target <- .ms_name_to_mz(target, adduct) # common name -> detected m/z
  }
  f[which.min(abs(v - target))]
}

## spatial map of a predicted metabolite, faceted by section (e.g. HCC-2 N / L / P / T).
## Uses metadata coordinates directly, so it does NOT need a Seurat image/UMAP slot.
## `target` is a number, a common name, or an "mz-..." string (same as ms_feature).
##   ms_spatial_plot(obj, "Glutathione")            # named; nearest [M-H]- ion auto-resolved
##   ms_spatial_plot(obj, "DHA", group = "sample.ident")
##   ms_spatial_plot(obj, 306.0765)                 # raw m/z still works
ms_spatial_plot <- function(obj, target, assay = "metaspatial", group = "sample.ident",
                            ncol = 4, pt.size = 1.4, option = "magma",
                            clip = c(0.02, 0.98), adduct = "[M-H]-") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", repos = "https://cloud.r-project.org")
  feat <- ms_feature(obj, target, assay, adduct)             # number | name | "mz-..." all accepted
  mzv  <- suppressWarnings(as.numeric(sub("^mz[-_]", "", feat)))
  ann  <- ms_annotate_mz(mzv)                                # common name(s) for the title, if known
  ttl  <- if (nzchar(ann)) sprintf("Predicted %s  (%s)", ann, feat) else paste0("Predicted ", feat)
  dat  <- tryCatch(SeuratObject::GetAssayData(obj, assay = assay, layer = "data"),
                   error = function(e) SeuratObject::GetAssayData(obj, assay = assay, slot = "data"))
  vals <- as.numeric(dat[feat, ])
  md   <- obj@meta.data
  cx <- grep("imagecol|pxl_col|^col$|^x$", colnames(md), ignore.case = TRUE, value = TRUE)[1]
  cy <- grep("imagerow|pxl_row|^row$|^y$", colnames(md), ignore.case = TRUE, value = TRUE)[1]
  if (is.na(cx) || is.na(cy)) stop("Could not find spatial coordinate columns in meta.data.")
  df <- data.frame(x = as.numeric(md[[cx]]), y = as.numeric(md[[cy]]),
                   value = vals, grp = as.character(md[[group]]))
  short <- sub("^.*([NLPT])$", "\\1", df$grp)                 # tidy N/L/P/T labels + order
  if (all(short %in% c("N","L","P","T"))) {
    lab <- c(N = "N · Normal", L = "L · Leading edge", P = "P · PVTT", T = "T · Tumour")
    df$grp <- factor(lab[short], levels = unname(lab[c("N","L","P","T")]))
  }
  lims <- suppressWarnings(stats::quantile(df$value, clip, na.rm = TRUE))  # 2/98% contrast stretch
  if (!all(is.finite(lims)) || diff(range(lims)) <= 0) lims <- range(df$value, na.rm = TRUE)
  ggplot2::ggplot(df, ggplot2::aes(x, y, color = value)) +
    ggplot2::geom_point(size = pt.size, stroke = 0) +
    ggplot2::scale_color_viridis_c(option = option, limits = lims, oob = scales::squish) +
    ggplot2::facet_wrap(~grp, ncol = ncol, scales = "free") +
    ggplot2::scale_y_reverse() +
    ggplot2::theme_void(base_size = 12) +
    ggplot2::labs(title = ttl, color = "level") +
    ggplot2::theme(strip.text = ggplot2::element_text(face = "bold"),
                   plot.title = ggplot2::element_text(face = "bold"),
                   aspect.ratio = 1)   # square panels (coord_fixed can't combine with free scales)
}
ms_run_rds <- function(in_rds, model, out_rds = NULL, assay = "RNA") {
  obj <- readRDS(in_rds); obj <- ms_predict_seurat(obj, model, assay = assay)
  if (!is.null(out_rds)) saveRDS(obj, out_rds); obj
}

# =====================================================================================
# EXAMPLE (full paths)
# model <- ms_load_model("C:/Users/pc/Desktop/spatial metabolism/MetaSpatial/metaspatial_model.pkl")
# obj   <- ms_run_rds("C:/Users/pc/Desktop/spatial metabolism/HCC-2-expr.RDS", model,
#                     out_rds = "HCC-2_with_metaspatial.RDS")   # section-aware; recovers the N->PVTT gradient
#
# ## work by COMMON NAME instead of raw m/z -------------------------------------------
# ms_list_metabolites()                       # names you can use
# ms_spatial_plot(obj, "Glutathione")         # N/L/P/T map, titled + 2/98% contrast
# ms_spatial_plot(obj, "DHA")                 # partial names resolve (-> DHA FA22:6)
# DefaultAssay(obj) <- "metaspatial"
# FeaturePlot(obj, ms_feature(obj, "Glutathione"))   # name -> nearest [M-H]- feature
# head(ms_annotation_table(obj), 20)          # every annotated ion: feature | m/z | metabolite
# =====================================================================================
