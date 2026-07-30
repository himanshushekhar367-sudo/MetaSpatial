#!/usr/bin/env python3
"""
Apply a trained MetaSpatial model to ANY spatial transcriptomics section (Visium, Xenium,
MERFISH, Slide-seq, ...) to PREDICT its spatial metabolome from the transcriptome alone.

You only need a query AnnData with:
    - gene expression (var_names = gene SYMBOLS, same species as the training data)
    - adata.obsm['spatial']  = per-spot (x, y) coordinates   (n_spots x 2)

USAGE
    python examples/predict_on_your_data.py --model my_model.pkl --query your_data.h5ad --out results/
    #  add --plot to also save spatial maps of the top predicted metabolites

Get a model bundle from examples/train_on_paired_data.py (train on your own paired data),
or train one on the public DESIUM cohort (see docs/USAGE.md).

IMPORTANT (honest use): cross-sample transfer is a research problem, not solved. Predictions on
tissue unlike the training data are hypothesis-generating MAPS, strongest for transcriptionally
programmed classes (lipids, nucleotides) and weak for flux-controlled polar metabolites. See the
caveats in docs/USAGE.md before interpreting quantitatively.
"""
import argparse, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad

# make `metaspatial` importable so the pickled model can be reconstructed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metaspatial  # noqa: F401  (needed by pickle.load)


def ensure_spatial(adata):
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], float)
    for xa, ya in [("pxl_col_in_fullres", "pxl_row_in_fullres"), ("x", "y"),
                   ("array_col", "array_row"), ("X", "Y"), ("imagecol", "imagerow")]:
        if xa in adata.obs and ya in adata.obs:
            xy = adata.obs[[xa, ya]].values.astype(float)
            adata.obsm["spatial"] = xy
            print(f"  spatial: built obsm['spatial'] from obs['{xa}','{ya}']")
            return xy
    sys.exit("Query has no adata.obsm['spatial']. Add per-spot (x, y) coordinates first "
             "(Visium h5ad from scanpy.read_visium already include them).")


def normalize(adata, mode):
    x = adata.X
    mx = float(x.max()) if not hasattr(x, "toarray") else float(x.max())
    is_raw = mx > 50 or np.allclose(x.data if hasattr(x, "data") else x,
                                    np.round(x.data if hasattr(x, "data") else x))
    if mode == "raw" or (mode == "auto" and is_raw):
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
        print("  normalisation: applied CP10k + log1p (data looked like raw counts)")
    else:
        print("  normalisation: none (data assumed already log-normalised)")
    adata.layers["log1p"] = adata.X.copy()


def main():
    ap = argparse.ArgumentParser(description="Predict the spatial metabolome for a transcriptome-only section.")
    ap.add_argument("--model", required=True, help="model bundle (.pkl) from train_on_paired_data.py")
    ap.add_argument("--query", required=True, help="query .h5ad (any spatial transcriptomics section)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--normalize", choices=["auto", "raw", "none"], default="auto",
                    help="auto (default): CP10k+log1p if data looks like raw counts")
    ap.add_argument("--top", type=int, default=25, help="number of top predicted metabolites to tabulate")
    ap.add_argument("--plot", action="store_true", help="also save spatial maps of the top predicted metabolites")
    ap.add_argument("--section_key", default=None,
                    help="obs column identifying tissue sections (e.g. 'sample'); keeps the transport graph "
                         "within each section for multi-section queries. Auto-detected if omitted.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    obj = pickle.load(open(args.model, "rb"))
    if isinstance(obj, dict):                          # bundle {model, mz, genes, conf_width, ...}
        model = obj["model"]
        mz = np.asarray(obj.get("mz", getattr(model, "mz_", [])), float)
        conf_width = obj.get("conf_width")
        train_genes = set(obj.get("genes", getattr(model, "genes_", [])))
    else:                                              # bare fitted MetaSpatial instance
        model = obj
        mz = np.asarray(getattr(model, "mz_", []), float)
        conf_width = None
        train_genes = set(getattr(model, "genes_", []))
    print(f"loaded model: {len(mz)} metabolites, {len(train_genes)} training genes, "
          f"conformal={'yes' if conf_width is not None else 'no'}")

    q = sc.read_h5ad(args.query)
    q.var_names_make_unique()
    print(f"query: {q.n_obs} spots x {q.n_vars} genes  ({os.path.basename(args.query)})")
    # species / panel guard
    overlap = len(set(q.var_names) & train_genes) / max(len(train_genes), 1)
    print(f"  gene overlap with training panel: {overlap:5.1%}")
    if overlap < 0.30:
        print("  [WARNING] low gene overlap — likely a SPECIES mismatch (e.g. human model on mouse data) "
              "or different gene IDs. Predictions will be unreliable. Match species / use gene SYMBOLS, "
              "or map orthologs first.")
    ensure_spatial(q)
    normalize(q, args.normalize)

    try:
        pred = model.predict_metabolome(q, section_key=args.section_key)   # section-aware transport graph
    except TypeError:
        pred = model.predict_metabolome(q)      # older model without section support
    # (n_spots x n_metab); also stored in q.obsm['metaspatial_pred']
    q.uns["metaspatial_mz"] = mz
    if conf_width is not None:
        q.uns["metaspatial_conf_width"] = conf_width

    # rank metabolites by predicted spatial dynamic range (structured predictions are the informative ones)
    dyn = pred.std(0)
    order = np.argsort(dyn)[::-1]
    tab = pd.DataFrame({
        "mz": np.round(mz[order], 4),
        "predicted_mean": pred[:, order].mean(0),
        "predicted_std": dyn[order],
        **({"conf_width": conf_width[order]} if conf_width is not None else {}),
    })
    tab.to_csv(os.path.join(args.out, "predicted_metabolites.csv"), index=False)
    q.write(os.path.join(args.out, "query_with_predictions.h5ad"))

    print(f"\nsaved:")
    print(f"  {args.out}/query_with_predictions.h5ad   (predictions in .obsm['metaspatial_pred'])")
    print(f"  {args.out}/predicted_metabolites.csv     ({len(mz)} metabolites, ranked by spatial structure)")
    print(f"\ntop {min(args.top, len(mz))} most spatially-structured predicted metabolites (by m/z):")
    print(tab.head(args.top).to_string(index=False))

    if args.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xy = np.asarray(q.obsm["spatial"], float); n = min(6, len(mz))
        fig, axs = plt.subplots(2, 3, figsize=(11, 7))
        for k, ax in enumerate(axs.ravel()):
            if k >= n: ax.axis("off"); continue
            j = order[k]; v = pred[:, j]
            lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            ax.scatter(xy[:, 0], xy[:, 1], c=v, s=6, cmap="magma", vmin=lo, vmax=hi, linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"predicted m/z {mz[j]:.3f}", fontsize=9)
        fig.suptitle("MetaSpatial predicted metabolome (top spatially-structured ions)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "predicted_maps.png"), dpi=150)
        print(f"  {args.out}/predicted_maps.png              (top-6 predicted metabolite maps)")


if __name__ == "__main__":
    main()
