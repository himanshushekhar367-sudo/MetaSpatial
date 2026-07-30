#!/usr/bin/env python3
"""
Train a MetaSpatial model on YOUR paired (spatial transcriptome + spatial metabolome) sections
and save a deployable model bundle (model + m/z axis + optional conformal uncertainty widths).

Each training section must be a MetaSpatial-format .h5ad:
    adata.layers['log1p'] (or adata.X)   genes, log1p(CP10k)   (n_spots x n_genes)
    adata.uns['msi']                     metabolite intensities (n_spots x n_metab)
    adata.uns['mz_features']             m/z values             (n_metab,)
    adata.obsm['spatial']                x, y coordinates       (n_spots x 2)

USAGE
    python examples/train_on_paired_data.py --data /path/to/folder_of_h5ad --out my_model.pkl
    python examples/train_on_paired_data.py --data ./sections --out m.pkl --kegg data/genesets/scMetab_KEGG.gmt

Then predict on any transcriptome-only data with examples/predict_on_your_data.py.
"""
import argparse, glob, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad

# make `import metaspatial` work whether or not the package is pip-installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metaspatial import MetaSpatial


def trainable(a):
    return ("msi" in a.uns) and ("mz_features" in a.uns) and ("spatial" in a.obsm) \
        and (("log1p" in a.layers) or (a.X is not None))


def main():
    ap = argparse.ArgumentParser(description="Train MetaSpatial on paired sections.")
    ap.add_argument("--data", required=True, help="folder containing MetaSpatial-format .h5ad sections")
    ap.add_argument("--out", default="metaspatial_model.pkl", help="output model bundle (.pkl)")
    ap.add_argument("--kegg", default=None, help="optional scMetabolism .gmt for the KEGG pathway prior")
    ap.add_argument("--alpha", type=float, default=200.0, help="ridge regularisation (default 200)")
    ap.add_argument("--no-conformal", action="store_true", help="skip conformal uncertainty estimation")
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(os.path.join(args.data, "**", "*.h5ad"), recursive=True))
             if "training_ready" not in p]
    ads, names = [], []
    for p in paths:
        try:
            a = ad.read_h5ad(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}"); continue
        if trainable(a):
            ads.append(a); names.append(os.path.basename(p))
            print(f"  [ok] {os.path.basename(p)}: {a.n_obs} spots x {a.n_vars} genes | "
                  f"{np.asarray(a.uns['msi']).shape[1]} metabolites")
    if not ads:
        sys.exit("No MetaSpatial-format .h5ad found. See docs/USAGE.md for the required fields.")
    # all sections must share the same metabolite (m/z) axis to pool
    sizes = {np.asarray(a.uns["msi"]).shape[1] for a in ads}
    if len(sizes) > 1:
        sys.exit(f"Sections have different metabolite-axis sizes {sizes}; pool only same-axis sections "
                 f"(convert/harmonise first — see pipeline/sma_to_metaspatial.py for an example).")

    use_kegg = args.kegg is not None
    conf_width = None
    # ---- conformal uncertainty: hold out one section, per-metabolite 90% abs-residual width ----
    if not args.no_conformal and len(ads) >= 3:
        cal = ads[-1]
        m = MetaSpatial(alpha=args.alpha, use_kegg=use_kegg, kegg_gmt=args.kegg).fit(ads[:-1])
        pred = m.predict_metabolome(cal)
        resid = np.abs(np.log1p(np.asarray(cal.uns["msi"], float)) - pred)
        conf_width = np.quantile(resid, 0.90, axis=0).astype(np.float32)
        print(f"  conformal: per-metabolite 90% interval widths from held-out '{names[-1]}' "
              f"(median width {np.median(conf_width):.3f})")
    elif not args.no_conformal:
        print("  conformal: need >=3 sections for a held-out calibration; skipping (point predictions only).")

    print(f"\nTraining final model on all {len(ads)} sections (alpha={args.alpha}, KEGG={use_kegg}) ...")
    model = MetaSpatial(alpha=args.alpha, use_kegg=use_kegg, kegg_gmt=args.kegg).fit(ads)

    bundle = dict(model=model, mz=np.asarray(model.mz_, float), conf_width=conf_width,
                  genes=list(model.genes_), n_train_sections=len(ads), train_sections=names)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nsaved model bundle -> {args.out}")
    print(f"  metabolites (m/z): {len(bundle['mz'])} | training genes: {len(bundle['genes'])} | "
          f"conformal: {'yes' if conf_width is not None else 'no'}")
    print("  predict on any transcriptome-only data:")
    print(f"    python examples/predict_on_your_data.py --model {args.out} --query your_data.h5ad --out results/")


if __name__ == "__main__":
    main()
