#!/usr/bin/env python3
"""
Quick start — predict the spatial metabolome for a transcriptome-only section
with the shipped pre-trained model. No mass-spectrometry data required.

    python examples/quickstart.py your_section.h5ad --model metaspatial_model.pkl

`your_section.h5ad` needs gene expression (symbols, in .X or .layers['log1p'])
and spatial coordinates in .obsm['spatial'].
"""
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from metaspatial import MetaSpatial


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", help="path to a .h5ad section (genes + obsm['spatial'])")
    ap.add_argument("--model", default="metaspatial_model.pkl", help="pre-trained model bundle")
    ap.add_argument("--out", default="predicted_metabolites.csv", help="output CSV of predictions")
    args = ap.parse_args()

    model = MetaSpatial.load(args.model)          # pre-trained; handles the shipped bundle
    adata = ad.read_h5ad(args.query)
    pred = model.predict_metabolome(adata)        # (n_spots x n_metab); also in adata.obsm['metaspatial_pred']

    mz = getattr(model, "mz_", np.arange(pred.shape[1]))
    cols = [f"mz_{m:.4f}" for m in np.asarray(mz)]
    pd.DataFrame(pred, columns=cols, index=adata.obs_names).to_csv(args.out)

    overlap = getattr(model, "last_gene_overlap_", float("nan"))
    print(f"Predicted {pred.shape[1]} metabolites for {pred.shape[0]} spots -> {args.out}")
    print(f"Gene overlap with the model: {overlap:.1%}"
          + ("  (low overlap -> interpret with caution)" if overlap == overlap and overlap < 0.5 else ""))
    print("Predictions are also stored in adata.obsm['metaspatial_pred'].")


if __name__ == "__main__":
    main()
