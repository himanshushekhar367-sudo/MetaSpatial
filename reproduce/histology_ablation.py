#!/usr/bin/env python3
"""DESIUM ablation: transcriptome-only MetaSpatial vs MetaSpatial + cheap H&E features.

This script uses the embedded Visium histology images in the DESIUM `.h5ad` files
when they can be matched safely to `obs['orig.ident']`. Sections without a matching
library image are skipped by default rather than sampled from the wrong slide.

Run:
  set DESIUM_DIR=C:\path\to\DESIUM\Correlation_NMF_analysis\data
  python reproduce/histology_ablation.py --protocol loso --out results/histology_ablation.csv
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from metaspatial import MetaSpatial, add_histology_features


DESIUM_DIR = os.environ.get(
    "DESIUM_DIR",
    r"C:\Users\pc\Desktop\spatial metabolism\DESIUM\Correlation_NMF_analysis\data",
)
SAMPLES = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]


def load_with_histology(sample, data_dir, allow_missing=False):
    path = os.path.join(data_dir, sample + ".h5ad")
    a = ad.read_h5ad(path)
    try:
        add_histology_features(a, key_added="histology")
        return a, None
    except Exception as e:
        if allow_missing:
            return a, str(e)
        raise


def score_pred(true, pred, ions):
    rr = []
    for k in ions:
        if true[:, k].std() > 1e-9 and pred[:, k].std() > 1e-9:
            rr.append(spearmanr(true[:, k], pred[:, k])[0])
    return float(np.nanmedian(rr)) if rr else float("nan")


def detected_top_ions(a, n=1000):
    raw = np.asarray(a.uns["msi"], float)
    y = np.log1p(raw)
    det = (raw > 0).mean(0) > 0.20
    keep = np.where(det)[0]
    if keep.size > n:
        keep = keep[np.argsort(y[:, keep].var(0))[-n:]]
    return keep


def loso(sections, samples, n_hvg):
    rows = []
    for i, sample in enumerate(samples):
        tr = [sections[j] for j in range(len(sections)) if j != i]
        te = sections[i]
        true = np.log1p(np.asarray(te.uns["msi"], float))
        ions = detected_top_ions(te)
        for label, extra_key in [("MetaSpatial", None), ("MetaSpatial+H&E", "histology")]:
            m = MetaSpatial(n_hvg=n_hvg, extra_key=extra_key).fit(tr)
            pred = m.predict_metabolome(te.copy())
            rows.append({
                "protocol": "LOSO",
                "sample": sample,
                "model": label,
                "n_ions": len(ions),
                "median_spearman": score_pred(true, pred, ions),
            })
            print(f"  {sample:18s} {label:16s} r={rows[-1]['median_spearman']:+.3f}", flush=True)
    return pd.DataFrame(rows)


def within_sample(sections, samples, n_hvg, n_metab, n_folds):
    rows = []
    for sample, a in zip(samples, sections):
        coords = np.asarray(a.obsm["spatial"], float)
        folds = KMeans(n_clusters=n_folds, n_init=10, random_state=0).fit_predict(coords)
        ions = detected_top_ions(a, n=n_metab)
        true_all = np.log1p(np.asarray(a.uns["msi"], float))
        for label, extra_key in [("MetaSpatial", None), ("MetaSpatial+H&E", "histology")]:
            fold_scores = []
            for f in range(n_folds):
                tr = a[folds != f].copy()
                te = a[folds == f].copy()
                m = MetaSpatial(n_hvg=n_hvg, extra_key=extra_key).fit([tr])
                pred = m.predict_metabolome(te)
                fold_scores.append(score_pred(true_all[folds == f], pred, ions))
            rows.append({
                "protocol": f"within_sample_{n_folds}fold",
                "sample": sample,
                "model": label,
                "n_ions": len(ions),
                "median_spearman": float(np.nanmean(fold_scores)),
            })
            print(f"  {sample:18s} {label:16s} r={rows[-1]['median_spearman']:+.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Ablate cheap H&E features as an optional MetaSpatial input.")
    ap.add_argument("--data", default=DESIUM_DIR, help="Folder containing DESIUM MetaSpatial-format .h5ad files.")
    ap.add_argument("--protocol", choices=["loso", "within"], default="loso")
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--n_metab", type=int, default=1000)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--max_sections", type=int, default=None, help="Debug/smoke limit.")
    ap.add_argument("--out", default=os.path.join("results", "histology_ablation.csv"))
    args = ap.parse_args()

    loaded, names, skipped = [], [], []
    for s in SAMPLES[: args.max_sections]:
        try:
            a, err = load_with_histology(s, args.data, allow_missing=True)
        except FileNotFoundError:
            skipped.append((s, "missing h5ad"))
            continue
        if err:
            skipped.append((s, err))
            continue
        loaded.append(a)
        names.append(s)
    if skipped:
        print("Skipped sections without safely matched embedded histology:")
        for s, err in skipped:
            print(f"  {s}: {err}")
    if len(loaded) < 2:
        raise SystemExit("Need at least two sections with matched histology for this ablation.")

    if args.protocol == "loso":
        df = loso(loaded, names, args.n_hvg)
    else:
        df = within_sample(loaded, names, args.n_hvg, args.n_metab, args.n_folds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    pivot = df.pivot(index="sample", columns="model", values="median_spearman")
    if {"MetaSpatial", "MetaSpatial+H&E"}.issubset(pivot.columns):
        pivot["delta_H&E"] = pivot["MetaSpatial+H&E"] - pivot["MetaSpatial"]
    print("\n=== Summary ===")
    print(pivot.round(4).to_string())
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
