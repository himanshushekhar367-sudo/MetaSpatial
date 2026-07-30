#!/usr/bin/env python3
"""
Honest validation of the metabolite-grounded "flux signature" (Warburg: glucose-down / lactate-up)
across every DESIUM section, comparing three tumour-vs-stroma readouts:

    (1) enzymes            -- glycolysis+TCA gene activity   (scFEA / Compass style, transcript-only)
    (2) measured metabolites -- from adata.uns['msi']         (needs MSI; the ceiling)
    (3) PREDICTED metabolites -- MetaSpatial, leave-one-section-out (what MetaSpatial actually delivers)

WHY THIS MATTERS: (3) is the honest test of whether MetaSpatial's predicted metabolites beat enzyme-based
flux. Our run gives mean AUC enzymes 0.79, measured 0.86, PREDICTED 0.69 -- predicted UNDERPERFORMS enzymes,
because lactate (the Warburg product) is transcriptionally uncoupled (the paper's central finding). Run it
yourself to confirm before putting any flux claim in the manuscript.

USAGE
    python flux_validation.py --data "C:/Users/pc/Desktop/spatial metabolism/DESIUM"
    # optional: also apply the DESIUM-trained model to a transcriptome-only section for QUALITATIVE maps
    #           (no measured metabolites there, so this is hypothesis-generating, NOT validation):
    python flux_validation.py --data ".../DESIUM" --apply my_gbm_section.h5ad --out flux_out

Requires metaspatial installed (pip install -e . in the repo), scanpy, scikit-learn, matplotlib.
"""
import argparse, glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad
try:
    from metaspatial import MetaSpatial
except Exception:
    sys.exit("Install the package first:  pip install -e .  (from the MetaSpatial repo root)")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ADD = (-1.0073, 34.9694, 44.9982)              # negative-mode DESI adducts [M-H]-, [M+Cl]-, [M+HCOO]-
def idx_for(mz, mass): return [i for i, v in enumerate(mz) if any(abs(v-(mass+d)) <= 0.01 for d in ADD)]
def z(v): v = np.asarray(v, float); return (v-v.mean())/(v.std()+1e-8)

def kegg_genes(gmt, prefix):
    for ln in open(gmt):
        p = ln.rstrip("\n").split("\t")
        if p[0].startswith(prefix): return [g for g in p[2:] if g]
    return []

def enzyme_features(a, gly, tca):
    X = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, float)
    if X.max() > 50: X = np.log1p(X/(X.sum(1, keepdims=True)+1)*1e4)
    Z = (X-X.mean(0))/(X.std(0)+1e-8); gp = {g: i for i, g in enumerate(a.var_names)}
    e = lambda gs: Z[:, [gp[g] for g in gs if g in gp]].mean(1)
    return np.c_[e(gly), e(tca)]

def auc(F, y): return cross_val_score(LogisticRegression(max_iter=500), F, y, cv=5, scoring="roc_auc").mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the DESIUM .h5ad sections")
    ap.add_argument("--gmt", default="data/genesets/scMetab_KEGG.gmt")
    ap.add_argument("--apply", default=None, help="optional transcriptome-only .h5ad for a QUALITATIVE predicted map")
    ap.add_argument("--out", default="flux_out")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.data, "**", "*.h5ad"), recursive=True))
    ads = {os.path.splitext(os.path.basename(p))[0]: ad.read_h5ad(p) for p in paths}
    for a in ads.values(): a.var_names_make_unique()
    secs = list(ads)
    gly, tca = kegg_genes(args.gmt, "Glycolysis"), kegg_genes(args.gmt, "Citrate cycle")

    print(f"{'section':20s} {'nTum':>5} {'nStr':>5} | {'enzyme':>7} {'measured':>8} {'PREDICTED':>9}")
    rows = []
    for h in secs:
        a = ads[h]
        if "annotation" not in a.obs: continue
        ann = a.obs["annotation"].values; tum = ann == "Tumor"; stro = ann == "Stroma"
        if tum.sum() < 25 or stro.sum() < 25:
            print(f"{h:20s}  (no Tumor/Stroma labels — skipped)"); continue
        mz = np.asarray(a.uns["mz_features"], float); Y = np.log1p(np.asarray(a.uns["msi"], float))
        meas = z(Y[:, idx_for(mz, 90.0317)].mean(1)) - z(Y[:, idx_for(mz, 180.0634)].mean(1))   # lactate-up - glucose-down
        m = MetaSpatial(use_kegg=False).fit([ads[s] for s in secs if s != h])   # leave-one-section-out
        pred = m.predict_metabolome(a); pmz = np.asarray(m.mz_, float)
        pr = z(pred[:, idx_for(pmz, 90.0317)].mean(1)) - z(pred[:, idx_for(pmz, 180.0634)].mean(1))
        mask = tum | stro; y = tum[mask].astype(int)
        ae, am, ap_ = auc(enzyme_features(a, gly, tca)[mask], y), auc(meas[mask, None], y), auc(pr[mask, None], y)
        rows.append((h, ae, am, ap_))
        print(f"{h:20s} {tum.sum():5d} {stro.sum():5d} | {ae:7.3f} {am:8.3f} {ap_:9.3f}")
    if rows:
        E, M, P = (np.mean([r[i] for r in rows]) for i in (1, 2, 3))
        print(f"\nMEAN across {len(rows)} sections:  enzymes {E:.3f} | measured {M:.3f} | PREDICTED {P:.3f}")
        print("VERDICT:", "predicted beats enzymes — report it." if P > E + 0.02 else
              "predicted does NOT beat enzymes — do not claim a flux advance (lactate is transcriptionally uncoupled).")

    if args.apply:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import scanpy as sc
        q = ad.read_h5ad(args.apply) if args.apply.endswith(".h5ad") else sc.read_10x_h5(args.apply)
        q.var_names_make_unique()
        if "spatial" not in q.obsm: sys.exit("query needs obsm['spatial'] for a map")
        m = MetaSpatial(use_kegg=False).fit(list(ads.values()))
        pred = m.predict_metabolome(q); pmz = np.asarray(m.mz_, float)
        sig = z(pred[:, idx_for(pmz, 90.0317)].mean(1)) - z(pred[:, idx_for(pmz, 180.0634)].mean(1))
        xy = np.asarray(q.obsm["spatial"], float)
        plt.figure(figsize=(5, 5)); lo, hi = np.percentile(sig, 2), np.percentile(sig, 98)
        plt.scatter(xy[:, 0], xy[:, 1], c=sig, s=6, cmap="RdBu_r", vmin=-max(abs(lo), hi), vmax=max(abs(lo), hi), linewidths=0)
        plt.gca().set_aspect("equal"); plt.gca().invert_yaxis(); plt.axis("off")
        plt.title("PREDICTED glycolytic signature (qualitative,\nNOT validated — no measured metabolites here)", fontsize=9)
        plt.savefig(os.path.join(args.out, "applied_predicted_signature.png"), dpi=150, bbox_inches="tight")
        print(f"saved {args.out}/applied_predicted_signature.png  (hypothesis-generating map, not validation)")

if __name__ == "__main__":
    main()
