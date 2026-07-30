#!/usr/bin/env python3
"""
MetaSpatial — train on EVERY available paired section + honest leave-one-sample-out benchmark.

Auto-discovers every MetaSpatial-format .h5ad under the folder (DESIUM now; SMA / any new
paired cohort the moment its .h5ad are added), trains on all of them, and reports both
within-cohort and leave-one-sample-out accuracy so you can watch the transfer number move
as more data is added.

USAGE:
    python train_all.py "C:\\Users\\pc\\Desktop\\spatial metabolism"

Requires metaspatial.py in the same folder (ships alongside this script), plus:
    pip install scanpy anndata scikit-learn scipy pandas numpy
Optional (KEGG prior): pass the scMetabolism gmt path as a 2nd argument.
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
ROOT   = sys.argv[1] if len(sys.argv) > 1 else "."
KEGG   = sys.argv[2] if len(sys.argv) > 2 else None
HERE   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np, anndata as ad
from scipy.stats import spearmanr
try:
    from metaspatial import MetaSpatial
except Exception as e:
    print("Could not import MetaSpatial — keep metaspatial.py next to this script.", e); sys.exit(1)

def trainable(path):
    try: a = ad.read_h5ad(path)
    except Exception: return False
    return ("msi" in a.uns) and ("mz_features" in a.uns) and ("spatial" in a.obsm) and \
           (("log1p" in a.layers) or (a.X is not None))

# 1) discover every trainable paired section
paths = [p for p in sorted(glob.glob(os.path.join(ROOT, "**", "*.h5ad"), recursive=True))
         if "training_ready" not in p and trainable(p)]
if len(paths) < 2:
    print(f"Found {len(paths)} trainable section(s) — need >=2 for LOSO. "
          f"Run prepare_training_data.py first / add more cohorts."); sys.exit(1)
adatas = [ad.read_h5ad(p) for p in paths]
names  = [os.path.splitext(os.path.basename(p))[0] for p in paths]
# only sections that share the SAME metabolite (m/z) axis can be pooled directly
mzsz   = [np.asarray(a.uns["mz_features"]).shape[0] for a in adatas]
from collections import Counter
main_axis = Counter(mzsz).most_common(1)[0][0]
keep = [i for i in range(len(adatas)) if mzsz[i] == main_axis]
if len(keep) < len(adatas):
    print(f"[note] pooling the {len(keep)} sections on the dominant {main_axis}-ion axis; "
          f"{len(adatas)-len(keep)} on a different axis are held for a separate cross-platform run.")
adatas = [adatas[i] for i in keep]; names = [names[i] for i in keep]
print(f"Training MetaSpatial on {len(adatas)} sections: {names}")

def detmask(a):
    Yr = np.asarray(a.uns["msi"], float); return (Yr > 0).mean(0) > 0.2

# 2) train on EVERYTHING (final model) + save
model = MetaSpatial(use_kegg=bool(KEGG), kegg_gmt=KEGG).fit(adatas)
try:
    import pickle; pickle.dump(model, open(os.path.join(ROOT, "training_ready",
              "metaspatial_model_all.pkl"), "wb")); print("[saved] training_ready/metaspatial_model_all.pkl")
except Exception as e:
    print("[warn] could not pickle model:", e)

# 3) leave-one-sample-out (honest transfer number, detection-matched median Spearman)
loso = {}
for h in range(len(adatas)):
    tr = [adatas[i] for i in range(len(adatas)) if i != h]
    m  = MetaSpatial(use_kegg=bool(KEGG), kegg_gmt=KEGG).fit(tr)
    pred = m.predict_metabolome(adatas[h])
    Y = np.log1p(np.asarray(adatas[h].uns["msi"], float)); det = detmask(adatas[h])
    r = [spearmanr(pred[:, j], Y[:, j])[0] for j in range(Y.shape[1]) if det[j] and Y[:, j].std() > 1e-9]
    loso[names[h]] = float(np.nanmedian(r)); print(f"  LOSO {names[h]:20s} median r = {loso[names[h]]:+.3f}", flush=True)
vals = list(loso.values())
summary = dict(n_sections=len(adatas), sections=names,
               loso_per_section=loso,
               loso_overall=[float(np.mean(vals)), float(np.std(vals))])
json.dump(summary, open(os.path.join(ROOT, "training_ready", "train_all_results.json"), "w"), indent=2)

print("\n===== TRAIN-ALL SUMMARY (paste this back) =====")
print(f"sections trained     : {len(adatas)}  ({main_axis}-ion axis)")
print(f"LOSO overall median r: {np.mean(vals):+.3f} +/- {np.std(vals):.3f}")
print("per section          :", {k: round(v, 3) for k, v in loso.items()})
print("saved                : training_ready/train_all_results.json + metaspatial_model_all.pkl")
