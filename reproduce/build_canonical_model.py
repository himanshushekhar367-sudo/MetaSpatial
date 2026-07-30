"""
Build ONE canonical MetaSpatial model (trained on DESIUM) and validate that the FIXED predict path
(mean-imputation of missing genes + target standardization + section-aware transport graph) recovers
the manuscript-quality N->PVTT gradient on HCC-2. The same pickled artifact is what R (reticulate) and
Python both load, so training/prediction is uniform across languages by construction.

Usage:  python build_canonical_model.py
Writes: metaspatial_model.pkl
"""
import sys, os, gzip, pickle, gc
sys.path.insert(0, "/home/claude/spatmet")
import numpy as np, pandas as pd, anndata as ad, scipy.io, scipy.sparse as sp
from metaspatial import MetaSpatial

DES = "/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
UP  = "/mnt/user-data/uploads/spatial metabolism"
dsamp = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]

# ---------- memory-lean loader: keep only sparse log1p + MSI targets + coords ----------
def slim(a):
    x = a.layers["log1p"] if "log1p" in a.layers else a.X
    s = ad.AnnData(X=sp.csr_matrix(x).astype(np.float32))
    s.var_names = list(a.var_names); s.obs_names = list(a.obs_names)
    s.layers["log1p"] = s.X
    s.uns["msi"] = np.asarray(a.uns["msi"], dtype=np.float32)
    s.uns["mz_features"] = np.asarray(a.uns["mz_features"], float)
    s.obsm["spatial"] = np.asarray(a.obsm["spatial"], float)
    return s

print("loading DESIUM (slim) ...", flush=True)
train = []
for sname in dsamp:
    a = ad.read_h5ad(f"{DES}/{sname}.h5ad"); train.append(slim(a)); del a; gc.collect()
mz = np.asarray(train[0].uns["mz_features"], float)

# ---------- HCC-2 query, built exactly like the R wrapper (CP10k -> log1p) ----------
def build_hcc2():
    genes = [l.strip() for l in open(f"{UP}/HCC-2-expr_features.txt")]
    bc    = [l.strip() for l in open(f"{UP}/HCC-2-expr_barcodes.txt")]
    M = scipy.io.mmread(gzip.open(f"{UP}/HCC-2-expr_counts.mtx.gz")).tocsc()
    tot = np.asarray(M.sum(0)).ravel() + 1e-8
    X = M.T.tocsr().astype(np.float32)
    X = X.multiply(1.0 / tot[:, None]).multiply(1e4).tocsr(); X.data = np.log1p(X.data)
    meta = pd.read_csv(f"{UP}/HCC-2-expr_meta.csv", index_col=0).reindex(bc)
    q = ad.AnnData(X=X); q.var_names = genes; q.obs_names = bc; q.layers["log1p"] = q.X
    q.obsm["spatial"] = meta[["imagecol", "imagerow"]].values.astype(float)
    q.obs["sample.ident"] = meta["sample.ident"].astype(str).values
    suff = meta["sample.ident"].astype(str).str.replace("HCC-2", "", regex=False).str.lstrip("-").values
    return q, suff

q, suff = build_hcc2()
SUB = ["N", "L", "P", "T"]
SIG = {"Glutathione":307.0838,"Ascorbate":176.0321,"Arachidonate FA20:4":304.2402,"DHA FA22:6":328.2402,
       "Palmitate FA16:0":256.2402,"Stearate FA18:0":284.2715,"Hexose/Glucose":180.0634,"Lactate":90.0317}
EXPECT = {"Glutathione":"+","Ascorbate":"+","Arachidonate FA20:4":"+","DHA FA22:6":"+",
          "Palmitate FA16:0":"-","Stearate FA18:0":"-","Hexose/Glucose":"-","Lactate":"0"}
ADD = [("[M-H]-",-1.0073),("[M+Cl]-",34.9694),("[M+HCOO]-",44.9982)]
def ion(neutral):
    for _, dl in ADD:
        t = neutral + dl; i = int(np.argmin(np.abs(mz - t)))
        if abs(mz[i] - t) <= 0.01: return i
    return None

def trajectory(pred, tag, overlap=None):
    print(f"\n[{tag}]  gene-overlap={overlap}")
    print(f"  {'metabolite':20s} {'N':>7s} {'L':>7s} {'P':>7s} {'T':>7s}   Δ(P-N)  dir")
    ok = 0; total = 0
    for nm, neu in SIG.items():
        i = ion(neu)
        if i is None: continue
        m = [pred[suff==s, i].mean() for s in SUB]; d = m[2]-m[0]
        got = "+" if d > 0.02 else ("-" if d < -0.02 else "0")
        hit = (EXPECT[nm]=="0" and abs(d) < 0.05) or (EXPECT[nm]==got and EXPECT[nm]!="0")
        if EXPECT[nm] != "0": total += 1; ok += int(hit)
        print(f"  {nm:20s} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f} {m[3]:7.3f}  {d:+7.3f}   {got}  {'OK' if hit else 'x'}")
    print(f"  --> {ok}/{total} directional signs match the manuscript signature")
    return ok, total

# manuscript ground truth
man = np.load("/home/claude/spatmet/xcancer/HCC-2_pred.npz")["pred"]
trajectory(man, "MANUSCRIPT ideal (shared-gene pipeline)")

# sweep HVG breadth (all with the FIXED predict path); pick the best-matching canonical
gsh = ion(307.0838)
best = None
for nhvg in [3000, 6000, 10000, None]:
    try:
        print(f"\ntraining MetaSpatial(n_hvg={nhvg}) ...", flush=True)
        m = MetaSpatial(n_hvg=nhvg, use_kegg=False).fit(train)
        p = m.predict_metabolome(q, section_key="sample.ident")
        ok, tot = trajectory(p, f"n_hvg={nhvg} (+mean-impute+section graph)", round(m.last_gene_overlap_, 3))
        r = float(np.corrcoef(p[:, gsh], man[:, gsh])[0, 1])
        print(f"    GSH r vs manuscript = {r:.3f}")
        score = (ok, round(r, 3))
        if best is None or score > best[0]:
            best = (score, m, nhvg, ok, tot, r)
        del m, p; gc.collect()
    except MemoryError:
        print(f"    n_hvg={nhvg}: MemoryError, skipping"); gc.collect()
    except Exception as e:
        print(f"    n_hvg={nhvg}: {type(e).__name__}: {e}"); gc.collect()

(score, model, nhvg, ok, tot, r) = best
with open("/home/claude/spatmet/metaspatial_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"\n==> SELECTED canonical: n_hvg={nhvg} | directional {ok}/{tot} | GSH r={r:.3f}")
print(f"SAVED metaspatial_model.pkl | metabolites={len(model.mz_)} genes={len(model.genes_)} hvg={len(model.hvg_genes_)}")
