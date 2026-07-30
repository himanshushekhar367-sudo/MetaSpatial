"""Transfer test: train the chemistry-conditioned head on CCLE bulk (cell lines), apply to DESIUM
tissue spots, and see if it recovers spatial metabolite patterns better than MetaSpatial's LOSO 0.03.
Per-gene z-scoring in each domain harmonises platform scale; Spearman is scale/shift invariant."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
UP = "/mnt/user-data/uploads/spatial metabolism"; DES = f"{UP}/DESIUM/Correlation_NMF_analysis/data"
DSS = ["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
DF, ALPHA = 60, 30.0

d = np.load("/tmp/ccle_desc.npz", allow_pickle=True)
names = list(d["names"]); E = d["D"].astype(float); mass = d["mass"].astype(float)
met = pd.read_csv(f"{UP}/CCLE_metabolomics_20190502.csv").set_index("DepMap_ID")
expr = pd.read_csv(f"{UP}/_expr_metab.csv", index_col=0)
common = met.index.intersection(expr.index); met = met.loc[common]; expr = expr.loc[common]
expr.columns = [c.split(" (")[0] for c in expr.columns]
expr = expr.loc[:, ~expr.columns.duplicated()]
Y = met[names].values.astype(np.float32); Z = ((Y-Y.mean(0))/(Y.std(0)+1e-8)).astype(np.float32)

print("loading DESIUM ...", flush=True)
S = {}
for s in DSS:
    a = ad.read_h5ad(f"{DES}/{s}.h5ad"); x = a.layers["log1p"] if "log1p" in a.layers else a.X
    S[s] = dict(Xg={g:i for i,g in enumerate(map(str,a.var_names))}, X=sp.csr_matrix(x).astype(np.float32),
                Y=np.log1p(np.asarray(a.uns["msi"],float)).astype(np.float32),
                mz=np.asarray(a.uns["mz_features"],float)); del a
mz = S[DSS[0]]["mz"]
des_shared = set.intersection(*[set(S[s]["Xg"]) for s in DSS])
G = sorted(set(expr.columns) & des_shared)
print(f"genes CCLE∩DESIUM: {len(G)} | CCLE lines: {len(common)}", flush=True)

# CCLE features: per-gene z, PCA, post-z; train bilinear W on ALL CCLE
Xc = expr[G].values.astype(np.float32); gmu, gsd = Xc.mean(0), Xc.std(0)+1e-8
pca = PCA(DF, svd_solver="randomized", random_state=0).fit((Xc-gmu)/gsd)
Xp = pca.transform((Xc-gmu)/gsd); pmu, psd = Xp.mean(0), Xp.std(0)+1e-8; Xp = (Xp-pmu)/psd
Estd = ((E-E.mean(0))/(E.std(0)+1e-8)).astype(np.float32); dE = Estd.shape[1]; P = DF*dE
def block(Xpc, em): return (Xpc[:, :, None]*em[None, None, :]).reshape(Xpc.shape[0], P)
XtX = np.zeros((P, P)); Xty = np.zeros(P)
for m in range(len(names)):
    b = block(Xp, Estd[m]); XtX += b.T@b; Xty += b.T@Z[:, m]
W = np.linalg.solve(XtX + ALPHA*np.eye(P), Xty)
print("trained bulk conditioned head on CCLE.", flush=True)

# DESIUM per-gene stats (pooled) for harmonisation
tot = np.zeros(len(G)); tot2 = np.zeros(len(G)); n = 0
Xd_sec = {}
for s in DSS:
    idx = [S[s]["Xg"][g] for g in G]; Xd = np.asarray(S[s]["X"][:, idx].todense(), np.float32)
    Xd_sec[s] = Xd; tot += Xd.sum(0); tot2 += (Xd*Xd).sum(0); n += Xd.shape[0]
dmu = tot/n; dsd = np.sqrt(np.maximum(tot2/n - dmu**2, 1e-8))

# match CCLE metabolites -> DESIUM ion
det = np.mean([(S[s]["Y"] > 0).mean(0) for s in DSS], 0)
matches = []
for m in range(len(names)):
    t = mass[m]-1.0073; j = int(np.argmin(np.abs(mz-t)))
    if abs(mz[j]-t) <= 0.01 and det[j] > 0.2: matches.append((m, j))
print(f"CCLE<->DESIUM metabolite matches: {len(matches)}", flush=True)

# precompute per-section spot projections ONCE (they don't depend on the metabolite)
secPC = {s: (pca.transform((Xd_sec[s]-dmu)/dsd) - pmu)/psd for s in DSS}
# predict each matched metabolite spatially per section, Spearman vs measured MSI
rows = []
for (m, j) in matches:
    rs = []
    for s in DSS:
        zhat = block(secPC[s], Estd[m]) @ W
        yv = S[s]["Y"][:, j]
        if yv.std() > 1e-9: rs.append(spearmanr(zhat, yv)[0])
    rows.append((names[m], float(np.nanmedian(rs))))
rows.sort(key=lambda r: -r[1])
tr = np.array([r[1] for r in rows])
print("\n==== CCLE-bulk -> DESIUM-tissue TRANSFER (per-metabolite median Spearman across sections) ====", flush=True)
for nm, r in rows: print(f"   {nm:22s} {r:+.3f}", flush=True)
print(f"\n  overall: median {np.nanmedian(tr):+.3f}  mean {np.nanmean(tr):+.3f}  %pos {100*np.mean(tr>0):.0f}%  (n={len(tr)})", flush=True)
print(f"  MetaSpatial spatial-LOSO transfer = 0.03  ->  CCLE-bulk transfer {'BEATS' if np.nanmedian(tr)>0.03 else 'does NOT beat'} it", flush=True)
print("DONE.", flush=True)
