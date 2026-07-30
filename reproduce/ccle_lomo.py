"""CCLE-scale leave-one-metabolite-out for the chemistry-conditioned head (bulk).
Efficient: precompute per-metabolite Gram matrices, so LOMO(m) = total - held-out. Compares the
conditioned bilinear head vs mean-pattern baseline vs in-distribution ceiling, overall and by class."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
UP = "/mnt/user-data/uploads/spatial metabolism"
DF, ALPHA = 60, 30.0

d = np.load("/tmp/ccle_desc.npz", allow_pickle=True)
names = list(d["names"]); E = d["D"].astype(float)
# classify for per-group reporting
import re
def kind(nm):
    if re.match(r"C\d+:\d+\s+(LPC|LPE|PC|SM|DAG|CE|TAG)", nm): return "lipid"
    if any(nm.lower().startswith(k) for k in ["acetyl","propionyl","butyr","hexanoyl","heptanoyl","lauroyl","myristoyl","palmitoyl","stearoyl","oleyl","arachidonyl","malonyl","valeryl"]) and "carnitine" in nm.lower(): return "carn"
    return "polar"
kinds = np.array([kind(n) for n in names])

met = pd.read_csv(f"{UP}/CCLE_metabolomics_20190502.csv").set_index("DepMap_ID")
expr = pd.read_csv(f"{UP}/_expr_metab.csv", index_col=0)
common = met.index.intersection(expr.index)
met = met.loc[common]; expr = expr.loc[common]
print(f"aligned cell lines: {len(common)} | genes: {expr.shape[1]} | metabolites mapped: {len(names)}", flush=True)

Y = met[names].values.astype(np.float32)                          # cell lines x metabolites
Z = ((Y - Y.mean(0)) / (Y.std(0) + 1e-8)).astype(np.float32)      # z-score per metabolite
Xg = expr.values.astype(np.float32)
Xg = np.log1p(Xg) if Xg.max() > 50 else Xg                        # already log2(TPM+1)
Xp = PCA(DF, svd_solver="randomized", random_state=0).fit_transform(Xg).astype(np.float32)
Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
Estd = ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)
Ncl, M, dE = Xp.shape[0], len(names), Estd.shape[1]; P = DF * dE
print(f"features: {DF} expr-PCs x {dE} chem-desc = {P} bilinear dims", flush=True)

# per-metabolite design block b_m = (Xp (x) e_m); precompute Gram pieces
def block(m): return (Xp[:, :, None] * Estd[m][None, None, :]).reshape(Ncl, P)
XtX = np.zeros((M, P, P), np.float32); Xty = np.zeros((M, P), np.float32)
for m in range(M):
    b = block(m); XtX[m] = b.T @ b; Xty[m] = b.T @ Z[:, m]
totX = XtX.sum(0); toty = Xty.sum(0); I = np.eye(P, dtype=np.float32)

def ceiling(m):                                                   # in-distribution: per-metabolite ridge, 5-fold
    idx = np.arange(Ncl); pred = np.zeros(Ncl)
    for b in range(5):
        te = idx % 5 == b; tr = ~te
        pred[te] = Ridge(ALPHA).fit(Xp[tr], Z[tr, m]).predict(Xp[te])
    return spearmanr(pred, Z[:, m])[0]

bi, mp, ce = [], [], []
for m in range(M):
    A = totX - XtX[m] + ALPHA * I; w = np.linalg.solve(A, toty - Xty[m])
    zhat = block(m) @ w
    bi.append(spearmanr(zhat, Z[:, m])[0])
    others = [k for k in range(M) if k != m]
    mp.append(spearmanr(Z[:, others].mean(1), Z[:, m])[0])
    ce.append(ceiling(m))
bi, mp, ce = map(np.array, (bi, mp, ce))

def summ(a, s=None):
    a = a if s is None else a[s]
    return f"median {np.nanmedian(a):+.3f}  mean {np.nanmean(a):+.3f}  %pos {100*np.mean(a>0):.0f}%  (n={len(a)})"
print("\n==== CCLE-scale LOMO (zero-shot metabolite prediction from bulk expression) ====", flush=True)
print(" in-distribution ceiling  :", summ(ce), flush=True)
print(" chem-conditioned bilinear:", summ(bi), flush=True)
print(" mean-pattern baseline    :", summ(mp), flush=True)
print(f" >>> bilinear - baseline gap (median): {np.nanmedian(bi)-np.nanmedian(mp):+.3f}   "
      f"(DESIUM-62 was -0.044; positive/near-zero = scale fixed it)", flush=True)
print("\n by metabolite class (bilinear zero-shot):", flush=True)
for k in ["lipid", "polar", "carn"]:
    s = kinds == k
    if s.sum(): print(f"   {k:6s}: {summ(bi, s)}   | baseline {np.nanmedian(mp[s]):+.3f}", flush=True)
print(f"\n chemistry check: corr(|logP|, bilinear) = {spearmanr(np.abs(E[:,1]), bi)[0]:+.2f}", flush=True)
print("DONE.", flush=True)
