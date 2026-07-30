"""
Reviewer-requested analyses on DESIUM (self-contained; inlines the transport pipeline):
  A. Empirical coverage of the 90% split-conformal intervals  (reviewer Q2 / concern #3)
  B. LOSO transfer within-breast, within-lung, cross-cancer-type (reviewer Q3 / weakness #3)
  C. Extra baselines vs the linear transport model: per-spot ridge (no transport), ridge+transport,
     RandomForest, HistGradientBoosting, best-single-gene correlation  (weakness #1,#6 / Q6)
Each block prints as it finishes (flush) so partial results survive a timeout.
"""
import os, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, anndata as ad, scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye
from scipy.stats import spearmanr

DES = "/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
BREAST = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823"]
LUNG   = ["LC_091", "LC_170", "LC_276"]
DS = BREAST + LUNG
ALPHA = 200.0; NPC = 100; NHVG = 3000; K = 6

def dense(x): return np.asarray(x.todense() if hasattr(x, "todense") else x, np.float32)
def norm_adj(xy, k=K):
    n = len(xy); _, idx = cKDTree(xy).query(xy, k=min(k+1, n))
    r = np.repeat(np.arange(n), idx.shape[1]-1); c = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)); A = ((A+A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A

print("loading DESIUM ...", flush=True)
S = {}
for s in DS:
    a = ad.read_h5ad(f"{DES}/{s}.h5ad")
    x = a.layers["log1p"] if "log1p" in a.layers else a.X
    S[s] = dict(X=sp.csr_matrix(x).astype(np.float32), pos={g: i for i, g in enumerate(map(str, a.var_names))},
                Y=np.log1p(np.asarray(a.uns["msi"], float)).astype(np.float32),
                mz=np.asarray(a.uns["mz_features"], float), xy=np.asarray(a.obsm["spatial"], float))
    del a; gc.collect()
shared = sorted(set.intersection(*[set(S[s]["pos"]) for s in DS]))
gi = {s: np.array([S[s]["pos"][g] for g in shared]) for s in DS}
mz = S[DS[0]]["mz"]; NMET = len(mz)
print(f"  {len(DS)} sections, {len(shared)} shared genes, {NMET} ions", flush=True)

# cache dense shared-gene matrices ONCE (the expensive step) so fit_basis/feats are cheap
print("  caching dense shared-gene matrices ...", flush=True)
XC = {s: dense(S[s]["X"][:, gi[s]]) for s in DS}
def Xg(s):
    return XC[s]

def fit_basis(train_sections):
    """HVG (pooled variance) -> PCA(100) on the given training sections."""
    G = len(shared); s1 = np.zeros(G); s2 = np.zeros(G); N = 0
    for s in train_sections:
        X = Xg(s); s1 += X.sum(0); s2 += (X*X).sum(0); N += X.shape[0]; del X
    hvg = np.sort(np.argsort(s2/N - (s1/N)**2)[-NHVG:])
    Xh = np.vstack([Xg(s)[:, hvg] for s in train_sections])
    pca = PCA(NPC, svd_solver="randomized", random_state=0).fit(Xh); del Xh
    return hvg, pca.components_.astype(np.float32), pca.mean_.astype(np.float32)

def feats(s, hvg, Vt, pmean):
    X = Xg(s)[:, hvg]; P = (X @ Vt.T - pmean @ Vt.T).astype(np.float32)
    A = norm_adj(S[s]["xy"]); P1 = A @ P
    return P, np.hstack([P, P1, A @ P1])   # P (no transport), full transport

def med_spear(pred, Y, detmask):
    r = [spearmanr(pred[:, j], Y[:, j])[0] for j in range(Y.shape[1]) if detmask[j] and Y[:, j].std() > 1e-9]
    return float(np.nanmedian(r))

# ---------------- B. transfer by cancer type (LOSO) ----------------
print("\n===== B. LOSO transfer by cancer type (detection-matched median Spearman) =====", flush=True)
def loso(train_pool, test_sections, label):
    res = []
    for te in test_sections:
        tr = [s for s in train_pool if s != te]
        hvg, Vt, pm = fit_basis(tr)
        Ftr = np.vstack([feats(s, hvg, Vt, pm)[1] for s in tr]); Ytr = np.vstack([S[s]["Y"] for s in tr])
        mu = Ytr.mean(0); sd = Ytr.std(0)+1e-8
        reg = Ridge(ALPHA).fit(Ftr, (Ytr-mu)/sd); del Ftr, Ytr
        pred = reg.predict(feats(te, hvg, Vt, pm)[1])*sd + mu
        det = (S[te]["Y"] > 0).mean(0) > 0.2
        res.append(med_spear(pred, S[te]["Y"], det))
    print(f"  {label:26s} median r = {np.nanmedian(res):+.3f}   per-section {[round(x,3) for x in res]}", flush=True)
    return res
try:
    loso(BREAST, BREAST, "within-breast")
    loso(LUNG,   LUNG,   "within-lung")
    # cross-type: train all breast -> predict each lung, and vice versa
    def cross(train, test, label):
        hvg, Vt, pm = fit_basis(train)
        Ftr = np.vstack([feats(s, hvg, Vt, pm)[1] for s in train]); Ytr = np.vstack([S[s]["Y"] for s in train])
        mu = Ytr.mean(0); sd = Ytr.std(0)+1e-8; reg = Ridge(ALPHA).fit(Ftr, (Ytr-mu)/sd); del Ftr, Ytr
        res = []
        for te in test:
            pred = reg.predict(feats(te, hvg, Vt, pm)[1])*sd+mu; det = (S[te]["Y"] > 0).mean(0) > 0.2
            res.append(med_spear(pred, S[te]["Y"], det))
        print(f"  {label:26s} median r = {np.nanmedian(res):+.3f}   per-section {[round(x,3) for x in res]}", flush=True)
    cross(BREAST, LUNG, "breast->lung (cross)")
    cross(LUNG, BREAST, "lung->breast (cross)")
except Exception as e:
    print("  B failed:", type(e).__name__, e, flush=True)

# ---------------- A. conformal coverage (within-sample, pooled spatial split) ----------------
print("\n===== A. Split-conformal 90% interval coverage (within-sample) =====", flush=True)
try:
    hvg, Vt, pm = fit_basis(DS)
    # per-section spatial 60/20/20 blocks via KMeans on coords
    tr_i, ca_i, te_i, F_all, Y_all = [], [], [], [], []
    off = 0
    for s in DS:
        _, Ff = feats(s, hvg, Vt, pm); F_all.append(Ff); Y_all.append(S[s]["Y"])
        lab = KMeans(5, n_init=3, random_state=0).fit_predict(S[s]["xy"])
        for b in range(5):
            idx = off + np.where(lab == b)[0]
            (tr_i if b < 3 else ca_i if b == 3 else te_i).extend(idx.tolist())
        off += S[s]["Y"].shape[0]
    F_all = np.vstack(F_all); Y_all = np.vstack(Y_all)
    tr_i, ca_i, te_i = map(np.array, (tr_i, ca_i, te_i))
    mu = Y_all[tr_i].mean(0); sd = Y_all[tr_i].std(0)+1e-8
    reg = Ridge(ALPHA).fit(F_all[tr_i], (Y_all[tr_i]-mu)/sd)
    predc = reg.predict(F_all[ca_i])*sd+mu; predt = reg.predict(F_all[te_i])*sd+mu
    det = (Y_all > 0).mean(0) > 0.2
    q = np.quantile(np.abs(predc - Y_all[ca_i]), 0.90, axis=0)   # per-ion 90% half-width
    inside = (np.abs(predt - Y_all[te_i]) <= q)                  # coverage on test
    cov_overall = float(inside[:, det].mean())
    cov_perion = inside[:, det].mean(0)
    print(f"  target 90%  ->  empirical coverage = {cov_overall:.3f}", flush=True)
    print(f"  per-ion coverage: median {np.median(cov_perion):.3f}, IQR [{np.percentile(cov_perion,25):.3f}, {np.percentile(cov_perion,75):.3f}]", flush=True)
except Exception as e:
    print("  A failed:", type(e).__name__, e, flush=True)

# ---------------- C. extra baselines (within-sample, pooled spatial 5-fold, capped ions) ----------------
print("\n===== C. Method comparison, within-sample spatial 5-fold (median Spearman, well-measured ions) =====", flush=True)
try:
    hvg, Vt, pm = fit_basis(DS)
    Plist, Flist, Ylist, fold = [], [], [], []
    for fi, s in enumerate(DS):
        P, Ff = feats(s, hvg, Vt, pm); Plist.append(P); Flist.append(Ff); Ylist.append(S[s]["Y"])
        fold.append(KMeans(5, n_init=3, random_state=0).fit_predict(S[s]["xy"]))
    P = np.vstack(Plist); F = np.vstack(Flist); Y = np.vstack(Ylist); fold = np.concatenate(fold)
    det_all = (Y > 0).mean(0) > 0.2
    score = (Y > 0).mean(0) * Y.var(0); ions = np.argsort(score)[-100:]   # top-100 well-measured
    Ysub = Y[:, ions]
    def cv(fit_predict):
        preds = np.zeros_like(Ysub)
        for b in range(5):
            tr = fold != b; te = ~tr
            preds[te] = fit_predict(tr, te)
        return med_spear(preds, Ysub, np.ones(Ysub.shape[1], bool))
    def ridge_on(Xfeat):
        def f(tr, te):
            mu = Ysub[tr].mean(0); sd = Ysub[tr].std(0)+1e-8
            return Ridge(ALPHA).fit(Xfeat[tr], (Ysub[tr]-mu)/sd).predict(Xfeat[te])*sd+mu
        return f
    print(f"  per-spot ridge (P, no transport) median r = {cv(ridge_on(P)):+.3f}", flush=True)
    print(f"  ridge + transport [P,AP,AAP]     median r = {cv(ridge_on(F)):+.3f}", flush=True)
    def rf(tr, te):
        m = RandomForestRegressor(n_estimators=50, max_depth=None, n_jobs=-1, random_state=0)
        return m.fit(P[tr], Ysub[tr]).predict(P[te])
    print(f"  RandomForest (P features)        median r = {cv(rf):+.3f}", flush=True)
    # HistGradientBoosting on a 20-ion subset (single-target, looped) as a boosting datapoint
    def hgb_cv(nj=20):
        preds = np.zeros((Ysub.shape[0], nj))
        for b in range(5):
            tr = fold != b; te = ~tr
            for j in range(nj):
                m = HistGradientBoostingRegressor(max_depth=6, max_iter=120, random_state=0)
                preds[te, j] = m.fit(P[tr], Ysub[tr, j]).predict(P[te])
        return med_spear(preds, Ysub[:, :nj], np.ones(nj, bool))
    print(f"  HistGradBoosting (top-20 ions)   median r = {hgb_cv():+.3f}  [subset]", flush=True)
    print("  (best-single-gene baseline is in the manuscript benchmark; omitted here for runtime)", flush=True)
except Exception as e:
    print("  C failed:", type(e).__name__, e, flush=True)

print("\nDONE.", flush=True)
