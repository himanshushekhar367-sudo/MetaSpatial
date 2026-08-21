"""flux_validation.py — the Warburg substrate->product flux test (Methods, "Pathway activity and
metabolite-grounded scoring").

Question: can a transcriptome-only MetaSpatial prediction recover a *flux* contrast (not just a
static pool)? We use the Warburg readout z[lactate, m/z 90.032] - z[glucose, m/z 180.063]
(negative-mode adducts) as a 1-D score and ask how well it separates tumour from stroma spots,
comparing three feature sources on the annotated DESIUM sections:
  (1) glycolysis+TCA ENZYME scores (transcript-only KEGG mean-z),
  (2) MEASURED metabolites (upper bound),
  (3) leave-one-section-out MetaSpatial-PREDICTED metabolites (what a transcriptome-only user gets).
Classifier: 5-fold CV logistic regression, reported as ROC-AUC. Run: python reproduce/flux_validation.py
"""
import sys, os, numpy as np, anndata as ad, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from metaspatial import MetaSpatial
warnings.filterwarnings("ignore")

DATA = os.environ.get("DESIUM_DIR", "/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data")
SAMPLES = ["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
LACTATE, GLUCOSE = 90.0317, 180.0634
NEG = (-1.0073, 34.9694, 44.9982)   # [M-H]-, [M+Cl]-, [M+HCOO]-

def ion(mz, neutral, tol=0.01):
    for dl in NEG:
        t = neutral + dl; i = int(np.argmin(np.abs(mz - t)))
        if abs(mz[i] - t) <= tol: return i
    return None

def zc(v): return (v - v.mean()) / (v.std() + 1e-8)

def tumour_labels(a):
    # pathology annotation column varies; accept common keys, else fall back to None
    for k in ("Type", "pathology", "region", "annotation"):
        if k in a.obs:
            lab = a.obs[k].astype(str).str.lower().values
            return np.array(["tumour" in x or "tumor" in x or "carcinoma" in x for x in lab])
    return None

def auc(score, y):
    s = np.asarray(score).reshape(-1, 1)
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=500), s, y, cv=5, scoring="roc_auc")))

def main():
    mz = np.asarray(ad.read_h5ad(f"{DATA}/{SAMPLES[0]}.h5ad").uns["mz_features"], float)
    iL, iG = ion(mz, LACTATE), ion(mz, GLUCOSE)
    if iL is None or iG is None:
        print("lactate/glucose ions not on this m/z axis; abort."); return
    rows = []
    for s in SAMPLES:
        a = ad.read_h5ad(f"{DATA}/{s}.h5ad")
        y = tumour_labels(a)
        if y is None or y.sum() < 20 or (~y).sum() < 20:
            print(f"  {s}: no usable tumour/stroma labels, skipped"); continue
        Ymeas = np.log1p(np.asarray(a.uns["msi"], float))
        meas = zc(Ymeas[:, iL]) - zc(Ymeas[:, iG])                      # measured flux readout
        # LOSO-predicted metabolome for this section
        tr = [ad.read_h5ad(f"{DATA}/{o}.h5ad") for o in SAMPLES if o != s]
        m = MetaSpatial(n_hvg=3000).fit(tr)
        P = m.predict_metabolome(a.copy())
        predflux = zc(P[:, iL]) - zc(P[:, iG])                          # predicted flux readout
        # enzyme-only: glycolysis+TCA KEGG mean-z (uses the shipped human map if available)
        enz = None
        if getattr(m, "kegg", None):
            pos = {g: i for i, g in enumerate(a.var_names)}
            X = a.layers["log1p"]; X = np.asarray(X.todense() if hasattr(X, "todense") else X, float)
            cols = []
            for name, gs in m.kegg.items():
                if ("glycolysis" in name.lower()) or ("citrate cycle" in name.lower()):
                    idx = [pos[g] for g in gs if g in pos]
                    if len(idx) >= 3: cols.append(((X[:, idx] - X[:, idx].mean(0)) / (X[:, idx].std(0) + 1e-8)).mean(1))
            if cols: enz = np.vstack(cols).T.mean(1)
        rows.append((s, auc(meas, y), auc(predflux, y), (auc(enz, y) if enz is not None else np.nan)))
        print(f"  {s:16s} AUC measured={rows[-1][1]:.3f}  predicted={rows[-1][2]:.3f}  enzyme-only={rows[-1][3]:.3f}", flush=True)
    if rows:
        M = np.array([r[1:] for r in rows], float)
        print("\n=== Warburg flux discrimination (5-fold CV ROC-AUC, mean over sections) ===")
        print(f"  measured metabolites : {np.nanmean(M[:,0]):.3f}")
        print(f"  MetaSpatial predicted: {np.nanmean(M[:,1]):.3f}")
        print(f"  enzyme-only (KEGG)   : {np.nanmean(M[:,2]):.3f}")

if __name__ == "__main__":
    main()
