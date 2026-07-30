#!/usr/bin/env python3
"""
MetaSpatial - all-in-one: inventory + train on EVERY paired section + leave-one-sample-out.

HOW TO USE (no download needed - just copy this whole file):
  1) Save this text as a file named:  metaspatial_all_in_one.py
  2) Install deps once:   pip install scanpy anndata scikit-learn scipy pandas numpy
  3) Run:                 python metaspatial_all_in_one.py "C:\\Users\\pc\\Desktop\\spatial metabolism"
     (optional KEGG prior, matches the paper's best result - put scMetab_KEGG.gmt in the folder)
  4) Paste the ===== SUMMARY ===== block back to me.

It auto-discovers every MetaSpatial-format .h5ad under the folder (the 7 DESIUM sections now;
SMA / any new paired cohort the moment its .h5ad are added), trains on all of them, and reports
within-cohort and honest leave-one-sample-out accuracy. Writes training_ready/ outputs.

MetaSpatial-format AnnData per section:
  adata.layers['log1p']     genes, log1p(CP10k)     (n_spots x n_genes)   [or adata.X]
  adata.uns['msi']          metabolite intensities  (n_spots x n_metab)
  adata.uns['mz_features']  m/z values              (n_metab,)
  adata.obsm['spatial']     x,y coordinates         (n_spots x 2)
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
try:
    import numpy as np, anndata as ad
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.stats import spearmanr
except ImportError as e:
    print("MISSING DEPENDENCY:", e)
    print("  Run:  pip install scanpy anndata scikit-learn scipy pandas numpy")
    sys.exit(1)


# ============================ MetaSpatial (memory-lean) ============================
def _norm_adj(coords, k=6):
    n = len(coords)
    d, idx = cKDTree(coords).query(coords, k=min(k + 1, n))
    r = np.repeat(np.arange(n), idx.shape[1] - 1); c = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A

def _dense(x):
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)

class MetaSpatial:
    def __init__(self, n_hvg=3000, n_pcs=100, alpha=200.0, k=6, kegg_gmt=None):
        self.n_hvg, self.n_pcs, self.alpha, self.k = n_hvg, n_pcs, alpha, k
        self.kegg = {}
        if kegg_gmt and os.path.exists(kegg_gmt):
            for ln in open(kegg_gmt):
                p = ln.rstrip("\n").split("\t"); self.kegg[p[0]] = [g for g in p[2:] if g]

    def _gm(self, a):
        v = a[:, self.genes_].layers["log1p"] if "log1p" in a.layers else a[:, self.genes_].X
        return _dense(v)

    def _pathways(self, X):
        pos = self._gpos; cols = []
        for gs in self.kegg.values():
            ix = [pos[g] for g in gs if g in pos]
            if len(ix) >= 3:
                z = (X[:, ix] - X[:, ix].mean(0)) / (X[:, ix].std(0) + 1e-8); cols.append(z.mean(1))
        return np.array(cols).T.astype(np.float32) if cols else np.zeros((X.shape[0], 0), np.float32)

    def _features(self, X, coords):
        Xh = X[:, self.hvg_]
        P = (Xh @ self.Vt_.T - self.pmean_ @ self.Vt_.T).astype(np.float32)
        A = _norm_adj(coords, self.k); P1 = A @ P; F = np.hstack([P, P1, A @ P1])
        if self.kegg:
            PW = self._pathways(X); F = np.hstack([F, PW, A @ PW])
        return F

    def fit(self, adatas, metab_key="msi", mz_key="mz_features"):
        self.genes_ = sorted(set.intersection(*[set(a.var_names) for a in adatas]))
        self._gpos = {g: i for i, g in enumerate(self.genes_)}; G = len(self.genes_)
        s1 = np.zeros(G); s2 = np.zeros(G); N = 0
        for a in adatas:
            X = self._gm(a); s1 += X.sum(0); s2 += (X * X).sum(0); N += X.shape[0]; del X
        self.hvg_ = np.argsort(s2 / N - (s1 / N) ** 2)[-self.n_hvg:]
        self.gene_mean_ = (s1 / N).astype(np.float32)   # per-gene training mean, for mean-imputation at predict
        Xh_all = np.vstack([self._gm(a)[:, self.hvg_] for a in adatas])
        npc = int(min(self.n_pcs, Xh_all.shape[1], Xh_all.shape[0] - 1))   # guard tiny gene/spot counts
        self.pca_ = PCA(npc, svd_solver="randomized", random_state=0).fit(Xh_all)
        self.Vt_ = self.pca_.components_.astype(np.float32); self.pmean_ = self.pca_.mean_.astype(np.float32)
        del Xh_all
        F = []; Y = []
        for a in adatas:
            X = self._gm(a)
            F.append(self._features(X, np.asarray(a.obsm["spatial"], float))); del X
            Y.append(np.log1p(np.asarray(a.uns[metab_key], float)))
        F = np.vstack(F); Y = np.vstack(Y).astype(np.float32)
        self.mz_ = np.asarray(adatas[0].uns[mz_key], float)
        self.reg_ = Ridge(alpha=self.alpha).fit(F, Y)
        return self

    def predict_metabolome(self, adata):
        pos = {g: i for i, g in enumerate(list(adata.var_names))}
        present = [k for k, g in enumerate(self.genes_) if g in pos]
        sub = adata[:, [g for g in self.genes_ if g in pos]]
        Xsub = _dense(sub.layers["log1p"] if "log1p" in sub.layers else sub.X)
        # mean-impute genes absent from the query (training mean -> zero centred PCA contribution).
        # naive zero-fill injects a -mean bias per missing gene and corrupts cross-dataset transfer.
        base = getattr(self, "gene_mean_", np.zeros(len(self.genes_), np.float32))
        X = np.tile(base, (adata.n_obs, 1)); X[:, present] = Xsub
        return self.reg_.predict(self._features(X, np.asarray(adata.obsm["spatial"], float))).astype(np.float32)


# ================================ driver ================================
def trainable(a):
    return ("msi" in a.uns) and ("mz_features" in a.uns) and ("spatial" in a.obsm) and \
           (("log1p" in a.layers) or (a.X is not None))

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    kegg = None
    for cand in (sys.argv[2] if len(sys.argv) > 2 else "", os.path.join(root, "scMetab_KEGG.gmt"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "scMetab_KEGG.gmt")):
        if cand and os.path.exists(cand): kegg = cand; break
    out = os.path.join(root, "training_ready"); os.makedirs(out, exist_ok=True)

    print("=" * 64 + "\nMetaSpatial — discovering paired sections under:\n  " + root + "\n" + "=" * 64)
    paths = [p for p in sorted(glob.glob(os.path.join(root, "**", "*.h5ad"), recursive=True))
             if "training_ready" not in p]
    ads, names = [], []
    for p in paths:
        try:
            a = ad.read_h5ad(p)
        except Exception as ex:
            print(f"  [skip] {os.path.relpath(p, root)}: {ex}"); continue
        if trainable(a):
            nm = int(np.asarray(a.uns["msi"]).shape[1])
            print(f"  [OK] {os.path.relpath(p, root):55s} {a.n_obs:5d} spots x {a.n_vars} genes | {nm} metab")
            ads.append(a); names.append(os.path.splitext(os.path.basename(p))[0])
        else:
            print(f"  [--] {os.path.relpath(p, root):55s} not MetaSpatial-format (no msi/mz/spatial)")
    # also note raw cohorts that still need conversion
    imz = glob.glob(os.path.join(root, "**", "*.imzML"), recursive=True)
    vis = glob.glob(os.path.join(root, "**", "*filtered_feature_bc_matrix.h5"), recursive=True)
    if imz or vis:
        print(f"\n  (found {len(vis)} Visium .h5 and {len(imz)} imzML not yet in paired .h5ad form — "
              f"SMA/new cohorts need conversion first; paste this and I'll write the converter.)")

    if len(ads) < 2:
        print("\nNeed >=2 trainable sections for LOSO. Stopping."); return
    print(f"\nUsing KEGG prior: {kegg if kegg else 'no (run without it — add scMetab_KEGG.gmt to enable)'}")

    from collections import defaultdict
    def detmask(a):
        Yr = np.asarray(a.uns["msi"], float); return (Yr > 0).mean(0) > 0.2

    def species_tag(a):
        # human gene symbols are UPPERCASE (TH, GFAP); mouse are Title-case (Th, Gfap).
        vn = [g for g in list(a.var_names[:3000]) if isinstance(g, str) and g.isalpha()]
        if not vn: return "sp"
        return "Hs" if sum(g.isupper() for g in vn) / len(vn) > 0.5 else "Mm"

    # cohort key = (metabolite m/z axis, species): only sections sharing BOTH can pool into one model.
    # mouse+human can't share a gene space without ortholog mapping, so they train as separate cohorts.
    groups = defaultdict(list)
    for i, a in enumerate(ads):
        groups[(int(np.asarray(a.uns["msi"]).shape[1]), species_tag(a))].append(i)
    order = sorted(groups, key=lambda k: -len(groups[k]))
    print("\nCohorts by metabolite axis + species (only compatible sections pool):")
    for k in order:
        print(f"  {len(groups[k]):2d} section(s): {k[0]}-ion / {k[1]}"
              + ("" if len(groups[k]) >= 2 else "   (singleton — needs a 2nd matching section for LOSO)"))

    allres = {}
    for key in order:
        gi = groups[key]
        if len(gi) < 2:
            continue
        cads = [ads[i] for i in gi]; cnames = [names[i] for i in gi]
        tag = f"{key[0]}ion_{key[1]}_n{len(gi)}"
        print("\n" + "-" * 64 + f"\nCOHORT {tag}: training on all {len(gi)} sections ...", flush=True)
        model = MetaSpatial(kegg_gmt=kegg).fit(cads)
        try:
            import pickle; pickle.dump(model, open(os.path.join(out, f"metaspatial_model_{tag}.pkl"), "wb"))
            print(f"  saved training_ready/metaspatial_model_{tag}.pkl")
        except Exception as ex:
            print("  [warn] could not pickle model:", ex)
        print("  leave-one-sample-out (honest transfer, detection-matched median Spearman):", flush=True)
        loso = {}
        for h in range(len(cads)):
            m = MetaSpatial(kegg_gmt=kegg).fit([cads[i] for i in range(len(cads)) if i != h])
            pred = m.predict_metabolome(cads[h])
            Y = np.log1p(np.asarray(cads[h].uns["msi"], float)); det = detmask(cads[h])
            r = [spearmanr(pred[:, j], Y[:, j])[0] for j in range(Y.shape[1]) if det[j] and Y[:, j].std() > 1e-9]
            loso[cnames[h]] = float(np.nanmedian(r)); print(f"    {cnames[h]:26s} median r = {loso[cnames[h]]:+.3f}", flush=True)
        vals = list(loso.values())
        allres[tag] = dict(axis=key[0], species=key[1], sections=cnames, spots=int(sum(a.n_obs for a in cads)),
                           loso=loso, loso_overall=[float(np.mean(vals)), float(np.std(vals))])
    json.dump(allres, open(os.path.join(out, "train_all_results.json"), "w"), indent=2)

    print("\n" + "=" * 64 + "\n===== SUMMARY (paste this back) =====\n" + "=" * 64)
    print(f"KEGG prior : {'yes' if kegg else 'no'}")
    for tag, r in allres.items():
        mu, sd = r["loso_overall"]
        print(f"  cohort {tag:18s}: {len(r['sections']):2d} sections, {r['spots']:6d} spots | "
              f"LOSO median r = {mu:+.3f} +/- {sd:.3f}")
        print("       per section: {" + ", ".join(f"{k}:{v:+.2f}" for k, v in r["loso"].items()) + "}")
    sing = {f"{k[0]}ion/{k[1]}": len(groups[k]) for k in order if len(groups[k]) < 2}
    if sing: print("  singletons (no LOSO yet):", sing)
    print("saved : training_ready/train_all_results.json + metaspatial_model_<cohort>.pkl")

if __name__ == "__main__":
    main()
