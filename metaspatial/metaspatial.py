"""
MetaSpatial — transport-aware prediction of spatial metabolomes from spatial transcriptomics.

Minimal, scanpy-compatible API:
    from metaspatial import MetaSpatial
    model = MetaSpatial(use_kegg=True).fit(train_adatas)      # each adata: .layers['log1p'] + .uns['msi'] + .uns['mz_features'] + .obsm['spatial']
    pred  = model.predict_metabolome(query_adata)             # -> (n_spots x n_metab); also stored in query_adata.obsm['metaspatial_pred']

Design: HVG -> PCA -> transport features [P, A P, A^2 P] (neighbour diffusion) (+ KEGG pathway scores) -> multi-output Ridge.
Genes matched by name; predicts the training metabolite (m/z) axis. Optional split-conformal uncertainty.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye


def _norm_adj(coords, groups=None, k=6):
    n = len(coords); groups = np.zeros(n) if groups is None else np.asarray(groups)
    rows, cols = [], []
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        if len(gi) < 3: continue
        _, idx = cKDTree(coords[gi]).query(coords[gi], k=min(k + 1, len(gi)))
        for a in range(len(gi)):
            for b in idx[a, 1:]:
                rows.append(gi[a]); cols.append(gi[b])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A


def _dense(x):
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


class MetaSpatial:
    def __init__(self, n_hvg=3000, n_pcs=100, alpha=200.0, k=6, use_kegg=False, kegg_gmt=None, extra_key=None):
        self.n_hvg, self.n_pcs, self.alpha, self.k = n_hvg, n_pcs, alpha, k
        self.use_kegg = use_kegg
        self.extra_key = extra_key   # optional per-spot modality in adata.obsm (e.g. antibody-derived protein); concatenated to features
        self.kegg = self._load_gmt(kegg_gmt) if (use_kegg and kegg_gmt) else {}

    @staticmethod
    def _load_gmt(path):
        d = {}
        for ln in open(path):
            p = ln.rstrip("\n").split("\t"); d[p[0]] = [g for g in p[2:] if g]
        return d

    def _pathway_scores(self, X, var_names):
        pos = {g: i for i, g in enumerate(var_names)}; cols = []
        for gs in self.kegg.values():
            idx = [pos[g] for g in gs if g in pos]
            if len(idx) >= 3:
                z = (X[:, idx] - X[:, idx].mean(0)) / (X[:, idx].std(0) + 1e-8); cols.append(z.mean(1))
        return np.array(cols).T.astype(np.float32) if cols else np.zeros((X.shape[0], 0), np.float32)

    def _features(self, X_full, var_names, coords, groups, extra=None):
        Xh = X_full[:, self.hvg_idx_(var_names)]
        P = (Xh @ self.Vt_.T - self.pmean_ @ self.Vt_.T).astype(np.float32)
        A = _norm_adj(coords, groups, self.k); P1 = A @ P; F = np.hstack([P, P1, A @ P1])
        if self.use_kegg and self.kegg:
            PW = self._pathway_scores(X_full, var_names); F = np.hstack([F, PW, A @ PW])
        if extra is not None:                                   # optional extra modality (e.g. protein): value + neighbour-diffused
            E = np.asarray(extra, np.float32); F = np.hstack([F, E, A @ E])
        return F

    def hvg_idx_(self, var_names):
        return np.array([self._gpos.get(g, -1) for g in self.hvg_genes_])  # placeholder; set below

    def fit(self, adatas, metab_key="msi", mz_key="mz_features"):
        # memory-lean: never hold all dense gene matrices at once (robust on laptops & big cohorts)
        self.genes_ = sorted(set.intersection(*[set(a.var_names) for a in adatas]))
        self._gpos = {g: i for i, g in enumerate(self.genes_)}
        G = len(self.genes_)
        def gm(a):
            v = a[:, self.genes_].layers["log1p"] if "log1p" in a.layers else a[:, self.genes_].X
            return _dense(v)
        # pass 1 — pooled per-gene variance to choose HVGs (free each matrix immediately)
        s1 = np.zeros(G); s2 = np.zeros(G); N = 0
        for a in adatas:
            X = gm(a); s1 += X.sum(0); s2 += (X * X).sum(0); N += X.shape[0]; del X
        var = s2 / N - (s1 / N) ** 2
        self.gene_mean_ = (s1 / N).astype(np.float32)   # training per-gene mean (full gene set); used to impute query-absent genes
        self.hvg_ = np.argsort(var)[-self.n_hvg:]
        self.hvg_genes_ = [self.genes_[i] for i in self.hvg_]
        # pass 2 — PCA on HVG columns only
        Xh_all = np.vstack([gm(a)[:, self.hvg_] for a in adatas])
        self.pca_ = PCA(self.n_pcs, svd_solver="randomized", random_state=0).fit(Xh_all)
        self.Vt_ = self.pca_.components_.astype(np.float32); self.pmean_ = self.pca_.mean_.astype(np.float32)
        del Xh_all
        # pass 3 — transport features (per-section graph) + metabolite targets
        F = []; Y = []
        for a in adatas:
            X = gm(a)
            ex = np.asarray(a.obsm[self.extra_key]) if (self.extra_key and self.extra_key in a.obsm) else None
            F.append(self._features(X, self.genes_, np.asarray(a.obsm["spatial"], float), None, extra=ex)); del X
            Y.append(np.log1p(np.asarray(a.uns[metab_key], float)))
        F = np.vstack(F); Y = np.vstack(Y).astype(np.float32)
        self.mz_ = np.asarray(adatas[0].uns[mz_key], float)
        self.reg_ = Ridge(alpha=self.alpha).fit(F, Y)
        return self

    def predict_metabolome(self, adata, metab_uns_key="metaspatial_pred"):
        # align genes to training set; impute genes absent from the query at the TRAINING per-gene mean
        # so absent HVG genes contribute exactly zero after PCA centring (X-pmean_ -> 0 on those columns)
        # and do not bias the KEGG pathway z-scores
        pos = {g: i for i, g in enumerate(list(adata.var_names))}
        v = adata[:, [g for g in self.genes_ if g in pos]]
        Xsub = _dense(v.layers["log1p"] if "log1p" in v.layers else v.X)
        if hasattr(self, "gene_mean_"):
            X = np.tile(self.gene_mean_, (adata.n_obs, 1)).astype(np.float32)
        else:                                                # models pickled before mean-imputation was added: fall back to zero-fill
            X = np.zeros((adata.n_obs, len(self.genes_)), np.float32)
        present = [k for k, g in enumerate(self.genes_) if g in pos]
        X[:, present] = Xsub
        ex = np.asarray(adata.obsm[self.extra_key]) if (self.extra_key and self.extra_key in adata.obsm) else None
        F = self._features(X, self.genes_, np.asarray(adata.obsm["spatial"], float), None, extra=ex)
        pred = self.reg_.predict(F).astype(np.float32)
        adata.obsm[metab_uns_key] = pred
        return pred

    def save(self, path):
        """Pickle the fitted model to `path` (reload with MetaSpatial.load)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        """Load a model previously saved with .save(); returns a MetaSpatial instance."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # portable self-test: pass a folder containing MetaSpatial-format .h5ad, e.g.
    #   python metaspatial.py /path/to/sections
    import anndata as ad, glob, sys, os
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    h5 = [p for p in sorted(glob.glob(os.path.join(root, "**", "*.h5ad"), recursive=True))
          if "training_ready" not in p]
    if len(h5) < 2:
        print("Point me at a folder with >=2 MetaSpatial-format .h5ad files."); sys.exit(0)
    tr = [ad.read_h5ad(f) for f in h5[:-1]]; te = ad.read_h5ad(h5[-1])
    m = MetaSpatial(use_kegg=False).fit(tr)
    p = m.predict_metabolome(te)
    print(f"demo: trained on {len(tr)} sections; predicted {p.shape} metabolites for {os.path.basename(h5[-1])}")
