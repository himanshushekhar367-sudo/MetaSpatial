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
        # pass 1 — choose HVGs by pooled per-gene variance (or use ALL genes if n_hvg is None / >= G)
        if self.n_hvg is None or self.n_hvg >= G:
            self.hvg_ = np.arange(G); self.hvg_genes_ = list(self.genes_)
        else:
            s1 = np.zeros(G); s2 = np.zeros(G); N = 0
            for a in adatas:
                X = gm(a); s1 += X.sum(0); s2 += (X * X).sum(0); N += X.shape[0]; del X
            var = s2 / N - (s1 / N) ** 2
            self.hvg_ = np.sort(np.argsort(var)[-self.n_hvg:])
            self.hvg_genes_ = [self.genes_[i] for i in self.hvg_]
        # pass 2 — PCA on HVG columns only
        Xh_all = np.vstack([gm(a)[:, self.hvg_] for a in adatas])
        self.pca_ = PCA(self.n_pcs, svd_solver="randomized", random_state=0).fit(Xh_all)
        self.Vt_ = self.pca_.components_.astype(np.float32); self.pmean_ = self.pca_.mean_.astype(np.float32)
        if len(self.hvg_genes_) == G: self.gene_mean_ = self.pmean_   # all-genes case: PCA mean == full-gene mean
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
        self.mu_ = Y.mean(0).astype(np.float32); self.sd_ = (Y.std(0) + 1e-8).astype(np.float32)  # standardize targets
        self.reg_ = Ridge(alpha=self.alpha).fit(F, (Y - self.mu_) / self.sd_)
        return self

    def predict_metabolome(self, adata, metab_uns_key="metaspatial_pred", section_key=None, groups=None):
        """Predict the metabolome for a query section.

        section_key : column in adata.obs identifying tissue sections/subsections. When several
            sections are concatenated (each with its own coordinate frame, e.g. HCC-2 N/L/P/T), the
            transport graph is built block-diagonally so neighbours never cross section boundaries.
            If None, we auto-detect a 'sample.ident'/'section'/'library'/'batch' column, else use one graph.
        groups : optional array of per-spot section labels passed directly (overrides section_key).
            Convenient from R/reticulate, which can hand in a character vector without building adata.obs.

        Gene alignment uses MEAN-IMPUTATION: genes the model expects but the query lacks are set to the
        training mean, so their centred PCA contribution is exactly zero (no spurious offset). This is
        the key to correct cross-dataset transfer — naive zero-fill injects a -pmean bias per missing gene.
        """
        pos = {g: i for i, g in enumerate(list(adata.var_names))}
        Xq = _dense(adata.layers["log1p"] if "log1p" in adata.layers else adata.X)   # n_obs x n_query_genes
        n = adata.n_obs
        # --- HVG matrix, mean-imputed: start at the training mean, overwrite genes present in the query ---
        Xh = np.tile(self.pmean_.astype(np.float32), (n, 1))
        present = 0
        for k, g in enumerate(self.hvg_genes_):
            j = pos.get(g)
            if j is not None:
                Xh[:, k] = Xq[:, j]; present += 1
        self.last_gene_overlap_ = present / max(1, len(self.hvg_genes_))
        P = (Xh @ self.Vt_.T - self.pmean_ @ self.Vt_.T).astype(np.float32)           # missing genes -> 0 contribution
        # --- section-aware transport graph ---
        if groups is not None:
            groups = np.asarray(groups).astype(str)
        else:
            if section_key is None:
                for cand in ("sample.ident", "section", "library", "batch", "orig.ident"):
                    if cand in adata.obs.columns: section_key = cand; break
            if section_key is not None and section_key in adata.obs.columns:
                groups = np.asarray(adata.obs[section_key]).astype(str)
        A = _norm_adj(np.asarray(adata.obsm["spatial"], float), groups, self.k)
        P1 = A @ P; F = np.hstack([P, P1, A @ P1])
        if self.use_kegg and self.kegg:            # pathway prior on the mean-imputed full-gene matrix
            Xfull = np.tile(getattr(self, "gene_mean_", np.zeros(len(self.genes_), np.float32)), (n, 1))
            for k, g in enumerate(self.genes_):
                j = pos.get(g)
                if j is not None: Xfull[:, k] = Xq[:, j]
            PW = self._pathway_scores(Xfull, self.genes_); F = np.hstack([F, PW, A @ PW])
        if self.extra_key and self.extra_key in adata.obsm:
            E = np.asarray(adata.obsm[self.extra_key], np.float32); F = np.hstack([F, E, A @ E])
        Z = self.reg_.predict(F).astype(np.float32)
        pred = (Z * self.sd_ + self.mu_).astype(np.float32) if hasattr(self, "mu_") else Z   # un-standardize
        adata.obsm[metab_uns_key] = pred
        return pred

    def save(self, path):
        """Pickle this fitted model to `path` (loadable with `MetaSpatial.load`)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path):
        """Load a pre-trained model. Accepts either a pickled MetaSpatial instance or a
        bundle dict (``{'model': <MetaSpatial>, 'mz': ..., 'genes': ...}``), so the shipped
        ``metaspatial_model.pkl`` works directly."""
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            m = obj.get("model") or obj.get("metaspatial")
            if isinstance(m, cls):
                if "mz" in obj and not hasattr(m, "mz_"):
                    m.mz_ = np.asarray(obj["mz"], float)
                if "genes" in obj and not hasattr(m, "genes_"):
                    m.genes_ = list(obj["genes"])
                return m
        raise ValueError(f"{path} does not contain a MetaSpatial model (got {type(obj).__name__}).")


if __name__ == "__main__":
    # portable self-test: pass a folder containing MetaSpatial-format .h5ad, e.g.
    #   python metaspatial.py "C:\\Users\\pc\\Desktop\\spatial metabolism"
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
