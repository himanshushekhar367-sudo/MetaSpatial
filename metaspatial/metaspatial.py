"""
MetaSpatial — transport-aware prediction of spatial metabolomes from spatial transcriptomics.

Minimal, scanpy-compatible API:
    from metaspatial import MetaSpatial
    model = MetaSpatial(use_kegg=True).fit(train_adatas)   # each adata: .layers['log1p'] + .uns['msi'] + .uns['mz_features'] + .obsm['spatial']
    pred  = model.predict_metabolome(query_adata)          # -> (n_spots x n_metab); also stored in query_adata.obsm['metaspatial_pred']

Design: HVG -> PCA -> transport features [P, A P, A^2 P] (neighbour diffusion) (+ KEGG pathway scores)
-> multi-output Ridge on per-metabolite standardized targets. Genes are matched by symbol; the model
predicts the training metabolite (m/z) axis. Query genes absent from the training panel are imputed at
the training per-gene mean, so they contribute exactly zero after PCA centring. Outputs are log1p MSI
intensities (non-negative). Split-conformal per-metabolite interval widths, when estimated at fit time,
are attached to the query as .uns['metaspatial_conf_width'].
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye


def _norm_adj(coords, groups=None, k=6):
    """Degree-normalized (D^{-1/2} A) symmetric kNN adjacency + self loops, built within each group.

    NOTE: this is the left/degree normalization used throughout the manuscript (see Methods); it is
    intentionally not the symmetric D^{-1/2} A D^{-1/2}. Changing it would alter the diffusion operator
    and break reproducibility with the reported results.
    """
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
        self.last_gene_overlap_ = float("nan")   # set on every predict_metabolome call
        self.conf_width_ = None                  # per-metabolite conformal interval half-width (if estimated)

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
        # X_full columns are aligned to self.genes_; select HVG columns by position
        hvg_cols = np.asarray([self._gpos[g] for g in self.hvg_genes_])
        Xh = X_full[:, hvg_cols]
        P = (Xh @ self.Vt_.T - self.pmean_ @ self.Vt_.T).astype(np.float32)
        A = _norm_adj(coords, groups, self.k); P1 = A @ P; F = np.hstack([P, P1, A @ P1])
        if self.use_kegg and self.kegg:
            PW = self._pathway_scores(X_full, var_names); F = np.hstack([F, PW, A @ PW])
        if extra is not None:                                   # optional extra modality (e.g. protein): value + neighbour-diffused
            E = np.asarray(extra, np.float32); F = np.hstack([F, E, A @ E])
        return F

    def fit(self, adatas, metab_key="msi", mz_key="mz_features"):
        # memory-lean: never hold all dense gene matrices at once (robust on laptops & big cohorts)
        self.genes_ = sorted(set.intersection(*[set(a.var_names) for a in adatas]))
        self._gpos = {g: i for i, g in enumerate(self.genes_)}
        G = len(self.genes_)
        n_hvg = min(self.n_hvg if self.n_hvg else G, G)
        def gm(a):
            v = a[:, self.genes_].layers["log1p"] if "log1p" in a.layers else a[:, self.genes_].X
            return _dense(v)
        # pass 1 — pooled per-gene mean/variance to choose HVGs (free each matrix immediately)
        s1 = np.zeros(G); s2 = np.zeros(G); N = 0
        for a in adatas:
            X = gm(a); s1 += X.sum(0); s2 += (X * X).sum(0); N += X.shape[0]; del X
        var = s2 / N - (s1 / N) ** 2
        self.gene_mean_ = (s1 / N).astype(np.float32)   # training per-gene mean (full gene set); imputes query-absent genes
        self.hvg_ = np.argsort(var)[-n_hvg:]
        self.hvg_genes_ = [self.genes_[i] for i in self.hvg_]
        # pass 2 — PCA on HVG columns only
        Xh_all = np.vstack([gm(a)[:, self.hvg_] for a in adatas])
        self.pca_ = PCA(min(self.n_pcs, Xh_all.shape[1], Xh_all.shape[0]),
                        svd_solver="randomized", random_state=0).fit(Xh_all)
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
        # per-metabolite standardization stabilises the multi-output ridge across very differently-scaled ions
        self.mu_ = Y.mean(0).astype(np.float32); self.sd_ = (Y.std(0) + 1e-8).astype(np.float32)
        self.reg_ = Ridge(alpha=self.alpha).fit(F, (Y - self.mu_) / self.sd_)
        self.last_gene_overlap_ = 1.0   # trained on its own gene panel
        return self

    def estimate_conformal(self, adatas, metab_key="msi", q=0.90):
        """Split-conformal per-metabolite interval half-widths from a held-out section.

        Refits on all-but-last section, predicts the held-out section, and stores the per-metabolite
        `q`-quantile of absolute residuals in self.conf_width_ (log1p-intensity units). Requires >=2
        sections. Call after fit(); leaves the full-data model in self.reg_ untouched.
        """
        if len(adatas) < 2:
            self.conf_width_ = None; return self
        cal = MetaSpatial(self.n_hvg, self.n_pcs, self.alpha, self.k, self.use_kegg,
                          None, self.extra_key)
        cal.kegg = self.kegg
        cal.fit(adatas[:-1], metab_key=metab_key)
        held = adatas[-1]
        pred = cal.predict_metabolome(held.copy())
        true = np.log1p(np.asarray(held.uns[metab_key], float))
        self.conf_width_ = np.quantile(np.abs(true - pred), q, axis=0).astype(np.float32)
        return self

    def predict_metabolome(self, adata, metab_uns_key="metaspatial_pred", section_key=None):
        # align genes to the training set; impute genes absent from the query at the TRAINING per-gene mean
        # so absent genes contribute exactly zero after PCA centring (X - pmean_ -> 0 on those columns)
        # and do not bias the KEGG pathway z-scores
        pos = {g: i for i, g in enumerate(list(adata.var_names))}
        present_genes = [g for g in self.genes_ if g in pos]
        self.last_gene_overlap_ = len(present_genes) / max(len(self.genes_), 1)
        v = adata[:, present_genes]
        Xsub = _dense(v.layers["log1p"] if "log1p" in v.layers else v.X)
        if getattr(self, "gene_mean_", None) is not None:
            X = np.tile(self.gene_mean_, (adata.n_obs, 1)).astype(np.float32)
        else:                                                # models pickled before mean-imputation: fall back to zero-fill
            X = np.zeros((adata.n_obs, len(self.genes_)), np.float32)
        present = [k for k, g in enumerate(self.genes_) if g in pos]
        X[:, present] = Xsub
        # section-aware transport graph: keep neighbours within each section for multi-section queries
        groups = None
        if section_key is not None and section_key in adata.obs:
            groups = np.asarray(adata.obs[section_key].values)
        ex = np.asarray(adata.obsm[self.extra_key]) if (self.extra_key and self.extra_key in adata.obsm) else None
        F = self._features(X, self.genes_, np.asarray(adata.obsm["spatial"], float), groups, extra=ex)
        pred = self.reg_.predict(F).astype(np.float32)
        if getattr(self, "mu_", None) is not None and getattr(self, "sd_", None) is not None:
            pred = pred * self.sd_ + self.mu_          # invert per-metabolite standardization -> log1p intensity
        pred = np.maximum(pred, 0.0).astype(np.float32)  # log1p MSI intensities are non-negative
        adata.obsm[metab_uns_key] = pred
        if getattr(self, "conf_width_", None) is not None:
            adata.uns["metaspatial_conf_width"] = np.asarray(self.conf_width_, np.float32)
        return pred

    def save(self, path, bundle=True):
        """Pickle the model to `path`. By default writes a self-describing bundle dict
        {model, mz, genes, conf_width, version, ...}; reload with MetaSpatial.load()."""
        import pickle
        obj = self
        if bundle:
            obj = dict(model=self,
                       mz=np.asarray(getattr(self, "mz_", []), float),
                       genes=list(getattr(self, "genes_", [])),
                       conf_width=(None if getattr(self, "conf_width_", None) is None
                                   else np.asarray(self.conf_width_, np.float32)),
                       n_hvg=self.n_hvg,
                       version="metaspatial-unified-1")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        """Load a model saved with .save() OR the shipped bundle dict. Always returns a MetaSpatial
        instance (unwrapping the {model, ...} bundle and exposing the extra keys as .bundle_meta_)."""
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("model"), cls):
            m = obj["model"]
            m.bundle_meta_ = {k: v for k, v in obj.items() if k != "model"}
            # surface a bundled conformal width onto the model so predict_metabolome attaches it
            if getattr(m, "conf_width_", None) is None and obj.get("conf_width") is not None:
                m.conf_width_ = np.asarray(obj["conf_width"], np.float32)
            return m
        return obj   # last resort: hand back whatever was pickled


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
