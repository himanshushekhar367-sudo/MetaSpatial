"""Smoke tests: the shipped model must load and predict via the advertised 3-line workflow,
and a from-scratch train->save->load cycle must round-trip. Run: pytest -q  (or python tests/test_smoke.py)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anndata as ad
from metaspatial import MetaSpatial, add_histology_features

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(REPO, "metaspatial_model.pkl")


def _synthetic_query(genes, n=200, seed=0):
    rng = np.random.RandomState(seed)
    use = list(genes[:max(1, int(len(genes) * 0.8))])          # 80% panel overlap
    X = rng.gamma(1.0, 1.0, size=(n, len(use))).astype(np.float32)
    q = ad.AnnData(X=np.log1p(X)); q.var_names = use
    q.layers["log1p"] = q.X
    q.obsm["spatial"] = rng.rand(n, 2) * 100
    return q


def test_shipped_model_loads_and_predicts():
    assert os.path.exists(PKL), "shipped model missing"
    model = MetaSpatial.load(PKL)                              # bug #1: must return a MetaSpatial, not a dict
    assert isinstance(model, MetaSpatial)
    assert getattr(model, "gene_mean_", None) is not None, "bug #2: gene_mean_ must be present (no zero-fill fallback)"
    q = _synthetic_query(model.genes_)
    pred = model.predict_metabolome(q)
    assert pred.shape == (q.n_obs, len(model.mz_))
    assert np.isfinite(pred).all()
    assert (pred >= 0).all(), "bug #4: predictions must be non-negative log1p intensities"
    assert 0.0 <= model.last_gene_overlap_ <= 1.0 and model.last_gene_overlap_ == model.last_gene_overlap_, "bug #5: overlap must be set"
    # bug #3: shipped conformal widths are attached to the query
    assert "metaspatial_conf_width" in q.uns
    assert np.asarray(q.uns["metaspatial_conf_width"]).shape[0] == len(model.mz_)


def test_train_save_load_roundtrip(tmp_path=None):
    import tempfile
    rng = np.random.RandomState(1)
    def paired(nspots, ngenes=300, nmz=40, seed=0):
        r = np.random.RandomState(seed)
        a = ad.AnnData(X=np.log1p(r.gamma(1, 1, (nspots, ngenes)).astype(np.float32)))
        a.var_names = [f"G{i}" for i in range(ngenes)]; a.layers["log1p"] = a.X
        a.obsm["spatial"] = r.rand(nspots, 2) * 50
        a.uns["msi"] = r.gamma(1, 1, (nspots, nmz)).astype(np.float32)
        a.uns["mz_features"] = np.linspace(100, 900, nmz)
        return a
    tr = [paired(150, seed=s) for s in range(3)]
    m = MetaSpatial(n_hvg=100, use_kegg=False).fit(tr)
    m.estimate_conformal(tr)
    q = paired(120, seed=99)
    p1 = m.predict_metabolome(q.copy())
    d = tempfile.mkdtemp(); path = os.path.join(d, "m.pkl")
    m.save(path)
    m2 = MetaSpatial.load(path)
    assert isinstance(m2, MetaSpatial)
    p2 = m2.predict_metabolome(q.copy())
    assert np.allclose(p1, p2), "save->load must round-trip numerically"
    assert (p1 >= 0).all()


def test_kegg_human_map_autoloads():
    """use_kegg=True must auto-load the shipped human KEGG map (no path) and train/predict."""
    import anndata as ad, numpy as np
    def paired(nspots, seed):
        rr = np.random.RandomState(seed)
        a = ad.AnnData(X=np.log1p(rr.gamma(1, 1, (nspots, 300)).astype(np.float32)))
        a.var_names = [f"G{i}" for i in range(300)]; a.layers["log1p"] = a.X
        a.obsm["spatial"] = rr.rand(nspots, 2) * 50
        a.uns["msi"] = rr.gamma(1, 1, (nspots, 30)).astype(np.float32)
        a.uns["mz_features"] = np.linspace(100, 900, 30)
        return a
    tr = [paired(120, s) for s in range(2)]
    m = MetaSpatial(n_hvg=100, use_kegg=True).fit(tr)
    assert len(m.kegg) > 50, "shipped human KEGG map did not auto-load with use_kegg=True"
    assert m.kegg_gmt_ is not None and str(m.kegg_gmt_).endswith(".gmt")
    p = m.predict_metabolome(paired(80, 99))
    assert p.shape[1] == len(m.mz_) and (p >= 0).all()


def test_histology_features_from_embedded_visium_image():
    rng = np.random.RandomState(3)
    a = ad.AnnData(X=np.ones((4, 3), np.float32))
    a.var_names = ["G1", "G2", "G3"]
    a.obs["orig.ident"] = ["lib1"] * a.n_obs
    a.obsm["spatial"] = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=float)
    img = rng.rand(80, 80, 3).astype(np.float32)
    a.uns["spatial"] = {
        "lib1": {
            "images": {"hires": img},
            "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 8.0},
        }
    }
    add_histology_features(a)
    assert "histology" in a.obsm
    assert a.obsm["histology"].shape == (a.n_obs, 13)
    assert np.isfinite(a.obsm["histology"]).all()
    assert a.uns["histology_source"]["library_id"] == "lib1"


if __name__ == "__main__":
    test_shipped_model_loads_and_predicts()
    test_train_save_load_roundtrip()
    test_kegg_human_map_autoloads()
    test_histology_features_from_embedded_visium_image()
    print("ALL SMOKE TESTS PASSED")
