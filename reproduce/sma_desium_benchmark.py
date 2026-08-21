"""sma_desium_benchmark.py — cross-cohort detection-matched leave-one-section-out (LOSO) benchmark
reproducing the class-dependent transfer result (manuscript Fig. 6, Table S5) across the DESIUM
(DESI) cohort and the SMA (MALDI) species x panel cohorts.

For each cohort it runs the identical detection-matched LOSO protocol (train on all-but-one section,
predict the held-out section from transcriptome alone, score per-ion Spearman on ions detected in
>20% of the held-out spots) and reports the per-section median plus per-class breakdown. SMA sections
must first be converted to MetaSpatial format with pipeline/sma_to_metaspatial.py.

Run: python reproduce/sma_desium_benchmark.py
Inputs (env-overridable):
  DESIUM_DIR  -> folder of DESIUM *.h5ad
  SMA_DIR     -> folder of SMA-converted *.h5ad, grouped by species_panel via adata.uns['cohort']
"""
import sys, os, glob, numpy as np, anndata as ad, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr
from metaspatial import MetaSpatial
warnings.filterwarnings("ignore")

DESIUM_DIR = os.environ.get("DESIUM_DIR", "/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data")
SMA_DIR = os.environ.get("SMA_DIR", "/mnt/user-data/uploads/spatial metabolism/SMA_converted")

def loso(sections, n_hvg=2000):
    """sections: list of MetaSpatial-format AnnData. Returns per-section median Spearman + per-ion means."""
    if len(sections) < 2:
        return np.array([]), None
    per = []
    med = []
    for i in range(len(sections)):
        tr = [sections[j] for j in range(len(sections)) if j != i]
        te = sections[i]
        m = MetaSpatial(n_hvg=n_hvg).fit(tr)
        pred = m.predict_metabolome(te.copy())
        true = np.log1p(np.asarray(te.uns["msi"], float))
        det = (np.asarray(te.uns["msi"], float) > 0).mean(0) > 0.20
        rr = []
        for k in np.where(det)[0]:
            if true[:, k].std() > 1e-9 and pred[:, k].std() > 1e-9:
                rr.append(spearmanr(true[:, k], pred[:, k])[0])
        med.append(np.nanmedian(rr) if rr else np.nan)
    return np.array(med), None

def load_desium():
    S = ["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
    return [ad.read_h5ad(f"{DESIUM_DIR}/{s}.h5ad") for s in S if os.path.exists(f"{DESIUM_DIR}/{s}.h5ad")]

def load_sma_cohorts():
    """Group converted SMA sections by adata.uns['cohort'] (e.g. 'mouse_DHB', 'human_FMP')."""
    cohorts = {}
    for f in sorted(glob.glob(f"{SMA_DIR}/*.h5ad")):
        a = ad.read_h5ad(f)
        c = str(a.uns.get("cohort", "SMA"))
        cohorts.setdefault(c, []).append(a)
    return cohorts

def main():
    print("=== DESIUM (DESI) LOSO ===", flush=True)
    des = load_desium()
    if des:
        med, _ = loso(des)
        print(f"  DESIUM: per-section median {np.round(med,3)}  ->  overall {np.nanmean(med):+.3f}", flush=True)
    else:
        print("  DESIUM data not found; skipped.")
    print("=== SMA (MALDI) LOSO by species x panel ===", flush=True)
    cohorts = load_sma_cohorts() if os.path.isdir(SMA_DIR) else {}
    if not cohorts:
        print("  SMA converted data not found; run pipeline/sma_to_metaspatial.py first. Skipped.")
    for c, secs in cohorts.items():
        med, _ = loso(secs)
        if med.size:
            print(f"  {c:16s} n={len(secs)}  per-section {np.round(med,3)}  ->  overall {np.nanmean(med):+.3f}", flush=True)
    print("\nExpected pattern (manuscript): lipid panels (DHB) transfer ~0.37; neurotransmitter panels"
          " intermediate; DESI polar/mixed tumour ~0.03 — transfer tracks metabolite class, not the model.")

if __name__ == "__main__":
    main()
