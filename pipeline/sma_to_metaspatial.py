#!/usr/bin/env python3
"""
SMA -> MetaSpatial converter  (Vicari et al., Nat Biotechnol 2024 SMA deposit).

Builds one MetaSpatial-format .h5ad per PAIRED section from the extracted SMA tree:
    <sample>_RNA/outs/filtered_feature_bc_matrix.h5      (Visium transcriptome)
    <sample>_MSI/<...>.Visium.<matrix>.<date>_smamsi.csv (MALDI metabolites, x,y raster)

The MSI csv is a dense raster (x,y + m/z columns). It is co-registered to the Visium
spots automatically: swap is fixed from tissue aspect ratio, x/y flips are chosen by
maximising transcriptome->metabolome coupling against a shuffled null (a built-in QC that
also proves the join is real). Sections are grouped by metabolite panel (DHB-lipids /
9AA-metabolites / FMP-neurotransmitters); within a panel all sections are harmonised to a
common m/z axis so they pool for leave-one-section-out.

USAGE:
    python sma_to_metaspatial.py "C:\\Users\\pc\\Desktop\\spatial metabolism\\SMA"
    (point at the folder that contains the 'sma' tree; outputs go to <that>\\..\\sma_h5ad\\
     ONE LEVEL UP so the main folder's metaspatial_all_in_one.py auto-discovers them)

Then paste the ===== QC SUMMARY ===== block back.

Deps: pip install scanpy anndata scikit-learn scipy pandas numpy
"""
import os, sys, glob, json, warnings, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
try:
    import scanpy as sc, anndata as ad
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
except ImportError as e:
    print("MISSING DEPENDENCY:", e); print("  pip install scanpy anndata scikit-learn scipy pandas numpy"); sys.exit(1)

def panel_of(fname):
    f = fname.lower()
    if ".dhb." in f: return "DHB"
    if ".9aa." in f or "_9aa" in f: return "9AA"
    if ".fmp." in f: return "FMP"
    return "OTHER"

def find_sections(root):
    """Return list of dicts {csv, sample, panel, h5, tp} for real paired sections."""
    out = []
    for csv in sorted(glob.glob(os.path.join(root, "**", "*smamsi.csv"), recursive=True)):
        if "__MACOSX" in csv or os.path.basename(csv).startswith("._"): continue
        msi_dir = os.path.dirname(csv)                       # ..._MSI
        sample  = os.path.basename(msi_dir)
        sample  = sample[:-4] if sample.endswith("_MSI") else sample
        odata   = os.path.dirname(msi_dir)                   # output_data
        outs    = os.path.join(odata, sample + "_RNA", "outs")
        h5      = os.path.join(outs, "filtered_feature_bc_matrix.h5")
        tp      = os.path.join(outs, "spatial", "tissue_positions_list.csv")
        if not os.path.exists(h5):
            h5c = glob.glob(os.path.join(odata, sample + "*_RNA", "outs", "filtered_feature_bc_matrix.h5"))
            h5 = h5c[0] if h5c else h5
            tp = os.path.join(os.path.dirname(h5), "spatial", "tissue_positions_list.csv")
        if not (os.path.exists(h5) and os.path.exists(tp)):
            print(f"  [skip] {sample}: no paired RNA outs (h5={os.path.exists(h5)}, tp={os.path.exists(tp)})"); continue
        out.append(dict(csv=csv, sample=sample, panel=panel_of(os.path.basename(csv)), h5=h5, tp=tp))
    return out

def read_positions(tp):
    p = pd.read_csv(tp, header=None)
    p.columns = ["barcode","in_tissue","array_row","array_col","pxl_row","pxl_col"][:p.shape[1]]
    return p[p.in_tissue == 1].copy()

def gene_pcs(g):
    q = g.copy(); sc.pp.normalize_total(q, target_sum=1e4); sc.pp.log1p(q)
    sc.pp.highly_variable_genes(q, n_top_genes=min(2000, q.n_vars-1))
    Xh = q[:, q.var.highly_variable].X
    Xh = np.asarray(Xh.todense()) if hasattr(Xh, "todense") else np.asarray(Xh)
    return PCA(min(30, Xh.shape[1]), random_state=0).fit_transform(Xh)

def couple(P, Y, reps=3):
    rng = np.random.RandomState(0); n = len(P); bios=[]; nulls=[]
    for _ in range(reps):
        o = rng.permutation(n); tr=o[:n//2]; te=o[n//2:]
        det = (np.expm1(Y[tr]) > 0).mean(0) > 0.2
        pred = Ridge(alpha=100).fit(P[tr], Y[tr]).predict(P[te])
        cols=[j for j in range(Y.shape[1]) if det[j] and Y[te][:,j].std()>1e-9]
        if not cols: return 0.0,0.0
        bios.append(np.nanmedian([spearmanr(pred[:,j],Y[te][:,j])[0] for j in cols]))
        pm=rng.permutation(len(te))
        nulls.append(np.nanmedian([spearmanr(pred[:,j],Y[te][pm][:,j])[0] for j in cols]))
    return float(np.mean(bios)), float(np.mean(nulls))

def bmap(u, tmn, tmx): return (u-u.min())/(u.max()-u.min()+1e-9)*(tmx-tmn)+tmn

def align(P, px, py, xy_nz, M_nz):
    """Fix swap from aspect ratio, choose x/y flips by max (bioR-null). Returns idx, meta."""
    xr = np.ptp(xy_nz[:,0]); yr = np.ptp(xy_nz[:,1])
    swap = ((np.ptp(px)/max(np.ptp(py),1)) > 1) != ((xr/max(yr,1)) > 1)
    best=None
    for fx in (False,True):
        for fy in (False,True):
            a = xy_nz[:,1].copy() if swap else xy_nz[:,0].copy()
            b = xy_nz[:,0].copy() if swap else xy_nz[:,1].copy()
            if fx: a = a.max()-a
            if fy: b = b.max()-b
            am=bmap(a,px.min(),px.max()); bm=bmap(b,py.min(),py.max())
            _,idx = cKDTree(np.c_[am,bm]).query(np.c_[px,py], k=1)
            bio,null = couple(P, np.log1p(M_nz[idx]))
            if best is None or (bio-null) > best[0]:
                best=(bio-null, idx, dict(swap=bool(swap),flipx=fx,flipy=fy,bioR=round(bio,3),nullR=round(null,3)))
    return best[1], best[2]

def convert_section(s, common_mz):
    g = sc.read_10x_h5(s["h5"]); g.var_names_make_unique()
    pos = read_positions(s["tp"])
    common = [b for b in g.obs_names if b in set(pos.barcode)]
    if len(common) < 200: return None, {"section":s["sample"],"status":"FEW_SPOTS"}
    g = g[common].copy(); pos = pos.set_index("barcode").loc[common].reset_index()
    px = pos.pxl_col.values.astype(float); py = pos.pxl_row.values.astype(float)
    P = gene_pcs(g)
    msi = pd.read_csv(s["csv"])
    xy = msi.iloc[:,:2].values.astype(float)
    mz = np.round(msi.columns[2:].astype(float).values, 4)
    M  = msi.iloc[:,2:].values.astype(np.float32)
    nz = M.sum(1) > 0; xy_nz = xy[nz]; M_nz = M[nz]
    idx, meta = align(P, px, py, xy_nz, M_nz)
    # subset metabolites to the panel's common axis
    keep = np.array([np.where(mz == m)[0][0] for m in common_mz])
    Y = M_nz[idx][:, keep].astype(np.float32)
    # log1p CP10k gene layer for MetaSpatial
    ln = g.copy(); sc.pp.normalize_total(ln, target_sum=1e4); sc.pp.log1p(ln)
    A = ad.AnnData(X=g.X.copy(), obs=pos.set_index("barcode"), var=g.var.copy())
    A.layers["log1p"] = ln.X.copy()
    A.obsm["spatial"] = np.c_[px, py]
    A.uns["msi"] = Y
    A.uns["mz_features"] = np.asarray(common_mz, float)
    A.uns["sma"] = dict(sample=s["sample"], panel=s["panel"], **meta)
    rec = {"section":s["sample"],"panel":s["panel"],"spots":int(A.n_obs),"n_mz":int(len(common_mz)),
           **meta, "status": "OK" if (meta["bioR"]-meta["nullR"])>=0.03 else "REVIEW_low_coupling"}
    return A, rec

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out  = os.path.join(root, "sma_h5ad"); os.makedirs(out, exist_ok=True)
    print("="*64+"\nSMA -> MetaSpatial converter\n root: "+root+"\n out : "+out+"\n"+"="*64)
    secs = find_sections(root)
    if not secs: print("No paired SMA sections found."); return
    from collections import defaultdict
    byp = defaultdict(list)
    for s in secs: byp[s["panel"]].append(s)
    # per-panel common m/z axis (intersection of rounded headers)
    common = {}
    for panel, lst in byp.items():
        sets=[]
        for s in lst:
            hdr = pd.read_csv(s["csv"], nrows=0).columns[2:]
            sets.append(set(np.round(hdr.astype(float).values, 4)))
        cm = sorted(set.intersection(*sets)) if sets else []
        common[panel] = np.array(cm)
        print(f"  panel {panel:4s}: {len(lst)} section(s), common m/z axis = {len(cm)}")
    recs=[]
    for panel, lst in byp.items():
        if len(common[panel]) < 5:
            for s in lst: recs.append({"section":s["sample"],"panel":panel,"status":"NO_COMMON_MZ"}); 
            continue
        for s in lst:
            try:
                A, rec = convert_section(s, common[panel])
            except Exception as ex:
                recs.append({"section":s["sample"],"panel":panel,"status":f"ERROR:{ex}"}); print("  [err]",s["sample"],ex); continue
            recs.append(rec)
            if A is not None:
                fp = os.path.join(out, f"SMA_{panel}_{s['sample']}.h5ad"); A.write(fp)
                print(f"  [OK] {os.path.basename(fp):40s} {rec['spots']:5d} spots x {rec['n_mz']} mz | "
                      f"swap={rec['swap']} fx={rec['flipx']} fy={rec['flipy']} | bioR={rec['bioR']:+.3f} null={rec['nullR']:+.3f} {rec['status']}")
    json.dump(recs, open(os.path.join(out,"sma_convert_qc.json"),"w"), indent=2)
    print("\n"+"="*64+"\n===== QC SUMMARY (paste this back) =====\n"+"="*64)
    ok=[r for r in recs if r.get("status")=="OK"]
    for panel in byp:
        pr=[r for r in recs if r.get("panel")==panel]
        print(f" {panel:4s}: {sum(1 for r in pr if r.get('status')=='OK')}/{len(pr)} OK, common m/z={len(common.get(panel,[]))}")
    print(f" total sections written OK : {len(ok)}")
    print(f" median transcriptome->metabolome coupling (bioR): "
          f"{np.median([r['bioR'] for r in ok]):+.3f}  vs null {np.median([r['nullR'] for r in ok]):+.3f}" if ok else " (none OK)")
    print(f" wrote .h5ad + sma_convert_qc.json to: {out}")
    print(" next: run  python metaspatial_all_in_one.py \"<your main data folder>\"")
    print("       (the folder that also holds the 7 DESIUM .h5ad; it will train DESIUM + each SMA panel)")

if __name__ == "__main__":
    main()
