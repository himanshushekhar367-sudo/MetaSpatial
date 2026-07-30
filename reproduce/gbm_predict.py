#!/usr/bin/env python3
# =====================================================================================
# GBM MetaSpatial — STEP 1 of 2 : train on DESIUM, predict metabolome for BOTH GBM
# sections (Parent Visium + CytAssist), export CytAssist antibody-protein.
# Run this once (a few minutes), then run gbm_analyze.py.
#
#   python gbm_predict.py
#
# Needs: numpy pandas scanpy anndata scikit-learn scipy   (same env as MetaSpatial)
# =====================================================================================
import os, gc, glob, tarfile, warnings
import numpy as np, anndata as ad, scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye
warnings.filterwarnings("ignore")

# ============ SET THIS TO YOUR DATA FOLDER ============
ROOT = r"C:\Users\pc\Desktop\spatial metabolism"
# =====================================================
DES  = os.path.join(ROOT, "DESIUM", "Correlation_NMF_analysis", "data")
OUT  = os.path.join(ROOT, "gbm_metaspatial"); os.makedirs(OUT, exist_ok=True)
DSAMP = ["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
VIS_H5 = "Parent_Visium_Human_Glioblastoma_filtered_feature_bc_matrix.h5"
CYT_H5 = "CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_filtered_feature_bc_matrix.h5"

def nadj(coords, k=6):
    n=len(coords); d,idx=cKDTree(coords).query(coords,k=min(k+1,n))
    rows=np.repeat(np.arange(n),idx.shape[1]-1); cols=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    dinv=diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)); return dinv@A@dinv

# ---- ingest: attach spatial coords from the *_spatial.tar.gz (or an already-extracted folder) ----
def spatial_positions(prefix):
    cands = glob.glob(os.path.join(ROOT,"**",f"{prefix}*",'**','tissue_positions*.csv'), recursive=True)
    if not cands:
        tgz=os.path.join(ROOT,f"{prefix}_spatial.tar.gz")
        if os.path.exists(tgz):
            dest=os.path.join(OUT,f"{prefix}_spatial")
            if not os.path.isdir(dest):
                with tarfile.open(tgz) as t: t.extractall(dest)
            cands=glob.glob(os.path.join(dest,'**','tissue_positions*.csv'),recursive=True)
    if not cands: return None
    import pandas as pd; f=cands[0]
    if "list" in os.path.basename(f):
        df=pd.read_csv(f,header=None,names=["barcode","in_tissue","array_row","array_col","pxl_row","pxl_col"])
    else:
        df=pd.read_csv(f); df=df.rename(columns={df.columns[0]:"barcode"})
    return df.set_index("barcode")

def build(h5, prefix, gex_only):
    a=sc.read_10x_h5(os.path.join(ROOT,h5), gex_only=gex_only); a.var_names_make_unique()
    pos=spatial_positions(prefix)
    if pos is None: raise SystemExit(f"Could not find tissue_positions for {prefix} (extract {prefix}_spatial.tar.gz)")
    common=a.obs_names.intersection(pos.index); a=a[common].copy(); p=pos.loc[a.obs_names]
    xc=[c for c in p.columns if 'pxl_col' in c or 'imagecol' in c.lower()]
    yc=[c for c in p.columns if 'pxl_row' in c or 'imagerow' in c.lower()]
    xcol=xc[0] if xc else p.columns[-1]; ycol=yc[0] if yc else p.columns[-2]
    a.obsm['spatial']=np.c_[p[xcol].astype(float).values, p[ycol].astype(float).values]
    print(f"  {prefix}: {a.n_obs} spots x {a.n_vars} feats", flush=True); return a

print("ingesting GBM sections ...", flush=True)
gv  = build(VIS_H5, "Parent_Visium_Human_Glioblastoma", True)
gcy = build(CYT_H5, "CytAssist_FFPE_Protein_Expression_Human_Glioblastoma", False)

# ---- train on DESIUM (streamed, 3000 HVG) ----
def dsec(s, genes):
    a=ad.read_h5ad(os.path.join(DES,f"{s}.h5ad"))
    v=a[:,genes].layers['log1p']; X=np.asarray(v.todense() if hasattr(v,'todense') else v,dtype=np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],float)).astype(np.float32); C=np.asarray(a.obsm['spatial'],float)
    mz=np.asarray(a.uns['mz_features'],float); del a; gc.collect(); return X,Y,C,mz
def vnames(path):
    a=ad.read_h5ad(path,backed='r'); v=list(a.var_names)
    try:a.file.close()
    except:pass
    return v

gene_sets=[set(vnames(os.path.join(DES,f"{s}.h5ad"))) for s in DSAMP]
gc_rna_names=[g for g,t in zip(list(gcy.var_names), gcy.var['feature_types'].astype(str)) if t=='Gene Expression']
shared=sorted(set.intersection(*(gene_sets+[set(gv.var_names),set(gc_rna_names)])))
print("shared genes:",len(shared),flush=True)

n=0; s1=np.zeros(len(shared)); s2=np.zeros(len(shared)); mz=None
for s in DSAMP:
    X,_,_,mz=dsec(s,shared); n+=X.shape[0]; s1+=X.sum(0); s2+=(X.astype(np.float64)**2).sum(0); del X; gc.collect()
hvg=np.argsort(s2/n-(s1/n)**2)[-3000:]; sharedH=[shared[i] for i in hvg]
Xs=[];Ys=[];Cs=[]
for s in DSAMP:
    X,Y,C,_=dsec(s,sharedH); Xs.append(X);Ys.append(Y);Cs.append(C)
Xtr=np.vstack(Xs); Ytr=np.vstack(Ys)
pca=PCA(100,svd_solver='randomized',random_state=0).fit(Xtr)
Vt=pca.components_.astype(np.float32); pmean=pca.mean_.astype(np.float32)
def tfeat(Xd,coords):
    P=(Xd@Vt.T-pmean@Vt.T).astype(np.float32); A=nadj(coords); return np.hstack([P,A@P,A@(A@P)])
Ftr=np.vstack([tfeat(Xs[i],Cs[i]) for i in range(len(DSAMP))])
mu=Ytr.mean(0); sd=Ytr.std(0)+1e-8
reg=Ridge(alpha=200).fit(Ftr,(Ytr-mu)/sd)
print(f"trained on {Xtr.shape[0]} DESIUM spots -> {Ytr.shape[1]} metabolites",flush=True)
del Xs,Xtr,Ftr; gc.collect()

def predict(a):
    gmap={g:i for i,g in enumerate(a.var_names)}; gi=np.array([gmap[g] for g in sharedH])
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X); Xs2=X[:,gi].astype(np.float32)
    if X.max()>50:
        tot=np.asarray(X.sum(1)).ravel()+1e-8
        Xs2=Xs2.multiply(1.0/tot[:,None]).multiply(1e4).tocsr(); Xs2.data=np.log1p(Xs2.data)
    P=(np.asarray(Xs2@Vt.T)-pmean@Vt.T).astype(np.float32)
    A=nadj(np.asarray(a.obsm['spatial'],float)); F=np.hstack([P,A@P,A@(A@P)])
    return (reg.predict(F)*sd+mu).astype(np.float32)

pv=predict(gv); np.savez_compressed(os.path.join(OUT,"gbm2_visium_pred.npz"),pred=pv,mz=mz,coords=np.asarray(gv.obsm['spatial']))
rna=gcy[:,gcy.var['feature_types'].astype(str)=='Gene Expression'].copy()
pc=predict(rna); np.savez_compressed(os.path.join(OUT,"gbm2_cyt_pred.npz"),pred=pc,mz=mz,coords=np.asarray(gcy.obsm['spatial']))
Pm=gcy[:,gcy.var['feature_types'].astype(str)=='Antibody Capture']
Pv=np.asarray(Pm.X.todense() if hasattr(Pm.X,'todense') else Pm.X,dtype=float)
np.savez_compressed(os.path.join(OUT,"gbm_cyt_protein.npz"),protein=Pv,names=np.array(list(Pm.var_names)),coords=np.asarray(gcy.obsm['spatial']))
print(f"Visium pred {pv.shape} | CytAssist pred {pc.shape} | protein {Pv.shape}",flush=True)
print("DONE -> saved to",OUT,"; now run gbm_analyze.py",flush=True)
