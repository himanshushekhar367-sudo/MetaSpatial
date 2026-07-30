"""Cross-cancer application (memory-lean): train MetaSpatial on DESIUM, apply to HCC + GBM,
summarise HCC tumor-vs-normal predicted-metabolite zonation. HCC kept SPARSE through PCA."""
import anndata as ad, numpy as np, pandas as pd, scipy.io, gzip, glob, os, gc, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix, diags, eye
warnings.filterwarnings('ignore')
DES="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
UP="/mnt/user-data/uploads/spatial metabolism"; OUT="/home/claude/spatmet/xcancer"; os.makedirs(OUT,exist_ok=True)
dsamp=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]

def nadj_grouped(coords, groups, k=6):
    """block-diagonal symmetric-normalized kNN adjacency, neighbors only within each group."""
    n=len(coords); rows=[]; cols=[]
    for g in np.unique(groups):
        gi=np.where(groups==g)[0];
        if len(gi)<3: continue
        d,idx=cKDTree(coords[gi]).query(coords[gi],k=min(k+1,len(gi)))
        for a in range(len(gi)):
            for b in idx[a,1:]:
                rows.append(gi[a]); cols.append(gi[b])
    A=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A

def read_lines(p): return [l.strip() for l in open(p)]

# ---- DESIUM training ----
D={s:ad.read_h5ad(f"{DES}/{s}.h5ad") for s in dsamp}
mz=np.asarray(D[dsamp[0]].uns['mz_features'],float)
gene_sets=[set(a.var_names) for a in D.values()]
gene_sets += [set(read_lines(f"{UP}/{s}-expr_features.txt")) for s in sorted({os.path.basename(f).split('-expr')[0] for f in glob.glob(f'{UP}/HCC-*-expr_counts.mtx.gz')})]
GBM=ad.read_h5ad("/home/claude/spatmet/gbm_visium.h5ad") if os.path.exists("/home/claude/spatmet/gbm_visium.h5ad") else None
if GBM is not None: gene_sets.append(set(GBM.var_names))
shared=sorted(set.intersection(*gene_sets)); print("shared genes:",len(shared),flush=True)

def desi_dense(a):
    v=a[:,shared].layers['log1p']; return np.asarray(v.todense() if hasattr(v,'todense') else v,dtype=np.float32)
Xtr=np.vstack([desi_dense(D[s]) for s in dsamp])
Ytr=np.vstack([np.log1p(np.asarray(D[s].uns['msi'],float)) for s in dsamp]).astype(np.float32)
pca=PCA(100,svd_solver='randomized',random_state=0).fit(Xtr)
Vt=pca.components_.astype(np.float32); pmean=pca.mean_.astype(np.float32)
def tfeat_dense(Xd,coords,groups):
    P=(Xd@Vt.T - pmean@Vt.T).astype(np.float32); A=nadj_grouped(coords,groups); return np.hstack([P,A@P,A@(A@P)])
Ftr=np.vstack([tfeat_dense(desi_dense(D[s]),np.asarray(D[s].obsm['spatial'],float),
              np.zeros(D[s].n_obs)) for s in dsamp])
mu=Ytr.mean(0); sd=Ytr.std(0)+1e-8
reg=Ridge(alpha=200).fit(Ftr,(Ytr-mu)/sd)
print(f"trained on {Xtr.shape[0]} DESIUM spots -> {Ytr.shape[1]} metabolites",flush=True)
del Xtr,Ftr,D; gc.collect()

gmapS={g:i for i,g in enumerate(shared)}
def sparse_proj(Msub_spots_by_shared):
    X=Msub_spots_by_shared.tocsr().astype(np.float32); X.data=np.log1p(X.data)   # CP10k already? apply below
    return X
# ---- HCC ----
zon=[]
for sid in sorted({os.path.basename(f).split('-expr')[0] for f in glob.glob(f'{UP}/HCC-*-expr_counts.mtx.gz')}):
    genes=read_lines(f"{UP}/{sid}-expr_features.txt"); gmap={g:i for i,g in enumerate(genes)}
    gi=np.array([gmap[g] for g in shared])
    M=scipy.io.mmread(gzip.open(f"{UP}/{sid}-expr_counts.mtx.gz")).tocsc()   # genes x spots
    tot=np.asarray(M.sum(0)).ravel()+1e-8
    Msub=M[gi,:].T.tocsr().astype(np.float32)                                # spots x shared (raw)
    Msub=Msub.multiply(1.0/tot[:,None]).multiply(1e4).tocsr(); Msub.data=np.log1p(Msub.data)   # CP10k-log1p
    P=(np.asarray(Msub@Vt.T) - pmean@Vt.T).astype(np.float32)
    meta=pd.read_csv(f"{UP}/{sid}-expr_meta.csv",index_col=0)
    coords=meta[['imagerow','imagecol']].values.astype(float)
    reg_lbl=meta['sample.ident'].astype(str).values
    suff=np.array([r.replace(sid,'').lstrip('-') for r in reg_lbl])
    A=nadj_grouped(coords,suff); F=np.hstack([P,A@P,A@(A@P)])
    pred=reg.predict(F)*sd+mu
    np.savez_compressed(f"{OUT}/{sid}_pred.npz", pred=pred.astype(np.float32), mz=mz)
    if 'N' in suff and 'T' in suff:
        lfc=pred[suff=='T'].mean(0)-pred[suff=='N'].mean(0)
        zon.append(pd.Series(lfc,index=np.round(mz,3),name=sid))
        print(f"  {sid}: {M.shape[1]} spots, regions {sorted(set(suff))} -> N->T zonation ok",flush=True)
    else:
        print(f"  {sid}: {M.shape[1]} spots, regions {sorted(set(suff))} (no N/T)",flush=True)
    del M,Msub,P,F,pred; gc.collect()
if zon:
    Z=pd.concat(zon,axis=1); Z.to_csv(f"{OUT}/hcc_zonation_TvsN_log.csv"); print("HCC zonation saved:",Z.shape,flush=True)
# ---- GBM ----
if GBM is not None:
    v=GBM[:,shared].X; Xg=np.asarray(v.todense() if hasattr(v,'todense') else v,dtype=np.float32)
    if Xg.max()>50:
        Xg=Xg/(Xg.sum(1,keepdims=True)+1e-8)*1e4; Xg=np.log1p(Xg)
    predg=reg.predict(tfeat_dense(Xg,np.asarray(GBM.obsm['spatial'],float),np.zeros(GBM.n_obs)))*sd+mu
    np.savez_compressed(f"{OUT}/GBM_pred.npz", pred=predg.astype(np.float32), mz=mz, coords=np.asarray(GBM.obsm['spatial']))
    print("GBM predicted:",predg.shape,flush=True)
print("DONE.",flush=True)
