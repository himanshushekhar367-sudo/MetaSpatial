"""Detection-matched LOSO: for each held-out sample, restrict scoring to metabolites
well-detected IN THAT SAMPLE (same det>0.2 rule as within-sample CV) so the LOSO
median r is directly comparable to the within-sample CV median r."""
import anndata as ad, numpy as np, json, gc, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge
from scipy.stats import spearmanr; from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
SP="/home/claude/spatmet"; samples=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
A={s:ad.read_h5ad(f"{DATA}/{s}.h5ad") for s in samples}
genes=sorted(set.intersection(*[set(v.var_names) for v in A.values()]))
coordsD={s:np.asarray(A[s].obsm['spatial'],float) for s in samples}
rawMSI={s:np.asarray(A[s].uns['msi'],np.float32) for s in samples}
YlD={s:np.log1p(rawMSI[s]) for s in samples}
detD={s:(rawMSI[s]>0).mean(0)>0.2 for s in samples}   # per-sample well-detected mask
def gm(a):
    v=a[:,genes].layers['log1p']; return (np.asarray(v.todense()) if hasattr(v,'todense') else np.asarray(v)).astype(np.float32)
s1=np.zeros(len(genes)); s2=np.zeros(len(genes)); N=0
for s in samples:
    M=gm(A[s]); s1+=M.sum(0); s2+=(M*M).sum(0); N+=M.shape[0]; del M; gc.collect()
var=s2/N-(s1/N)**2; hv=np.argsort(var)[-2000:]
Xg={s:gm(A[s])[:,hv] for s in samples}; adjs={s:nadj(coordsD[s]) for s in samples}; gc.collect()
loso_all={}; loso_det={}
for held in samples:
    tr=[s for s in samples if s!=held]
    pca=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s] for s in tr]))
    def F(s):
        P=pca.transform(Xg[s]).astype(np.float32); return np.hstack([P,adjs[s]@P,adjs[s]@(adjs[s]@P)])
    reg=Ridge(alpha=200).fit(np.vstack([F(s) for s in tr]),np.vstack([YlD[s] for s in tr]))
    pr=reg.predict(F(held)); Yh=YlD[held]; det=detD[held]
    rall=[spearmanr(pr[:,j],Yh[:,j])[0] for j in range(Yh.shape[1]) if Yh[:,j].std()>1e-9]
    rdet=[spearmanr(pr[:,j],Yh[:,j])[0] for j in range(Yh.shape[1]) if det[j] and Yh[:,j].std()>1e-9]
    loso_all[held]=float(np.nanmedian(rall)); loso_det[held]=float(np.nanmedian(rdet))
    print(f"LOSO {held:18s} all-mz r={loso_all[held]:+.3f}  |  detected(n={int(det.sum())}) r={loso_det[held]:+.3f}",flush=True)
bc=[loso_det[s] for s in samples if s.startswith('BC')]; lc=[loso_det[s] for s in samples if s.startswith('LC')]
al=[loso_det[s] for s in samples]
out=dict(loso_all=loso_all,loso_det=loso_det,
    loso_det_overall=[float(np.mean(al)),float(np.std(al))],
    loso_det_breast=[float(np.mean(bc)),float(np.std(bc))],
    loso_det_lung=[float(np.mean(lc)),float(np.std(lc))])
json.dump(out,open(f"{SP}/loso_matched.json","w"),indent=2)
print(f"\nDETECTION-MATCHED LOSO median r: overall {np.mean(al):.3f}+-{np.std(al):.3f} | breast {np.mean(bc):.3f}+-{np.std(bc):.3f} | lung {np.mean(lc):.3f}+-{np.std(lc):.3f}")
