"""Does simple per-section harmonization close the LOSO gap? Empirical test of the roadmap claim.
Compares LOSO under (a) baseline features, (b) per-section z-scored transport features,
(c) per-section-centred PCA (remove section centroid). Detection-matched median Spearman."""
import anndata as ad, numpy as np, json, gc, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
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
YlD={s:np.log1p(rawMSI[s]) for s in samples}; detD={s:(rawMSI[s]>0).mean(0)>0.2 for s in samples}
def gm(a):
    v=a[:,genes].layers['log1p']; return (np.asarray(v.todense()) if hasattr(v,'todense') else np.asarray(v)).astype(np.float32)
s1=np.zeros(len(genes)); s2=np.zeros(len(genes)); N=0
for s in samples:
    M=gm(A[s]); s1+=M.sum(0); s2+=(M*M).sum(0); N+=M.shape[0]; del M; gc.collect()
hv=np.argsort(s2/N-(s1/N)**2)[-2000:]
Xg={s:gm(A[s])[:,hv] for s in samples}; adjs={s:nadj(coordsD[s]) for s in samples}; gc.collect()
def feats(Pd,s,mode):
    P=Pd[s]
    if mode=='center': P=P-P.mean(0)
    T=np.hstack([P,adjs[s]@P,adjs[s]@(adjs[s]@P)])
    if mode=='zscore': T=(T-T.mean(0))/(T.std(0)+1e-8)
    return T
def run(mode):
    out={}
    for held in samples:
        tr=[s for s in samples if s!=held]
        pca=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s] for s in tr]))
        Pd={s:pca.transform(Xg[s]).astype(np.float32) for s in samples}
        reg=Ridge(alpha=200).fit(np.vstack([feats(Pd,s,mode) for s in tr]),np.vstack([YlD[s] for s in tr]))
        pr=reg.predict(feats(Pd,held,mode)); Yh=YlD[held]; det=detD[held]
        out[held]=float(np.nanmedian([spearmanr(pr[:,j],Yh[:,j])[0] for j in range(Yh.shape[1]) if det[j] and Yh[:,j].std()>1e-9]))
    return out
res={}
for mode in ['baseline','center','zscore']:
    o=run(mode); res[mode]=o; v=list(o.values())
    print(f"{mode:9s} LOSO median r overall {np.mean(v):+.3f}+-{np.std(v):.3f} | breast {np.mean([o[s] for s in samples if s[:2]=='BC']):+.3f} | lung {np.mean([o[s] for s in samples if s[:2]=='LC']):+.3f}",flush=True)
json.dump(res,open(f"{SP}/loso_harmonize.json","w"),indent=2)
print("saved loso_harmonize.json")
