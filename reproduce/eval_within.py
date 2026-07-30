"""Within-sample spatially-blocked 5-fold CV: can genes predict the metabolite map within a tissue?
Restricted to well-detected metabolites (>20% of spots). per-spot vs transport-aware."""
import anndata as ad, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
samples=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]

def norm_adj(coords,k=6):
    n=len(coords); d,idx=cKDTree(coords).query(coords,k=min(k+1,n))
    rows=np.repeat(np.arange(n),idx.shape[1]-1); cols=idx[:,1:].ravel()
    Adj=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); Adj=((Adj+Adj.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    dinv=diags((1/np.sqrt(np.asarray(Adj.sum(1)).ravel())).astype(np.float32)); return dinv@Adj@dinv

res={False:[],True:[]}; ndet=[]
for s in samples:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad")
    X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],dtype=np.float32)); coords=np.asarray(a.obsm['spatial'],dtype=float)
    det=(Y>0).mean(0)>0.2; Yd=Y[:,det]; ndet.append(int(det.sum()))
    Adj=norm_adj(coords); hvg=np.argsort(X.var(0))[-3000:]; Xh=X[:,hvg]
    blocks=KMeans(5,n_init=3,random_state=0).fit_predict(coords)
    for transport in (False,True):
        preds=np.zeros_like(Yd)
        for fb in range(5):
            trm=blocks!=fb; tem=blocks==fb
            if tem.sum()==0 or trm.sum()<50: continue
            pca=PCA(60,svd_solver='randomized',random_state=0).fit(Xh[trm])
            P=pca.transform(Xh).astype(np.float32)
            if transport: P=np.hstack([P, Adj@P, Adj@(Adj@P)])
            preds[tem]=Ridge(alpha=200.0).fit(P[trm], Yd[trm]).predict(P[tem])
        rs=[spearmanr(preds[:,m],Yd[:,m])[0] for m in range(Yd.shape[1]) if Yd[:,m].std()>1e-9]
        res[transport].append(np.nanmedian(rs))
    print(f"  {s:18s} detected metab={ndet[-1]:4d} | within-sample per-spot r={res[False][-1]:+.3f} | transport r={res[True][-1]:+.3f}", flush=True)

print("\n=== WITHIN-SAMPLE (spatially-blocked CV, detected metabolites) ===")
print(f"  per-spot (baseline)   : mean median Spearman = {np.mean(res[False]):+.3f}")
print(f"  transport-aware (ours): mean median Spearman = {np.mean(res[True]):+.3f}")
print(f"  transport gain: {np.mean(res[True])-np.mean(res[False]):+.3f} across {len(samples)} samples")
