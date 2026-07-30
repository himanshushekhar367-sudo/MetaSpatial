"""Model-ceiling test: Ridge(per-spot) vs Ridge(transport) vs MLP(transport), within-sample
spatial-block CV, on detected + top-1000-variance metabolites. median per-metabolite Spearman."""
import anndata as ad, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

agg={k:[] for k in ["ridge_ps","ridge_tr","mlp_tr"]}
for s in samples:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad")
    X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],dtype=np.float32)); coords=np.asarray(a.obsm['spatial'],dtype=float)
    det=(Y>0).mean(0)>0.2
    var=Y[:,det].var(0); topidx=np.where(det)[0][np.argsort(var)[-1000:]]     # top-1000 variance detected
    Yd=Y[:,topidx]
    Adj=norm_adj(coords); hvg=np.argsort(X.var(0))[-3000:]; Xh=X[:,hvg]
    blocks=KMeans(5,n_init=3,random_state=0).fit_predict(coords)
    P_ps=np.zeros((len(X),100),np.float32)  # filled per fold
    pred={k:np.zeros_like(Yd) for k in agg}
    for fb in range(5):
        trm=blocks!=fb; tem=blocks==fb
        if tem.sum()==0 or trm.sum()<50: continue
        pca=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[trm])
        P=pca.transform(Xh).astype(np.float32); Ptr=np.hstack([P,Adj@P,Adj@(Adj@P)])
        sc=StandardScaler().fit(Ptr[trm]); Ptr_s=sc.transform(Ptr)
        pred["ridge_ps"][tem]=Ridge(alpha=200).fit(P[trm],Yd[trm]).predict(P[tem])
        pred["ridge_tr"][tem]=Ridge(alpha=200).fit(Ptr[trm],Yd[trm]).predict(Ptr[tem])
        mlp=MLPRegressor(hidden_layer_sizes=(128,),alpha=1e-3,max_iter=200,early_stopping=True,
                         n_iter_no_change=8,random_state=0).fit(Ptr_s[trm],Yd[trm])
        pred["mlp_tr"][tem]=mlp.predict(Ptr_s[tem])
    for k in agg:
        rs=[spearmanr(pred[k][:,m],Yd[:,m])[0] for m in range(Yd.shape[1]) if Yd[:,m].std()>1e-9]
        agg[k].append(np.nanmedian(rs))
    print(f"  {s:18s} ridge-ps={agg['ridge_ps'][-1]:+.3f} ridge-tr={agg['ridge_tr'][-1]:+.3f} MLP-tr={agg['mlp_tr'][-1]:+.3f}", flush=True)

print("\n=== MODEL CEILING (within-sample, top-1000 metabolites, mean median Spearman) ===")
for k,name in [("ridge_ps","Ridge per-spot"),("ridge_tr","Ridge transport"),("mlp_tr","MLP transport (nonlinear)")]:
    print(f"  {name:26s}: {np.mean(agg[k]):+.3f}")
