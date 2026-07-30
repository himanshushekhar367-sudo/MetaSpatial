"""Can we beat the transport-ridge ceiling (0.304)? Test accuracy levers, within-sample spatial CV,
top-1000 metabolites. baseline vs +ensemble(KNN) vs +more PCs vs +deeper diffusion."""
import anndata as ad, numpy as np, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from scipy.stats import spearmanr, rankdata; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
samples=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); rows=np.repeat(np.arange(n),idx.shape[1]-1); cols=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    dinv=diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)); return dinv@A@dinv
def z(M): return (M-M.mean(0))/(M.std(0)+1e-8)
methods=["base transport-ridge","+ensemble (KNN)","+200 PCs","+deeper diffusion"]
agg={m:[] for m in methods}
for s in samples:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad")
    X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],dtype=np.float32)); coords=np.asarray(a.obsm['spatial'],dtype=float)
    det=(Y>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-1000:]]; Yd=Y[:,top]
    Adj=nadj(coords); hvg=np.argsort(X.var(0))[-3000:]; Xh=X[:,hvg]; blocks=KMeans(5,n_init=3,random_state=0).fit_predict(coords)
    pred={m:np.zeros_like(Yd) for m in methods}
    for fb in range(5):
        trm=blocks!=fb; tem=blocks==fb
        if tem.sum()==0 or trm.sum()<50: continue
        pca=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[trm]); P=pca.transform(Xh).astype(np.float32)
        T2=np.hstack([P,Adj@P,Adj@(Adj@P)])
        rid=Ridge(alpha=200).fit(T2[trm],Yd[trm]).predict(T2[tem]); pred["base transport-ridge"][tem]=rid
        tree=cKDTree(P[trm]); _,nn=tree.query(P[tem],k=15); knn=Yd[trm][nn].mean(1)
        pred["+ensemble (KNN)"][tem]=z(rid)+z(knn)
        pca2=PCA(200,svd_solver='randomized',random_state=0).fit(Xh[trm]); P2=pca2.transform(Xh).astype(np.float32)
        T2b=np.hstack([P2,Adj@P2,Adj@(Adj@P2)]); pred["+200 PCs"][tem]=Ridge(alpha=300).fit(T2b[trm],Yd[trm]).predict(T2b[tem])
        P1=Adj@P; P3=Adj@(Adj@(Adj@P)); T3=np.hstack([P,P1,Adj@P1,P3]); pred["+deeper diffusion"][tem]=Ridge(alpha=200).fit(T3[trm],Yd[trm]).predict(T3[tem])
    for m in methods:
        rs=[spearmanr(pred[m][:,j],Yd[:,j])[0] for j in range(Yd.shape[1]) if Yd[:,j].std()>1e-9]
        agg[m].append(np.nanmedian(rs))
    print(f"  {s:18s} "+" ".join(f"{m.split()[0][:6]}={agg[m][-1]:+.3f}" for m in methods), flush=True)
print("\n=== ACCURACY LEVERS (within-sample, top-1000 metab, mean median Spearman) ===")
for m in methods: print(f"  {m:24s}: {np.mean(agg[m]):+.3f}")
