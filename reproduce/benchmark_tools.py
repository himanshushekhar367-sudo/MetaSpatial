"""Head-to-head: MetaSpatial (transport-ridge) vs the ACTUAL competing methods, within-sample
spatial 5-fold CV, top-1000 well-measured metabolites, mean median Spearman over 7 DESIUM samples.
Baselines: per-spot ridge; joint-NMF cross-modal (DESIUM/MANTIS school); single-best-gene correlation."""
import anndata as ad, numpy as np, warnings
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
samples=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
methods=["MetaSpatial (ours)","per-spot ridge","joint-NMF (DESIUM/MANTIS)","single-gene corr"]
agg={m:[] for m in methods}
for s in samples:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad")
    X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],dtype=np.float32)); coords=np.asarray(a.obsm['spatial'],dtype=float)
    det=(Y>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-1000:]]; Yd=Y[:,top]
    hvg=np.argsort(X.var(0))[-3000:]; Xh=X[:,hvg]; Adj=nadj(coords); blk=KMeans(5,n_init=3,random_state=0).fit_predict(coords)
    pred={m:np.zeros_like(Yd) for m in methods}
    for fb in range(5):
        tr=blk!=fb; te=blk==fb
        if te.sum()==0 or tr.sum()<50: continue
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32)
        T=np.hstack([P,Adj@P,Adj@(Adj@P)])
        pred["MetaSpatial (ours)"][te]=Ridge(alpha=200).fit(T[tr],Yd[tr]).predict(T[te])
        pred["per-spot ridge"][te]=Ridge(alpha=200).fit(P[tr],Yd[tr]).predict(P[te])
        # joint-NMF cross-modal (shared spot factors; predict metab from genes)
        gs=Xh/ (Xh[tr].mean()+1e-8); ms=Yd/(Yd[tr].mean()+1e-8)
        Xcat=np.hstack([gs[tr],ms[tr]]).clip(0,None)
        nmf=NMF(n_components=20,init='nndsvda',max_iter=250,random_state=0).fit(Xcat)
        H=nmf.components_; Hg=H[:,:gs.shape[1]]; Hm=H[:,gs.shape[1]:]
        Wte=gs[te]@np.linalg.pinv(Hg); pred["joint-NMF (DESIUM/MANTIS)"][te]=(Wte@Hm)*(Yd[tr].mean()+1e-8)
        # single-best-gene correlation (train), predict = that gene
        Xz=(Xh[tr]-Xh[tr].mean(0))/(Xh[tr].std(0)+1e-8); Yz=(Yd[tr]-Yd[tr].mean(0))/(Yd[tr].std(0)+1e-8)
        C=(Xz.T@Yz)/len(Yz); best=np.abs(C).argmax(0); pred["single-gene corr"][te]=Xh[te][:,best]
    for m in methods:
        rs=[spearmanr(pred[m][:,j],Yd[:,j])[0] for j in range(Yd.shape[1]) if Yd[:,j].std()>1e-9]
        agg[m].append(np.nanmedian(rs))
    print(f"  {s:18s} "+" | ".join(f"{m.split()[0][:9]}={agg[m][-1]:+.3f}" for m in methods), flush=True)
print("\n=== TOOL BENCHMARK (within-sample spatial CV, top-1000 metab, mean median Spearman) ===")
for m in methods: print(f"  {m:28s}: {np.mean(agg[m]):+.3f}")
