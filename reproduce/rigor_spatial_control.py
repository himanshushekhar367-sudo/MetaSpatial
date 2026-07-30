"""Spatial-structure destruction control. Keep the true train/test split and the metabolite values
fixed; only destroy the GRAPH (permuted coordinates / random edges) or the gene->metabolite PAIRING.
If the transport prior derives value from real spatial organization, its gain over per-spot ridge
collapses when the graph is scrambled — while total signal collapses only when the pairing is destroyed."""
import anndata as ad, numpy as np, json, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore'); rng=np.random.RandomState(0)
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"; SP="/home/claude/spatmet"
secs=["BC_525","BC_823","LC_091"]; NMET=1000
def normalize(A):
    A=((A+A.T)>0).astype(np.float32)+eye(A.shape[0],dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
def knn_adj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    return csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n))
def rand_adj(n,nedges):
    r=rng.randint(0,n,nedges); c=rng.randint(0,n,nedges)
    return csr_matrix((np.ones(len(r)),(r,c)),shape=(n,n))
def medS(P,Y,te): return float(np.nanmedian([spearmanr(P[te,j],Y[te,j])[0] for j in range(Y.shape[1]) if Y[te,j].std()>1e-9]))
def load(s):
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Yr=np.asarray(a.uns['msi'],np.float32); Y=np.log1p(Yr); co=np.asarray(a.obsm['spatial'],float)
    det=(Yr>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-NMET:]]; hv=np.argsort(X.var(0))[-3000:]
    return X[:,hv],Y[:,top],co
conds=["intact graph (transport)","per-spot ridge (no graph)","permuted coordinates","random graph edges","pairing shuffled (neg. ctrl)"]
res={c:[] for c in conds}
for s in secs:
    Xh,Yd,co=load(s); n=len(co); blk=KMeans(3,n_init=3,random_state=0).fit_predict(co)   # true spatial blocks (fixed)
    Areal=normalize(knn_adj(co)); nE=int(knn_adj(co).nnz)
    perm=rng.permutation(n); Aperm=normalize(knn_adj(co)[perm][:,perm]); Arand=normalize(rand_adj(n,nE))
    for fb in range(3):
        tr=np.where(blk!=fb)[0]; te=blk==fb
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32)
        def tf(A): return np.hstack([P,A@P,A@(A@P)])
        res["intact graph (transport)"].append(medS(Ridge(alpha=200).fit(tf(Areal)[tr],Yd[tr]).predict(tf(Areal)),Yd,te))
        res["per-spot ridge (no graph)"].append(medS(Ridge(alpha=200).fit(P[tr],Yd[tr]).predict(P),Yd,te))
        res["permuted coordinates"].append(medS(Ridge(alpha=200).fit(tf(Aperm)[tr],Yd[tr]).predict(tf(Aperm)),Yd,te))
        res["random graph edges"].append(medS(Ridge(alpha=200).fit(tf(Arand)[tr],Yd[tr]).predict(tf(Arand)),Yd,te))
        Ysh=Yd[tr][rng.permutation(len(tr))]                                              # destroy gene->metab pairing
        res["pairing shuffled (neg. ctrl)"].append(medS(Ridge(alpha=200).fit(tf(Areal)[tr],Ysh).predict(tf(Areal)),Yd,te))
    print(f"  {s} done",flush=True)
out={c:[float(np.mean(res[c])),float(np.std(res[c]))] for c in conds}
base=out["per-spot ridge (no graph)"][0]
print("\n=== SPATIAL-STRUCTURE DESTRUCTION (median Spearman, top-%d, spatial 3-fold) ==="%NMET)
for c in conds: print(f"  {c:34s}: {out[c][0]:+.3f} ± {out[c][1]:.3f}   (transport gain vs per-spot: {out[c][0]-base:+.3f})")
json.dump(dict(conditions=out,per_spot=base),open(f"{SP}/rigor_spatial_control.json","w"),indent=2)
print("saved rigor_spatial_control.json")
