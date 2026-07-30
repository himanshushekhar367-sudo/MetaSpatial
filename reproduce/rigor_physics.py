"""Physics-informed & structured extensions (closed-form, exact):
(1) transport loss as a spatial-Laplacian penalty on predictions -> enforces spatial consistency;
(2) structured multi-task output via reduced-rank regression (couples metabolites vs independent);
(3) non-negativity constraint; (4) representation (PCA/NMF/AE/random-proj); (5) enzyme-gene restriction."""
import anndata as ad, numpy as np, json, warnings
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import Ridge; from sklearn.neural_network import MLPRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"; SP="/home/claude/spatmet"
secs=["BC_525","LC_091"]; NMET=200
kegg_genes=set()
for ln in open(f"{SP}/genesets/scMetab_KEGG.gmt"):
    p=ln.rstrip("\n").split("\t"); kegg_genes.update(g for g in p[2:] if g)
def adj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
def medS(pred,Y,te): return float(np.nanmedian([spearmanr(pred[te,j],Y[te,j])[0] for j in range(Y.shape[1]) if Y[te,j].std()>1e-9]))
def load(s):
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); vn=list(a.var_names); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Yr=np.asarray(a.uns['msi'],np.float32); Y=np.log1p(Yr); co=np.asarray(a.obsm['spatial'],float)
    det=(Yr>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-NMET:]]
    hv=np.argsort(X.var(0))[-3000:]; enz=np.array([i for i in range(X.shape[1]) if vn[i] in kegg_genes])
    return X,hv,enz,Y[:,top],co
DAT={s:load(s) for s in secs}
def transport(P,A): return np.hstack([P,A@P,A@(A@P)])

# ---- (1) physics-informed spatial-Laplacian penalty (closed form) ----
def fit_phys(F,Y,L,alpha,lam):
    FtF=F.T@F; n=F.shape[1]
    if lam>0:
        M=L@F; FtF=FtF+lam*(M.T@M)
    W=np.linalg.solve(FtF+alpha*np.eye(n),F.T@Y); return W
phys={}
for lam in [0.0,1.0,10.0]:
    acc=[]; rough=[]
    for s in secs:
        X,hv,enz,Yd,co=DAT[s]; A=adj(co); L=eye(len(co),dtype=np.float32)-A; blk=KMeans(3,n_init=3,random_state=0).fit_predict(co)
        for fb in range(3):
            tr=np.where(blk!=fb)[0]; te=np.where(blk==fb)[0]
            P=PCA(100,svd_solver='randomized',random_state=0).fit(X[tr][:,hv]).transform(X[:,hv]).astype(np.float32); F=transport(P,A)
            Atr=adj(co[tr]); Ltr=(eye(len(tr),dtype=np.float32)-Atr).toarray().astype(np.float32)
            W=fit_phys(F[tr],Yd[tr],Ltr,200.0,lam); pr=F@W
            acc.append(medS(pr,Yd,te))
            prt=pr[te]; Lte=(eye(len(te),dtype=np.float32)-adj(co[te])).astype(np.float32)
            rough.append(float(np.sum((Lte@prt)**2)/(np.sum(prt**2)+1e-8)))
    phys[lam]=dict(acc=[float(np.mean(acc)),float(np.std(acc))],roughness=float(np.mean(rough)))
    print(f"physics lam={lam:4.0f}: acc={phys[lam]['acc'][0]:+.3f}  pred-roughness={phys[lam]['roughness']:.3f}",flush=True)

# ---- (2) structured output: reduced-rank regression (couples metabolites) vs independent ridge ----
struct={}
for tag,rank in [("independent ridge",None),("RRR rank-20",20),("RRR rank-50",50)]:
    vv=[]
    for s in secs:
        X,hv,enz,Yd,co=DAT[s]; A=adj(co); blk=KMeans(3,n_init=3,random_state=0).fit_predict(co)
        for fb in range(3):
            tr=np.where(blk!=fb)[0]; te=np.where(blk==fb)[0]
            P=PCA(100,svd_solver='randomized',random_state=0).fit(X[tr][:,hv]).transform(X[:,hv]).astype(np.float32); F=transport(P,A)
            if rank is None: pr=Ridge(alpha=200).fit(F[tr],Yd[tr]).predict(F)
            else:
                W=np.linalg.solve(F[tr].T@F[tr]+200*np.eye(F.shape[1]),F[tr].T@Yd[tr]); Yh=F[tr]@W
                Uc=Yh-Yh.mean(0); _,_,Vt=np.linalg.svd(Uc,full_matrices=False); Vr=Vt[:rank].T
                pr=(F@W-Yd[tr].mean(0))@ (Vr@Vr.T)+Yd[tr].mean(0)
            vv.append(medS(pr,Yd,te))
    struct[tag]=[float(np.mean(vv)),float(np.std(vv))]; print(f"structured {tag:18s}: {struct[tag][0]:+.3f} ± {struct[tag][1]:.3f}",flush=True)

# ---- (3) non-negativity, (4) representation, (5) enzyme restriction ----
rep={};
for tag in ["PCA","NMF","autoencoder","random-proj","enzyme-only-PCA","+non-neg(clip)"]:
    vv=[]
    for s in secs:
        X,hv,enz,Yd,co=DAT[s]; A=adj(co); blk=KMeans(3,n_init=3,random_state=0).fit_predict(co)
        for fb in range(3):
            tr=np.where(blk!=fb)[0]; te=np.where(blk==fb)[0]; Xh=X[:,hv]
            if tag=="PCA" or tag=="+non-neg(clip)": E=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32)
            elif tag=="NMF": mo=NMF(100,init='nndsvda',max_iter=200,random_state=0).fit(np.clip(Xh[tr],0,None)); E=mo.transform(np.clip(Xh,0,None)).astype(np.float32)
            elif tag=="autoencoder":
                aem=MLPRegressor(hidden_layer_sizes=(100,),alpha=1e-3,max_iter=80,random_state=0).fit(Xh[tr],Xh[tr])
                W1=aem.coefs_[0]; b1=aem.intercepts_[0]; E=np.maximum(Xh@W1+b1,0).astype(np.float32)   # hidden activations
            elif tag=="random-proj": E=GaussianRandomProjection(100,random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32)
            elif tag=="enzyme-only-PCA": Xe=X[:,enz]; E=PCA(min(100,Xe.shape[1]-1),svd_solver='randomized',random_state=0).fit(Xe[tr]).transform(Xe).astype(np.float32)
            F=transport(E,A); pr=Ridge(alpha=200).fit(F[tr],Yd[tr]).predict(F)
            if tag=="+non-neg(clip)": pr=np.clip(pr,0,None)
            vv.append(medS(pr,Yd,te))
    rep[tag]=[float(np.mean(vv)),float(np.std(vv))]; print(f"representation {tag:16s}: {rep[tag][0]:+.3f} ± {rep[tag][1]:.3f}",flush=True)
json.dump(dict(physics=phys,structured=struct,representation=rep,n_metab=NMET),open(f"{SP}/rigor_physics.json","w"),indent=2)
print("saved rigor_physics.json")
