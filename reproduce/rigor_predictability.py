"""What makes a metabolite predictable? Regress per-metabolite predictability (within-sample CV r)
on measurable properties: abundance, coefficient of variation, spatial autocorrelation (Moran's I),
detection reliability, and molecular class. Feature importance + variance explained -> reinforces the
conceptual 'predictability' framing."""
import anndata as ad, numpy as np, pandas as pd, json, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"; SP="/home/claude/spatmet"
secs=["BC_525","BC_823","LC_091","LC_170"]
ann=pd.read_csv(f"{SP}/metabolite_annotation.csv"); ann["mz"]=ann["mz"].round(3); CL=dict(zip(ann.mz,ann['class']))
def clsof(m): v=CL.get(round(float(m),3),'unannotated'); return v if isinstance(v,str) else 'unannotated'
def A_bin(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    B=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); B=((B+B.T)>0).astype(np.float32); return B
def norm(A): return diags((1/np.sqrt(np.asarray((A+eye(A.shape[0])).sum(1)).ravel())).astype(np.float32))@(A+eye(A.shape[0]))
rows=[]
for s in secs:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Yr=np.asarray(a.uns['msi'],np.float32); Y=np.log1p(Yr); co=np.asarray(a.obsm['spatial'],float); mz=np.round(np.asarray(a.uns['mz_features'],float),3)
    det=(Yr>0).mean(0); keep=np.where(det>0.2)[0]; Yd=Y[:,keep]; mzk=mz[keep]
    B=A_bin(co); An=norm(B); W=B.sum()
    hv=np.argsort(X.var(0))[-3000:]; Xh=X[:,hv]; blk=KMeans(5,n_init=3,random_state=0).fit_predict(co)
    pred=np.zeros_like(Yd)
    for fb in range(5):
        tr=blk!=fb; te=blk==fb
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32); T=np.hstack([P,An@P,An@(An@P)])
        pred[te]=Ridge(alpha=200).fit(T[tr],Yd[tr]).predict(T[te])
    r=np.array([spearmanr(pred[:,j],Yd[:,j])[0] if Yd[:,j].std()>1e-9 else np.nan for j in range(Yd.shape[1])])
    # properties (on measured metabolite)
    Yc=Yd-Yd.mean(0); AY=An@Yc                                   # for Moran's I use binary B
    Bc=B@Yc; moran=(Yd.shape[0]/W)*(np.sum(Yc*Bc,0)/(np.sum(Yc*Yc,0)+1e-9))
    abund=Y[:,keep].mean(0); cv=Yr[:,keep].std(0)/(Yr[:,keep].mean(0)+1e-9); detf=det[keep]
    for j in range(len(mzk)):
        rows.append(dict(sec=s,r=r[j],abundance=abund[j],cv=cv[j],moran=moran[j],detection=detf[j],cls=clsof(mzk[j])))
df=pd.DataFrame(rows).dropna(subset=['r'])
# univariate Spearman of predictability vs each property
uni={p:float(spearmanr(df['r'],df[p])[0]) for p in ['abundance','cv','moran','detection']}
print("Univariate Spearman(predictability r, property):")
for p,v in uni.items(): print(f"  {p:11s}: {v:+.3f}")
# multivariate: gradient boosting importance + R^2
lip=['sphingolipid','phospholipid','lysophospholipid','fatty acid','lipid headgroup','glycerolipid','nucleobase','nucleoside','nucleotide']
df['is_lipid_nuc']=df['cls'].isin(lip).astype(float)
feats=['abundance','cv','moran','detection','is_lipid_nuc']
Xg=df[feats].values; yg=df['r'].values
from sklearn.model_selection import cross_val_predict
gb=GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=0)
yp=cross_val_predict(gb,Xg,yg,cv=5); R2=1-np.sum((yg-yp)**2)/np.sum((yg-yg.mean())**2)
gb.fit(Xg,yg); imp=dict(zip(feats,[float(x) for x in gb.feature_importances_]))
print(f"\nMultivariate model of predictability: 5-fold R²={R2:.2f}")
print("feature importance:",{k:round(v,2) for k,v in sorted(imp.items(),key=lambda x:-x[1])})
json.dump(dict(univariate=uni,importance=imp,cv_r2=float(R2),n=len(df),
    scatter={p:[df[p].tolist(),df['r'].tolist()] for p in ['moran','abundance']}),open(f"{SP}/rigor_predictability.json","w"),indent=2)
df.to_csv(f"{SP}/predictability_determinants.csv",index=False)
print("saved rigor_predictability.json")
