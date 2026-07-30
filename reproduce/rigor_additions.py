"""Added rigor for revision: (1) effect sizes + paired bootstrap 95% CIs on benchmark deltas;
(2) high- vs low-confidence example maps (measured/predicted) for the Fig 4 confidence-contrast panel."""
import anndata as ad, numpy as np, pandas as pd, json, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore'); rng=np.random.RandomState(0)
SP="/home/claude/spatmet"; DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
ann=pd.read_csv(f"{SP}/metabolite_annotation.csv"); ann["mz"]=ann["mz"].round(3)
ANNn=ann.set_index('mz')['annotation'].to_dict(); ANNc=ann.set_index('mz')['class'].to_dict()
def nameof(m): v=ANNn.get(round(float(m),3),''); return v if isinstance(v,str) else ''
def clsof(m): v=ANNc.get(round(float(m),3),'unannotated'); return v if isinstance(v,str) else 'unannotated'

# ---------- (1) effect sizes + paired bootstrap CI ----------
B=json.load(open(f"{SP}/benchmark_all.json")); ps=B['per_sample']; n=7
def boot(a,b,nit=5000):
    d=np.array([a[i]-b[i] for i in range(n)]); idx=rng.randint(0,n,(nit,n))
    means=d[idx].mean(1); return float(d.mean()),float(np.percentile(means,2.5)),float(np.percentile(means,97.5))
eff={}
for base in ["joint-NMF (MANTIS-class)","scMetabolism (KEGG)","per-spot ridge"]:
    for top in ["MetaSpatial","MetaSpatial + KEGG"]:
        m,lo,hi=boot(ps[top],ps[base]); eff[f"{top} - {base}"]=dict(delta=round(m,3),ci=[round(lo,3),round(hi,3)])
print("Effect sizes (Δ median Spearman, paired bootstrap 95% CI):")
for k,v in eff.items(): print(f"  {k:42s} Δr={v['delta']:+.3f}  95%CI [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")

# ---------- (2) confidence-contrast example maps (BC_525) ----------
a=ad.read_h5ad(f"{DATA}/BC_525.h5ad")
X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
mz=np.round(np.asarray(a.uns['mz_features'],float),3); Y=np.log1p(np.asarray(a.uns['msi'],np.float32))
coords=np.asarray(a.obsm['spatial'],float)
det=(Y>0).mean(0)>0.2; keep=np.where(det)[0]; Yd=Y[:,keep]; mzk=mz[keep]
def nadj(c,k=6):
    nn=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,nn)); r=np.repeat(np.arange(nn),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(nn,nn)); A=((A+A.T)>0).astype(np.float32)+eye(nn,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
hvg=np.argsort(X.var(0))[-3000:]; Xh=X[:,hvg]; Adj=nadj(coords); blk=KMeans(5,n_init=3,random_state=0).fit_predict(coords)
pred=np.zeros_like(Yd); wsum=np.zeros(Yd.shape[1]); wct=0
for fb in range(5):
    tr=blk!=fb; te=blk==fb
    rs=np.random.RandomState(fb).permutation(np.where(tr)[0]); nc=len(rs)//4; cal=rs[:nc]; fit=rs[nc:]
    P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[fit]).transform(Xh).astype(np.float32)
    T=np.hstack([P,Adj@P,Adj@(Adj@P)]); reg=Ridge(alpha=200).fit(T[fit],Yd[fit])
    q=np.quantile(np.abs(Yd[cal]-reg.predict(T[cal])),0.9,axis=0); pred[te]=reg.predict(T[te]); wsum+=2*q; wct+=1
width=wsum/wct; scale=Yd.std(0)+1e-8; relwidth=width/scale
rmet=np.array([spearmanr(pred[:,j],Yd[:,j])[0] if Yd[:,j].std()>1e-9 else np.nan for j in range(Yd.shape[1])])
names=np.array([nameof(m) for m in mzk]); cls=np.array([clsof(m) for m in mzk])
ok=np.isfinite(rmet)&(np.array([len(x)>0 for x in names]))
# high-confidence: high r AND low relative width (annotated); low-confidence: low r AND high relative width
zr=(rmet-np.nanmean(rmet))/np.nanstd(rmet); zw=(relwidth-np.nanmean(relwidth))/np.nanstd(relwidth)
hi=np.where(ok)[0][np.argmax((zr-zw)[ok])]; lo=np.where(ok)[0][np.argmin((zr-zw)[ok])]
def emap(j): return dict(meas=Yd[:,j].tolist(),pred=pred[:,j].tolist(),name=(names[j] or f"m/z {mzk[j]:.2f}"),
                         cls=str(cls[j]),r=float(rmet[j]),relwidth=float(relwidth[j]))
json.dump(dict(high=emap(hi),low=emap(lo),coords=coords.tolist()),open(f"{SP}/conf_contrast.json","w"))
print(f"\nHIGH-confidence example: {names[hi]} ({cls[hi]}) r={rmet[hi]:.2f} relwidth={relwidth[hi]:.2f}")
print(f"LOW-confidence  example: {names[lo]} ({cls[lo]}) r={rmet[lo]:.2f} relwidth={relwidth[lo]:.2f}")
json.dump(dict(effect_sizes=eff,
    highconf=dict(name=str(names[hi]),cls=str(cls[hi]),r=float(rmet[hi]),relwidth=float(relwidth[hi])),
    lowconf=dict(name=str(names[lo]),cls=str(cls[lo]),r=float(rmet[lo]),relwidth=float(relwidth[lo]))),
    open(f"{SP}/rigor_summary.json","w"),indent=2)
print("saved conf_contrast.json, rigor_summary.json")
