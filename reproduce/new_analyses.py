"""Reviewer quick-wins (fixed): scale-normalized uncertainty, named top annotations,
memory-safe LOSO per-sample/per-cancer, multi-class example maps."""
import anndata as ad, numpy as np, pandas as pd, json, gc, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
SP="/home/claude/spatmet"; samples=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
ann=pd.read_csv(f"{SP}/metabolite_annotation.csv"); ann["mz"]=ann["mz"].round(3)
ANNc=ann.set_index('mz')['class'].to_dict(); ANNn=ann.set_index('mz')['annotation'].to_dict()
def clsof(m): v=ANNc.get(round(float(m),3),'unannotated'); return v if isinstance(v,str) else 'unannotated'
def nameof(m): v=ANNn.get(round(float(m),3),''); return v if isinstance(v,str) else ''
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
def dense(a):
    v=a.layers['log1p']; return (np.asarray(v.todense()) if hasattr(v,'todense') else np.asarray(v)).astype(np.float32)

# ---- representative within-sample CV (BC_525): per-metab r, conformal width, example maps ----
a=ad.read_h5ad(f"{DATA}/BC_525.h5ad"); X=dense(a); mz=np.round(np.asarray(a.uns['mz_features'],float),3)
Y=np.log1p(np.asarray(a.uns['msi'],np.float32)); coords=np.asarray(a.obsm['spatial'],float)
det=(Y>0).mean(0)>0.2; keep=np.where(det)[0]; Yd=Y[:,keep]; mzk=mz[keep]
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
cls=np.array([clsof(m) for m in mzk]); nm=np.array([nameof(m) for m in mzk])
ok=np.isfinite(rmet)
r_abs,_=spearmanr(width[ok],rmet[ok]); r_rel,p_rel=spearmanr(relwidth[ok],rmet[ok])
print(f"(i) Spearman(ABS width, r)={r_abs:+.3f} (tracks dynamic range) ; Spearman(RELATIVE width, r)={r_rel:+.3f} p={p_rel:.1e} (informative confidence)")
lip=['phospholipid','sphingolipid','fatty acid','lysophospholipid']; pol=['amino acid','TCA acid','sugar phosphate','carboxylic acid']
print(f"    mean RELATIVE interval width: lipids={np.nanmean(relwidth[np.isin(cls,lip)]):.2f}  polar={np.nanmean(relwidth[np.isin(cls,pol)]):.2f} (higher=less confident)")
def pick(cc):
    m=np.where(np.isin(cls,cc)&ok)[0]; return int(m[np.argmax(rmet[m])]) if len(m) else None
ex={'lipid':pick(lip),'nucleotide/base':pick(['nucleotide','nucleoside','nucleobase']),'polar':pick(pol)}
exmaps={}
for lab,j in ex.items():
    if j is None: continue
    exmaps[lab]=dict(meas=Yd[:,j].tolist(),pred=pred[:,j].tolist(),name=(nm[j] or f"m/z {mzk[j]:.3f}"),r=float(rmet[j]))
np.savez_compressed(f"{SP}/fig_arrays.npz",coords=coords,r=rmet,relwidth=relwidth,cls=cls); json.dump(exmaps,open(f"{SP}/example_maps.json","w"))
tab=pd.DataFrame({'mz':mzk,'name':nm,'class':cls,'r':np.round(rmet,3)})
tab=tab[(tab.name.astype(str).str.len()>0)&np.isfinite(tab.r)].sort_values('r',ascending=False)
tab.head(20).to_csv(f"{SP}/top_annotated.csv",index=False)
print("(ii) top named predictable:", list(tab.head(8)[['name','r']].itertuples(index=False,name=None)))
del X,Xh,Adj,pred,Yd; gc.collect()

# ---- memory-safe LOSO (2000-gene two-pass) ----
A={s:ad.read_h5ad(f"{DATA}/{s}.h5ad") for s in samples}
genes=sorted(set.intersection(*[set(v.var_names) for v in A.values()]))
coordsD={s:np.asarray(A[s].obsm['spatial'],float) for s in samples}
YlD={s:np.log1p(np.asarray(A[s].uns['msi'],float)).astype(np.float32) for s in samples}
def gm(a):
    v=a[:,genes].layers['log1p']; return (np.asarray(v.todense()) if hasattr(v,'todense') else np.asarray(v)).astype(np.float32)
s1=np.zeros(len(genes)); s2=np.zeros(len(genes)); N=0
for s in samples:
    M=gm(A[s]); s1+=M.sum(0); s2+=(M*M).sum(0); N+=M.shape[0]; del M; gc.collect()
var=s2/N-(s1/N)**2; hv=np.argsort(var)[-2000:]
Xg={s:gm(A[s])[:,hv] for s in samples}; adjs={s:nadj(coordsD[s]) for s in samples}; gc.collect()
loso={}
for held in samples:
    tr=[s for s in samples if s!=held]
    pca=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s] for s in tr]))
    def F(s):
        P=pca.transform(Xg[s]).astype(np.float32); return np.hstack([P,adjs[s]@P,adjs[s]@(adjs[s]@P)])
    reg=Ridge(alpha=200).fit(np.vstack([F(s) for s in tr]),np.vstack([YlD[s] for s in tr]))
    pr=reg.predict(F(held)); Yh=YlD[held]
    loso[held]=float(np.nanmedian([spearmanr(pr[:,j],Yh[:,j])[0] for j in range(Yh.shape[1]) if Yh[:,j].std()>1e-9]))
    print(f"(iii) LOSO {held:18s} median r={loso[held]:+.3f}",flush=True)
bc=[loso[s] for s in samples if s.startswith('BC')]; lc=[loso[s] for s in samples if s.startswith('LC')]
json.dump(dict(unc_rel_r=[float(r_rel),float(p_rel)],unc_abs_r=float(r_abs),loso=loso,
   loso_overall=[float(np.mean(list(loso.values()))),float(np.std(list(loso.values())))],
   loso_breast=[float(np.mean(bc)),float(np.std(bc))],loso_lung=[float(np.mean(lc)),float(np.std(lc))],
   relwidth_lipid=float(np.nanmean(relwidth[np.isin(cls,lip)])),relwidth_polar=float(np.nanmean(relwidth[np.isin(cls,pol)]))),
   open(f"{SP}/new_summary.json","w"),indent=2)
print(f"\nLOSO overall {np.mean(list(loso.values())):.3f}+-{np.std(list(loso.values())):.3f} | breast {np.mean(bc):.3f}+-{np.std(bc):.3f} | lung {np.mean(lc):.3f}+-{np.std(lc):.3f}")
print("saved fig_arrays.npz, example_maps.json, top_annotated.csv, new_summary.json")
