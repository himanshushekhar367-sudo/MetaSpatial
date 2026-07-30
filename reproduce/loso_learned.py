"""Learned harmonization vs the LOSO gap. Linear per-section normalization did not help (0.032->0.033);
here we test (1) Harmony-integrated embedding and (2) a nonlinear MLP head, isolating each effect.
Harmony is unsupervised (transcriptome+batch only, never metabolome), run once on the pooled 7 sections,
then LOSO by splitting -- a valid transductive protocol (query transcriptome is available at predict time)."""
import anndata as ad, numpy as np, pandas as pd, json, gc, warnings, time
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
import harmonypy
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
lens={s:Xg[s].shape[0] for s in samples}; order=samples
Xall=np.vstack([Xg[s] for s in order]); batch=np.concatenate([[s]*lens[s] for s in order])
pca=PCA(100,svd_solver='randomized',random_state=0).fit(Xall); Pall=pca.transform(Xall).astype(np.float32)
# Harmony (unsupervised) on pooled embedding
t0=time.time(); ho=harmonypy.run_harmony(Pall,pd.DataFrame({'batch':batch}),['batch'],max_iter_harmony=10)
Zc=np.asarray(ho.Z_corr).astype(np.float32)
if Zc.shape[0]!=Xall.shape[0]: Zc=Zc.T           # ensure (cells x PCs)
print(f"Harmony done in {time.time()-t0:.0f}s, Zc {Zc.shape} (want {Xall.shape[0]} cells)",flush=True)
# split back to sections
def split(Mat):
    out={}; i=0
    for s in order: out[s]=Mat[i:i+lens[s]]; i+=lens[s]
    return out
Pd=split(Pall); Zd=split(Zc)
def tfeat(emb,s): P=emb[s]; return np.hstack([P,adjs[s]@P,adjs[s]@(adjs[s]@P)])
def loso(embd,head):
    out={}
    for held in order:
        tr=[s for s in order if s!=held]
        Xtr=np.vstack([tfeat(embd,s) for s in tr]); Ytr=np.vstack([YlD[s] for s in tr]); Xte=tfeat(embd,held)
        if head=='ridge': mdl=Ridge(alpha=200).fit(Xtr,Ytr); pr=mdl.predict(Xte)
        else:
            mdl=MLPRegressor(hidden_layer_sizes=(128,),alpha=1e-2,max_iter=60,early_stopping=True,random_state=0).fit(Xtr,Ytr); pr=mdl.predict(Xte)
        Yh=YlD[held]; det=detD[held]
        out[held]=float(np.nanmedian([spearmanr(pr[:,j],Yh[:,j])[0] for j in range(Yh.shape[1]) if det[j] and Yh[:,j].std()>1e-9]))
    return out
def summ(o): v=list(o.values()); return dict(overall=[float(np.mean(v)),float(np.std(v))],
    breast=float(np.mean([o[s] for s in order if s[:2]=='BC'])),lung=float(np.mean([o[s] for s in order if s[:2]=='LC'])),per=o)
res={}
for name,embd,head in [("ridge + shared-PCA",Pd,'ridge'),("ridge + Harmony",Zd,'ridge')]:
    o=loso(embd,head); res[name]=summ(o); print(f"{name:22s} LOSO {res[name]['overall'][0]:+.3f}+-{res[name]['overall'][1]:.3f} | breast {res[name]['breast']:+.3f} | lung {res[name]['lung']:+.3f}",flush=True)
for name,embd,head in [("MLP  + shared-PCA",Pd,'mlp'),("MLP  + Harmony",Zd,'mlp')]:
    t0=time.time(); o=loso(embd,head); res[name]=summ(o); print(f"{name:22s} LOSO {res[name]['overall'][0]:+.3f}+-{res[name]['overall'][1]:.3f} | breast {res[name]['breast']:+.3f} | lung {res[name]['lung']:+.3f}  ({time.time()-t0:.0f}s)",flush=True)
json.dump(dict(baseline_linear=0.032,results=res),open(f"{SP}/loso_learned.json","w"),indent=2)
print("saved loso_learned.json")
