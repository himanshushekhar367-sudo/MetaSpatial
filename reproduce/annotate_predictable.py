"""Which metabolites are predictable? Per-m/z Spearman (ridge transport-aware, within-sample CV),
averaged across the 7 tumours (shared m/z axis), then annotated to metabolites by exact mass."""
import anndata as ad, numpy as np, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
samples=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
def norm_adj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); rows=np.repeat(np.arange(n),idx.shape[1]-1); cols=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A@A.__class__(A)  # placeholder
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); rows=np.repeat(np.arange(n),idx.shape[1]-1); cols=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    dinv=diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)); return dinv@A@dinv
mz=np.asarray(ad.read_h5ad(f"{DATA}/{samples[0]}.h5ad").uns['mz_features'],dtype=float); nmz=len(mz)
S=np.full((len(samples),nmz),np.nan)
for si,s in enumerate(samples):
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad")
    X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],dtype=np.float32)); coords=np.asarray(a.obsm['spatial'],dtype=float)
    det=(Y>0).mean(0)>0.2; Adj=nadj(coords); hvg=np.argsort(X.var(0))[-3000:]; Xh=X[:,hvg]
    blocks=KMeans(5,n_init=3,random_state=0).fit_predict(coords); pred=np.zeros_like(Y)
    for fb in range(5):
        trm=blocks!=fb; tem=blocks==fb
        if tem.sum()==0 or trm.sum()<50: continue
        pca=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[trm]); P=pca.transform(Xh).astype(np.float32)
        Ptr=np.hstack([P,Adj@P,Adj@(Adj@P)]); pred[tem]=Ridge(alpha=200).fit(Ptr[trm],Y[trm]).predict(Ptr[tem])
    for m in range(nmz):
        if det[m] and Y[:,m].std()>1e-9: S[si,m]=spearmanr(pred[:,m],Y[:,m])[0]
    print(f"  {s} done", flush=True)
meanS=np.nanmean(S,0)

# curated neutral monoisotopic masses (Da) for common tissue metabolites
DB={'Lactate':90.0317,'Pyruvate':88.0160,'Alanine':89.0477,'Serine':105.0426,'Glycine':75.0320,
 'Succinate':118.0266,'Fumarate':116.0110,'Malate':134.0215,'Aspartate':133.0375,'Glutamate':147.0532,
 'Glutamine':146.0691,'a-Ketoglutarate':146.0215,'Citrate/Isocitrate':192.0270,'Hexose(Glc/Fru/Gal)':180.0634,
 'Glucose-6-P':260.0297,'Taurine':125.0147,'Creatine':131.0695,'Creatinine':113.0589,'Hypoxanthine':136.0385,
 'Inosine':268.0808,'AMP':347.0631,'ADP':427.0294,'ATP':506.9957,'Glutathione(GSH)':307.0838,'GSSG':612.1519,
 'Ascorbate':176.0321,'Urate':168.0283,'Palmitate(FA16:0)':256.2402,'Stearate(FA18:0)':284.2715,
 'Oleate(FA18:1)':282.2559,'Linoleate(FA18:2)':280.2402,'Arachidonate(FA20:4)':304.2402,'DHA(FA22:6)':328.2402,
 'Sphingosine':299.2824,'Cholesterol-sulfate':466.3111,'Taurocholate':515.2917,'Inositol':180.0634,
 'N-acetylaspartate':175.0481,'Phosphocreatine':211.0358,'UDP-GlcNAc':607.0817,'Spermidine':145.1579,
 'Carnitine':161.1052,'Acetylcarnitine':203.1158,'Kynurenine':208.0848,'Adenosine':267.0968}
def annotate(v):
    hits=[]
    for name,mass in DB.items():
        for ad_,dl in [("[M-H]-",-1.0073),("[M+Cl]-",34.9694),("[M+HCOO]-",44.9982)]:
            if abs(v-(mass+dl))<=0.01: hits.append(f"{name} {ad_}")
    return "; ".join(hits)

order=np.argsort(-np.nan_to_num(meanS,nan=-1))
print("\n=== TOP predictable metabolites (mean Spearman across 7 tumours, transport-aware) ===")
shown=0
for i in order:
    if not np.isfinite(meanS[i]): continue
    ann=annotate(mz[i])
    if ann:
        print(f"  m/z {mz[i]:9.4f}  r={meanS[i]:+.3f}  ->  {ann}")
        shown+=1
    if shown>=25: break
print(f"\n  detected metabolites with mean r>0.3: {int(np.sum(meanS>0.3))} / {int(np.sum(np.isfinite(meanS)))}")
np.savez("/home/claude/spatmet/permz_spearman.npz", meanS=meanS, mz=mz)
print("saved permz_spearman.npz")
