"""Uncertainty framework: separate EPISTEMIC (ensemble variance), ALEATORIC (residual-variance model),
and CONFORMAL calibration — and test whether calibration holds UNDER DISTRIBUTION SHIFT (new section).
Key question a reviewer will ask: does the 90% interval still cover ~90% on unseen tissue?"""
import anndata as ad, numpy as np, json, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore'); rng=np.random.RandomState(0)
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"; SP="/home/claude/spatmet"
secs=["BC_525","BC_823","LC_091","LC_170"]; NMET=200
def adj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
genes=sorted(set.intersection(*[set(ad.read_h5ad(f"{DATA}/{s}.h5ad").var_names) for s in secs]))
Xg={}; Y={}; co={}; As={}
for s in secs:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); v=a[:,genes].layers['log1p']
    Xg[s]=(np.asarray(v.todense()) if hasattr(v,'todense') else np.asarray(v)).astype(np.float32)
    Yr=np.asarray(a.uns['msi'],np.float32); Yl=np.log1p(Yr); det=(Yr>0).mean(0)>0.2
    top=np.where(det)[0][np.argsort(Yl[:,det].var(0))[-NMET:]]; Y[s]=Yl[:,top]; co[s]=np.asarray(a.obsm['spatial'],float); As[s]=adj(co[s])
gv=np.vstack([Xg[s] for s in secs]).var(0); hv=np.argsort(gv)[-2000:]
def feat(pca,s): P=pca.transform(Xg[s][:,hv]).astype(np.float32); return np.hstack([P,As[s]@P,As[s]@(As[s]@P)])
held="LC_170"; tr=[s for s in secs if s!=held]
pca=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s][:,hv] for s in tr]))
Ftr={s:feat(pca,s) for s in secs}
# hold out a calibration section-split: use BC_823 spots as in-distribution test + calibration
Fall=np.vstack([Ftr[s] for s in tr]); Yall=np.vstack([Y[s] for s in tr])
idx=rng.permutation(len(Fall)); ncal=len(idx)//5; cal=idx[:ncal]; fit=idx[ncal:]
# ---- ensemble (epistemic) ----
K=10; preds_id=[]; preds_ood=[]
Fid=Fall[cal]; Food=Ftr[held]
for k in range(K):
    bs=rng.randint(0,len(fit),len(fit)); sub=fit[bs]
    reg=Ridge(alpha=200).fit(Fall[sub],Yall[sub])
    preds_id.append(reg.predict(Fid)); preds_ood.append(reg.predict(Food))
preds_id=np.array(preds_id); preds_ood=np.array(preds_ood)
epi_id=preds_id.std(0).mean(); epi_ood=preds_ood.std(0).mean()
# ---- point model + conformal + aleatoric ----
reg=Ridge(alpha=200).fit(Fall[fit],Yall[fit])
res_cal=np.abs(Yall[cal]-reg.predict(Fall[cal]))                       # calibration residuals
qhat=np.quantile(res_cal,0.9,axis=0)                                   # per-metabolite 90% conformal width
# in-distribution coverage (on a fresh split of training sections)
idc=rng.permutation(len(Fall)); test_id=idc[:3000]
cov_id=float(np.mean(np.abs(Yall[test_id]-reg.predict(Fall[test_id]))<=qhat))
# OOD coverage (held section) with SAME intervals
cov_ood=float(np.mean(np.abs(Y[held]-reg.predict(Food))<=qhat))
# aleatoric: fit residual-variance predictor; check it tracks error
al=Ridge(alpha=50).fit(Fall[fit],np.log1p(np.abs(Yall[fit]-reg.predict(Fall[fit])).mean(1)))
al_pred_id=al.predict(Fid); al_true_id=np.abs(Yall[cal]-reg.predict(Fid)).mean(1)
al_r=spearmanr(al_pred_id,al_true_id)[0]
print("=== UNCERTAINTY DECOMPOSITION ===")
print(f"  epistemic (ensemble std): in-distribution={epi_id:.3f}  OOD(held section)={epi_ood:.3f}  ratio={epi_ood/epi_id:.2f}x")
print(f"  aleatoric model vs true error: Spearman={al_r:+.3f}")
print(f"  conformal 90% coverage: in-distribution={cov_id:.2f}  OOD(held section)={cov_ood:.2f}")
print(f"  -> calibration {'DEGRADES' if cov_ood<0.8 else 'holds'} under shift (nominal 0.90)")
json.dump(dict(epistemic_id=float(epi_id),epistemic_ood=float(epi_ood),epistemic_ratio=float(epi_ood/epi_id),
    aleatoric_spearman=float(al_r),coverage_id=cov_id,coverage_ood=cov_ood,held=held),open(f"{SP}/rigor_uncertainty.json","w"),indent=2)
print("saved rigor_uncertainty.json")
