"""Cross-cohort liver zonation validation (unpaired, directional):
does MetaSpatial's HCC tumor-vs-normal predicted metabolite shift agree with the MEASURED
liver lipidome shift (cirrhotic 'ac' vs normal 'hc')?  Matched by m/z (+-0.01 Da)."""
import numpy as np, pandas as pd, glob, warnings, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
warnings.filterwarnings('ignore')
UP="/mnt/user-data/uploads/spatial metabolism"

def load_lip(tag):
    f=glob.glob(f"{UP}/*liver_{tag}*_lipidome.npz")[0]; d=np.load(f)
    return d['mz'].astype(float), d['intensity'].astype(float)   # (n_mz,), (pixels, n_mz)
hc_mz, hc_I = load_lip("hc"); ac_mz, ac_I = load_lip("ac")
def match(a,b,tol=0.01):
    idx=np.searchsorted(b,a); out=[]
    for i,m in enumerate(a):
        cand=[j for j in (idx[i]-1,idx[i]) if 0<=j<len(b) and abs(b[j]-m)<tol]
        out.append(min(cand,key=lambda j:abs(b[j]-m)) if cand else -1)
    return np.array(out)
order=np.argsort(ac_mz); ac_mz=ac_mz[order]; ac_I=ac_I[:,order]
j=match(hc_mz,ac_mz); keep=j>=0
liver_mz=hc_mz[keep]
liver_lfc=np.log1p(ac_I[:,j[keep]].mean(0))-np.log1p(hc_I[:,keep].mean(0))    # cirrhosis - normal (measured)
print(f"liver: {len(hc_mz)} hc peaks, {len(ac_mz)} ac peaks, {keep.sum()} matched (hc<->ac)")

# HCC predicted tumor-vs-normal (consensus across 4 patients)
Z=pd.read_csv("/home/claude/spatmet/xcancer/hcc_zonation_TvsN_log.csv",index_col=0)
hcc_mz=Z.index.values.astype(float); hcc_tn=Z.mean(1).values
# match liver <-> HCC(DESIUM) m/z
o2=np.argsort(hcc_mz); hcc_mz_s=hcc_mz[o2]; hcc_tn_s=hcc_tn[o2]
k=match(liver_mz,hcc_mz_s); com=k>=0
xv=hcc_tn_s[k[com]]; yv=liver_lfc[com]
rS,pS=spearmanr(xv,yv); rP,pP=pearsonr(xv,yv)
print(f"common m/z (liver<->HCC-DESIUM): {com.sum()}")
print(f"CONCORDANCE  HCC(tumor-normal, predicted) vs liver(cirrhosis-normal, measured):")
print(f"   Spearman r={rS:+.3f} p={pS:.1e} | Pearson r={rP:+.3f} p={pP:.1e}  (n={com.sum()} matched lipids)")
# save + figure
pd.DataFrame({'mz':liver_mz[com],'HCC_pred_TvsN':xv,'liver_meas_cirr_vs_norm':yv}).to_csv("/home/claude/spatmet/xcancer/liver_concordance.csv",index=False)
fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].scatter(xv,yv,s=14,alpha=.6,color="#2e6f9e"); ax[0].axhline(0,color='k',lw=.6); ax[0].axvline(0,color='k',lw=.6)
m,b=np.polyfit(xv,yv,1); xs=np.linspace(xv.min(),xv.max(),50); ax[0].plot(xs,m*xs+b,color="#b23a48",lw=2)
ax[0].set_xlabel("HCC predicted  (tumor − normal-adjacent), log"); ax[0].set_ylabel("liver measured  (cirrhosis − normal), log")
ax[0].set_title(f"Cross-cohort lipid-zonation concordance\nSpearman r={rS:+.2f}, p={pS:.0e} (n={com.sum()} matched lipids)",fontsize=11,loc='left')
# measured liver: top differential lipids
di=np.argsort(liver_lfc); topn=np.r_[di[:8],di[-8:]]
ax[1].barh(range(len(topn)),liver_lfc[topn],color=["#2e6f9e" if v<0 else "#b0763a" for v in liver_lfc[topn]])
ax[1].set_yticks(range(len(topn))); ax[1].set_yticklabels([f"{m:.3f}" for m in liver_mz[topn]],fontsize=8)
ax[1].axvline(0,color='k',lw=.6); ax[1].set_xlabel("cirrhosis − normal (log)"); ax[1].set_ylabel("m/z")
ax[1].set_title("Measured liver lipidome: top shifted species (cirrhosis vs normal)",fontsize=11,loc='left')
plt.tight_layout(); fig.savefig("/home/claude/spatmet/xcancer/Fig_liver_concordance.png",dpi=150,bbox_inches='tight',facecolor='white')
print("saved liver_concordance.csv, Fig_liver_concordance.png")
