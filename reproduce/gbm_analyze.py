#!/usr/bin/env python3
# =====================================================================================
# GBM MetaSpatial — STEP 2 of 2 : analyse the predicted metabolomes.
#   (A) Parent Visium GBM : spatial metabolite maps + unsupervised metabolic niches
#   (B) CytAssist GBM     : predicted-metabolite  x  measured-protein  co-localization
# Run after gbm_predict.py:   python gbm_analyze.py
#
# HONEST CAVEATS (read me):
#  * GBM is fully cross-cohort (model trained on DESI breast/lung tumour) -> QUALITATIVE.
#  * The metabolite<->protein correlations share an RNA-driven component (both derive from
#    the transcriptome), so they are suggestive co-localization, NOT independent validation.
#  * N-acetylaspartate is a NEURONAL metabolite (high in normal brain, low in tumour); if the
#    model predicts it tracking proliferation, that is a cross-cohort ERROR, not biology.
# =====================================================================================
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from scipy.stats import rankdata
warnings.filterwarnings("ignore")

# ============ SET THIS TO YOUR DATA FOLDER (same as gbm_predict.py) ============
ROOT = r"C:\Users\pc\Desktop\spatial metabolism"
# ==============================================================================
OUT = os.path.join(ROOT, "gbm_metaspatial")

mz = np.load(os.path.join(OUT,"gbm2_visium_pred.npz"))["mz"]
# curated metabolites (neutral monoisotopic mass) + DESIUM within-sample predictability r (fallback)
DB={'Glutathione(GSH)':307.0838,'Ascorbate':176.0321,'Arachidonate(FA20:4)':304.2402,'DHA(FA22:6)':328.2402,
 'Glutamate':147.0532,'Palmitate(FA16:0)':256.2402,'Stearate(FA18:0)':284.2715,'Oleate(FA18:1)':282.2559,
 'Hexose(glucose)':180.0634,'Inosine':268.0808,'Hypoxanthine':136.0385,'Urate':168.0283,'Taurine':125.0147,
 'N-acetylaspartate':175.0481,'AMP':347.0631,'Aspartate':133.0375}
REL={'Glutathione(GSH)':0.58,'Ascorbate':0.58,'Arachidonate(FA20:4)':0.58,'DHA(FA22:6)':0.51,'Glutamate':0.40,
 'Palmitate(FA16:0)':0.47,'Stearate(FA18:0)':0.47,'Oleate(FA18:1)':0.46,'Hexose(glucose)':0.53,'Inosine':0.55,
 'Hypoxanthine':0.41,'Urate':0.49,'Taurine':0.54,'N-acetylaspartate':0.40,'AMP':0.53,'Aspartate':0.46}
ADD=[("[M-H]-",-1.0073),("[M+Cl]-",34.9694),("[M+HCOO]-",44.9982)]
def ion_index(mass):
    for _,dl in ADD:
        i=int(np.argmin(np.abs(mz-(mass+dl))))
        if abs(mz[i]-(mass+dl))<=0.01: return i
    return None
IONS={n:ion_index(m) for n,m in DB.items()}; IONS={k:v for k,v in IONS.items() if v is not None}
def zc(a): a=np.asarray(a,float); return (a-a.mean())/(a.std()+1e-9)

# ================= (A) VISIUM: maps + niches =================
d=np.load(os.path.join(OUT,"gbm2_visium_pred.npz")); pv=d["pred"]; xy=d["coords"].astype(float)
relvec=np.array([REL.get(n,0.4) for n in IONS])
idxs=list(IONS.values())
Z=zc(pv[:,idxs]) if False else (pv[:,idxs]-pv[:,idxs].mean(0))/(pv[:,idxs].std(0)+1e-9)
niche=KMeans(5,n_init=5,random_state=0).fit_predict(Z)

fig=plt.figure(figsize=(15,8)); gs=GridSpec(2,4,figure=fig,hspace=0.35,wspace=0.3)
for i,nm in enumerate(["Glutathione(GSH)","DHA(FA22:6)","Arachidonate(FA20:4)","Palmitate(FA16:0)"]):
    ion=IONS.get(nm); ax=fig.add_subplot(gs[0,i])
    if ion is None: continue
    v=pv[:,ion]; lo,hi=np.percentile(v,2),np.percentile(v,98)
    ax.scatter(xy[:,0],xy[:,1],c=v,s=8,cmap="magma",vmin=lo,vmax=hi,linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); ax.set_title(f"Visium: {nm}\n(r~{REL.get(nm,0.4):.2f})",fontsize=9)
axn=fig.add_subplot(gs[1,0]); axn.scatter(xy[:,0],xy[:,1],c=niche,s=8,cmap="tab10",linewidths=0)
axn.set_aspect("equal"); axn.invert_yaxis(); axn.axis("off"); axn.set_title("Metabolic niches (k=5)",fontsize=9)
axh=fig.add_subplot(gs[1,1:])
NM=pd.DataFrame({nm:[zc(pv[:,i])[niche==k].mean() for k in range(5)] for nm,i in IONS.items()},
                index=[f"niche {k}" for k in range(5)]).T
im=axh.imshow(NM.values,cmap="RdBu_r",vmin=-1.2,vmax=1.2,aspect="auto")
axh.set_xticks(range(5)); axh.set_xticklabels(NM.columns,fontsize=8); axh.set_yticks(range(len(NM)))
axh.set_yticklabels(NM.index,fontsize=8); axh.set_title("Predicted-metabolite mean (z) per niche",fontsize=9)
plt.colorbar(im,ax=axh,fraction=0.025)
fig.suptitle("Parent Visium GBM — predicted spatial metabolome",fontsize=13,fontweight="bold")
fig.savefig(os.path.join(OUT,"GBM_visium_metabolome.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("saved GBM_visium_metabolome.png")

# ================= (B) CYTASSIST: metabolite <-> protein =================
dc=np.load(os.path.join(OUT,"gbm2_cyt_pred.npz")); pc=dc["pred"]; xyc=dc["coords"].astype(float)
dp=np.load(os.path.join(OUT,"gbm_cyt_protein.npz"),allow_pickle=True); P=dp["protein"].astype(float); pn=list(dp["names"])
keep=[i for i,n in enumerate(pn) if not (n.lower().startswith("mouse_ig") or n.lower().startswith("rat_ig"))]
P=P[:,keep]; pn=[pn[i].replace("-1","").replace("-2","_2") for i in keep]
Rp=np.apply_along_axis(rankdata,0,np.log1p(P))
mets=[(n,i) for n,i in IONS.items() if REL.get(n,0.4)>=0.30]
Rm=np.apply_along_axis(rankdata,0,np.column_stack([pc[:,i] for _,i in mets]))
def corr(a,b): return float((zc(a)*zc(b)).mean())
C=np.array([[corr(Rm[:,mi],Rp[:,pi]) for pi in range(len(pn))] for mi in range(len(mets))])
pd.DataFrame(C,index=[m[0] for m in mets],columns=pn).to_csv(os.path.join(OUT,"cyt_metabolite_protein_spearman.csv"))

fig2=plt.figure(figsize=(14,8)); gs2=GridSpec(1,3,figure=fig2,width_ratios=[2.4,1,1],wspace=0.35)
axc=fig2.add_subplot(gs2[0,0]); vmax=np.nanmax(np.abs(C))
im=axc.imshow(C,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
axc.set_xticks(range(len(pn))); axc.set_xticklabels(pn,rotation=90,fontsize=7)
axc.set_yticks(range(len(mets))); axc.set_yticklabels([m[0] for m in mets],fontsize=8)
axc.set_title("CytAssist GBM: predicted metabolite x measured protein (Spearman)",fontsize=10)
plt.colorbar(im,ax=axc,fraction=0.03)
flat=[(m[0],pn[j],C[i,j]) for i,m in enumerate(mets) for j in range(len(pn))]; flat.sort(key=lambda t:-abs(t[2]))
print("\nTOP metabolite<->protein co-localizations (CytAssist GBM):")
for a,b,c in flat[:15]: print(f"  {a:22s} ~ {b:10s}  Spearman {c:+.2f}")
ex=[t for t in flat if t[2]>0][:2]
for k,(mnm,pnm,cc) in enumerate(ex):
    mi=[i for i,m in enumerate(mets) if m[0]==mnm][0]; pj=pn.index(pnm); axm=fig2.add_subplot(gs2[0,1+k])
    axm.scatter(xyc[:,0],xyc[:,1],c=zc(pc[:,mets[mi][1]]),s=6,cmap="magma",vmin=-2,vmax=2,linewidths=0)
    axm.set_aspect("equal"); axm.invert_yaxis(); axm.axis("off"); axm.set_title(f"{mnm}\nvs protein {pnm} (r={cc:+.2f})",fontsize=8)
fig2.savefig(os.path.join(OUT,"GBM_cytassist_metab_protein.png"),dpi=150,bbox_inches="tight"); plt.close(fig2)
print("\nsaved GBM_cytassist_metab_protein.png and cyt_metabolite_protein_spearman.csv -> ",OUT)
print("\nREMINDER: cross-cohort predictions are qualitative; NAA tracking proliferation is a known cross-cohort error.")
