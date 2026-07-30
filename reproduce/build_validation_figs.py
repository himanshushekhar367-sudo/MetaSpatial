#!/usr/bin/env python3
"""Build manuscript validation figures:
 Figure 8 (HCC cross-patient validation) and Figure 9 (GBM cross-platform extension)."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from scipy.stats import rankdata
SP="/home/claude/spatmet"; UP="/mnt/user-data/uploads/spatial metabolism"; FIG=f"{SP}/manuscript_v3/figures"
mz=np.load(f"{SP}/xcancer/HCC-2_pred.npz")["mz"]; relS=np.load(f"{SP}/permz_spearman.npz")["meanS"]
def zc(a): a=np.asarray(a,float); return (a-a.mean())/(a.std()+1e-9)
DB={'Glutathione(GSH)':307.0838,'Ascorbate':176.0321,'Arachidonate(FA20:4)':304.2402,'DHA(FA22:6)':328.2402,
 'Glutamate':147.0532,'Palmitate(FA16:0)':256.2402,'Stearate(FA18:0)':284.2715,'Hexose(glucose)':180.0634,
 'Oleate(FA18:1)':282.2559}
ADD=[("[M-H]-",-1.0073),("[M+Cl]-",34.9694),("[M+HCOO]-",44.9982)]
def ii(mass):
    for _,dl in ADD:
        i=int(np.argmin(np.abs(mz-(mass+dl))));
        if abs(mz[i]-(mass+dl))<=0.01: return i
    return None
ION={n:ii(m) for n,m in DB.items()}

# ---- HCC-2 subsections ----
pred=np.load(f"{SP}/xcancer/HCC-2_pred.npz")["pred"]
bc=[l.strip() for l in open(f"{UP}/HCC-2-expr_barcodes.txt")]
meta=pd.read_csv(f"{UP}/HCC-2-expr_meta.csv",index_col=0).reindex(bc)
suff=meta["sample.ident"].astype(str).str.replace("HCC-2","",regex=False).str.lstrip("-").values
xy=meta[["imagecol","imagerow"]].values.astype(float)
SUB=["N","L","P","T"]; SUBN={"N":"Normal","L":"Leading edge","P":"PVTT","T":"Tumour"}
mask={s:suff==s for s in SUB}

# ================= FIGURE 8 : HCC validation =================
fig=plt.figure(figsize=(13,13)); gs=GridSpec(3,4,figure=fig,height_ratios=[0.9,1.15,0.9],hspace=0.5,wspace=0.35)
# panel a: GSH maps across N/L/P/T
gsh=ION["Glutathione(GSH)"]; v=pred[:,gsh]; lo,hi=np.percentile(v,2),np.percentile(v,98)
for ci,s in enumerate(SUB):
    ax=fig.add_subplot(gs[0,ci]); m=mask[s]
    sc=ax.scatter(xy[m,0],xy[m,1],c=v[m],s=5,cmap="magma",vmin=lo,vmax=hi,linewidths=0)
    ax.set_aspect("equal");ax.invert_yaxis();ax.axis("off")
    ax.set_title(f"{s} · {SUBN[s]}",fontsize=9,fontweight="bold" if s=="P" else "normal",color="crimson" if s=="P" else "k")
fig.text(0.09,0.905,"a",fontsize=15,fontweight="bold"); fig.text(0.5,0.905,"HCC-2 predicted glutathione — rises to a peak in the PVTT",ha="center",fontsize=10)
cax=fig.add_axes([0.92,0.72,0.01,0.15]); plt.colorbar(sc,cax=cax); cax.tick_params(labelsize=7)

# panel b: N->PVTT signature bar
from scipy.stats import ttest_ind
sign={}
for nm,i in ION.items():
    a,b=pred[mask["N"],i],pred[mask["P"],i]; sign[nm]=b.mean()-a.mean()
order=["Glutathione(GSH)","DHA(FA22:6)","Arachidonate(FA20:4)","Ascorbate","Glutamate","Oleate(FA18:1)","Hexose(glucose)","Stearate(FA18:0)","Palmitate(FA16:0)"]
order=[o for o in order if o in sign]
axb=fig.add_subplot(gs[1,0:2]); vals=[sign[o] for o in order]
cols=["#c0392b" if x>0 else "#2471a3" for x in vals]
axb.barh(range(len(order)),vals,color=cols,edgecolor="k",lw=0.4)
axb.set_yticks(range(len(order))); axb.set_yticklabels([o.split("(")[0] for o in order],fontsize=8)
axb.invert_yaxis(); axb.axvline(0,color="k",lw=0.6); axb.set_xlabel("Δ predicted (log)  Normal → PVTT",fontsize=8.5)
axb.set_title("b   HCC-2 Normal→PVTT signature",fontsize=10,loc="left",fontweight="bold")

# panel c: cross-patient reproducibility heatmap
S=pd.read_csv(f"{SP}/hcc_repro_out/signature_deltas.csv",index_col=0)
axc=fig.add_subplot(gs[1,2:4]); vmax=np.nanmax(np.abs(S.values))
im=axc.imshow(S.values,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
axc.set_xticks(range(len(S.columns))); axc.set_xticklabels([c.split("\n")[0]+"\n"+c.split("\n")[1] for c in S.columns],fontsize=7)
axc.set_yticks(range(len(S.index))); axc.set_yticklabels(S.index,fontsize=7.5)
for yi in range(len(S.index)):
    for xi in range(len(S.columns)):
        val=S.values[yi,xi]
        if not np.isnan(val): axc.text(xi,yi,f"{val:+.2f}",ha="center",va="center",fontsize=6,color="white" if abs(val)>vmax*0.6 else "k")
axc.set_title("c   Cross-patient reproducibility (Δ tumour/PVTT − normal)",fontsize=10,loc="left",fontweight="bold")
plt.colorbar(im,ax=axc,fraction=0.03)

# panel d: literature coherence scorecard
axd=fig.add_subplot(gs[2,:]); axd.axis("off")
rows=[("Glutathione ↑","antioxidant / GPX4-ferroptosis resistance⁷⁻⁹","Coherent (2/4 patients)","#1a7a3a"),
      ("Ascorbate ↑","co-antioxidant","Coherent","#1a7a3a"),
      ("Arachidonate ↑","eicosanoid / suppressed FAO accumulation","Coherent (2/4)","#1a7a3a"),
      ("DHA ↑","ferroptosis substrate; some report PUFA↓ with progression","Contested","#b8860b"),
      ("Glutamate ↑","glutaminolysis → GSH / α-KG","Coherent","#1a7a3a"),
      ("Palmitate ↓, Stearate ↓","free saturated-FA depletion (vs bulk lipogenesis↑)","Defensible (3/4)","#1a7a3a"),
      ("Hexose/glucose ↓","glycolytic consumption (Warburg)","Coherent","#1a7a3a"),
      ("Lactate  (flat)","known Warburg ↑ but transcriptionally uncoupled","Model-blind (honest)","#7a1a1a")]
axd.set_title("d   Coherence with known HCC metabolic reprogramming",fontsize=10,loc="left",fontweight="bold")
y=0.9
axd.text(0.02,y,"MetaSpatial prediction",fontsize=8,fontweight="bold"); axd.text(0.34,y,"Known HCC biology",fontsize=8,fontweight="bold"); axd.text(0.80,y,"Verdict",fontsize=8,fontweight="bold")
for lab,bio,verd,col in rows:
    y-=0.108
    axd.text(0.02,y,lab,fontsize=8); axd.text(0.34,y,bio,fontsize=7.6); axd.text(0.80,y,verd,fontsize=8,color=col,fontweight="bold")
fig.suptitle("Figure 8 | Validation: MetaSpatial recovers known HCC metabolic reprogramming",fontsize=13,fontweight="bold",y=0.965)
fig.savefig(f"{FIG}/Figure8.png",dpi=160,bbox_inches="tight"); plt.close(fig); print("saved Figure8.png")

# ================= FIGURE 9 : GBM cross-platform =================
dv=np.load(f"{SP}/gbm2_visium_pred.npz"); pv=dv["pred"]; xyv=dv["coords"].astype(float)
GDB={'Glutathione(GSH)':307.0838,'Arachidonate(FA20:4)':304.2402,'DHA(FA22:6)':328.2402,'Palmitate(FA16:0)':256.2402,
 'Stearate(FA18:0)':284.2715,'Glutamate':147.0532,'Hexose(glucose)':180.0634,'Inosine':268.0808,'Taurine':125.0147,
 'N-acetylaspartate':175.0481,'Urate':168.0283,'Ascorbate':176.0321}
GION={n:ii(m) for n,m in GDB.items()}; GION={k:v for k,v in GION.items() if v is not None}
idxs=list(GION.values()); Z=(pv[:,idxs]-pv[:,idxs].mean(0))/(pv[:,idxs].std(0)+1e-9)
niche=KMeans(5,n_init=5,random_state=0).fit_predict(Z)
fig9=plt.figure(figsize=(15,6.2)); gs9=GridSpec(1,3,figure=fig9,width_ratios=[1,1.25,2.1],wspace=0.35)
axm=fig9.add_subplot(gs9[0,0]); axm.scatter(xyv[:,0],xyv[:,1],c=niche,s=7,cmap="tab10",linewidths=0)
axm.set_aspect("equal");axm.invert_yaxis();axm.axis("off"); axm.set_title("a  GBM metabolic niches (k=5)",fontsize=10,loc="left",fontweight="bold")
axh=fig9.add_subplot(gs9[0,1])
NMv=pd.DataFrame({nm:[zc(pv[:,i])[niche==k].mean() for k in range(5)] for nm,i in GION.items()},index=[f"n{k}" for k in range(5)]).T
im=axh.imshow(NMv.values,cmap="RdBu_r",vmin=-1.2,vmax=1.2,aspect="auto")
axh.set_xticks(range(5));axh.set_xticklabels(NMv.columns,fontsize=8);axh.set_yticks(range(len(NMv)));axh.set_yticklabels(NMv.index,fontsize=7.5)
axh.set_title("predicted metabolite (z) per niche",fontsize=9); plt.colorbar(im,ax=axh,fraction=0.045)
# CytAssist metabolite x protein
dc=np.load(f"{SP}/gbm2_cyt_pred.npz"); pc=dc["pred"]
dp=np.load(f"{SP}/gbm_cyt_protein.npz",allow_pickle=True); P=dp["protein"].astype(float); pn=list(dp["names"])
keep=[i for i,n in enumerate(pn) if not (n.lower().startswith("mouse_ig") or n.lower().startswith("rat_ig"))]
P=P[:,keep]; pn=[pn[i].replace("-1","").replace("-2","_2") for i in keep]
Rp=np.apply_along_axis(rankdata,0,np.log1p(P)); mets=list(GION.items())
Rm=np.apply_along_axis(rankdata,0,np.column_stack([pc[:,i] for _,i in mets]))
C=np.array([[float((zc(Rm[:,mi])*zc(Rp[:,pi])).mean()) for pi in range(len(pn))] for mi in range(len(mets))])
axp=fig9.add_subplot(gs9[0,2]); vmax=np.nanmax(np.abs(C))
im=axp.imshow(C,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
axp.set_xticks(range(len(pn)));axp.set_xticklabels(pn,rotation=90,fontsize=6.5)
axp.set_yticks(range(len(mets)));axp.set_yticklabels([m[0].split("(")[0] for m in mets],fontsize=7.5)
# flag NAA row
naa=[i for i,m in enumerate(mets) if "acetylaspartate" in m[0]]
if naa: axp.add_patch(plt.Rectangle((-0.5,naa[0]-0.5),len(pn),1,fill=False,edgecolor="red",lw=1.5))
axp.set_title("b  CytAssist: predicted metabolite × measured protein (Spearman)   [red = NAA, cross-cohort error]",fontsize=9,loc="left",fontweight="bold")
plt.colorbar(im,ax=axp,fraction=0.03)
fig9.suptitle("Figure 9 | Cross-platform extension to glioblastoma (qualitative): organized metabolic niches and protein co-localization",fontsize=12,fontweight="bold",y=1.02)
fig9.savefig(f"{FIG}/Figure9.png",dpi=160,bbox_inches="tight"); plt.close(fig9); print("saved Figure9.png")
