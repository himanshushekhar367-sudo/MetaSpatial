#!/usr/bin/env python3
"""
Robustness check: Normal-CELL vs Tumour-CELL contrast (per-spot Type label, pooled across ALL subsections),
compared against the subsection-level Normal(N)->Tumour(T) contrast.
If the saturated-FA-down / GSH-up pattern survives BOTH spot definitions, the signature is not an artifact
of region mixture.
"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

UP="/mnt/user-data/uploads/spatial metabolism"; OUT="hcc_repro_out"; os.makedirs(OUT,exist_ok=True)
PATS=["HCC-1","HCC-2","HCC-3","HCC-4"]
mz=np.asarray(np.load("xcancer/HCC-2_pred.npz")["mz"],float)

DB={'Glutathione(GSH)':307.0838,'Ascorbate':176.0321,'Arachidonate(FA20:4)':304.2402,'DHA(FA22:6)':328.2402,
    'Glutamate':147.0532,'Palmitate(FA16:0)':256.2402,'Stearate(FA18:0)':284.2715,'Hexose(glucose)':180.0634,
    'Oleate(FA18:1)':282.2559,'Linoleate(FA18:2)':280.2402}
EXPECT={'Glutathione(GSH)':+1,'Ascorbate':+1,'Arachidonate(FA20:4)':+1,'DHA(FA22:6)':+1,'Glutamate':+1,
        'Palmitate(FA16:0)':-1,'Stearate(FA18:0)':-1,'Hexose(glucose)':-1,'Oleate(FA18:1)':0,'Linoleate(FA18:2)':0}
ADD=[("[M-H]-",-1.0073),("[M+Cl]-",34.9694),("[M+HCOO]-",44.9982)]
def ion_index(mass):
    for _,dl in ADD:
        i=int(np.argmin(np.abs(mz-(mass+dl))))
        if abs(mz[i]-(mass+dl))<=0.01: return i
    return None
IONS={n:ion_index(m) for n,m in DB.items()}

def load(pid):
    pred=np.load(f"xcancer/{pid}_pred.npz")["pred"].astype(float)
    bc=[l.strip() for l in open(f"{UP}/{pid}-expr_barcodes.txt")]
    meta=pd.read_csv(f"{UP}/{pid}-expr_meta.csv",index_col=0).reindex(bc)
    suff=meta["sample.ident"].astype(str).str.replace(pid,"",regex=False).str.lstrip("-").values
    typ=meta["Type"].astype(str).values
    return pred,suff,typ

def contrast(pred,mA,mB):
    out={}
    for name,i in IONS.items():
        if i is None: out[name]=(np.nan,np.nan); continue
        a,b=pred[mA,i],pred[mB,i]
        delta=b.mean()-a.mean(); t,p=ttest_ind(b,a,equal_var=False)
        out[name]=(delta,p)
    return out

typ_cols={}; sub_cols={}; sig={}
for pid in PATS:
    pred,suff,typ=load(pid)
    cT=contrast(pred, typ=="Normal", typ=="Tumor")            # Type-level: normal-cell vs tumour-cell
    cS=contrast(pred, suff=="N", suff=="T")                    # subsection-level
    typ_cols[pid]={k:v[0] for k,v in cT.items()}; sub_cols[pid]={k:v[0] for k,v in cS.items()}
    sig[pid]={k:v[1] for k,v in cT.items()}
    print(f"{pid}: Type-level Normal-cell={int((typ=='Normal').sum())} vs Tumour-cell={int((typ=='Tumor').sum())}")

T=pd.DataFrame(typ_cols).reindex(list(DB)); S=pd.DataFrame(sub_cols).reindex(list(DB))
T.to_csv(f"{OUT}/typelevel_deltas.csv")

# ---- concordance comparison ----
print("\n=== CONCORDANCE: subsection-level vs Type-level (Normal->Tumour, 4 patients) ===")
print(f"{'metabolite':22s} {'expect':6s} {'subsection':>11s} {'Type-level':>11s}")
rows=[]
for name in DB:
    if EXPECT[name]==0: continue
    ks=int((np.sign(S.loc[name].values)==EXPECT[name]).sum())
    kt=int((np.sign(T.loc[name].values)==EXPECT[name]).sum())
    exp="UP" if EXPECT[name]>0 else "DOWN"
    rows.append((name,exp,ks,kt))
    print(f"{name:22s} {exp:6s} {ks:>8d}/4 {kt:>8d}/4   Type deltas={np.round(T.loc[name].values,2)}")

# ---- figure: Type-level heatmap ----
fig,ax=plt.subplots(figsize=(8.2,7))
vmax=np.nanmax(np.abs(T.values))
im=ax.imshow(T.values,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
ax.set_xticks(range(len(T.columns))); ax.set_xticklabels(T.columns,fontsize=9)
ax.set_yticks(range(len(T.index))); ax.set_yticklabels(T.index,fontsize=9)
for yi,name in enumerate(T.index):
    for xi,pid in enumerate(T.columns):
        v=T.values[yi,xi]
        if np.isnan(v): continue
        star="*" if sig[pid][name]<0.05 else ""
        ax.text(xi,yi,f"{v:+.2f}{star}",ha="center",va="center",fontsize=8,
                color="white" if abs(v)>vmax*0.55 else "black")
cb=plt.colorbar(im,fraction=0.046,pad=0.04); cb.set_label("Δ predicted (log)  tumour-cell − normal-cell",fontsize=9)
ax.set_title("Robustness: Normal-CELL vs Tumour-CELL (per-spot Type, pooled)\nHCC-1..4  (* q<0.05)",fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/HCC_typelevel_reproducibility.png",dpi=155,bbox_inches="tight")
print(f"\nsaved {OUT}/HCC_typelevel_reproducibility.png and typelevel_deltas.csv")
