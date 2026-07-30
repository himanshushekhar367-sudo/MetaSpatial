#!/usr/bin/env python3
"""
Cross-patient reproducibility of the HCC predicted-metabolite signature.

Signature discovered in HCC-2 Normal->PVTT: glutathione / ascorbate / long-chain PUFA (DHA, AA) UP,
saturated fatty acids (palmitate, stearate) + hexose DOWN.

PVTT (P subsection) exists ONLY in HCC-2. HCC-1/3/4 have N (Normal) and T (Tumour) but no PVTT;
HCC-5 is A/B/C/D tumour zones with no Normal reference -> excluded.
So the portable test is Normal(N) -> Tumour(T) across HCC-1,2,3,4, plus HCC-2's N->PVTT for reference.
"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

UP="/mnt/user-data/uploads/spatial metabolism"; OUT="hcc_repro_out"; os.makedirs(OUT,exist_ok=True)
PATS=["HCC-1","HCC-2","HCC-3","HCC-4"]                      # have both N and T
d0=np.load("xcancer/HCC-2_pred.npz"); mz=np.asarray(d0["mz"],float)

DB={'Glutathione(GSH)':307.0838,'Ascorbate':176.0321,'Arachidonate(FA20:4)':304.2402,'DHA(FA22:6)':328.2402,
    'Glutamate':147.0532,'Palmitate(FA16:0)':256.2402,'Stearate(FA18:0)':284.2715,'Hexose(glucose)':180.0634,
    'Oleate(FA18:1)':282.2559,'Linoleate(FA18:2)':280.2402}
EXPECT={'Glutathione(GSH)':+1,'Ascorbate':+1,'Arachidonate(FA20:4)':+1,'DHA(FA22:6)':+1,'Glutamate':+1,
        'Palmitate(FA16:0)':-1,'Stearate(FA18:0)':-1,'Hexose(glucose)':-1,'Oleate(FA18:1)':0,'Linoleate(FA18:2)':0}
ADD=[("[M-H]-",-1.0073),("[M+Cl]-",34.9694),("[M+HCOO]-",44.9982)]
def ion_index(mass):                                        # prefer [M-H]-
    for _,dl in ADD:
        i=int(np.argmin(np.abs(mz-(mass+dl))))
        if abs(mz[i]-(mass+dl))<=0.01: return i
    return None
IONS={n:ion_index(m) for n,m in DB.items()}

def load(pid):
    pred=np.load(f"xcancer/{pid}_pred.npz")["pred"].astype(float)
    bc=[l.strip() for l in open(f"{UP}/{pid}-expr_barcodes.txt")]
    meta=pd.read_csv(f"{UP}/{pid}-expr_meta.csv",index_col=0)
    aligned = list(meta.index)==bc
    meta=meta.reindex(bc)
    suff=meta["sample.ident"].astype(str).str.replace(pid,"",regex=False).str.lstrip("-").values
    return pred, suff, aligned

# ---- compute N->T (all patients) and N->P (HCC-2) deltas + significance ----
cols=[]; colnames=[]
sig={}   # (patient, contrast) -> {metabolite: (delta, cohen_d, q)}
for pid in PATS:
    pred,suff,aligned=load(pid)
    if not aligned: print(f"WARNING {pid}: meta order != barcode order (pred alignment suspect)")
    for cl,(A,B) in {"N→T":("N","T"), **({"N→P":("N","P")} if "P" in set(suff) else {})}.items():
        mA=suff==A; mB=suff==B
        col={}; res={}
        for name,i in IONS.items():
            if i is None: col[name]=np.nan; continue
            a,b=pred[mA,i],pred[mB,i]
            delta=b.mean()-a.mean()
            cd=delta/np.sqrt((a.var()+b.var())/2+1e-9)
            t,p=ttest_ind(b,a,equal_var=False)
            col[name]=delta; res[name]=(delta,cd,p)
        cols.append(col); colnames.append(f"{pid}\n{cl}\n(n {mA.sum()}v{mB.sum()})"); sig[(pid,cl)]=res
        print(f"{pid} {cl}: N={mA.sum()} vs {B if cl=='N→T' else 'P'}={mB.sum()}")

M=pd.DataFrame(cols,index=colnames).T                      # metabolites x (patient,contrast)
M=M.reindex(list(DB))                                       # signature order
M=M[[c for c in M.columns if "N→T" in c] + [c for c in M.columns if "N→P" in c]]  # N->T first, PVTT last
M.to_csv(f"{OUT}/signature_deltas.csv")

# ---- concordance across the 4 N->T contrasts ----
NT=[c for c in M.columns if "N→T" in c]
print("\n=== CONCORDANCE (Normal->Tumour, 4 patients) ===")
concord={}
for name in DB:
    if EXPECT[name]==0: continue
    signs=np.sign(M.loc[name,NT].values); k=int((signs==EXPECT[name]).sum())
    concord[name]=k
    exp="UP" if EXPECT[name]>0 else "DOWN"
    print(f"  {name:22s} expected {exp:4s}: {k}/4 patients concordant   deltas={np.round(M.loc[name,NT].values,2)}")
print("  (controls, expected ~flat):")
for name in DB:
    if EXPECT[name]==0:
        print(f"  {name:22s} deltas={np.round(M.loc[name,NT].values,2)}")

# ---- figure: heatmap ----
fig,ax=plt.subplots(figsize=(9.5,7))
vmax=np.nanmax(np.abs(M.values))
im=ax.imshow(M.values,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns,fontsize=8)
ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index,fontsize=9)
for yi,name in enumerate(M.index):
    for xi,cn in enumerate(M.columns):
        pid=cn.split("\n")[0]; cl=cn.split("\n")[1]
        v=M.values[yi,xi]
        if np.isnan(v): continue
        q=sig[(pid,cl)][name][2]; star="*" if q<0.05 else ""
        ax.text(xi,yi,f"{v:+.2f}{star}",ha="center",va="center",fontsize=7.5,
                color="white" if abs(v)>vmax*0.55 else "black")
# separator before HCC-2 N->P
if any("N→P" in c for c in M.columns):
    xi=[i for i,c in enumerate(M.columns) if "N→P" in c][0]
    ax.axvline(xi-0.5,color="k",lw=1.5)
cb=plt.colorbar(im,fraction=0.046,pad=0.04); cb.set_label("Δ predicted (log)  tumour/PVTT − normal",fontsize=9)
ax.set_title("Reproducibility of the HCC predicted-metabolite signature\n(N→T across 4 patients; HCC-2 N→PVTT at right; * q<0.05)",fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/HCC_signature_reproducibility.png",dpi=155,bbox_inches="tight")
print(f"\nsaved {OUT}/HCC_signature_reproducibility.png and signature_deltas.csv")
