#!/usr/bin/env python3
"""
HCC patient-2 subsection metabolome analysis with MetaSpatial predictions.

Subsections in HCC-2 (from sample.ident suffix; biology from the `Type` column):
    N = adjacent Normal liver
    L = Leading edge (tumour/normal interface)
    P = PVTT  (portal vein tumour thrombus)   <-- the aggressive vascular-invasion region
    T = primary Tumour

Inputs (all produced earlier / on disk):
    xcancer/HCC-2_pred.npz         # pred (spots x 2086 predicted metabolome, log scale) + mz  [from hcc_gbm_predict.py]
    <UP>/HCC-2-expr_barcodes.txt   # spot order of `pred`
    <UP>/HCC-2-expr_meta.csv       # sample.ident (subsection), Type, imagerow/imagecol
    permz_spearman.npz             # per-ion within-sample DESIUM predictability (mean Spearman across 7 tumours)

What it does:
    * mean predicted intensity per metabolite per subsection (N,L,P,T)
    * N->P (normal->PVTT) differential: delta(log), Cohen's d, Welch t, BH-FDR q
    * annotate ions by exact neutral mass + negative-mode adducts (same DB as annotate_predictable.py)
    * cross-reference each mover with its predictability r  -> flag what MetaSpatial can *reliably* map
    * figure: GSH + top-lipid spatial maps across N/L/P/T, top N->P movers, N->L->P->T trajectory
"""
import numpy as np, pandas as pd, os, argparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="xcancer/HCC-2_pred.npz")
ap.add_argument("--up",   default="/mnt/user-data/uploads/spatial metabolism")
ap.add_argument("--rel",  default="permz_spearman.npz")
ap.add_argument("--out",  default="hcc2_out")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

# ---- load predicted metabolome + spot metadata ----
d = np.load(a.pred); pred = d["pred"].astype(float); mz = np.asarray(d["mz"], float)
bc = [l.strip() for l in open(f"{a.up}/HCC-2-expr_barcodes.txt")]
meta = pd.read_csv(f"{a.up}/HCC-2-expr_meta.csv", index_col=0).reindex(bc)
suff = meta["sample.ident"].astype(str).str.replace("HCC-2","",regex=False).str.lstrip("-").values
xy   = meta[["imagecol","imagerow"]].values.astype(float)   # x=imagecol, y=imagerow
assert pred.shape[0] == len(bc) == len(meta), "spot alignment broke"

# per-ion predictability (same 2086 mz axis)
rel = np.load(a.rel); relS = np.asarray(rel["meanS"], float)
assert len(relS) == len(mz)

SUBS = ["N","L","P","T"]; SUBNAME = {"N":"Normal","L":"Leading edge","P":"PVTT","T":"Tumour"}
mask = {s: suff == s for s in SUBS}
mean_by = {s: pred[mask[s]].mean(0) for s in SUBS}

# ---- annotation: exact neutral mass + negative-mode adducts (identical DB to annotate_predictable.py) ----
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
ADD=[("[M-H]-",-1.0073),("[M+Cl]-",34.9694),("[M+HCOO]-",44.9982)]
def annotate(v):
    hits=[f"{n} {ad}" for n,ms in DB.items() for ad,dl in ADD if abs(v-(ms+dl))<=0.01]
    return "; ".join(hits)
ann = np.array([annotate(v) for v in mz])
def ion_index(name, adduct="[M-H]-"):
    dl=dict(ADD)[adduct]; target=DB[name]+dl
    i=int(np.argmin(np.abs(mz-target)))
    return i if abs(mz[i]-target)<=0.01 else None

# ---- N -> P (normal -> PVTT) differential ----
def contrast(sA, sB):
    A, B = pred[mask[sA]], pred[mask[sB]]
    delta = B.mean(0) - A.mean(0)                                   # log-fold-change B vs A
    ps = A.std(0)**2; qs = B.std(0)**2
    d_cohen = delta / np.sqrt((ps+qs)/2 + 1e-9)
    t, p = ttest_ind(B, A, axis=0, equal_var=False)
    # BH-FDR
    order=np.argsort(p); q=np.empty_like(p); n=len(p)
    q[order]=np.minimum.accumulate((p[order]*n/np.arange(n,0,-1))[::-1])[::-1]
    return delta, d_cohen, p, q

dNP, dNPd, pNP, qNP = contrast("N","P")

df = pd.DataFrame({
    "mz": np.round(mz,4), "annotation": ann, "predictability_r": np.round(relS,3),
    "mean_N": mean_by["N"], "mean_L": mean_by["L"], "mean_P": mean_by["P"], "mean_T": mean_by["T"],
    "delta_NtoP_log": dNP, "cohens_d_NtoP": dNPd, "p_NtoP": pNP, "q_NtoP": qNP,
})
df.to_csv(f"{a.out}/HCC2_NtoP_differential_ALL.csv", index=False)

# reliably-mappable movers = annotated AND per-ion predictability above a floor
ANN = df["annotation"]!=""
REL = df["predictability_r"] >= 0.30           # DESIUM within-sample reliable ions
sig = df["q_NtoP"] < 0.05
rel_movers = df[ANN & REL].copy().sort_values("delta_NtoP_log")
rel_movers.to_csv(f"{a.out}/HCC2_NtoP_reliable_annotated.csv", index=False)

print("="*78)
print("HCC-2  Normal(N) -> PVTT(P)  predicted-metabolite differential")
print(f"spots: N={mask['N'].sum()}  L={mask['L'].sum()}  P={mask['P'].sum()}  T={mask['T'].sum()}")
print(f"ions: {len(mz)} total | {int(ANN.sum())} annotated | {int((ANN&REL).sum())} annotated & reliably-predicted (r>=0.30)")
print("-"*78)
print("TOP RELIABLE ANNOTATED movers N->P (by |delta log|, q<0.05 flagged *):")
tops = rel_movers.reindex(rel_movers["delta_NtoP_log"].abs().sort_values(ascending=False).index)
for _,r in tops.head(18).iterrows():
    star = "*" if r["q_NtoP"]<0.05 else " "
    arrow = "UP  " if r["delta_NtoP_log"]>0 else "DOWN"
    print(f"  {arrow}{star} d(log)={r['delta_NtoP_log']:+.3f}  d'={r['cohens_d_NtoP']:+.2f}  r={r['predictability_r']:+.2f}  m/z {r['mz']:9.4f}  {r['annotation']}")

# ---- figure ----
gsh = ion_index("Glutathione(GSH)")
# top reliable lipid mover for the 2nd map (largest |delta| among FA/lipid annotated reliable ions)
lipid_kw = ("FA","DHA","Oleate","Linoleate","Palmitate","Stearate","Arachid","Sphing","Cholesterol")
cand = tops[tops["annotation"].str.contains("|".join(lipid_kw))]
lip_i = int(np.argmin(np.abs(mz - cand.iloc[0]["mz"]))) if len(cand) else int(np.argmax(np.abs(dNP)))
maps = [("Predicted glutathione (GSH)", gsh), (f"Predicted {ann[lip_i] or ('m/z %.3f'%mz[lip_i])}", lip_i)]

fig = plt.figure(figsize=(15, 11)); gs = GridSpec(3, 4, figure=fig, height_ratios=[1,1,1.15], hspace=0.42, wspace=0.28)
for ri,(title,ion) in enumerate(maps):
    if ion is None: continue
    vals = pred[:,ion]
    vmin,vmax = np.percentile(vals,2), np.percentile(vals,98)
    for ci,s in enumerate(SUBS):
        ax=fig.add_subplot(gs[ri,ci]); m=mask[s]
        sc=ax.scatter(xy[m,0], xy[m,1], c=vals[m], s=5, cmap="magma", vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{s} · {SUBNAME[s]}", fontsize=10, fontweight="bold" if s=="P" else "normal",
                     color="crimson" if s=="P" else "black")
        if ci==0: ax.set_ylabel(title, fontsize=10)
    cax=fig.add_axes([0.915, 0.66-ri*0.285, 0.011, 0.19]); plt.colorbar(sc,cax=cax); cax.tick_params(labelsize=7)

# bottom-left: top N->P reliable movers bar
axb=fig.add_subplot(gs[2,0:2])
bar=tops.head(12).iloc[::-1]
labels=[a2.split(";")[0].replace(" [M-H]-","") for a2 in bar["annotation"]]
colors=["#c0392b" if v>0 else "#2471a3" for v in bar["delta_NtoP_log"]]
axb.barh(range(len(bar)), bar["delta_NtoP_log"], color=colors, edgecolor="black", linewidth=0.4)
axb.set_yticks(range(len(bar))); axb.set_yticklabels(labels, fontsize=8)
axb.axvline(0,color="k",lw=0.6); axb.set_xlabel("Δ predicted (log)  Normal → PVTT", fontsize=9)
axb.set_title("Top reliably-mapped N→PVTT movers", fontsize=10)
axb.tick_params(axis="x",labelsize=8)

# bottom-right: N->L->P->T trajectory for key reliable metabolites
axt=fig.add_subplot(gs[2,2:4])
key=["Glutathione(GSH)","DHA(FA22:6)","Arachidonate(FA20:4)","Oleate(FA18:1)","Palmitate(FA16:0)","Inosine"]
for name in key:
    i=ion_index(name)
    if i is None: continue
    traj=[mean_by[s][i] for s in SUBS]
    traj=[(t-traj[0]) for t in traj]   # relative to Normal
    axt.plot(range(4), traj, marker="o", label=name.split("(")[0], linewidth=1.8)
axt.axhline(0,color="grey",lw=0.6,ls="--")
axt.set_xticks(range(4)); axt.set_xticklabels([f"{s}\n{SUBNAME[s]}" for s in SUBS], fontsize=8)
axt.set_ylabel("Δ predicted vs Normal (log)", fontsize=9)
axt.set_title("Metabolic trajectory  Normal→Leading→PVTT→Tumour", fontsize=10)
axt.legend(fontsize=7, ncol=2, frameon=False)

fig.suptitle("HCC-2 predicted spatial metabolome across subsections (MetaSpatial)", fontsize=13, fontweight="bold", y=0.995)
fig.savefig(f"{a.out}/HCC2_subsection_metabolome.png", dpi=155, bbox_inches="tight")
print(f"\nsaved {a.out}/HCC2_subsection_metabolome.png")
print(f"saved {a.out}/HCC2_NtoP_differential_ALL.csv  and  HCC2_NtoP_reliable_annotated.csv")
