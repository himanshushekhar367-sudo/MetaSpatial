"""Cross-platform external check (no paired transcriptome for liver, so distribution-level):
do the lipid species/classes MetaSpatial predicts best in DESI reappear in an INDEPENDENT
MALDI lipidome platform on independent liver tissue? Match the 600 MALDI m/z to our annotated
DESI ions and compare class compositions."""
import numpy as np, pandas as pd, json
SP="/home/claude/spatmet"; UP="/mnt/user-data/uploads/spatial metabolism"
ann=pd.read_csv(f"{SP}/metabolite_annotation.csv"); ann["mz"]=ann["mz"].astype(float)
annI=ann[ann.annotation.astype(str).str.strip().str.len()>0].copy()   # 141 annotated DESI ions
desi_mz=ann["mz"].values
# well-predicted DESI classes (from pooled class predictability)
cls_r=json.load(open(f"{SP}/improve_summary.json"))['class_median_r']
lipfam=['sphingolipid','phospholipid','lysophospholipid','fatty acid','lipid headgroup','glycerolipid']
res={}
for tag,f in [('normal(ac)','20260309_mv_liver_ac_bq82b32_lipidome.npz'),('cirrhosis(hc)','20260309_mv_liver_hc_bx81c31_lipidome.npz')]:
    d=np.load(f"{UP}/{f}"); mz=np.asarray(d['mz'],float); inten=d['intensity']
    # match each MALDI m/z to nearest DESI ion within 0.02 Da
    match=0; annmatch=0; classes=[]
    for m in mz:
        j=int(np.argmin(np.abs(desi_mz-m)))
        if abs(desi_mz[j]-m)<=0.02:
            match+=1
            a=ann.iloc[j]
            if isinstance(a['annotation'],str) and a['annotation'].strip():
                annmatch+=1; classes.append(a['class'])
    cc=pd.Series(classes).value_counts()
    liphits=int(sum(cc.get(c,0) for c in lipfam))
    res[tag]=dict(n_maldi=int(len(mz)),mz_range=[float(mz.min()),float(mz.max())],
        matched_desi=match,annotated=annmatch,lipid_class_hits=liphits,
        top_classes=cc.head(6).to_dict())
    print(f"{tag}: {len(mz)} MALDI ions, m/z {mz.min():.0f}-{mz.max():.0f}; {match} match a DESI ion (<=0.02 Da), "
          f"{annmatch} annotated, {liphits} in lipid families")
    print(f"   shared classes: {dict(cc.head(6))}")
json.dump(res,open(f"{SP}/liver_crossplatform.json","w"),indent=2)
print("\nDESI predictability (lipid families predicted best):",{c:cls_r[c][0] for c in lipfam if c in cls_r})
print("saved liver_crossplatform.json")
