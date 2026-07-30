"""
Execute roadmap items on DESIUM paired data:
 (1) putative metabolite ANNOTATION of the 2086 DESI m/z (exact-mass, negative + positive adducts)
 (2) UNCERTAINTY quantification (split-conformal per metabolite; empirical coverage)
 (3) PER-METABOLITE PREDICTABILITY by chemical class (does the model know what it can predict?)
"""
import anndata as ad, numpy as np, pandas as pd, warnings, json
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
warnings.filterwarnings('ignore')
DATA="/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
samples=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
OUT="/home/claude/spatmet";

# ---------- (1) metabolite reference (formula -> monoisotopic) ----------
AM={'C':12.0,'H':1.0078250319,'O':15.9949146221,'N':14.0030740052,'P':30.97376151,'S':31.97207069,'Na':22.98976928,'K':38.96370649,'Cl':34.96885268}
me=0.00054858; mp=1.00727646688
def mass(f):
    import re; return sum(AM[e]*(int(n) if n else 1) for e,n in re.findall(r'([A-Z][a-z]?)(\d*)',f) if e)
# curated metabolite + lipid class reference (name, formula, class)
REF=[("Lactate","C3H6O3","carboxylic acid"),("Pyruvate","C3H4O3","carboxylic acid"),("Succinate","C4H6O4","TCA acid"),
("Fumarate","C4H4O4","TCA acid"),("Malate","C4H6O5","TCA acid"),("Citrate","C6H8O7","TCA acid"),("aKG","C5H6O5","TCA acid"),
("Alanine","C3H7NO2","amino acid"),("Serine","C3H7NO3","amino acid"),("Proline","C5H9NO2","amino acid"),
("Valine","C5H11NO2","amino acid"),("Threonine","C4H9NO3","amino acid"),("Leucine/Ile","C6H13NO2","amino acid"),
("Asparagine","C4H8N2O3","amino acid"),("Aspartate","C4H7NO4","amino acid"),("Glutamine","C5H10N2O3","amino acid"),
("Glutamate","C5H9NO4","amino acid"),("Lysine","C6H14N2O2","amino acid"),("Arginine","C6H14N4O2","amino acid"),
("Histidine","C6H9N3O2","amino acid"),("Phenylalanine","C9H11NO2","amino acid"),("Tyrosine","C9H11NO3","amino acid"),
("Tryptophan","C11H12N2O2","amino acid"),("Methionine","C5H11NO2S","amino acid"),("Cysteine","C3H7NO2S","amino acid"),
("Taurine","C2H7NO3S","amino acid"),("Glutathione","C10H17N3O6S","peptide/redox"),("GSSG","C20H32N6O12S2","peptide/redox"),
("Glucose","C6H12O6","sugar"),("Glucose-6-P","C6H13O9P","sugar phosphate"),("Fructose-1,6-BP","C6H14O12P2","sugar phosphate"),
("Ribose-5-P","C5H11O8P","sugar phosphate"),("Lactose/Maltose","C12H22O11","sugar"),
("AMP","C10H14N5O7P","nucleotide"),("ADP","C10H15N5O10P2","nucleotide"),("ATP","C10H16N5O13P3","nucleotide"),
("GMP","C10H14N5O8P","nucleotide"),("GDP","C10H15N5O11P2","nucleotide"),("GTP","C10H16N5O14P3","nucleotide"),
("UMP","C9H13N2O9P","nucleotide"),("UDP","C9H14N2O12P2","nucleotide"),("NAD+","C21H27N7O14P2","cofactor"),
("NADH","C21H29N7O14P2","cofactor"),("NADP","C21H28N7O17P3","cofactor"),("FAD","C27H33N9O15P2","cofactor"),
("Inosine","C10H12N4O5","nucleoside"),("Hypoxanthine","C5H4N4O","nucleobase"),("Xanthine","C5H4N4O2","nucleobase"),
("Urate","C5H4N4O3","nucleobase"),("Creatine","C4H9N3O2","other"),("Creatinine","C4H7N3O","other"),
("Carnitine","C7H15NO3","acylcarnitine"),("Acetylcarnitine","C9H17NO4","acylcarnitine"),("Choline","C5H13NO","lipid headgroup"),
("Glycerophosphocholine","C8H20NO6P","lipid headgroup"),("Ascorbate","C6H8O6","vitamin"),
# fatty acids
("FA16:0","C16H32O2","fatty acid"),("FA18:0","C18H36O2","fatty acid"),("FA18:1","C18H34O2","fatty acid"),
("FA18:2","C18H32O2","fatty acid"),("FA20:4(AA)","C20H32O2","fatty acid"),("FA22:6(DHA)","C22H32O2","fatty acid"),
("FA20:5(EPA)","C20H30O2","fatty acid"),("FA24:1","C24H46O2","fatty acid"),
# lipids (representative species; MS1 class-level)
("LPC16:0","C24H50NO7P","lysophospholipid"),("LPC18:1","C26H52NO7P","lysophospholipid"),
("PC34:1","C42H82NO8P","phospholipid"),("PC36:2","C44H84NO8P","phospholipid"),("PC38:4","C46H84NO8P","phospholipid"),
("PE36:2","C41H78NO8P","phospholipid"),("PE38:4","C43H78NO8P","phospholipid"),("PI38:4","C47H83O13P","phospholipid"),
("PS36:1","C42H80NO10P","phospholipid"),("PG34:1","C40H77O10P","phospholipid"),("SM d18:1/16:0","C39H79N2O6P","sphingolipid"),
("Cer d18:1/16:0","C34H67NO3","sphingolipid"),("Cholesterol","C27H46O","sterol"),("TAG52:2","C55H102O6","glycerolipid"),
]
# adducts: negative (DESI neg common) + positive
ADD_NEG={"[M-H]-":-mp,"[M+Cl]-":AM['Cl']+me,"[M+FA-H]-":mass("CH2O2")-mp}
ADD_POS={"[M+H]+":mp,"[M+Na]+":AM['Na']-me,"[M+K]+":AM['K']-me}
ref_rows=[]
for name,f,cls in REF:
    M=mass(f)
    for ad_name,dm in {**ADD_NEG,**ADD_POS}.items():
        ref_rows.append((name,cls,ad_name,M+dm))
refdf=pd.DataFrame(ref_rows,columns=["name","class","adduct","mz"]).sort_values("mz").reset_index(drop=True)

mz=np.round(np.asarray(ad.read_h5ad(f"{DATA}/{samples[0]}.h5ad").uns['mz_features'],float),4)
TOL=0.01
ann=[]
rmz=refdf['mz'].values
for i,m in enumerate(mz):
    j=np.argmin(np.abs(rmz-m)); d=rmz[j]-m
    if abs(d)<=TOL: ann.append((m,refdf.iloc[j]['name'],refdf.iloc[j]['class'],refdf.iloc[j]['adduct'],round(d*1000,2)))
    else: ann.append((m,"","unannotated","",np.nan))
anndf=pd.DataFrame(ann,columns=["mz","annotation","class","adduct","ppm_mDa"])
anndf.to_csv(f"{OUT}/metabolite_annotation.csv",index=False)
nA=int((anndf['annotation']!="").sum())
print(f"(1) ANNOTATION: {nA}/{len(mz)} m/z putatively annotated ({100*nA/len(mz):.0f}%) at +-{TOL} Da")
print("    class counts:", anndf[anndf.annotation!=""]['class'].value_counts().to_dict())

# ---------- helpers for CV predictions ----------
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A

# ---------- (2)+(3): per-metabolite predictability + conformal uncertainty (within-sample spatial CV) ----------
allr=[]; cover=[]; classr={}
for s in samples:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad")
    X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],np.float32)); coords=np.asarray(a.obsm['spatial'],float)
    det=(Y>0).mean(0)>0.2; keepidx=np.where(det)[0]; Yd=Y[:,keepidx]
    hvg=np.argsort(X.var(0))[-3000:]; Xh=X[:,hvg]; Adj=nadj(coords); blk=KMeans(5,n_init=3,random_state=0).fit_predict(coords)
    pred=np.zeros_like(Yd); lo=np.zeros_like(Yd); hi=np.zeros_like(Yd)
    for fb in range(5):
        tr=blk!=fb; te=blk==fb
        if te.sum()==0 or tr.sum()<80: continue
        # inner split of train for conformal calibration
        rs=np.random.RandomState(fb).permutation(np.where(tr)[0]); ncal=len(rs)//4; cal=rs[:ncal]; fit=rs[ncal:]
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[fit]).transform(Xh).astype(np.float32)
        T=np.hstack([P,Adj@P,Adj@(Adj@P)])
        reg=Ridge(alpha=200).fit(T[fit],Yd[fit])
        rcal=np.abs(Yd[cal]-reg.predict(T[cal]))            # calib residuals (spots x metab)
        q=np.quantile(rcal,0.9,axis=0)                       # 90% conformal half-width per metabolite
        pt=reg.predict(T[te]); pred[te]=pt; lo[te]=pt-q; hi[te]=pt+q
    # per-metabolite Spearman + empirical 90% coverage
    rlist=[]; clist=[]
    for j in range(Yd.shape[1]):
        if Yd[:,j].std()<1e-9: continue
        r,_=spearmanr(pred[:,j],Yd[:,j]); rlist.append(r if np.isfinite(r) else 0.0)
        cov=np.mean((Yd[:,j]>=lo[:,j])&(Yd[:,j]<=hi[:,j])); clist.append(cov)
        cls=anndf.iloc[keepidx[j]]['class']
        classr.setdefault(cls,[]).append(r if np.isfinite(r) else 0.0)
    allr.append(np.array(rlist)); cover.append(np.array(clist))
    print(f"    {s:18s} median r={np.nanmedian(rlist):+.3f}  empirical 90% coverage={np.mean(clist):.2f}", flush=True)

allr=np.concatenate(allr); cover=np.concatenate(cover)
# uncertainty "knows what it knows": correlate interval width vs accuracy across metabolites (use last sample maps saved)
print(f"\n(2) UNCERTAINTY: mean empirical coverage of 90% conformal interval = {cover.mean():.2f} (nominal 0.90)")
print(f"(3) PREDICTABILITY: overall median r={np.median(allr):+.3f}; frac r>0.3={np.mean(allr>0.3):.2f}")
print("    median Spearman by chemical class (annotated only):")
cl_summary={}
for cls,v in sorted(classr.items(), key=lambda kv:-np.median(kv[1])):
    if cls=="unannotated" or len(v)<5: continue
    cl_summary[cls]=[round(float(np.median(v)),3),len(v)]
    print(f"      {cls:18s} median r={np.median(v):+.3f}  (n={len(v)})")
un=classr.get("unannotated",[])
print(f"      {'unannotated':18s} median r={np.median(un):+.3f}  (n={len(un)})")
json.dump({"coverage":float(cover.mean()),"overall_median_r":float(np.median(allr)),
           "class_median_r":cl_summary,"n_annotated":nA,"frac_annotated":round(nA/len(mz),3)},
          open(f"{OUT}/improve_summary.json","w"),indent=2)
np.savez(f"{OUT}/improve_arrays.npz",allr=allr,cover=cover)
print("\nsaved metabolite_annotation.csv, improve_summary.json, improve_arrays.npz")
