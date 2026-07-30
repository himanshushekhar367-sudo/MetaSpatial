"""Build chemical descriptors for the 225 CCLE metabolites: curated SMILES for polar species +
a constructor that turns the Cx:y lipid / acylcarnitine notation into representative structures.
Saves descriptor table to /tmp/ccle_desc.npz. Robust: any metabolite whose SMILES fails is skipped."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, re
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
UP = "/mnt/user-data/uploads/spatial metabolism"
mets = list(pd.read_csv(f"{UP}/CCLE_metabolomics_20190502.csv", nrows=1).columns[2:])

# ---- curated SMILES for clean polar / small-molecule CCLE names ----
POLAR = {
 "2-aminoadipate":"NC(CCCC(=O)O)C(=O)O","3-phosphoglycerate":"OC(COP(=O)(O)O)C(=O)O","alpha-glycerophosphate":"OCC(O)COP(=O)(O)O",
 "aconitate":"OC(=O)CC(=CC(=O)O)C(=O)O","adenine":"Nc1ncnc2[nH]cnc12","adipate":"OC(=O)CCCCC(=O)O",
 "alpha-ketoglutarate":"OC(=O)CCC(=O)C(=O)O","AMP":"Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O","citrate":"OC(=O)CC(O)(C(=O)O)CC(=O)O",
 "isocitrate":"OC(=O)C(O)C(CC(=O)O)C(=O)O","CMP":"Nc1ccn(C2OC(COP(=O)(O)O)C(O)C2O)c(=O)n1","cystathionine":"NC(CCSCC(N)C(=O)O)C(=O)O",
 "cytidine":"Nc1ccn(C2OC(CO)C(O)C2O)c(=O)n1","dCMP":"Nc1ccn(C2CC(O)C(COP(=O)(O)O)O2)c(=O)n1","glucuronate":"OC1OC(C(=O)O)C(O)C(O)C1O",
 "glutathione oxidized":"NC(CCC(=O)NC(CSSCC(NC(=O)CCC(N)C(=O)O)C(=O)NCC(=O)O)C(=O)NCC(=O)O)C(=O)O",
 "glutathione reduced":"NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O","GMP":"Nc1nc2c(ncn2C2OC(COP(=O)(O)O)C(O)C2O)c(=O)[nH]1",
 "guanosine":"Nc1nc2c(ncn2C2OC(CO)C(O)C2O)c(=O)[nH]1","hippurate":"OC(=O)CNC(=O)c1ccccc1","hypoxanthine":"O=c1[nH]cnc2[nH]cnc12",
 "inosine":"O=c1[nH]cnc2c1ncn2C1OC(CO)C(O)C1O","kynurenine":"NC(CC(=O)c1ccccc1N)C(=O)O","lactate":"CC(O)C(=O)O",
 "lactose":"OCC1OC(OC2C(O)C(O)C(O)OC2CO)C(O)C(O)C1O","malate":"OC(=O)CC(O)C(=O)O",
 "NAD":"NC(=O)c1ccc[n+](C2OC(COP(=O)([O-])OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1",
 "oxalate":"OC(=O)C(=O)O","pantothenate":"CC(C)(CO)C(O)C(=O)NCCC(=O)O","PEP":"OC(=O)C(=C)OP(=O)(O)O",
 "sorbitol":"OCC(O)C(O)C(O)C(O)CO","sucrose":"OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O","thymine":"Cc1c[nH]c(=O)[nH]c1=O",
 "UMP":"OC1C(O)C(COP(=O)(O)O)OC1n1ccc(=O)[nH]c1=O","uracil":"O=c1cc[nH]c(=O)[nH]1","urate":"O=c1[nH]c(=O)c2[nH]c(=O)[nH]c2[nH]1",
 "uridine":"OCC1OC(n2ccc(=O)[nH]c2=O)C(O)C1O","xanthine":"O=c1[nH]c(=O)c2[nH]cnc2[nH]1","phosphocreatine":"CN(CC(=O)O)C(=N)NP(=O)(O)O",
 "6-phosphogluconate":"OCC(O)C(O)C(O)C(O)C(=O)O","alpha-hydroxybutyrate":"CCC(O)C(=O)O","2-hydroxyglutarate":"OC(=O)CCC(O)C(=O)O",
 "inositol":"OC1C(O)C(O)C(O)C(O)C1O","malondialdehyde":"O=CCC=O","glycine":"NCC(=O)O","alanine":"CC(N)C(=O)O","serine":"OCC(N)C(=O)O",
 "threonine":"CC(O)C(N)C(=O)O","methionine":"CSCCC(N)C(=O)O","aspartate":"OC(=O)CC(N)C(=O)O","glutamate":"OC(=O)CCC(N)C(=O)O",
 "asparagine":"NC(=O)CC(N)C(=O)O","glutamine":"NC(=O)CCC(N)C(=O)O","histidine":"OC(=O)C(N)Cc1c[nH]cn1","arginine":"NC(=N)NCCCC(N)C(=O)O",
 "lysine":"NCCCCC(N)C(=O)O","valine":"CC(C)C(N)C(=O)O","leucine":"CC(C)CC(N)C(=O)O","isoleucine":"CCC(C)C(N)C(=O)O",
 "phenylalanine":"OC(=O)C(N)Cc1ccccc1","tyrosine":"OC(=O)C(N)Cc1ccc(O)cc1","tryptophan":"OC(=O)C(N)Cc1c[nH]c2ccccc12",
 "proline":"OC(=O)C1CCCN1","ornithine":"NCCCC(N)C(=O)O","citrulline":"NC(=O)NCCCC(N)C(=O)O","taurine":"NCCS(=O)(=O)O",
 "5-HIAA":"OC(=O)Cc1c[nH]c2ccc(O)cc12","serotonin":"NCCc1c[nH]c2ccc(O)cc12","GABA":"NCCCC(=O)O","acetylglycine":"CC(=O)NCC(=O)O",
 "dimethylglycine":"CN(C)CC(=O)O","homocysteine":"NC(CCS)C(=O)O","allantoin":"NC(=O)NC1NC(=O)NC1=O","anthranilic acid":"Nc1ccccc1C(=O)O",
 "kynurenic acid":"OC(=O)c1cc(=O)c2ccccc2[nH]1","carnosine":"NCCC(=O)NC(Cc1c[nH]cn1)C(=O)O","thiamine":"Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1",
 "niacinamide":"NC(=O)c1cccnc1","betaine":"C[N+](C)(C)CC(=O)[O-]","choline":"C[N+](C)(C)CCO","alpha-glycerophosphocholine":"OCC(O)COP(=O)([O-])OCC[N+](C)(C)C",
 "acetylcholine":"CC(=O)OCC[N+](C)(C)C","creatine":"CN(CC(=O)O)C(=N)N","creatinine":"CN1CC(=O)N=C1N","trimethylamine-N-oxide":"C[N+](C)(C)[O-]",
 "adenosine":"Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O","thymidine":"Cc1cn(C2CC(O)C(CO)O2)c(=O)[nH]c1=O","xanthosine":"O=c1[nH]c(=O)c2nc(n(c2[nH]1)C1OC(CO)C(O)C1O)",
 "2-deoxyadenosine":"Nc1ncnc2c1ncn2C1CC(O)C(CO)O1","2-deoxycytidine":"Nc1ccn(C2CC(O)C(CO)O2)c(=O)n1","cAMP":"Nc1ncnc2c1ncn2C1OC2COP(=O)(O)OC2C1O",
 "cotinine":"CN1CCC(c2cccnc2)C1=O","pipecolic acid":"OC(=O)C1CCCCN1","pyroglutamic acid":"OC(=O)C1CCC(=O)N1","1-methylnicotinamide":"C[n+]1cccc(C(N)=O)c1",
 "butyrobetaine":"C[N+](C)(C)CCCC(=O)[O-]","putrescine":"NCCCCN","carnitine":"C[N+](C)(C)CC(O)CC(=O)[O-]","sarcosine":"CNCC(=O)O",
 "beta-alanine":"NCCC(=O)O","5-adenosylhomocysteine":"Nc1ncnc2c1ncn2C1OC(CSCCC(N)C(=O)O)C(O)C1O","4-pyridoxate":"Cc1ncc(CO)c(C(=O)O)c1O",
 "N-carbamoyl-beta-alanine":"NC(=O)NCCC(=O)O","methionine sulfoxide":"CS(=O)CCC(N)C(=O)O","anserine":"Cn1cnc(CC(NC(=O)CCN)C(=O)O)c1",
}
# ---- lipid / acylcarnitine constructors from Cx:y notation ----
def acyl(n, d):                                   # O-acyl fragment: C(=O) + (n-1) carbons, d C=C
    s = "C(=O)" + "C"*(max(n-1,1))
    for _ in range(d): s = s.replace("CC", "C=C", 1)
    return s
CHOL = "CC(C)CCCC(C)C1CCC2C1(C)CCC1C2CC=C2CC(CCC12C)"  # cholesteryl (attach ester O)
def lipid_smiles(name):
    m = re.match(r"C(\d+):(\d+)\s+(LPC|LPE|PC|SM|DAG|CE|TAG)", name)
    if not m: return None
    n, d, cls = int(m.group(1)), int(m.group(2)), m.group(3)
    if cls == "LPC": return "OCC(COP(=O)([O-])OCC[N+](C)(C)C)O" + acyl(n, d)
    if cls == "LPE": return "OCC(COP(=O)(O)OCCN)O" + acyl(n, d)
    if cls == "PC":
        n1, d1 = n//2, d//2; n2, d2 = n-n1, d-d1
        return "O"+acyl(n1,d1)[1:]+"CC(COP(=O)([O-])OCC[N+](C)(C)C)O"+acyl(n2,d2)  # two esters
    if cls == "DAG":
        n1,d1=n//2,d//2; n2,d2=n-n1,d-d1
        return "OCC(CO"+acyl(n1,d1)+")O"+acyl(n2,d2)
    if cls == "CE": return CHOL + "O" + acyl(n, d)
    if cls == "SM":
        na=max(n-18,2); da=max(d-1,0)
        return "CCCCCCCCCCCCCC=CC(O)C(COP(=O)([O-])OCC[N+](C)(C)C)N"+acyl(na,da)
    if cls == "TAG":
        n1=n//3;n2=n//3;n3=n-n1-n2; d1=d//3;d2=d//3;d3=d-d1-d2
        return "C(CO"+acyl(n1,d1)+")(CO"+acyl(n2,d2)+")O"+acyl(n3,d3)
    return None
CARN = {"acetyl":(2,0),"propionyl":(3,0),"butyryl":(4,0),"hexanoyl":(6,0),"heptanoyl":(7,0),"lauroyl":(12,0),
        "myristoyl":(14,0),"palmitoyl":(16,0),"stearoyl":(18,0),"oleyl":(18,1),"arachidonyl":(20,4),"malonyl":(3,0),"valeryl":(5,0)}
def carn_smiles(name):
    for k,(n,d) in CARN.items():
        if name.lower().startswith(k):
            return "[O-]C(=O)CC(C[N+](C)(C)C)O"+acyl(n,d)
    return None

DESC = ["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors","FractionCSP3","NumRotatableBonds","RingCount"]
def desc(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol)]

names, D, kind, mass, smis = [], [], [], [], []
for nm in mets:
    smi = POLAR.get(nm) or lipid_smiles(nm) or carn_smiles(nm)
    if smi is None: continue
    dd = desc(smi)
    if dd is None: continue
    mol = Chem.MolFromSmiles(smi)
    names.append(nm); D.append(dd); mass.append(Descriptors.ExactMolWt(mol)); smis.append(smi)
    kind.append("lipid" if lipid_smiles(nm) else ("carn" if carn_smiles(nm) else "polar"))
D = np.array(D)
np.savez("/tmp/ccle_desc.npz", names=np.array(names), D=D, cols=np.array(DESC),
         mass=np.array(mass), smiles=np.array(smis))
from collections import Counter
print(f"mapped {len(names)}/225 metabolites  |  by kind: {dict(Counter(kind))}")
print("descriptor cols:", DESC)
print("example logP range:", round(D[:,1].min(),1), "to", round(D[:,1].max(),1))
PY = None
