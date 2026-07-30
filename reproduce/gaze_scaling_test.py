"""
Scaling test for the metabolite-conditioned head (GAZE idea) on DESIUM.
Does the chemistry-conditioned zero-shot model close the gap on the mean-pattern baseline
as the number of training metabolites grows?  If yes -> CCLE-scale (225) is worth it.

For N in a grid, subsample N annotated metabolites, run leave-one-metabolite-out with the
bilinear head (spatial features (x) chemical descriptors) and the chemistry-blind mean-pattern
baseline; report median LOMO Spearman and the (bilinear - baseline) gap vs N.
"""
import os, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, anndata as ad, scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DES = "/mnt/user-data/uploads/spatial metabolism/DESIUM/Correlation_NMF_analysis/data"
DS = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
NHVG, NPC, K, DF = 3000, 100, 6, 30

# broad curated metabolite panel (name -> SMILES); RDKit computes exact mass. Invalid SMILES are skipped.
SMI = {
 "Glycine":"NCC(=O)O","Alanine":"CC(N)C(=O)O","Serine":"OCC(N)C(=O)O","Threonine":"CC(O)C(N)C(=O)O",
 "Valine":"CC(C)C(N)C(=O)O","Leucine":"CC(C)CC(N)C(=O)O","Isoleucine":"CCC(C)C(N)C(=O)O","Proline":"OC(=O)C1CCCN1",
 "Phenylalanine":"OC(=O)C(N)Cc1ccccc1","Tyrosine":"OC(=O)C(N)Cc1ccc(O)cc1","Tryptophan":"OC(=O)C(N)Cc1c[nH]c2ccccc12",
 "Histidine":"OC(=O)C(N)Cc1c[nH]cn1","Lysine":"NCCCCC(N)C(=O)O","Arginine":"NC(=N)NCCCC(N)C(=O)O",
 "Aspartate":"OC(=O)CC(N)C(=O)O","Glutamate":"OC(=O)CCC(N)C(=O)O","Asparagine":"NC(=O)CC(N)C(=O)O",
 "Glutamine":"NC(=O)CCC(N)C(=O)O","Methionine":"CSCCC(N)C(=O)O","Cysteine":"SCC(N)C(=O)O",
 "Ornithine":"NCCCC(N)C(=O)O","Citrulline":"NC(=O)NCCCC(N)C(=O)O","Taurine":"NCCS(=O)(=O)O","GABA":"NCCCC(=O)O",
 "Betaine":"C[N+](C)(C)CC(=O)[O-]","Creatine":"CN(CC(=O)O)C(=N)N","Creatinine":"CN1CC(=O)N=C1N","Sarcosine":"CNCC(=O)O",
 "Lactate":"CC(O)C(=O)O","Pyruvate":"CC(=O)C(=O)O","Succinate":"OC(=O)CCC(=O)O","Fumarate":"OC(=O)/C=C/C(=O)O",
 "Malate":"OC(=O)CC(O)C(=O)O","Citrate":"OC(=O)CC(O)(C(=O)O)CC(=O)O","aKG":"OC(=O)CCC(=O)C(=O)O",
 "2HG":"OC(=O)CCC(O)C(=O)O","Itaconate":"C=C(CC(=O)O)C(=O)O","3HB":"CC(O)CC(=O)O","Acetoacetate":"CC(=O)CC(=O)O",
 "Glycolate":"OCC(=O)O","Glycerate":"OCC(O)C(=O)O","Malonate":"OC(=O)CC(=O)O","Glutarate":"OC(=O)CCCC(=O)O",
 "Adipate":"OC(=O)CCCCC(=O)O","Hippurate":"OC(=O)CNC(=O)c1ccccc1","Kynurenate":"OC(=O)c1cc(=O)c2ccccc2[nH]1",
 "Glucose":"OCC1OC(O)C(O)C(O)C1O","Fructose":"OCC1(O)OCC(O)C(O)C1O","Mannose":"OCC1OC(O)C(O)C(O)C1O",
 "Ribose":"OCC1OC(O)C(O)C1O","Inositol":"OC1C(O)C(O)C(O)C(O)C1O","Sorbitol":"OCC(O)C(O)C(O)C(O)CO",
 "G6P":"OCC1OC(O)C(O)C(O)C1OP(=O)(O)O","F6P":"OC1OCC(OP(=O)(O)O)C(O)C1O","R5P":"OCC(O)C(O)C(O)COP(=O)(O)O",
 "G3P":"OCC(O)COP(=O)(O)O","PEP":"OC(=O)C(=C)OP(=O)(O)O",
 "Adenine":"Nc1ncnc2[nH]cnc12","Guanine":"Nc1nc2[nH]cnc2c(=O)[nH]1","Hypoxanthine":"O=c1[nH]cnc2[nH]cnc12",
 "Xanthine":"O=c1[nH]c(=O)c2[nH]cnc2[nH]1","Urate":"O=c1[nH]c(=O)c2[nH]c(=O)[nH]c2[nH]1","Cytosine":"Nc1cc[nH]c(=O)n1",
 "Uracil":"O=c1cc[nH]c(=O)[nH]1","Thymine":"Cc1c[nH]c(=O)[nH]c1=O","Adenosine":"Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O",
 "Guanosine":"Nc1nc2c(ncn2C2OC(CO)C(O)C2O)c(=O)[nH]1","Inosine":"O=c1[nH]cnc2c1ncn2C1OC(CO)C(O)C1O",
 "Uridine":"OCC1OC(n2ccc(=O)[nH]c2=O)C(O)C1O","Cytidine":"Nc1ccn(C2OC(CO)C(O)C2O)c(=O)n1",
 "AMP":"Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O","GMP":"Nc1nc2c(ncn2C2OC(COP(=O)(O)O)C(O)C2O)c(=O)[nH]1",
 "UMP":"OC1C(O)C(COP(=O)(O)O)OC1n1ccc(=O)[nH]c1=O","IMP":"O=c1[nH]cnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O",
 "Laurate":"CCCCCCCCCCCC(=O)O","Myristate":"CCCCCCCCCCCCCC(=O)O","Palmitate":"CCCCCCCCCCCCCCCC(=O)O",
 "Palmitoleate":"CCCCCC/C=C\\CCCCCCCC(=O)O","Stearate":"CCCCCCCCCCCCCCCCCC(=O)O","Oleate":"CCCCCCCC/C=C\\CCCCCCCC(=O)O",
 "Linoleate":"CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O","Linolenate":"CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)O",
 "Arachidate":"CCCCCCCCCCCCCCCCCCCC(=O)O","Arachidonate":"CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O",
 "EPA":"CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O","DHA":"CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O",
 "Behenate":"CCCCCCCCCCCCCCCCCCCCCC(=O)O","Sphingosine":"CCCCCCCCCCCCC/C=C/C(O)C(N)CO",
 "Glutathione":"NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O","GSSG":"NC(CCC(=O)NC(CSSCC(NC(=O)CCC(N)C(=O)O)C(=O)NCC(=O)O)C(=O)NCC(=O)O)C(=O)O",
 "Ascorbate":"OCC(O)C1OC(=O)C(O)=C1O","Carnitine":"C[N+](C)(C)CC(O)CC(=O)[O-]","Acetylcarnitine":"CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
 "Pantothenate":"CC(C)(CO)C(O)C(=O)NCCC(=O)O","Nicotinate":"OC(=O)c1cccnc1","Choline":"C[N+](C)(C)CCO",
 "Spermidine":"NCCCNCCCCN","Putrescine":"NCCCCN","NAA":"CC(=O)NC(CC(=O)O)C(=O)O","Cystathionine":"NC(CCSCC(N)C(=O)O)C(=O)O",
}
DESC = ["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors","FractionCSP3"]
def feat_chem(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None, None
    d = np.array([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                  Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.FractionCSP3(m)], float)
    return Descriptors.ExactMolWt(m), d

def dense(x): return np.asarray(x.todense() if hasattr(x, "todense") else x, np.float32)
def norm_adj(xy, k=K):
    n = len(xy); _, idx = cKDTree(xy).query(xy, k=min(k+1, n))
    r = np.repeat(np.arange(n), idx.shape[1]-1); c = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)); A = ((A+A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A

print("loading DESIUM ...", flush=True)
S = {}
for s in DS:
    a = ad.read_h5ad(f"{DES}/{s}.h5ad"); x = a.layers["log1p"] if "log1p" in a.layers else a.X
    S[s] = dict(X=sp.csr_matrix(x).astype(np.float32), pos={g: i for i, g in enumerate(map(str, a.var_names))},
                Y=np.log1p(np.asarray(a.uns["msi"], float)).astype(np.float32),
                mz=np.asarray(a.uns["mz_features"], float), xy=np.asarray(a.obsm["spatial"], float)); del a
shared = sorted(set.intersection(*[set(S[s]["pos"]) for s in DS]))
gi = {s: np.array([S[s]["pos"][g] for g in shared]) for s in DS}
XC = {s: dense(S[s]["X"][:, gi[s]]) for s in DS}
mz = S[DS[0]]["mz"]
G = len(shared); s1 = np.zeros(G); s2 = np.zeros(G); N = 0
for s in DS: X = XC[s]; s1 += X.sum(0); s2 += (X*X).sum(0); N += X.shape[0]
hvg = np.sort(np.argsort(s2/N - (s1/N)**2)[-NHVG:])
pca = PCA(NPC, svd_solver="randomized", random_state=0).fit(np.vstack([XC[s][:, hvg] for s in DS]))
Vt = pca.components_.astype(np.float32); pm = pca.mean_.astype(np.float32)
Flist, Ylist, secid = [], [], []
for si, s in enumerate(DS):
    Xh = XC[s][:, hvg]; P = (Xh @ Vt.T - pm @ Vt.T).astype(np.float32)
    A = norm_adj(S[s]["xy"]); P1 = A @ P; Flist.append(np.hstack([P, P1, A @ P1])); Ylist.append(S[s]["Y"]); secid.append(np.full(S[s]["Y"].shape[0], si))
F = np.vstack(Flist); Y = np.vstack(Ylist); secid = np.concatenate(secid)
Fr = PCA(DF, svd_solver="randomized", random_state=0).fit_transform(F).astype(np.float32); Fr = (Fr-Fr.mean(0))/(Fr.std(0)+1e-8)
det_all = (Y > 0).mean(0)

# map metabolites -> detected ion, collect descriptors + z-scored measured pattern
names, E, Z = [], [], []
for nm, smi in SMI.items():
    mass, d = feat_chem(smi)
    if mass is None: continue
    t = mass - 1.0073; j = int(np.argmin(np.abs(mz - t)))
    if abs(mz[j]-t) > 0.01 or det_all[j] < 0.2: continue
    y = Y[:, j]
    if y.std() < 1e-9: continue
    names.append(nm); E.append(d); Z.append(((y-y.mean())/(y.std()+1e-8)).astype(np.float32))
E = np.array(E); Z = np.array(Z).T; Estd = (E-E.mean(0))/(E.std(0)+1e-8); M = len(names)
print(f"  mapped metabolites (detected ion + valid SMILES): {M}", flush=True)

# subsample spots for speed
sub = np.concatenate([np.where(secid == si)[0][:600] for si in range(len(DS))])
FrS = Fr[sub]; Zs = Z[sub]; dE = Estd.shape[1]; nS = len(sub)

def lomo(idx, seed=0):
    """LOMO over the metabolite subset idx: returns (median bilinear r, median mean-pattern r)."""
    blocks = {m: (FrS[:, :, None]*Estd[m][None, None, :]).reshape(nS, DF*dE).astype(np.float32) for m in idx}
    bi, mp = [], []
    for m in idx:
        tr = [k for k in idx if k != m]
        Xtr = np.vstack([blocks[k] for k in tr]); ytr = np.concatenate([Zs[:, k] for k in tr])
        w = Ridge(20.0).fit(Xtr, ytr).coef_
        zhat = (Fr[:, :, None]*Estd[m][None, None, :]).reshape(Fr.shape[0], DF*dE) @ w
        bi.append(spearmanr(zhat, Z[:, m])[0])
        mp.append(spearmanr(Z[:, tr].mean(1), Z[:, m])[0])
    return np.nanmedian(bi), np.nanmedian(mp)

rng = np.random.RandomState  # deterministic per-N via seed
grid = [15, 25, 35, 45, min(60, M), M]
grid = sorted(set([g for g in grid if g <= M]))
print("\n  N   bilinear  mean-pattern   gap(bi-mp)   [median LOMO Spearman]", flush=True)
for Ni in grid:
    if Ni == M:
        b, m = lomo(list(range(M))); draws = [(b, m)]
    else:
        draws = []
        for rep in range(3):
            idx = list(rng(rep).permutation(M)[:Ni])
            draws.append(lomo(idx))
    b = np.mean([d[0] for d in draws]); mm = np.mean([d[1] for d in draws])
    print(f"  {Ni:3d}   {b:+.3f}     {mm:+.3f}       {b-mm:+.3f}", flush=True)
print("\n  Interpretation: if gap(bi-mp) rises toward >=0 as N grows -> chemistry-conditioning learns with scale", flush=True)
print("  -> CCLE (225 metabolites) worth pursuing. If gap stays strongly negative/flat -> scale alone won't fix it.", flush=True)
print("DONE.", flush=True)
