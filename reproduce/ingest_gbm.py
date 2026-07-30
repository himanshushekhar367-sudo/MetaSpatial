"""Ingest GBM Visium + CytAssist(gene+protein) into AnnData with spatial coords."""
import scanpy as sc, pandas as pd, numpy as np, glob, os, warnings
warnings.filterwarnings('ignore')
UP="/mnt/user-data/uploads/spatial metabolism"; OUT="/home/claude/spatmet"; os.makedirs(OUT, exist_ok=True)

def load_positions(spatial_dir):
    for pat in ["tissue_positions.csv","tissue_positions_list.csv"]:
        fs=glob.glob(f"{spatial_dir}/**/{pat}", recursive=True)
        if fs:
            f=fs[0]
            if "list" in os.path.basename(f):
                df=pd.read_csv(f, header=None,
                   names=["barcode","in_tissue","array_row","array_col","pxl_row","pxl_col"])
            else:
                df=pd.read_csv(f); df=df.rename(columns={df.columns[0]:"barcode"})
            return df.set_index("barcode"), f
    return None, None

def build(h5, spatial_dir, name, gex_only=True):
    a=sc.read_10x_h5(h5, gex_only=gex_only); a.var_names_make_unique()
    pos,f=load_positions(spatial_dir)
    print(f"{name}: {a.shape[0]} spots x {a.shape[1]} feats; positions={os.path.basename(f) if f else None} cols={list(pos.columns) if pos is not None else None}")
    if pos is not None:
        common=a.obs_names.intersection(pos.index); a=a[common].copy(); p=pos.loc[a.obs_names]
        xc=[c for c in p.columns if 'pxl_col' in c or 'imagecol' in c.lower()]
        yc=[c for c in p.columns if 'pxl_row' in c or 'imagerow' in c.lower()]
        xcol=xc[0] if xc else p.columns[-1]; ycol=yc[0] if yc else p.columns[-2]
        a.obsm['spatial']=np.c_[p[xcol].values.astype(float), p[ycol].values.astype(float)]
        if 'in_tissue' in p.columns:
            a.obs['in_tissue']=p['in_tissue'].values
    return a

gbm=build(f"{UP}/Parent_Visium_Human_Glioblastoma_filtered_feature_bc_matrix.h5", f"{OUT}/spatial_visium","GBM Visium")
gbm.write(f"{OUT}/gbm_visium.h5ad")
print("  -> saved gbm_visium.h5ad; coords in obsm:", 'spatial' in gbm.obsm)

cyt=build(f"{UP}/CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_filtered_feature_bc_matrix.h5", f"{OUT}/spatial_cyt","GBM CytAssist", gex_only=False)
cyt.write(f"{OUT}/gbm_cyt.h5ad")
prot=cyt.var_names[cyt.var['feature_types']=='Antibody Capture'].tolist()
print(f"  -> saved gbm_cyt.h5ad; {len(prot)} proteins, coords:", 'spatial' in cyt.obsm)
