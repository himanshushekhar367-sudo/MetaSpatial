#!/usr/bin/env python3
"""
MetaSpatial — unified training-data preparation + inventory.

GOAL: assemble EVERY available paired (spatial transcriptome + spatial metabolite) section
into MetaSpatial's training format, and deep-inspect anything not yet ingestible so we can
finalise its loader.

USAGE (Windows PowerShell / any terminal):
    python prepare_training_data.py "/path/to/data"

Then paste the ===== SUMMARY ===== block back to me.

MetaSpatial training format — one AnnData (.h5ad) per section:
    adata.layers['log1p']     genes,  log1p(CP10k)      (n_spots x n_genes)
    adata.uns['msi']          metabolite intensities     (n_spots x n_metab)
    adata.uns['mz_features']  m/z values                 (n_metab,)
    adata.obsm['spatial']     x,y coordinates            (n_spots x 2)

Dependencies:  pip install scanpy anndata pandas numpy   (optional: pyimzml, imageio)
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT  = os.path.join(ROOT, "training_ready")
os.makedirs(OUT, exist_ok=True)
REPORT = {"root": ROOT, "paired_sections": [], "needs_review": [], "notes": []}

try:
    import numpy as np, anndata as ad, pandas as pd
    import scanpy as sc
except ImportError as e:
    print("MISSING DEPENDENCY:", e)
    print("  Run:  pip install scanpy anndata pandas numpy")
    sys.exit(1)

def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)
def ok(m):  print("  [OK]  " + m, flush=True)
def warn(m):print("  [!!]  " + m, flush=True)

# ----------------------------------------------------------------------------
# 1) VALIDATE any MetaSpatial-format .h5ad already present (DESIUM = 7 sections)
# ----------------------------------------------------------------------------
hdr("1. PAIRED SECTIONS ALREADY IN MetaSpatial FORMAT (.h5ad)")

def validate_h5ad(path):
    """Return a dict describing whether an h5ad is MetaSpatial-trainable."""
    try:
        a = ad.read_h5ad(path)
    except Exception as e:
        return {"path": path, "loadable": False, "error": str(e)}
    has_genes = ("log1p" in a.layers) or (a.X is not None)
    has_msi   = "msi" in a.uns
    has_mz    = "mz_features" in a.uns
    has_xy    = "spatial" in a.obsm
    info = dict(path=os.path.relpath(path, ROOT), spots=int(a.n_obs), genes=int(a.n_vars),
                has_log1p="log1p" in a.layers, has_msi=has_msi, has_mz=has_mz, has_spatial=has_xy,
                n_metab=int(np.asarray(a.uns["msi"]).shape[1]) if has_msi else 0)
    info["trainable"] = bool(has_genes and has_msi and has_mz and has_xy)
    return info

h5s = sorted(glob.glob(os.path.join(ROOT, "**", "*.h5ad"), recursive=True))
h5s = [p for p in h5s if "training_ready" not in p]
if not h5s:
    warn("No .h5ad found. If the DESIUM zip isn't unpacked yet, extract it first (see section 3).")
for p in h5s:
    v = validate_h5ad(p)
    if v.get("trainable"):
        ok(f"{v['path']}  |  {v['spots']} spots x {v['genes']} genes  |  {v['n_metab']} metabolites")
        REPORT["paired_sections"].append(v)
    else:
        warn(f"{os.path.relpath(p, ROOT)}  ->  NOT trainable: "
             f"{'genes ' if not (v.get('has_log1p') or True) else ''}"
             f"{'no msi ' if not v.get('has_msi') else ''}"
             f"{'no mz ' if not v.get('has_mz') else ''}"
             f"{'no spatial' if not v.get('has_spatial') else ''}")
        REPORT["needs_review"].append(v)

# ----------------------------------------------------------------------------
# 2) INVENTORY every other data folder (so nothing is missed)
# ----------------------------------------------------------------------------
hdr("2. FULL DATA INVENTORY")
def scan(pattern):
    return sorted(glob.glob(os.path.join(ROOT, "**", pattern), recursive=True))
inv = {
    "Visium h5 (spaceranger)": scan("*filtered_feature_bc_matrix.h5"),
    "Visium tissue_positions": scan("*tissue_positions*.csv") + scan("*tissue_positions*.parquet"),
    "MALDI/DESI imzML":        scan("*.imzML"),
    "metabolite matrices":     [p for p in scan("*.npz") if any(k in os.path.basename(p).lower()
                                    for k in ("lipid","metab","msi","intensity","targeted"))],
    "HCC/GBM counts (mtx/h5)": scan("*counts*.mtx*") + scan("*expr*.RDS") + scan("Parent_Visium*_filtered_feature_bc_matrix.h5"),
    "zips (raw deposits)":     scan("*.zip"),
    "R data (.rdata)":         scan("*.rdata") + scan("*.RData"),
}
for k, v in inv.items():
    print(f"  {k:26s}: {len(v)} file(s)")
    for p in v[:6]:
        print(f"       - {os.path.relpath(p, ROOT)}  ({os.path.getsize(p)//10**6} MB)")
    if len(v) > 6: print(f"       ... (+{len(v)-6} more)")
REPORT["inventory_counts"] = {k: len(v) for k, v in inv.items()}

# ----------------------------------------------------------------------------
# 3) SMA (or any new Visium+MALDI cohort): detect, inspect, and ingest if possible
# ----------------------------------------------------------------------------
hdr("3. SMA / NEW PAIRED COHORT (Visium + MALDI) — detect & inspect")

def peek_imzml(path):
    try:
        from pyimzml.ImzMLParser import ImzMLParser
    except ImportError:
        return {"path": os.path.relpath(path, ROOT), "note": "install pyimzml to read (pip install pyimzml)"}
    if not os.path.exists(path[:-6] + ".ibd"):
        return {"path": os.path.relpath(path, ROOT),
                "error": "missing companion .ibd file (imzML needs BOTH .imzML and .ibd side-by-side)"}
    try:
        p = ImzMLParser(path)
        if len(p.coordinates) == 0:
            return {"path": os.path.relpath(path, ROOT), "error": "0 pixels parsed"}
        mzs, ints = p.getspectrum(0)
        return dict(path=os.path.relpath(path, ROOT), n_pixels=len(p.coordinates),
                    n_mz_spectrum0=len(mzs), mz_min=float(np.min(mzs)), mz_max=float(np.max(mzs)))
    except Exception as e:
        return {"path": os.path.relpath(path, ROOT), "error": str(e)}

imzmls = inv["MALDI/DESI imzML"]
vis_h5 = inv["Visium h5 (spaceranger)"]
if imzmls:
    print("  MALDI/DESI imzML files found — inspecting spectra:")
    REPORT["imzml"] = [peek_imzml(p) for p in imzmls]
    for r in REPORT["imzml"]:
        if "n_pixels" in r:
            ok(f"{r['path']}: {r['n_pixels']} pixels, m/z {r['mz_min']:.1f}-{r['mz_max']:.1f}")
        else:
            warn(f"{r['path']}: {r.get('note') or r.get('error')}")
if vis_h5:
    print("  Visium spaceranger .h5 found:")
    for p in vis_h5: ok(os.path.relpath(p, ROOT))

# Heuristic auto-pairing: only if a section clearly has BOTH an aligned metabolite matrix
# (spot-level, same n as Visium spots) AND Visium. Raw-imzML->spot co-registration is dataset-
# specific and must be verified, so we DO NOT silently guess it — we report and stop for review.
if imzmls and vis_h5:
    REPORT["notes"].append(
        "SMA-like data present (Visium + imzML). MALDI pixels must be co-registered to Visium "
        "spots. If the download already contains an ALIGNED per-spot metabolite matrix "
        "(a table with one row per Visium barcode), point me to it and I'll build the paired "
        ".h5ad in one step. If only raw imzML is present, paste this SUMMARY and I'll write the "
        "exact co-registration loader for this cohort's structure.")
    warn("SMA ingestion needs one confirmation (aligned matrix vs raw imzML) — see SUMMARY.")
else:
    REPORT["notes"].append("No new Visium+MALDI cohort detected yet. Download SMA (Mendeley "
        "10.17632/w7nw4km7xd.1 + Figshare 10.17044/scilifelab.22770161) into this folder and re-run.")

# ----------------------------------------------------------------------------
# 4) WRITE the training manifest
# ----------------------------------------------------------------------------
hdr("4. TRAINING MANIFEST")
manifest = {"trainable_sections": [s["path"] for s in REPORT["paired_sections"]],
            "n_trainable": len(REPORT["paired_sections"]),
            "total_spots": int(sum(s["spots"] for s in REPORT["paired_sections"])),
            "metabolite_axis_sizes": sorted({s["n_metab"] for s in REPORT["paired_sections"]})}
json.dump({**REPORT, "manifest": manifest},
          open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
ok(f"wrote {os.path.join('training_ready','manifest.json')}")

print("\n" + "=" * 70 + "\n===== SUMMARY (paste this back) =====\n" + "=" * 70)
print(f"Trainable paired sections : {manifest['n_trainable']}")
print(f"Total paired spots        : {manifest['total_spots']}")
print(f"Metabolite axis size(s)   : {manifest['metabolite_axis_sizes']}")
print(f"Visium-only / imzML files : {len(vis_h5)} Visium h5, {len(imzmls)} imzML")
print("Sections:")
for s in REPORT["paired_sections"]:
    print(f"   - {s['path']}  ({s['spots']} spots, {s['n_metab']} metab)")
if REPORT["needs_review"]:
    print("Needs review:", [r["path"] for r in REPORT["needs_review"]])
for n in REPORT["notes"]:
    print("NOTE:", n)
