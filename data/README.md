# Datasets

MetaSpatial is trained and evaluated on **public paired** spatial-transcriptome + spatial-metabolome data. Only the gene-set priors (`genesets/`) are bundled in this repo; the imaging datasets are large and must be downloaded from their archives.

## DESIUM — DESI-MSI + Visium (human breast & lung cancer)

- **DOI:** [10.7910/DVN/GZFCWC](https://doi.org/10.7910/DVN/GZFCWC) (Harvard Dataverse)
- **Reference:** Godfrey T.M. et al. *Angew. Chem. Int. Ed.* **2025**, 64, e202502028.
- **Contents:** 7 paired sections (breast BC_515/525/823, lung LC_091/170/276), 14,816 spots, an identical 2,086-ion m/z axis. The provided `Correlation_NMF_analysis/data/*.h5ad` are already in MetaSpatial format (`.uns['msi']`, `.uns['mz_features']`, `.obsm['spatial']`).
- **Use:** download, unzip, and point the pipeline at the folder:
  `python pipeline/metaspatial_all_in_one.py "/path/to/DESIUM"`

## SMA — MALDI-MSI + Visium (mouse & human brain)

- **DOI:** [10.17632/w7nw4km7xd.1](https://doi.org/10.17632/w7nw4km7xd.1) (Mendeley Data) — bundles the aligned per-spot metabolite matrices used here.
- **Also:** SciLifeLab Figshare (MALDI + spatial transcriptomics deposits); Reference: Vicari M. et al. *Nat. Biotechnol.* **2024**. [10.1038/s41587-023-01937-y](https://doi.org/10.1038/s41587-023-01937-y).
- **Contents:** 13 paired sections across three MALDI chemistries — DHB (lipids, 2,754 ions), 9-AA (metabolites, 3,658 ions), FMP-10 (neurotransmitters, 1,538 ions) — in mouse and human striatum / substantia nigra. The download bundle contains `sma.zip` with, per section, a spaceranger `outs/` folder and an aligned `*.Visium.<matrix>.smamsi.csv` metabolite raster.
- **Convert to MetaSpatial format** (auto co-registration of MALDI raster → Visium spots, validated by transcriptome↔metabolite coupling):
  `python pipeline/sma_to_metaspatial.py "/path/to/SMA"`
  then train with `pipeline/metaspatial_all_in_one.py`.

## Cross-cancer application (transcriptome-only, no paired MSI)

- **Glioblastoma:** 10x Genomics public Visium and CytAssist FFPE protein-expression glioblastoma datasets.
- **Hepatocellular carcinoma / liver:** public Visium cohorts (see manuscript Data availability).

## Bundled gene-set priors (`genesets/`)

| File | Source |
|---|---|
| `scMetab_KEGG.gmt` | scMetabolism KEGG metabolic pathways |
| `scMetab_REACT.gmt` | scMetabolism Reactome metabolic pathways |
| `scFEA_mod.csv`, `scFEA_cmpd.csv` | scFEA metabolic-module / compound tables |

Pass a `.gmt` to the model via `MetaSpatial(use_kegg=True, kegg_gmt="data/genesets/scMetab_KEGG.gmt")`, or drop `scMetab_KEGG.gmt` beside your data for the pipeline to auto-detect it.
