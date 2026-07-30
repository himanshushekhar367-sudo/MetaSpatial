# Using MetaSpatial on your own spatial transcriptomics data

This guide shows how **anyone** can apply MetaSpatial to **any** spatial transcriptomics section — Visium, Xenium, MERFISH, Slide-seq — to predict its spatial metabolome from the transcriptome alone.

---

## 0. Install

```bash
git clone https://github.com/himanshushekhar367-sudo/MetaSpatial.git
cd MetaSpatial
pip install -r requirements.txt
pip install -e .        # makes `from metaspatial import MetaSpatial` work anywhere
```

---

## 1. Which path are you on?

MetaSpatial learns a **gene → metabolite** map from *paired* data, then predicts. So there are two situations:

| You have… | Do this | What you get |
|---|---|---|
| **Paired** transcriptome **+** metabolome on the same sections | **Train** on it, predict within your data | Strong, quantitative (within-sample median Spearman ≈ 0.28) |
| **Transcriptome only** (the common case) | **Apply a trained model** to your section | Predicted metabolite **maps + uncertainty** — hypothesis-generating (see caveats) |

If you only have transcriptomics, a **pre-trained model ships with the repo** — `metaspatial_model.pkl` (trained once on the 7-section DESIUM human-tumour cohort, predicting 2,086 DESI-MSI m/z channels). Just apply it — **no training, no MSI, no paired data needed**. You can also train your own on the public **DESIUM** or **SMA** cohorts (see [`../data/README.md`](../data/README.md)) if you want a *different* metabolite panel.

> **R / Seurat users:** every step below has an R equivalent. `source("R/metaspatial.R")`, then `ms_run_rds("your_section.RDS", ms_load_model("metaspatial_model.pkl"))` adds a `metaspatial` assay to your Seurat object. The R layer dispatches to the same Python model via reticulate, so results are byte-identical. See [`../R/metaspatial.R`](../R/metaspatial.R) and `examples/predict_on_your_data.R`. Address metabolites by common name with `ms_feature(obj, "Glutathione")` / `ms_spatial_plot(obj, "Glutathione")`.

---

## 2. Get your section into MetaSpatial format

**For prediction, a query only needs two things:** gene expression (with `var_names` = gene **symbols**, same **species** as the model) and per-spot coordinates in `adata.obsm['spatial']`.

```python
import scanpy as sc
q = sc.read_visium("path/to/spaceranger/outs")   # obsm['spatial'] is filled automatically
# or: q = sc.read_h5ad("your_section.h5ad")
q.var_names_make_unique()
```

A **training** section additionally needs the measured metabolome:

| field | contents | shape |
|---|---|---|
| `adata.layers['log1p']` (or `adata.X`) | genes, log1p(CP10k) | `n_spots × n_genes` |
| `adata.uns['msi']` | metabolite ion intensities | `n_spots × n_metab` |
| `adata.uns['mz_features']` | m/z values | `n_metab` |
| `adata.obsm['spatial']` | x, y coordinates | `n_spots × 2` |

> The example scripts normalise for you (CP10k + log1p if the data looks like raw counts). If your query has no `obsm['spatial']`, they try common fallbacks (`pxl_col_in_fullres`, `array_col`, `x`/`y`, …).

---

## 3. Predict on transcriptome-only data (the common case)

### 3a. Command line

```bash
# 1) get a model bundle — train on your own paired data, or on the public DESIUM cohort:
python examples/train_on_paired_data.py --data /path/to/paired_sections --out model.pkl \
       --kegg data/genesets/scMetab_KEGG.gmt

# 2) apply it to ANY section (Visium/Xenium/MERFISH/... .h5ad):
python examples/predict_on_your_data.py --model model.pkl --query your_section.h5ad --out results/ --plot
```

Outputs in `results/`:
- `query_with_predictions.h5ad` — your section with the predicted metabolome in `.obsm['metaspatial_pred']` (columns follow `.uns['metaspatial_mz']`).
- `predicted_metabolites.csv` — every metabolite ranked by predicted spatial structure, with the conformal interval width.
- `predicted_maps.png` — spatial maps of the top predicted metabolites (with `--plot`).

### 3b. Python API

```python
import pickle, scanpy as sc
import metaspatial                                    # noqa: needed to unpickle the model

bundle = pickle.load(open("model.pkl", "rb"))
model, mz = bundle["model"], bundle["mz"]

q = sc.read_h5ad("your_section.h5ad"); q.var_names_make_unique()
sc.pp.normalize_total(q, target_sum=1e4); sc.pp.log1p(q)
q.layers["log1p"] = q.X.copy()                        # model reads layers['log1p'] or X
# q.obsm['spatial'] must hold (x, y) — Visium h5ad already have it

pred = model.predict_metabolome(q)                    # (n_spots × n_metab); also in q.obsm['metaspatial_pred']
print(pred.shape, "metabolites at m/z:", mz[:5], "...")
```

Each column of `pred` is the predicted intensity of the metabolite at `mz[column]`.

---

## 4. Train on your own paired data (strongest, quantitative)

```python
from metaspatial import MetaSpatial
import anndata as ad

train = [ad.read_h5ad(f) for f in ["sectionA.h5ad", "sectionB.h5ad", "sectionC.h5ad"]]
model = MetaSpatial(use_kegg=True, kegg_gmt="data/genesets/scMetab_KEGG.gmt").fit(train)

query = ad.read_h5ad("sectionD.h5ad")                 # held out
pred  = model.predict_metabolome(query)               # evaluate vs query.uns['msi']
```

or one line via the CLI: `python examples/train_on_paired_data.py --data ./sections --out model.pkl`. With ≥3 sections it also computes split-conformal per-metabolite uncertainty (a held-out section as the calibration set) and stores it in the bundle.

To benchmark honestly across all your sections (leave-one-section-out), use the pipeline:
```bash
python pipeline/prepare_training_data.py ./sections     # validate
python pipeline/metaspatial_all_in_one.py ./sections    # train every cohort + report LOSO
```

---

## 5. Optional: multimodal input

If you also have a per-spot modality on the same spots (e.g. antibody-derived protein, `adata.obsm['protein']`), concatenate it into the features:

```python
model = MetaSpatial(use_kegg=True, extra_key="protein").fit(train)
```

---

## Metabolic pathway activity (a spatially-aware scMetabolism)

MetaSpatial also scores per-spot **metabolic pathway activity** directly from the transcriptome — the same output [scMetabolism](https://github.com/wu-yc/scMetabolism) gives (KEGG/Reactome gene sets scored per spot, no metabolomics required) — with a spatial-smoothing option scMetabolism lacks.

**Command line**
```bash
python examples/pathway_scores.py --query your_section.h5ad \
       --gmt data/genesets/scMetab_KEGG.gmt --out pathways/ --method aucell --smooth 1 --plot
```
→ `pathways/pathway_scores.csv` (spots × pathways), the section with scores in `.obsm['pathway_scores']`, and `pathway_maps.png`.

**Python API**
```python
from metaspatial import PathwayScorer
ps = PathwayScorer("data/genesets/scMetab_KEGG.gmt", method="aucell")   # or "mean_z"
scores = ps.score(adata, spatial_smooth=1)     # DataFrame: spots × metabolic pathways
```

- `method="aucell"` reproduces scMetabolism's rank-based AUCell scoring; `"mean_z"` is a fast mean-of-z-scores.
- `spatial_smooth=1` (or 2) diffuses activity over the tissue k-NN graph — the improvement over per-spot scoring, giving spatially-coherent pathway maps.
- Because the same model predicts measured metabolites, you can cross-check a pathway's transcript activity against the predicted abundance of its metabolites (`PathwayScorer.consistency_with_prediction`) — a validation scMetabolism cannot do.

Bundled sets: `data/genesets/scMetab_KEGG.gmt` (85 KEGG metabolic pathways, the scMetabolism sets) and `scMetab_REACT.gmt`; or pass your own `.gmt`.

---

## Metabolite-grounded metabolic activity (beyond scMetabolism)

The pathway scores above — like scMetabolism / AUCell / ssGSEA — come from **enzyme transcripts only**, which confuses "enzymes are expressed" with "the pathway is metabolically active". Because MetaSpatial predicts the metabolites too, it scores a pathway high **only where its enzymes *and* its metabolites agree** (the flux signature), and reports a per-pathway **enzyme–metabolite coherence** that tells you which pathway calls to trust.

```bash
# with a trained model (predicts metabolites for any transcriptome-only section):
python examples/metabolic_activity.py --query your_section.h5ad --model model.pkl \
       --gmt data/genesets/scMetab_KEGG.gmt --out metabolic/ --plot
# or validate on a paired section that already has measured metabolites in .uns['msi']:
python examples/metabolic_activity.py --query paired_section.h5ad --measured --out metabolic/ --plot
```

```python
from metaspatial import MetabolicActivity
ma = MetabolicActivity("data/genesets/scMetab_KEGG.gmt", adducts="neg")
activity, confidence, source = ma.score(adata, model=model)   # model=None → use measured .uns['msi']
```

- `activity` (spots × pathways) scores each pathway by **enzyme × metabolite** agreement; `confidence` ranks pathways by `enzyme_metabolite_coherence` and flags `metabolite_supported`.
- **It corrects enzyme-only scoring.** On a breast-tumour section it ranks glutathione (coherence r=0.51), TCA (0.38) and unsaturated-fatty-acid (0.38) metabolism as genuinely active, while **flagging glycolysis / galactose / fructose-mannose as decoupled** (enzymes high but metabolites elsewhere — coherence negative) — pathways scMetabolism would call active. Aggregated, the coherence recovers the paper's lipid/nucleotide-vs-polar split at the pathway level (mean coherence: lipid +0.20, nucleotide +0.25, polar +0.09).
- m/z annotation defaults to negative-mode DESI adducts; pass `adducts="pos"` for positive mode, or your own `mass_db` / `met2path` dicts. Pathways with no annotated metabolites fall back to the enzyme-only score, clearly flagged (`metabolite_supported=False`).

This is the part no other pathway tool can do, because none predict the metabolites: it turns "which enzymes are on" into "which pathways are actually running, and how confident are we."

---

## 6. Read this before interpreting (honest use)

MetaSpatial predicts the part of the metabolome that transcription controls. Its own paper is explicit about the limits — respect them:

- **Species / gene IDs must match.** A human-trained model on mouse data (or Ensembl IDs vs symbols) shares almost no genes and returns noise. `predict_on_your_data.py` prints the gene overlap and warns below 30%.
- **Cross-sample transfer is weak and unsolved.** Leave-one-section-out accuracy on the DESI tumour panel is ~0.03; predictions on tissue unlike the training data are **qualitative maps / hypotheses**, not validated measurements. It is strongest when the query resembles the training tissue and platform.
- **Predictability is class-dependent.** Structural **lipids and nucleotides** are recoverable (median r 0.3–0.62); fast-turnover **polar/central-carbon metabolites** are not (r < 0.1). Trust lipid/nucleotide predictions far more than amino-acid/TCA predictions. The predicted spatial *dynamic range* (the CSV's `predicted_std`) and the conformal `conf_width` both help you triage which predictions to believe.
- **Coordinates matter.** The transport prior uses a spatial k-NN graph, so `obsm['spatial']` must be real geometry (not shuffled).

For the full evidence behind these statements see the manuscript in [`../manuscript/`](../manuscript/) and the reproduction scripts in [`../reproduce/`](../reproduce/).

---

## 7. End-to-end example you can copy

```bash
# download the DESIUM paired cohort (see data/README.md), then:
python examples/train_on_paired_data.py --data /path/to/DESIUM --out desium_model.pkl \
       --kegg data/genesets/scMetab_KEGG.gmt

# apply the trained model to your own Visium section:
python examples/predict_on_your_data.py --model desium_model.pkl \
       --query my_visium_section.h5ad --out my_results/ --plot

# inspect: my_results/predicted_metabolites.csv  and  my_results/predicted_maps.png
```

That's it — one model bundle, one command per new section, works on any spatial transcriptomics section whose species matches the model.
