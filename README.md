<!-- omit in toc -->
# MetaSpatial

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Interfaces](https://img.shields.io/badge/interfaces-Python%20%7C%20R-blue.svg)](#use-it-on-your-own-data)
[![Status](https://img.shields.io/badge/status-research%20software-informational.svg)](#)

**Predict the spatial metabolome from spatial transcriptomics.**

MetaSpatial learns a **gene → metabolite** map from paired tissue sections and then *predicts* the per‑spot metabolite profile for **transcriptome‑only** tissue — with calibrated, out‑of‑distribution‑aware uncertainty. A pre‑trained model ships with the package, so you can go from a Visium/Xenium/Slide‑seq section to predicted metabolite maps in a few lines, with **no mass‑spectrometry data required**.

![MetaSpatial overview](docs/overview.png)

- [Why MetaSpatial](#why-metaspatial)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Use it on your own data](#use-it-on-your-own-data)
- [Input data format](#input-data-format)
- [How it works](#how-it-works)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

## Why MetaSpatial

Spatial transcriptomics is abundant; spatial metabolomics is scarce and hard to acquire. Existing tools only *correlate* the two modalities where both were measured — they cannot be applied to the thousands of transcriptome‑only sections where a metabolic readout is actually needed. MetaSpatial is, to our knowledge, the first tool that **predicts** metabolite abundances from transcriptome alone.

- **Predicts** a full per‑spot metabolite (m/z) profile from gene expression + coordinates.
- **Knows when it is wrong** — split‑conformal per‑metabolite intervals plus an epistemic out‑of‑distribution flag that localises where predictions should not be trusted.
- **Honest about what is predictable** — accuracy is chemical‑class dependent (structural lipids and nucleotides are recoverable; fast‑turnover polar metabolites such as lactate are not), and the model reports this per ion rather than hiding it in an average.
- **Python and R** — one pickled model, called from either language, with byte‑identical results.
- **Lightweight** — CPU‑only `scikit‑learn`; trains and predicts in seconds to minutes.

Transport‑aware features (gene principal components diffused over the spatial neighbourhood graph) give the model its name; a separately verified mass‑conserving reaction–diffusion solver is included to seed future physics‑coupled prediction but is **not** used by the released model.

## Installation

```bash
git clone https://github.com/himanshushekhar367-sudo/MetaSpatial.git
cd MetaSpatial

# option A — conda
conda env create -f environment.yml
conda activate metaspatial
pip install -e .

# option B — pip only
pip install -r requirements.txt
pip install -e .
```

Python ≥ 3.9, CPU only (no GPU). For the R interface, install [`reticulate`](https://rstudio.github.io/reticulate/) and point it at the same environment.

## Quick start

A pre‑trained model (`metaspatial_model.pkl`, trained once on the 7‑section DESIUM human‑cancer cohort) ships with the repo. Predicting on your own transcriptome‑only section takes three lines:

```python
import anndata as ad
from metaspatial import MetaSpatial

model = MetaSpatial.load("metaspatial_model.pkl")     # pre-trained; no MSI needed
adata = ad.read_h5ad("your_section.h5ad")             # genes (symbols) + adata.obsm['spatial']
pred  = model.predict_metabolome(adata)               # (n_spots × n_metab); also in adata.obsm['metaspatial_pred']
```

Prediction returns the metabolite matrix, stores it in `adata.obsm['metaspatial_pred']`, and attaches per‑metabolite confidence intervals. Genes are matched by symbol and the transport graph is built per section automatically.

## Use it on your own data

**Python — Visium / Xenium / MERFISH / Slide‑seq `.h5ad`:**

```bash
python examples/predict_on_your_data.py --model metaspatial_model.pkl --query your_section.h5ad --out results/ --plot
```

**R / Seurat — a `.RDS` object, no conversion needed:**

```r
source("R/metaspatial.R")
model <- ms_load_model("metaspatial_model.pkl")
seu   <- ms_run_rds("your_section.RDS", model)   # adds a 'metaspatial' assay
```

Both entry points call the **same** pickled model — the R wrapper dispatches to Python through `reticulate`, so R and Python predictions are byte‑identical. You get a ranked `predicted_metabolites.csv` and spatial maps; the step checks gene/species overlap and warns on mismatch.

> **Read the caveats.** Predictions are strongest for lipids, nucleotides and redox metabolites, and for tissue resembling the human‑tumour training data. Fast‑turnover polar species (lactate, glucose) are deliberately flagged as unpredictable rather than guessed. See [`docs/USAGE.md`](docs/USAGE.md).

To train MetaSpatial on **your own** paired MSI + transcriptomics instead of using the shipped model:

```python
from metaspatial import MetaSpatial
model = MetaSpatial(use_kegg=True).fit(train_adatas)   # list of paired AnnData (see below)
model.save("my_model.pkl")
```

## Input data format

One `AnnData` (`.h5ad`) per section. For **prediction** you only need the gene expression and coordinates; the `msi` fields are required only for **training**.

| field | contents | shape | needed for |
|---|---|---|---|
| `adata.layers['log1p']` (or `adata.X`) | genes, `log1p(CP10k)` | `n_spots × n_genes` | predict + train |
| `adata.obsm['spatial']` | x, y coordinates | `n_spots × 2` | predict + train |
| `adata.uns['msi']` | metabolite ion intensities | `n_spots × n_metab` | train only |
| `adata.uns['mz_features']` | m/z values | `n_metab` | train only |

## How it works

Gene expression is reduced to principal components, then augmented with **transport‑aware features** — the components diffused over a spatial k‑nearest‑neighbour graph (`[P, ÂP, Â²P]`), a lightweight surrogate for nutrient transport — and an optional KEGG metabolic‑pathway prior. A multi‑output ridge model maps these features to metabolite intensities, and split‑conformal calibration adds per‑metabolite uncertainty. Evaluation uses spatially‑blocked cross‑validation and leave‑one‑section‑out transfer (never random‑spot splits, which inflate accuracy through spatial autocorrelation).

The central empirical finding is a **map of predictability**: which parts of the metabolome are recoverable from transcription (structural lipids, nucleotides, locally‑synthesised neurotransmitters) versus which are not (fast‑flux polar metabolites), reproduced across two ionisation platforms (DESI‑MSI and MALDI‑MSI) and two species.

## Datasets

MetaSpatial is validated on public paired data — see [`data/README.md`](data/README.md) for download instructions.

| Cohort | Assay | Tissue | DOI |
|---|---|---|---|
| **DESIUM** | DESI‑MSI + Visium | human breast / lung tumour | [10.7910/DVN/GZFCWC](https://doi.org/10.7910/DVN/GZFCWC) (Godfrey et al. 2025) |
| **SMA** | MALDI‑MSI + Visium | mouse & human brain | [10.17632/w7nw4km7xd.1](https://doi.org/10.17632/w7nw4km7xd.1) (Vicari et al. 2024) |

## Citation

If you use MetaSpatial, please cite the accompanying manuscript (details on publication) and the paired datasets above.

```bibtex
@software{metaspatial,
  title  = {MetaSpatial: transport-aware prediction of spatial metabolomes from spatial transcriptomics},
  year   = {2025},
  url    = {https://github.com/himanshushekhar367-sudo/MetaSpatial}
}
```

## License

[MIT](LICENSE).
