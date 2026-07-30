# Reproducing the paper

Every quantitative claim in the manuscript maps to a script here and a cached result in `results/`. These are the **exact analysis scripts** used, kept as a reproducibility record.

> **Paths:** most scripts contain the local data path used for the paper (e.g. a DESIUM folder). Edit the `ROOT` / path variable near the top of a script to your own download (see `../data/README.md`) before running. Cached outputs in `results/` let you regenerate figures without re-running the heavy analyses.

## Analyses

| Script | Output | Reproduces |
|---|---|---|
| `benchmark_all.py` | `results/benchmark_all.json` | Head-to-head vs NMF / scMetabolism / SpaGE-style / single-gene (Fig. 2a,b) |
| `benchmark.py`, `benchmark_tools.py` | — | Earlier head-to-head variants |
| `annotate_predictable.py` | — | Per-metabolite predictability + class gradient (Fig. 3d) |
| `rigor_ceiling.py` | `results/rigor_ceiling.json` | Model-family ceiling: is 0.28 biology or the model? (ED Fig. 4a) |
| `rigor_splits.py` | `results/rigor_splits.json` | Spatial-leakage audit + diffusion-hop ablation (ED Fig. 4b,c) |
| `rigor_baselines.py` | `results/rigor_baselines.json` | kNN / SpaGE-style / MLP baselines |
| `rigor_predictability.py` | `results/rigor_predictability.json` | What makes a metabolite predictable — Moran's I dominates (ED Fig. 5a,b) |
| `rigor_uncertainty.py` | `results/rigor_uncertainty.json` | Calibrated + epistemic/aleatoric uncertainty (Fig. 4, ED Fig. 5d) |
| `rigor_shift.py` | `results/rigor_shift.json` | Why transfer fails — metabolite-side distribution shift (ED Fig. 5c) |
| `rigor_gcn.py` | `results/rigor_gcn.json` | Learned GCN vs fixed diffusion prior (ED Fig. 5e) |
| `rigor_spatial_control.py` | `results/rigor_spatial_control.json` | Spatial-structure destruction control (ED Fig. 6) |
| `rigor_additions.py` | — | Paired-bootstrap effect sizes / CIs (Table S2) |
| `loso_matched.py` | `results/loso_matched.json` | Detection-matched leave-one-sample-out (Table S1) |
| `loso_harmonize.py`, `loso_learned.py` | `results/loso_*.json` | Harmonization does not close the LOSO gap (Table S3, Fig. S1) |
| `fvm_solver.py` | `results/fvm_summary.json` | Verified reaction–diffusion finite-volume solver (Fig. 6) |
| `hcc_gbm_predict.py`, `ingest_gbm.py` | — | Cross-cancer application to GBM & HCC (Fig. 5b,c) |
| `gbm_multimodal.py` | `results/gbm_multimodal.json` | Antibody-protein multimodal extensibility |
| `liver_crossplatform.py`, `liver_validation.py` | `results/liver_crossplatform.json` | Cross-platform liver lipidome check |
| `flux_kegg_experiment.py`, `new_analyses.py`, `improve_analyses.py` | `results/*.json` | KEGG/flux prior + reviewer quick-wins |

## Figures (`figures/`)

| Script | Builds |
|---|---|
| `make_nature_figs.py` | Main Figures 1–6 |
| `make_rigor_figs.py`, `make_rigor2_figs.py` | Extended Data Figs. 4, 5 |
| `make_control_fig.py` | Extended Data Fig. 6 |
| `make_bio_figs.py` | Extended Data Figs. 2, 3 |
| `make_supp_figs.py` | Supplementary Figs. S1, S2 |
| `make_sma_figure.py` | Figure 7 (cross-platform transfer) |
| `make_figS3.py` | Supplementary Fig. S3 (SMA co-registration QC) |
| `ed7_maps.py` → `ed7_plot.py` | Extended Data Fig. 7 (nigra-vs-striatum dopamine maps; run `ed7_maps.py` first to compute, then `ed7_plot.py`) |
| `build_s4.py` | Table S4 |
| `build_html.py` | Renders `manuscript/*.md` → self-contained `.html` with embedded figures |
