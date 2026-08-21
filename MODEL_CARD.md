# Model Card — MetaSpatial (`metaspatial_model.pkl`)

A concise, honest summary of what the shipped model is, what it should and should not be used for,
and how it fails. Read this before applying it to your data.

## Model details
- **Task:** predict a per-spot spatial metabolome (mass-spectrometry-imaging ion intensities) from
  spatial transcriptomics (gene expression + spot coordinates) alone.
- **Architecture:** HVG → PCA(100) → transport-aware features `[P, ÂP, Â²P]` (degree-normalised
  spatial k-NN diffusion, k=6) → multi-output ridge (α=200) on per-metabolite standardised targets.
  Optional human KEGG metabolic-map prior (`use_kegg=True`, off in the shipped model).
- **Training data:** DESIUM cohort — 7 paired DESI-MSI + Visium sections, human breast and lung
  carcinoma, 14,816 spots (Godfrey et al. 2025; Dataverse [10.7910/DVN/GZFCWC](https://doi.org/10.7910/DVN/GZFCWC)).
- **Output:** **2,086 m/z channels**, negative-mode DESI — these are *ion channels*, not confirmed named
  metabolites. Exact-mass annotations (141/2,086) are putative (confidence level 2–3), not MS/MS-confirmed.
- **Uncertainty:** split-conformal per-ion interval widths (`.uns['metaspatial_conf_width']`), an
  overlap-adaptive widened width (`.uns['metaspatial_conf_width_adj']`), and a per-prediction
  gene-panel overlap (`model.last_gene_overlap_`). A per-ion `reliability_table()` reports trust tiers.
- **Optional extra modalities:** `MetaSpatial(extra_key=...)` can append co-measured per-spot covariates
  such as cheap H&E patch features from `add_histology_features()`. The shipped model does **not** use
  histology; H&E support is for retraining and ablation studies.
- **Version:** `metaspatial-unified-1`. Pickled under the versions in `requirements-lock.txt`; loading
  under other scikit-learn versions works but may emit an `InconsistentVersionWarning`.

## Intended use
- **Hypothesis generation** on transcriptome-only tissue: prioritising spatially- and patient-resolved
  metabolic hypotheses for targeted metabolomics follow-up.
- Strongest for tissue **resembling the human-tumour training data**, and for **transcriptionally
  programmed, spatially-structured** ion classes (structural/signalling lipids, nucleotides, redox
  metabolites).

## Out-of-scope / known failure modes
- **Fast-turnover polar metabolites** (lactate, glucose, TCA acids, sugar-phosphates): predicted poorly
  (median r < 0.1) because their pools are set by flux, transport and allostery, not transcript level.
  The model flags these as low-confidence rather than guessing.
- **Cross-sample / cross-tissue transfer is weak** (leave-one-sample-out median ≈ 0.03 on the DESI tumour
  panel). The model transfers the *within-section* gene→metabolite relationship, not a section's absolute
  metabolite pattern. Treat cross-cohort predictions as demonstrations, not measurements.
- **Low gene-panel overlap** with the training genes → extrapolation; `last_gene_overlap_` warns below ~0.5.
- **Non-human tissue** without ortholog symbol mapping: gene symbols will not match; predictions degrade.
  Retrain per species/panel (see `pipeline/` and `reproduce/sma_desium_benchmark.py`).
- **Named-metabolite / structural claims** require MS/MS or FDR-controlled annotation (e.g. METASPACE);
  do not report exact-mass hits as confirmed identifications.

## Safe-use decision tree
Report a prediction as usable (rather than hypothesis-only) when **all** hold:
1. input tissue resembles the training tissue (human tumour / epithelial);
2. gene-panel overlap with the model is above ~0.5 (`model.last_gene_overlap_`);
3. the ion's chemical class is lipid / nucleotide / redox / locally-synthesised;
4. the predicted map is spatially structured (non-random);
5. epistemic/conformal uncertainty for that ion is low (`reliability_table()` trust tier ≥ medium).
Otherwise, report as a **hypothesis only**.

## Ethics / provenance
Trained on public, de-identified human cancer data under the source licence. Predictions are
computational and not a clinical or diagnostic device.
