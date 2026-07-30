# CCLE-pretrained chemistry-conditioned head — findings

Experiments testing whether a GAZE-style metabolite-conditioned head (spatial/expression features ⊗
chemical descriptors) can (1) predict metabolites held out of training and (2) transfer to tissue.
Scripts: `ccle_descriptors.py`, `ccle_lomo.py`, `ccle_transfer.py`. Data: CCLE metabolomics (225
metabolites × 928 cell lines, Li et al. 2019) + CCLE expression (DepMap) + DESIUM.

## 1. Does chemistry-conditioning scale? (leave-one-metabolite-out) — YES

| setting | N metabolites | conditioned head | baseline | gap |
|---|---|---|---|---|
| DESIUM tissue | 62 | 0.207 | 0.252 | **−0.044** |
| CCLE bulk | 190 | **0.235** | 0.016 | **+0.220** |

At scale the head clearly beats the baseline (median LOMO Spearman 0.235; in-distribution ceiling 0.394;
83% of held-out metabolites positive). By class: **lipid 0.44** (92% positive), polar 0.13, acylcarnitine 0.15.
`corr(|logP|, accuracy) = +0.52` — the model learns a real chemistry→predictability map. Scale, not
representation, was the binding constraint (it was +0.04 at N=62).

## 2. Does bulk pre-training fix cross-sample transfer? — NO

CCLE-bulk head applied to DESIUM tissue, on the 51 metabolites overlapping both datasets:
**median Spearman +0.028**, mean +0.041, 71% positive — statistically tied with MetaSpatial's
spatial-LOSO transfer (0.03), **not beating it**.

## 3. Does CCLE overcome the lactate / polar limitation? — NO (important)

The transfer overlap is polar-dominated, and those are exactly the transcriptionally-uncoupled species:

| metabolite | CCLE→DESIUM transfer Spearman |
|---|---|
| lactate | +0.024 |
| glutathione (reduced) | −0.069 |
| taurine | −0.109 |
| inosine | −0.120 |

The lactate/polar limit is **information-theoretic** (transcription does not encode fast metabolic flux)
and persists at any training scale. CCLE integration expands *metabolite scope* (arbitrary metabolites,
strongest for lipids) with calibrated uncertainty — it does **not** make flux-controlled polar metabolites
predictable. Claims to the contrary are refuted by `ccle_transfer.py` above.

## Recommendation
Build the CCLE-pretrained conditioned head as a **scope expander** (predict metabolites beyond the MSI
panel, best for lipids, uncertainty-gated), not as a transfer or polar-metabolite fix. The 0.03
cross-sample transfer number and the polar/lactate blind spot remain honestly stated open limitations.
