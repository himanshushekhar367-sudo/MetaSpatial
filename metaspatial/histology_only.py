"""
EXPERIMENTAL - morphology-only metabolite prediction.

This is a companion to the default transcriptome-based MetaSpatial model, not
a replacement. It predicts the spatial metabolome from H&E histology features
alone, using the same cheap, auditable per-spot features as
add_histology_features: RGB, optical-density, texture, and tissue-fraction
summaries mapped through a ridge model to the metabolite m/z axis.

Why it exists - empirical finding from the DESIUM/HCC control analyses:
  * In cross-section transfer on DESIUM, morphology-only prediction is
    competitive with, and on this cohort slightly beats, transcriptome-only
    (LOSO median Spearman about 0.14 vs 0.08), because gross tissue morphology
    transfers across sections better than the high-dimensional
    transcriptome-to-metabolite map.
  * Within a section the transcriptome model is richer (0.22 vs 0.20), so
    morphology-only is a transfer tool, not a universally better one.
  * A shuffle control collapses it to about zero, confirming that the signal is
    real spatial structure.
  * Cross-tissue coherence: a DESIUM-trained morphology-only model reproduced
    20/28 of the expected HCC tumour-vs-normal directional metabolite changes
    across four liver patients (vs 18/28 for the transcriptome model), with no
    liver MSI supervision.

Strong caveats: this is a research/experimental mode. It is a coherence-level
signal, validated without measured MSI on the external HCC cohort; it uses
cheap features rather than a histology foundation model; and it is cross-tissue.
Do not treat its outputs as measured metabolites. The shipped
metaspatial_model.pkl is unaffected and remains transcriptome-only.
"""

import numpy as np
from sklearn.linear_model import Ridge


class MorphologyMetabolitePredictor:
    """Experimental H&E-only metabolite predictor.

    Train on paired sections that already carry per-spot histology features
    from add_histology_features and a metabolite matrix in .uns["msi"]. Predict
    on any section with histology features, with no transcriptome or MSI needed.
    """

    def __init__(self, alpha=200.0, key="histology"):
        self.alpha = alpha
        self.key = key
        self.experimental_ = True

    def _feat(self, adata):
        if self.key not in adata.obsm:
            raise ValueError(f"adata.obsm['{self.key}'] missing - run add_histology_features(adata) first.")
        return np.asarray(adata.obsm[self.key], np.float32)

    def fit(self, adatas, metab_key="msi", mz_key="mz_features", standardize_features="global"):
        """Fit the morphology-only ridge.

        Parameters
        ----------
        adatas
            List of AnnData objects with .obsm[key] histology features and
            .uns[metab_key] shaped n_spots x n_metabolites.
        standardize_features
            "global" standardizes the stacked training feature matrix and
            preserves between-section contrast. "none" uses the features as is.
            Targets are per-metabolite standardized, as in MetaSpatial.
        """
        if standardize_features not in {"global", "none"}:
            raise ValueError("standardize_features must be 'global' or 'none'.")
        X = np.vstack([self._feat(a) for a in adatas]).astype(np.float32)
        Y = np.vstack([np.log1p(np.asarray(a.uns[metab_key], float)) for a in adatas]).astype(np.float32)
        self.feat_mode_ = standardize_features
        if standardize_features == "global":
            self.fmu_ = X.mean(0)
            self.fsd_ = X.std(0) + 1e-8
            X = (X - self.fmu_) / self.fsd_
        else:
            self.fmu_ = None
            self.fsd_ = None
        self.mz_ = np.asarray(adatas[0].uns[mz_key], float)
        self.mu_ = Y.mean(0).astype(np.float32)
        self.sd_ = (Y.std(0) + 1e-8).astype(np.float32)
        self.reg_ = Ridge(alpha=self.alpha).fit(X, (Y - self.mu_) / self.sd_)
        return self

    def predict_from_histology(self, adata, metab_uns_key="metaspatial_he_pred"):
        """Predict the metabolome from histology features alone.

        For a coherent between-condition signal on a multi-condition query,
        features are standardized across the whole query so condition contrast
        survives. Do not standardize per condition before calling this method.
        """
        X = self._feat(adata)
        if getattr(self, "feat_mode_", "global") == "global":
            X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        pred = self.reg_.predict(X.astype(np.float32)) * self.sd_ + self.mu_
        pred = np.maximum(pred, 0.0).astype(np.float32)
        adata.obsm[metab_uns_key] = pred
        return pred

    def save(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
