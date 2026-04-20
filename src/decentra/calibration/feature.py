import warnings

import numpy as np


class FeatureCalibrator:
    """Feature-level attribution calibration.

    Rescales each feature's contribution to match the black-box's *relative*
    attribution shape, while (optionally) preserving the surrogate's total
    magnitude so that predictions remain in a comparable range.

    With ``magnitude_preserving=True`` (default) the per-feature scaling is

        alpha_raw_j = E|phi_j^bb| / E|phi_j^surr|
        alpha_j     = alpha_raw_j * (sum_j E|phi_j^surr|) / (sum_j alpha_raw_j * E|phi_j^surr|)

    guaranteeing ``sum_j alpha_j * E|phi_j^surr| == sum_j E|phi_j^surr|``. This
    fixes the R² collapse that occurs when ``bb_shap`` (e.g. logit-scale
    TreeSHAP) and ``surr_contribs`` (e.g. score-scale surrogate) differ in
    absolute magnitude by orders of magnitude.

    Setting ``magnitude_preserving=False`` restores the legacy behavior
    (``alpha_j = E|phi_j^bb| / E|phi_j^surr|`` without rescaling) and emits a
    deprecation warning; it is retained only for backward reproducibility.

    Examples
    --------
    >>> cal = FeatureCalibrator()
    >>> cal.fit(surr_contribs, bb_shap)
    >>> cal_contribs, cal_pred = cal.transform(surr_contribs, surr_pred)
    """

    def __init__(self, magnitude_preserving=True, eps=1e-10):
        self.magnitude_preserving = magnitude_preserving
        self.eps = eps
        self.alpha_ = None
        self.alpha_raw_ = None

    def fit(self, surr_contribs, bb_shap):
        """Compute per-feature scaling factors.

        Parameters
        ----------
        surr_contribs : ndarray of shape (n_samples, n_features)
        bb_shap : ndarray of shape (n_samples, n_features)
            Black-box SHAP values (ground truth).
        """
        bb_abs_mean = np.mean(np.abs(bb_shap), axis=0)
        su_abs_mean = np.mean(np.abs(surr_contribs), axis=0)
        alpha_raw = np.where(
            su_abs_mean > self.eps, bb_abs_mean / su_abs_mean, 1.0
        )
        self.alpha_raw_ = alpha_raw

        if self.magnitude_preserving:
            num = su_abs_mean.sum()
            den = (alpha_raw * su_abs_mean).sum()
            scale = num / den if den > self.eps else 1.0
            self.alpha_ = alpha_raw * scale
        else:
            warnings.warn(
                "magnitude_preserving=False is deprecated and causes R² "
                "collapse when bb_shap and surr_contribs are on different "
                "scales. Default is now True.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.alpha_ = alpha_raw
        return self

    def transform(self, surr_contribs, surr_pred):
        """Apply calibration.

        Returns
        -------
        new_contribs : ndarray of shape (n_samples, n_features)
        new_pred : ndarray of shape (n_samples,)
        """
        new_contribs = surr_contribs * self.alpha_[np.newaxis, :]
        new_pred = surr_pred + (
            new_contribs.sum(axis=1) - surr_contribs.sum(axis=1)
        )
        return new_contribs, new_pred

    def fit_transform(self, surr_contribs, bb_shap, surr_pred):
        """Fit and transform in one step."""
        self.fit(surr_contribs, bb_shap)
        return self.transform(surr_contribs, surr_pred)
