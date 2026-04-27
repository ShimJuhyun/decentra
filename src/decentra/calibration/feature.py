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

    def __init__(self, magnitude_preserving=True, sign_align=False,
                 sign_threshold=0.05, eps=1e-10):
        """
        Parameters
        ----------
        magnitude_preserving : bool, default=True
            Normalize alphas so that total magnitude is preserved.
        sign_align : {False, True, "per_feature"}, default=False
            If True or "per_feature": detect per-feature sign mismatch between
            bb_shap and surr_contribs via Spearman correlation and flip alpha
            sign for features with ``|rho| > sign_threshold`` and wrong
            direction. Only features with statistically relevant correlation
            are flipped — near-zero correlations retain the positive alpha.
            Addresses the case where a surrogate's contribution sign
            disagrees with the black-box's SHAP direction (e.g., Ridge with
            multicollinearity-induced sign redistribution).
        sign_threshold : float, default=0.05
            Minimum |Spearman rho| required to trigger sign flip.
        eps : float, default=1e-10
        """
        self.magnitude_preserving = magnitude_preserving
        self.sign_align = sign_align
        self.sign_threshold = sign_threshold
        self.eps = eps
        self.alpha_ = None
        self.alpha_raw_ = None
        self.sign_factors_ = None
        self.sign_correlations_ = None

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

        # Sign alignment: detect mis-direction features
        if self.sign_align:
            from scipy.stats import spearmanr
            n_features = surr_contribs.shape[1]
            rhos = np.zeros(n_features)
            signs = np.ones(n_features)
            for j in range(n_features):
                su = surr_contribs[:, j]
                bb = bb_shap[:, j]
                if np.std(su) < self.eps or np.std(bb) < self.eps:
                    rhos[j] = np.nan
                    continue
                rho, _ = spearmanr(su, bb)
                rhos[j] = rho
                # Expected convention: surr in score space (negative=adverse)
                # vs bb in log-odds (positive=adverse). Well-aligned feature
                # should have rho < 0. If rho > sign_threshold, surr's sign
                # is flipped relative to convention → apply -1.
                if not np.isnan(rho) and rho > self.sign_threshold:
                    signs[j] = -1.0
            self.sign_correlations_ = rhos
            self.sign_factors_ = signs
            alpha_raw = alpha_raw * signs

        if self.magnitude_preserving:
            num = su_abs_mean.sum()
            den = (np.abs(alpha_raw) * su_abs_mean).sum()
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
