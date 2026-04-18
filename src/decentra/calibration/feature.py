import numpy as np


class FeatureCalibrator:
    """Feature-level attribution calibration.

    Rescales each feature's contribution by
    ``alpha_j = E[|phi_j^bb|] / E[|phi_j^surr|]``
    so that the surrogate's mean absolute attribution matches the
    black-box SHAP on a per-feature basis.

    Examples
    --------
    >>> cal = FeatureCalibrator()
    >>> cal.fit(surr_contribs, bb_shap)
    >>> cal_contribs, cal_pred = cal.transform(surr_contribs, surr_pred)
    """

    def __init__(self):
        self.alpha_ = None

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
        self.alpha_ = np.where(su_abs_mean > 1e-10, bb_abs_mean / su_abs_mean, 1.0)
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
