import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV,
    Lasso, LassoCV,
    ElasticNet, ElasticNetCV,
)
from sklearn.preprocessing import StandardScaler

from .base import BaseSurrogate

_VALID_METHODS = [
    "ols", "ridge", "ridgecv",
    "lasso", "lassocv",
    "elasticnet", "elasticnetcv",
]


def _make_sklearn_model(method, alpha=1.0, l1_ratio=0.5):
    """Create an sklearn regression model by name.

    Parameters
    ----------
    method : str
        One of ``_VALID_METHODS``.
    alpha : float
        Regularization strength (ignored for "ols" and "*cv" methods).
    l1_ratio : float
        ElasticNet mixing parameter.

    Returns
    -------
    sklearn estimator
    """
    if method == "ols":
        return LinearRegression()
    if method == "ridge":
        return Ridge(alpha=alpha)
    if method == "ridgecv":
        return RidgeCV()
    if method == "lasso":
        return Lasso(alpha=alpha, max_iter=10000)
    if method == "lassocv":
        return LassoCV(max_iter=10000, cv=5)
    if method == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    if method == "elasticnetcv":
        return ElasticNetCV(l1_ratio=l1_ratio, max_iter=10000, cv=5)
    raise ValueError(
        f"method must be one of {_VALID_METHODS}, got '{method}'"
    )


class LinearSurrogate(BaseSurrogate):
    """Linear regression surrogate.

    Contributions are ``beta_j * x_j`` for each feature.

    When monotone constraints are active, coefficient signs are
    enforced by flipping feature values (``x_j → -x_j``) for features
    whose unconstrained coefficient disagrees with the constraint,
    then refitting.

    Parameters
    ----------
    method : str, default="ridge"
        One of:

        - ``"ols"``: Ordinary Least Squares (no regularization).
        - ``"ridge"``: Ridge (L2). Uses ``alpha``.
        - ``"ridgecv"``: RidgeCV (alpha auto-tuned via LOO-CV).
        - ``"lasso"``: Lasso (L1). Uses ``alpha``.
        - ``"lassocv"``: LassoCV (alpha auto-tuned via CV).
        - ``"elasticnet"``: ElasticNet (L1+L2). Uses ``alpha``, ``l1_ratio``.
        - ``"elasticnetcv"``: ElasticNetCV (alpha auto-tuned via CV).
          Uses ``l1_ratio``.
    alpha : float, default=1.0
        Regularization strength. Ignored for ``"ols"`` and ``*cv`` methods.
    l1_ratio : float, default=0.5
        ElasticNet mixing (0 = Ridge, 1 = Lasso). Only used when
        ``method`` is ``"elasticnet"`` or ``"elasticnetcv"``.
    monotone_detect_mode : {"auto", "none"}, default="none"
    monotone_constraints : dict[str, int], optional

    Attributes
    ----------
    alpha_ : float
        Effective alpha after fitting. For CV methods, this is the
        selected alpha. For fixed-alpha methods, same as ``alpha``.

    Examples
    --------
    >>> LinearSurrogate(method="ridge")
    >>> LinearSurrogate(method="lassocv")
    >>> LinearSurrogate(method="elasticnetcv", l1_ratio=0.7)
    >>> LinearSurrogate(method="ols")
    """

    def __init__(self, method="ridge", alpha=1.0, l1_ratio=0.5,
                 scale=True, monotone_detect_mode="none",
                 monotone_constraints=None):
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scale = scale
        self.monotone_detect_mode = monotone_detect_mode
        self.monotone_constraints = monotone_constraints
        self.model_ = None
        self.scaler_ = None
        self.alpha_ = None
        self.monotone_constraints_ = None
        self.monotone_correlations_ = None

    def _make_model(self):
        return _make_sklearn_model(self.method, self.alpha, self.l1_ratio)

    def _scale(self, X_arr, fit=False):
        """Scale features if self.scale is True."""
        if not self.scale:
            return X_arr
        if fit:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X_arr)
        return self.scaler_.transform(X_arr)

    def fit(self, X, y_logit, *, eval_set=None, sample_weight=None):
        X_arr = self._scale(np.asarray(X), fit=True)

        mc = self._resolve_monotone(X, y_logit)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if mc is not None and any(c != 0 for c in mc):
            # Fit unconstrained first to get coefficient signs
            self.model_ = self._make_model()
            self.model_.fit(X_arr, y_logit, **fit_kwargs)
            coefs = self.model_.coef_

            # Flip features whose coefficient sign disagrees
            X_fit = X_arr.copy()
            self._sign_flips_ = np.ones(X_arr.shape[1])
            for j, c in enumerate(mc):
                if c == 0:
                    continue
                if (c == 1 and coefs[j] < 0) or (c == -1 and coefs[j] > 0):
                    X_fit[:, j] = -X_fit[:, j]
                    self._sign_flips_[j] = -1.0

            self.model_ = self._make_model()
            self.model_.fit(X_fit, y_logit, **fit_kwargs)
        else:
            self._sign_flips_ = np.ones(X_arr.shape[1])
            self.model_ = self._make_model()
            self.model_.fit(X_arr, y_logit, **fit_kwargs)

        self.alpha_ = getattr(self.model_, "alpha_", self.alpha)
        self._set_feature_info(X)

        # Mean contribution on training data (for SHAP-like centering)
        # Use raw (unscaled) X to compute mean, so centering works in original space
        X_raw = np.asarray(X)
        self.mean_contributions_ = np.mean(
            self._raw_contributions(X_raw), axis=0
        )
        self._store_training_stats(X, y_logit)
        return self

    def _set_feature_info(self, X):
        """Set selected_features_ and coef_summary_ after fit."""
        if hasattr(X, "columns"):
            names = list(X.columns)
        else:
            names = [f"Feature_{i}" for i in range(np.asarray(X).shape[1])]
        coefs = self.model_.coef_
        self.selected_features_ = [
            name for name, c in zip(names, coefs) if abs(c) > 1e-8
        ]
        self.coef_summary_ = pd.DataFrame({
            "feature": names,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
            "selected": np.abs(coefs) > 1e-8,
        }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    @property
    def feature_importances_(self):
        """Absolute coefficient values (standardised if scale=True)."""
        return np.abs(self.model_.coef_)

    def predict(self, X):
        self._check_is_fitted()
        X_s = self._scale(np.asarray(X)) * self._sign_flips_
        return self.model_.predict(X_s)

    def _effective_coef(self):
        """Coefficients in original (unscaled) feature space."""
        coef = self.model_.coef_ * self._sign_flips_
        if self.scale and self.scaler_ is not None:
            return coef / self.scaler_.scale_
        return coef

    def _raw_contributions(self, X):
        """Uncentered per-feature contributions in original X space."""
        X_raw = np.asarray(X)
        return X_raw * self._effective_coef()

    def contributions(self, X):
        """Centered contributions (SHAP-like: relative to training mean)."""
        self._check_is_fitted()
        return self._raw_contributions(X) - self.mean_contributions_


class BinningSurrogate(BaseSurrogate):
    """Binning + Linear surrogate.

    Each feature is discretized into bins, encoded, then a linear model
    is fitted on the encoded features.

    Parameters
    ----------
    method : {"ols","ridge","ridgecv","lasso","lassocv","elasticnet",
        "elasticnetcv"}, default="ridge"
    alpha : float, default=1.0
    l1_ratio : float, default=0.5
    encoding : {"woe", "dummy"}, default="woe"
        - ``"woe"``: Weight of Evidence encoding. Each feature remains
          1 column. Contribution = ``beta_j * woe_j``.
        - ``"dummy"``: One-hot (dummy) encoding per bin. Each bin gets
          its own coefficient. More flexible but higher dimensionality.
    binning : {"optbinning", "custom"}, default="optbinning"
        - ``"optbinning"``: automatic optimal binning via OptimalBinning.
        - ``"custom"``: use ``custom_edges`` dict for bin edges.
    custom_edges : dict[str, list[float]], optional
        Per-feature bin edges (sorted, excluding -inf/+inf).
        Keys are feature names. Only used when ``binning="custom"``.
        Example: ``{"age": [25, 35, 45, 55, 65]}``
    max_n_bins : int, default=10
        Maximum bins per feature (OptimalBinning only).
    min_bin_size : float, default=0.05
        Minimum bin size as proportion (OptimalBinning only).
    monotone_detect_mode : {"auto", "none"}, default="none"
    monotone_constraints : dict[str, int], optional

    Examples
    --------
    >>> BinningSurrogate(encoding="woe", method="lassocv")
    >>> surr.fit(X, y_logit)                         # continuous binning
    >>> surr.fit(X, y_logit, binning_y=y_binary)     # binary binning
    >>> BinningSurrogate(binning="custom",
    ...     custom_edges={"age": [25, 45, 65]}, encoding="dummy")
    """

    def __init__(self, method="ridge", alpha=1.0, l1_ratio=0.5,
                 encoding="woe", binning="optbinning",
                 custom_edges=None, max_n_bins=10, min_bin_size=0.01,
                 monotone_detect_mode="none", monotone_constraints=None):
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.encoding = encoding
        self.binning = binning
        self.custom_edges = custom_edges
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.monotone_detect_mode = monotone_detect_mode
        self.monotone_constraints = monotone_constraints
        self.model_ = None
        self.alpha_ = None
        self.binners_ = None          # {j: OptimalBinning/ContinuousOB}
        self.bin_edges_ = None         # {j: [edge1, edge2, ...]}
        self.feature_names_ = None
        self.n_original_features_ = None
        self.active_features_ = None
        self.removed_features_ = None
        self.dummy_map_ = None
        self.monotone_constraints_ = None
        self.monotone_correlations_ = None

    def _make_model(self):
        return _make_sklearn_model(self.method, self.alpha, self.l1_ratio)

    # ── Binning ───────────────────────────────────────────────

    def _fit_bins(self, X_arr, y_logit, feature_names, mc, binning_y=None):
        """Determine bin edges per feature.

        binning_y=None → ContinuousOptimalBinning with y_logit.
        binning_y=array → OptimalBinning with that binary target.
        Fallback: if optbinning fails, median split (2 bins).
        """
        n_features = X_arr.shape[1]
        self.bin_edges_ = {}

        if self.binning == "custom":
            if not self.custom_edges:
                raise ValueError("custom_edges required when binning='custom'")
            name_to_idx = {name: i for i, name in enumerate(feature_names)}
            for fname, edges in self.custom_edges.items():
                idx = name_to_idx.get(fname)
                if idx is None:
                    raise KeyError(f"Feature '{fname}' not found")
                self.bin_edges_[idx] = sorted(edges)

        elif self.binning == "optbinning":
            _TREND = {1: "ascending", -1: "descending", 0: "auto"}
            self.binners_ = {}
            failed = []

            use_binary = binning_y is not None
            if use_binary:
                from optbinning import OptimalBinning
                y_target = np.asarray(binning_y).ravel()
            else:
                from optbinning import ContinuousOptimalBinning
                y_target = y_logit

            for j in range(n_features):
                trend = _TREND.get(mc[j], "auto") if mc is not None else "auto"
                try:
                    if use_binary:
                        ob = OptimalBinning(
                            name=feature_names[j],
                            max_n_bins=self.max_n_bins,
                            min_n_bins=2,
                            dtype="numerical",
                            monotonic_trend=trend,
                            min_bin_size=self.min_bin_size,
                        )
                    else:
                        ob = ContinuousOptimalBinning(
                            name=feature_names[j],
                            max_n_bins=self.max_n_bins,
                            min_n_bins=2,
                            dtype="numerical",
                            monotonic_trend=trend,
                            min_bin_size=self.min_bin_size,
                        )
                    ob.fit(X_arr[:, j], y_target)
                    if len(ob.splits) > 0:
                        self.bin_edges_[j] = sorted(ob.splits.tolist())
                        self.binners_[j] = ob
                    else:
                        raise ValueError("0 splits")
                except Exception:
                    # Fallback: median split (guarantees 2 bins)
                    med = float(np.median(X_arr[:, j]))
                    if np.ptp(X_arr[:, j]) > 0:
                        self.bin_edges_[j] = [med]
                    failed.append(feature_names[j])

            if failed:
                import warnings
                warnings.warn(
                    f"OptBinning failed for {len(failed)}/{n_features} "
                    f"features (median fallback): {failed}",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            raise ValueError(f"binning must be 'optbinning' or 'custom', "
                             f"got '{self.binning}'")

    def _assign_bin_idx(self, x_col, edges):
        """Assign each value to a bin index. Returns int array."""
        # bins: (-inf, e0], (e0, e1], ..., (e_{n-1}, inf)
        return np.searchsorted(edges, x_col, side="left")

    # ── Encoding (active features only) ─────────────────────────

    def _encode(self, X_arr):
        """Encode active features of X. Returns encoded matrix."""
        active = self.active_features_
        if self.encoding == "woe":
            return self._encode_woe(X_arr, active)
        elif self.encoding == "dummy":
            return self._encode_dummy(X_arr, active)
        raise ValueError(f"encoding must be 'woe' or 'dummy', "
                         f"got '{self.encoding}'")

    def _encode_woe(self, X_arr, active):
        """WoE encoding: 1 column per active feature.

        - OptimalBinning (binary target): metric="woe"
        - ContinuousOptimalBinning: metric="mean" (WoE not available)
        - Fallback (bin_edges, no binner): ordinal bin index
        - Unprocessed: raw X
        """
        from optbinning import ContinuousOptimalBinning

        cols = []
        for j in active:
            if self.binners_ and j in self.binners_:
                ob = self.binners_[j]
                if isinstance(ob, ContinuousOptimalBinning):
                    cols.append(ob.transform(X_arr[:, j], metric="mean"))
                else:
                    cols.append(ob.transform(X_arr[:, j], metric="woe"))
            elif j in self.bin_edges_:
                cols.append(
                    self._assign_bin_idx(X_arr[:, j], self.bin_edges_[j])
                    .astype(float)
                )
            else:
                cols.append(X_arr[:, j])
        return np.column_stack(cols) if cols else np.empty((len(X_arr), 0))

    def _encode_dummy(self, X_arr, active):
        """Dummy encoding: n_bins columns per active binned feature."""
        cols = []
        self.dummy_map_ = {}

        for j in active:
            if j in self.bin_edges_:
                edges = self.bin_edges_[j]
                bin_idx = self._assign_bin_idx(X_arr[:, j], edges)
                n_bins = len(edges) + 1
                dummies = np.zeros((len(X_arr), n_bins), dtype=float)
                for b in range(n_bins):
                    dummies[:, b] = (bin_idx == b).astype(float)
                cols.append(dummies)
                self.dummy_map_[j] = n_bins
            else:
                cols.append(X_arr[:, j:j + 1])
                self.dummy_map_[j] = 1

        return np.hstack(cols) if cols else np.empty((len(X_arr), 0))

    # ── Sign enforcement (WoE backward elimination) ───────────

    def _enforce_monotone_signs(self, X_arr, y_logit, mc):
        """Backward elimination of features whose coefficient sign
        violates the monotone constraint.

        For WoE encoding:
        - monotone=1  → WoE ascending → beta must be > 0
        - monotone=-1 → WoE descending → beta must be > 0
        - If beta < 0, the feature's contribution direction is reversed.

        Removal priority: smallest ``|beta × std(woe)|`` first
        (least contribution → least model degradation).
        """
        self.removed_features_ = []

        while True:
            X_enc = self._encode(X_arr)
            self.model_ = self._make_model()
            self.model_.fit(X_enc, y_logit)

            # Check sign violations among active features
            violations = []
            for idx, j in enumerate(self.active_features_):
                if mc[j] == 0:
                    continue
                beta = self.model_.coef_[idx]
                if beta < 0:
                    impact = abs(beta) * np.std(X_enc[:, idx])
                    violations.append((j, idx, impact))

            if not violations or len(self.active_features_) <= 1:
                break

            # Remove feature with smallest impact
            violations.sort(key=lambda x: x[2])
            remove_j = violations[0][0]
            self.active_features_.remove(remove_j)
            self.removed_features_.append(self.feature_names_[remove_j])

    # ── Fit / Predict / Contributions ─────────────────────────

    def fit(self, X, y_logit, *, eval_set=None, sample_weight=None,
            feature_names=None, binning_y=None):
        """Fit the binning + linear surrogate.

        Parameters
        ----------
        X : array-like
        y_logit : array-like
            Surrogate target (continuous log-odds from base model).
            Always used for the linear model fitting.
        binning_y : array-like, optional
            Separate target for OptimalBinning bin construction.
            ``None`` (default): use ``y_logit`` directly with
            ContinuousOptimalBinning.
            If provided (e.g. original binary target): use
            OptimalBinning with this binary target for bin edges.
        feature_names : list of str, optional
        """
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        self.n_original_features_ = n_features

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"f{i}" for i in range(n_features)]
        self.feature_names_ = feature_names

        mc = self._resolve_monotone(X, y_logit)
        self._fit_bins(X_arr, y_logit, feature_names, mc, binning_y)

        self.active_features_ = list(range(n_features))
        self.removed_features_ = []

        if (self.encoding == "woe" and mc is not None
                and any(c != 0 for c in mc)):
            self._enforce_monotone_signs(X_arr, y_logit, mc)
        else:
            X_enc = self._encode(X_arr)
            self.model_ = self._make_model()
            self.model_.fit(X_enc, y_logit)

        self.alpha_ = getattr(self.model_, "alpha_", self.alpha)

        # Mean contribution on training data (for SHAP-like centering)
        self.mean_contributions_ = np.mean(
            self._raw_contributions(X_arr), axis=0
        )
        self._store_training_stats(X, y_logit)
        return self

    @property
    def feature_importances_(self):
        """Absolute coefficient values mapped to original feature space.

        Removed features get importance = 0.
        For dummy encoding, per-feature importance is the sum of
        absolute bin coefficients.
        """
        imp = np.zeros(self.n_original_features_)
        coef = self.model_.coef_

        if self.encoding == "woe":
            for idx, j in enumerate(self.active_features_):
                imp[j] = abs(coef[idx])
        else:
            col = 0
            for j in self.active_features_:
                n_cols = self.dummy_map_.get(j, 1)
                imp[j] = np.sum(np.abs(coef[col:col + n_cols]))
                col += n_cols
        return imp

    def predict(self, X):
        self._check_is_fitted()
        return self.model_.predict(self._encode(np.asarray(X)))

    def _raw_contributions(self, X):
        """Uncentered per-feature contributions."""
        X_arr = np.asarray(X)
        X_enc = self._encode(X_arr)
        coef = self.model_.coef_
        n = X_arr.shape[0]
        contribs = np.zeros((n, self.n_original_features_))

        if self.encoding == "woe":
            for idx, j in enumerate(self.active_features_):
                contribs[:, j] = X_enc[:, idx] * coef[idx]
        else:
            col = 0
            for j in self.active_features_:
                n_cols = self.dummy_map_.get(j, 1)
                contribs[:, j] = (X_enc[:, col:col + n_cols] *
                                  coef[col:col + n_cols]).sum(axis=1)
                col += n_cols
        return contribs

    def contributions(self, X):
        """Centered contributions (SHAP-like: relative to training mean).

        Removed features get contribution = 0.
        """
        self._check_is_fitted()
        return self._raw_contributions(X) - self.mean_contributions_

    # ── Bin structure for scorecard ───────────────────────────

    def _get_feature_bins(self, X, n_bins=10):
        """Use stored bin edges for exact boundaries."""
        X_arr = np.asarray(X)
        contribs = self.contributions(X)
        result = {}

        for j in range(X_arr.shape[1]):
            if j in self.bin_edges_:
                edges_inner = self.bin_edges_[j]
                edges = [-np.inf] + edges_inner + [np.inf]
                bins = []
                for k in range(len(edges) - 1):
                    if k == 0:
                        mask = X_arr[:, j] <= edges[k + 1]
                    else:
                        mask = (X_arr[:, j] > edges[k]) & (
                            X_arr[:, j] <= edges[k + 1]
                        )
                    if mask.sum() == 0:
                        continue
                    bins.append({
                        "mask": mask,
                        "score": float(np.mean(contribs[mask, j])),
                    })
                if len(bins) > 1:
                    result[j] = bins
            else:
                scores = contribs[:, j]
                unique = np.unique(np.round(scores, 8))
                if len(unique) <= 1:
                    continue
                if len(unique) <= self._DISCRETE_THRESHOLD:
                    rounded = np.round(scores, 8)
                    bins = []
                    for us in unique:
                        mask = np.abs(rounded - us) < 1e-7
                        bins.append({"mask": mask, "score": float(us)})
                    result[j] = bins
                else:
                    result[j] = self._quantile_bin(X_arr[:, j], scores, n_bins)

        return result


# Backward compatibility
OptBinningSurrogate = BinningSurrogate
