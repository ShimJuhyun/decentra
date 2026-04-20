"""Choi & Cha (2026) SHAP-PDP scorecard baseline.

Implements the model-replacement pipeline from Choi, Y. & Cha, E. (2026),
"LightGBM Scorecard based on SHAP Values", Computational Economics,
DOI: 10.1007/s10614-025-11194-7.

Pipeline (paper Section 4, Steps 4a-4c):

  1. Bin each feature with OptBinning.
  2. Compute the black-box (base) model's SHAP values on training X.
  3. For each feature j:
       SHAP PDP = mean SHAP value over samples falling in each bin.
  4. Fit a monotonic LightGBM regressor that maps x_j -> SHAP PDP value
     (one tiny regressor per feature, used as a smoother).
  5. Evaluate the regressor at each bin center to obtain f_j(bin_k),
     the bin-level representative score.

Unlike other decentra surrogates, this class does **not** train a
regression on y_logit. It observes the base model's SHAP attributions
and reshapes them into a binned, monotone, scorecard-compatible form
-- faithful to Choi & Cha's "SHAP values construct the model scorecard"
framing. In P5's surrogate zoo this is the D5 baseline: binning axis
identical to OptBin+Ridge (D4), scoring axis replaced by BB SHAP PDP.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import BaseSurrogate


class ShapPdpSurrogate(BaseSurrogate):
    """Choi & Cha (2026) SHAP-PDP scorecard baseline.

    Parameters
    ----------
    binning : {"optbinning", "custom"}, default="optbinning"
    custom_edges : dict[str, list[float]], optional
        Per-feature bin edges (sorted, excluding -inf/+inf).
        Only used when ``binning="custom"``.
    max_n_bins : int, default=10
    min_bin_size : float, default=0.01
    smoother_n_estimators : int, default=200
        Number of trees for the per-feature monotone LightGBM smoother.
    smoother_max_depth : int, default=2
    smoother_learning_rate : float, default=0.05
    monotone_detect_mode : {"auto", "none"}, default="auto"
        Choi & Cha determine the monotone direction from the sign of a
        linear regression on the SHAP PDP. We follow the same idea via
        Spearman correlation on (x, shap_mean) pairs, same threshold
        logic as elsewhere in decentra.
    monotone_constraints : dict[str, int], optional
        Overrides per feature. Keys are feature names.
    random_state : int, default=317

    Attributes
    ----------
    bin_edges_ : dict[int, list[float]]
        Inner bin edges per feature index (from OptBinning).
    bin_scores_ : dict[int, np.ndarray]
        ``bin_scores_[j][k]`` = f_j(bin k) after monotone smoothing.
    bin_centers_ : dict[int, np.ndarray]
        Bin center x-values (used to evaluate the smoother).
    mean_contributions_ : ndarray
        Mean f_j over training data -- used to center ``contributions()``.
    """

    _FALLBACK_EDGES = "median"  # match BinningSurrogate behaviour

    def __init__(
        self,
        binning: str = "optbinning",
        custom_edges: dict | None = None,
        max_n_bins: int = 10,
        min_bin_size: float = 0.01,
        smoother_n_estimators: int = 200,
        smoother_max_depth: int = 2,
        smoother_learning_rate: float = 0.05,
        monotone_detect_mode: str = "auto",
        monotone_constraints: dict | None = None,
        random_state: int = 317,
    ):
        self.binning = binning
        self.custom_edges = custom_edges
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.smoother_n_estimators = smoother_n_estimators
        self.smoother_max_depth = smoother_max_depth
        self.smoother_learning_rate = smoother_learning_rate
        self.monotone_detect_mode = monotone_detect_mode
        self.monotone_constraints = monotone_constraints
        self.random_state = random_state

        self.model_ = None  # marker for _check_is_fitted
        self.bin_edges_ = None
        self.bin_scores_ = None
        self.bin_centers_ = None
        self.mean_contributions_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self.monotone_constraints_ = None
        self.monotone_correlations_ = None
        self.smoothers_ = None

    # ── Binning (reuse OptBinning logic) ──────────────────────────

    def _fit_bins(self, X_arr, binning_target, feature_names, mc,
                  is_binary_target):
        """Fit per-feature bin edges."""
        n_features = X_arr.shape[1]
        self.bin_edges_ = {}

        if self.binning == "custom":
            if not self.custom_edges:
                raise ValueError("custom_edges required when binning='custom'")
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            for fname, edges in self.custom_edges.items():
                idx = name_to_idx.get(fname)
                if idx is None:
                    raise KeyError(f"Feature '{fname}' not found")
                self.bin_edges_[idx] = sorted(edges)
            return

        if self.binning != "optbinning":
            raise ValueError(
                f"binning must be 'optbinning' or 'custom', got '{self.binning}'"
            )

        _TREND = {1: "ascending", -1: "descending", 0: "auto"}
        if is_binary_target:
            from optbinning import OptimalBinning as _OB
        else:
            from optbinning import ContinuousOptimalBinning as _OB

        failed = []
        for j in range(n_features):
            trend = _TREND.get(mc[j], "auto") if mc is not None else "auto"
            try:
                ob = _OB(
                    name=feature_names[j],
                    max_n_bins=self.max_n_bins,
                    min_n_bins=2,
                    dtype="numerical",
                    monotonic_trend=trend,
                    min_bin_size=self.min_bin_size,
                )
                ob.fit(X_arr[:, j], binning_target)
                if len(ob.splits) > 0:
                    self.bin_edges_[j] = sorted(ob.splits.tolist())
                else:
                    raise ValueError("0 splits")
            except Exception:
                med = float(np.median(X_arr[:, j]))
                if np.ptp(X_arr[:, j]) > 0:
                    self.bin_edges_[j] = [med]
                failed.append(feature_names[j])

        if failed:
            warnings.warn(
                f"OptBinning failed for {len(failed)}/{n_features} features "
                f"(median fallback): {failed}",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _assign_bin_idx(x_col, edges):
        return np.searchsorted(edges, x_col, side="left")

    def _bin_center(self, j, k, x_col):
        """Midpoint of bin k for feature j, clipped to observed x range."""
        edges = self.bin_edges_[j]
        n_bins = len(edges) + 1
        x_min, x_max = float(x_col.min()), float(x_col.max())
        if k == 0:
            upper = edges[0]
            samples = x_col[x_col <= upper]
            return (
                float(samples.mean()) if samples.size > 0 else (x_min + upper) / 2
            )
        if k == n_bins - 1:
            lower = edges[-1]
            samples = x_col[x_col > lower]
            return (
                float(samples.mean()) if samples.size > 0 else (lower + x_max) / 2
            )
        lower, upper = edges[k - 1], edges[k]
        samples = x_col[(x_col > lower) & (x_col <= upper)]
        return (
            float(samples.mean()) if samples.size > 0 else (lower + upper) / 2
        )

    # ── SHAP PDP + monotone smoother ──────────────────────────────

    def _compute_shap_pdp(self, X_arr, shap_values, j):
        """Compute SHAP PDP for feature j as (bin_centers, mean_shap)."""
        edges = self.bin_edges_[j]
        n_bins = len(edges) + 1
        bin_idx = self._assign_bin_idx(X_arr[:, j], edges)

        centers, means = [], []
        for k in range(n_bins):
            mask = bin_idx == k
            if mask.sum() == 0:
                continue
            centers.append(self._bin_center(j, k, X_arr[:, j]))
            means.append(float(shap_values[mask, j].mean()))
        return np.array(centers), np.array(means)

    def _detect_shap_monotone(self, centers, shap_means, constraint):
        """Return 1/-1/0 to use for LightGBM monotone_constraints.

        If the user/auto-detection already fixed a direction, use it.
        Otherwise infer from the sign of Spearman rho between x and shap.
        """
        if constraint in (1, -1):
            return int(constraint)
        if len(centers) < 3:
            return 0
        from scipy.stats import spearmanr
        rho, pval = spearmanr(centers, shap_means)
        if np.isnan(rho) or pval >= 0.05:
            return 0
        return 1 if rho > 0 else -1

    def _fit_feature_smoother(self, centers, shap_means, constraint):
        """Fit a tiny monotone LightGBM regressor on the SHAP PDP."""
        if len(centers) < 2:
            return None

        import lightgbm as lgb

        params = dict(
            n_estimators=self.smoother_n_estimators,
            max_depth=self.smoother_max_depth,
            learning_rate=self.smoother_learning_rate,
            num_leaves=max(2, 2 ** self.smoother_max_depth),
            min_child_samples=1,
            min_data_in_bin=1,
            random_state=self.random_state,
            verbose=-1,
        )
        if constraint != 0:
            params["monotone_constraints"] = [constraint]

        reg = lgb.LGBMRegressor(**params)
        reg.fit(centers.reshape(-1, 1), shap_means)
        return reg

    def _evaluate_smoother(self, reg, centers, shap_means):
        """Evaluate regressor at training centers; fallback to raw PDP."""
        if reg is None:
            return shap_means.astype(float)
        return reg.predict(centers.reshape(-1, 1)).astype(float)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(
        self,
        X,
        y_logit,
        *,
        base_model=None,
        shap_values=None,
        binning_y=None,
        eval_set=None,
        sample_weight=None,
    ):
        """Fit the SHAP-PDP surrogate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (raw, unbinned features).
        y_logit : array-like of shape (n_samples,)
            Teacher log-odds. Only used as a fallback binning target
            when ``binning_y`` is not given. The surrogate does **not**
            fit a regression on y_logit.
        base_model : fitted estimator, optional
            Used with :class:`shap.TreeExplainer` to compute SHAP values
            when ``shap_values`` is not supplied.
        shap_values : ndarray of shape (n_samples, n_features), optional
            Precomputed SHAP values (for BB's log-odds output). Takes
            precedence over ``base_model``.
        binning_y : array-like of shape (n_samples,), optional
            Binary target for OptBinning. Choi & Cha use the original
            binary default label, so pass it when available.
            If ``None``, falls back to continuous binning on y_logit.
        eval_set, sample_weight : present for API compatibility; unused.
        """
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        self.n_features_ = n_features

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"f{i}" for i in range(n_features)]
        self.feature_names_ = feature_names

        mc = self._resolve_monotone(X, y_logit)

        if binning_y is not None:
            binning_target = np.asarray(binning_y).ravel()
            is_binary = True
        else:
            binning_target = np.asarray(y_logit).ravel()
            is_binary = False
        self._fit_bins(X_arr, binning_target, feature_names, mc, is_binary)

        if shap_values is None:
            if base_model is None:
                raise ValueError(
                    "ShapPdpSurrogate.fit requires either base_model or "
                    "shap_values."
                )
            import shap
            explainer = shap.TreeExplainer(base_model)
            raw = explainer.shap_values(X_arr)
            if isinstance(raw, list):
                # Binary classifier returns [class0, class1]; use class 1.
                raw = raw[1] if len(raw) == 2 else raw[-1]
            shap_values = np.asarray(raw)
        else:
            shap_values = np.asarray(shap_values)

        if shap_values.shape[:2] != X_arr.shape:
            raise ValueError(
                f"shap_values shape {shap_values.shape} does not match "
                f"X shape {X_arr.shape}."
            )

        self.bin_scores_ = {}
        self.bin_centers_ = {}
        self.smoothers_ = {}

        for j in range(n_features):
            if j not in self.bin_edges_:
                continue
            centers, shap_means = self._compute_shap_pdp(X_arr, shap_values, j)
            if len(centers) < 2:
                continue
            constraint = self._detect_shap_monotone(
                centers, shap_means,
                mc[j] if mc is not None else 0,
            )
            reg = self._fit_feature_smoother(centers, shap_means, constraint)
            smoothed = self._evaluate_smoother(reg, centers, shap_means)

            self.bin_centers_[j] = centers
            self.bin_scores_[j] = smoothed
            self.smoothers_[j] = reg

        self.model_ = self  # mark as fitted for _check_is_fitted

        # Mean contributions on training data → used to center contributions.
        raw = self._raw_contributions(X_arr)
        self.mean_contributions_ = raw.mean(axis=0)

        self._store_training_stats(X, y_logit)
        return self

    # ── Predict / contributions ───────────────────────────────────

    def _lookup_score(self, j, x_col):
        """Return the bin representative score for each sample of feature j."""
        if j not in self.bin_edges_ or j not in self.bin_scores_:
            return np.zeros(len(x_col))

        edges = self.bin_edges_[j]
        bin_idx = self._assign_bin_idx(x_col, edges)
        scores = self.bin_scores_[j]
        n_bins_fitted = len(scores)

        # Not all bins may have been observed during training. Map missing
        # bin indices to the nearest observed bin on the x-axis.
        if n_bins_fitted == len(edges) + 1:
            safe_idx = np.clip(bin_idx, 0, n_bins_fitted - 1)
            return scores[safe_idx]

        # Fallback: use bin_centers_ to find the nearest fitted bin.
        centers = self.bin_centers_[j]
        out = np.empty(len(x_col), dtype=float)
        for i, x in enumerate(x_col):
            out[i] = scores[int(np.argmin(np.abs(centers - x)))]
        return out

    def _raw_contributions(self, X_arr):
        """Uncentered per-feature scores f_j(x_j)."""
        X_arr = np.asarray(X_arr)
        out = np.zeros((X_arr.shape[0], self.n_features_))
        for j in range(self.n_features_):
            out[:, j] = self._lookup_score(j, X_arr[:, j])
        return out

    def contributions(self, X):
        """Centered contributions (SHAP-like, relative to training mean)."""
        self._check_is_fitted()
        return self._raw_contributions(np.asarray(X)) - self.mean_contributions_

    def predict(self, X):
        """Predict log-odds = base score + sum of centered contributions.

        base score = mean training prediction, recovered from the fact
        that `sum(mean_contributions_) + base_score = mean BB logit`.
        Here we approximate base score with the mean of total raw scores
        so that predict(X_train).mean() ≈ total raw mean.
        """
        self._check_is_fitted()
        raw = self._raw_contributions(np.asarray(X))
        # base = mean of raw row sums on training (captured via mean_contribs)
        # row = base + (raw - mean_raw) = raw + (base - mean_raw) = raw itself
        # if base := mean_raw. We use: predict = sum of raw contributions.
        return raw.sum(axis=1)

    # ── Scorecard bin structure (override) ────────────────────────

    def _get_feature_bins(self, X, n_bins=10):
        """Return bins using stored edges + smoothed scores.

        Produces the same structure as BinningSurrogate for the
        ScorecardModel conversion.
        """
        X_arr = np.asarray(X)
        contribs = self.contributions(X_arr)
        result = {}

        for j in range(X_arr.shape[1]):
            if j not in self.bin_edges_:
                continue
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
        return result

    @property
    def feature_importances_(self):
        """Range of f_j values across bins (proxy for feature influence)."""
        if self.bin_scores_ is None:
            raise RuntimeError("Not fitted.")
        imp = np.zeros(self.n_features_)
        for j, scores in self.bin_scores_.items():
            imp[j] = float(np.ptp(scores))
        return imp
