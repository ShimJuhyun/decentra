from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import spearmanr

from ..stats import TrainingStats


class BaseSurrogate(ABC):
    """Base class for all surrogate models.

    A surrogate approximates a black-box model's log-odds output
    with an interpretable model that provides per-feature contributions.

    Follows the sklearn estimator pattern: fit → predict / contributions.

    All surrogates store training data statistics in ``training_stats_``
    (a :class:`TrainingStats` instance) for downstream use
    (effort normalization in SIC, distribution monitoring, etc.).

    Monotone constraints
    --------------------
    All surrogates share monotone detection via Spearman correlation.
    Each subclass applies the detected constraints in its own way.

    Parameters (set by subclasses via __init__)
    -------------------------------------------
    monotone_detect_mode : {"auto", "none"}
    monotone_constraints : dict[str, int] or None
    """

    training_stats_: TrainingStats = None

    @abstractmethod
    def fit(self, X, y_logit, *, eval_set=None, sample_weight=None):
        """Fit the surrogate on teacher's log-odds predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_logit : array-like of shape (n_samples,)
            Teacher model's log-odds predictions on X.
        eval_set : tuple (X_val, y_logit_val), optional
            Validation set for early stopping.
        sample_weight : array-like of shape (n_samples,), optional
        """

    @abstractmethod
    def predict(self, X):
        """Predict log-odds.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """

    @abstractmethod
    def contributions(self, X):
        """Per-feature contribution matrix.

        Returns
        -------
        contribs : ndarray of shape (n_samples, n_features)
            contribs[i, j] = contribution of feature j for sample i.
        """

    def _feature_names(self, X):
        """Extract feature names from X."""
        if hasattr(X, 'columns'):
            return np.array(list(X.columns))
        return np.array([f'Feature_{i}' for i in range(np.asarray(X).shape[1])])

    def contribution_ranking(self, X):
        """Feature names ranked by |contribution| descending, per sample.

        Returns
        -------
        ndarray of shape (n_samples, n_features), dtype=str
            ranking[i, 0] = most important feature for sample i.
        """
        contribs = self.contributions(X)
        feature_names = self._feature_names(X)
        order = np.argsort(np.abs(contribs), axis=1)[:, ::-1]
        return feature_names[order]

    def adverse_features(self, X):
        """Features with negative contribution (감점요소), per sample.

        Higher score = better (가점). Negative contribution pushes
        score down = 감점.  Sorted by contribution ascending
        (largest deduction first).

        Returns
        -------
        list of ndarray[str]
            adverse[i] = feature names with contrib < 0 for sample i,
            sorted by most negative first.  Empty array if none.
        """
        contribs = self.contributions(X)
        feature_names = self._feature_names(X)
        result = []
        for i in range(len(contribs)):
            neg_idx = np.where(contribs[i] < 0)[0]
            if len(neg_idx) == 0:
                result.append(np.array([], dtype='<U1'))
            else:
                neg_order = neg_idx[np.argsort(contribs[i][neg_idx])]
                result.append(feature_names[neg_order])
        return result

    def _store_training_stats(self, X, y_logit=None):
        """Store training data statistics. Call at end of fit()."""
        self.training_stats_ = TrainingStats.from_data(
            X, y=y_logit, feature_names=self._feature_names(X).tolist()
        )

    def transform(self, X):
        """Apply fitted surrogate to new data.

        Returns
        -------
        dict with keys:
            predictions : ndarray of shape (n_samples,)
            contributions : ndarray of shape (n_samples, n_features)
            ranking : ndarray of shape (n_samples, n_features), dtype=str
            adverse : list of ndarray[str]
        """
        return {
            'predictions': self.predict(X),
            'contributions': self.contributions(X),
            'ranking': self.contribution_ranking(X),
            'adverse': self.adverse_features(X),
        }

    def fit_transform(self, X, y_logit, *, eval_set=None, sample_weight=None):
        """fit + transform on the same data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_logit : array-like of shape (n_samples,)
        eval_set : tuple (X_val, y_logit_val), optional
        sample_weight : array-like, optional

        Returns
        -------
        dict (same as transform)
        """
        self.fit(X, y_logit, eval_set=eval_set, sample_weight=sample_weight)
        return self.transform(X)

    def predict_with_contributions(self, X):
        """Predict and return contributions simultaneously.

        Returns
        -------
        pred : ndarray of shape (n_samples,)
        contribs : ndarray of shape (n_samples, n_features)
        """
        return self.predict(X), self.contributions(X)

    @property
    def is_additive(self):
        """Whether this surrogate is a GAM (purely additive, no interactions)."""
        return True  # subclasses override if needed

    @property
    def feature_importances_(self):
        """Per-feature importance (ndarray of shape n_features).

        Subclasses should override with model-appropriate implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement feature_importances_"
        )

    @staticmethod
    def filter_by_base_model(X, base_model, threshold=0):
        """Filter X to features used by the base model.

        Parameters
        ----------
        X : DataFrame
        base_model : fitted model with ``feature_importances_`` and
            ``feature_name_`` (e.g. LGBMClassifier).
        threshold : float, default=0
            Features with importance <= threshold are excluded.

        Returns
        -------
        X_filtered : DataFrame
            Columns with importance > threshold only.

        Examples
        --------
        >>> X_used = TreeSurrogate.filter_by_base_model(X_train, base_lgb)
        >>> surr.fit(X_used, y_logit)  # surrogate on used features only
        >>> # Or bypass: surr.fit(X_train, y_logit)  # all features
        """
        importances = base_model.feature_importances_
        names = base_model.feature_name_
        used = [name for name, imp in zip(names, importances) if imp > threshold]
        return X[used]

    def predict_with_contributions(self, X):
        """Return predictions and contributions together.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        contribs : ndarray of shape (n_samples, n_features)
        """
        return self.predict(X), self.contributions(X)

    # ── Monotone detection (shared) ───────────────────────────────

    @staticmethod
    def detect_monotone(X, y_logit, threshold=0.05):
        """Auto-detect monotone direction per feature via Spearman
        correlation with significance test.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_logit : array-like of shape (n_samples,)
        threshold : float, default=0.05
            p-value threshold.  If the Spearman correlation is not
            statistically significant (p >= threshold), the feature
            is treated as having no monotone relationship (0).

        Returns
        -------
        constraints : list[int]
            1 (increasing), -1 (decreasing), or 0 (not significant).
        correlations : list[float]
            Spearman correlation for each feature.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y_logit).ravel()
        constraints = []
        correlations = []

        for j in range(X_arr.shape[1]):
            corr, pval = spearmanr(X_arr[:, j], y_arr)
            if np.isnan(corr):
                corr, pval = 0.0, 1.0
            correlations.append(float(corr))
            if pval >= threshold:
                constraints.append(0)
            else:
                constraints.append(1 if corr > 0 else -1)

        return constraints, correlations

    def _resolve_monotone(self, X, y_logit):
        """Resolve monotone_detect_mode + monotone_constraints → list.

        Reads ``self.monotone_detect_mode`` and
        ``self.monotone_constraints`` (set by subclass ``__init__``).

        Sets ``self.monotone_constraints_`` and
        ``self.monotone_correlations_``.

        Returns
        -------
        list[int] or None
        """
        mode = getattr(self, "monotone_detect_mode", "none")
        user_mc = getattr(self, "monotone_constraints", None)

        if mode == "none" and not user_mc:
            self.monotone_constraints_ = None
            self.monotone_correlations_ = None
            return None

        n_features = np.asarray(X).shape[1]

        # Build name → index mapping
        if hasattr(X, "columns"):
            names = list(X.columns)
        else:
            names = [f"Feature_{i}" for i in range(n_features)]
        name_to_idx = {name: i for i, name in enumerate(names)}

        # Parse user-specified constraints
        user_specified = {}
        if user_mc:
            for key, val in user_mc.items():
                idx = name_to_idx.get(key)
                if idx is None:
                    raise KeyError(
                        f"Feature name '{key}' not found in columns: "
                        f"{names}"
                    )
                user_specified[idx] = int(val)

        if mode == "auto":
            auto_mc, corrs = self.detect_monotone(X, y_logit)
            self.monotone_correlations_ = corrs
            mc = [
                user_specified[j] if j in user_specified else auto_mc[j]
                for j in range(n_features)
            ]
        else:
            self.monotone_correlations_ = None
            mc = [user_specified.get(j, 0) for j in range(n_features)]

        self.monotone_constraints_ = mc
        return mc

    # ── Scorecard support ─────────────────────────────────────────

    _DISCRETE_THRESHOLD = 200

    def _get_feature_bins(self, X, n_bins=10):
        """Return per-feature bins for scorecard generation.

        Auto-detects discrete vs continuous contributions.
        - Discrete (unique values <= _DISCRETE_THRESHOLD): exact values.
        - Continuous: quantile-bin by feature values, mean score per bin.

        Override in subclasses for surrogate-specific binning.

        Returns
        -------
        dict : {feature_idx: [{'mask': ndarray[bool], 'score': float}, ...]}
        """
        contribs = self.contributions(X)
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        result = {}

        for j in range(n_features):
            scores = contribs[:, j]
            unique_rounded = np.unique(np.round(scores, 8))

            if len(unique_rounded) <= 1:
                continue

            if len(unique_rounded) <= self._DISCRETE_THRESHOLD:
                # Discrete: preserve every unique contribution value
                rounded = np.round(scores, 8)
                bins = []
                for us in unique_rounded:
                    mask = np.abs(rounded - us) < 1e-7
                    bins.append({"mask": mask, "score": float(us)})
            else:
                # Truly continuous: pre-bin by quantile
                bins = self._quantile_bin(X_arr[:, j], scores, n_bins)

            if bins:
                result[j] = bins

        return result

    @staticmethod
    def _quantile_bin(x_vals, contrib_vals, n_bins):
        """Quantile-bin by feature values, mean contribution per bin."""
        edges = np.unique(np.percentile(x_vals, np.linspace(0, 100, n_bins + 1)))

        bins = []
        for k in range(len(edges) - 1):
            if k == len(edges) - 2:
                mask = (x_vals >= edges[k]) & (x_vals <= edges[k + 1])
            else:
                mask = (x_vals >= edges[k]) & (x_vals < edges[k + 1])

            if mask.sum() == 0:
                continue

            bins.append({
                "mask": mask,
                "score": float(np.mean(contrib_vals[mask])),
            })
        return bins

    # ── Bin pruning ───────────────────────────────────────────────

    @staticmethod
    def _prune_bins(ranges, max_bins, min_ratio, n_samples,
                    max_bins_criterion="mse",
                    min_ratio_criterion="lower_count"):
        """Merge bins to satisfy max_bins and min_ratio constraints.

        Parameters
        ----------
        ranges : list of dict
            Sorted by x_min.
            Keys: x_min, x_max, score, count, (target_count optional).
        max_bins : int
        min_ratio : float
        n_samples : int
        max_bins_criterion : {"score_diff", "mse", "chi2"}
            Criterion for choosing which adjacent pair to merge when
            reducing bin count below max_bins.

            - ``"score_diff"``: merge the pair with smallest
              ``|s_i - s_j|``.
            - ``"mse"``: merge the pair with smallest Ward's SSE
              increase ``(n_i*n_j)/(n_i+n_j)*(s_i-s_j)²``.
            - ``"chi2"``: merge the pair with smallest chi-squared
              statistic on target rates (requires ``target_count``).
        min_ratio_criterion : {"lower_count", "mse", "chi2"}
            Criterion for choosing which adjacent neighbour to merge
            with when a bin violates min_ratio.

            - ``"lower_count"``: merge with the neighbour that has
              fewer samples (default — unreliable bin's score is not
              trusted, so minimise contamination of large bins).
            - ``"mse"`` / ``"chi2"``: merge with the neighbour that
              produces the smaller cost.

        Returns
        -------
        list of dict : pruned bins (sorted by x_min).
        """
        bins = [dict(r) for r in ranges]

        # ── cost functions ────────────────────────────────────

        def _score_diff(i):
            return abs(bins[i]["score"] - bins[i + 1]["score"])

        def _ward(i):
            a, b = bins[i], bins[i + 1]
            n = a["count"] + b["count"]
            if n == 0:
                return 0.0
            return (a["count"] * b["count"]) / n * (a["score"] - b["score"]) ** 2

        def _chi2(i):
            a, b = bins[i], bins[i + 1]
            n_a, n_b = a["count"], b["count"]
            t_a = a.get("target_count", 0)
            t_b = b.get("target_count", 0)
            if n_a == 0 or n_b == 0:
                return 0.0
            n = n_a + n_b
            r_pool = (t_a + t_b) / n if n > 0 else 0.0
            if r_pool == 0.0 or r_pool == 1.0:
                return 0.0
            chi2 = 0.0
            for obs, exp in [
                (t_a, n_a * r_pool),
                (n_a - t_a, n_a * (1 - r_pool)),
                (t_b, n_b * r_pool),
                (n_b - t_b, n_b * (1 - r_pool)),
            ]:
                if exp > 0:
                    chi2 += (obs - exp) ** 2 / exp
            return chi2

        _COST_FNS = {"score_diff": _score_diff, "mse": _ward, "chi2": _chi2}
        max_cost = _COST_FNS[max_bins_criterion]

        # ── merge helper ──────────────────────────────────────

        def _merge_at(bins, i):
            a, b = bins[i], bins[i + 1]
            total = a["count"] + b["count"]
            merged = {
                "x_min": a["x_min"],
                "x_max": b["x_max"],
                "score": (
                    (a["score"] * a["count"] + b["score"] * b["count"]) / total
                    if total > 0
                    else 0.0
                ),
                "count": total,
                "target_count": (
                    a.get("target_count", 0) + b.get("target_count", 0)
                ),
            }
            if "lower_edge" in a:
                merged["lower_edge"] = a["lower_edge"]
            if "upper_edge" in b:
                merged["upper_edge"] = b["upper_edge"]
            return bins[:i] + [merged] + bins[i + 2 :]

        # ── Stage 1: merge bins violating min_ratio ───────────

        if min_ratio_criterion == "lower_count":
            def _pick_neighbour(idx):
                if idx == 0:
                    return 0
                if idx == len(bins) - 1:
                    return idx - 1
                if bins[idx - 1]["count"] <= bins[idx + 1]["count"]:
                    return idx - 1
                return idx
        else:
            min_cost = _COST_FNS[min_ratio_criterion]

            def _pick_neighbour(idx):
                if idx == 0:
                    return 0
                if idx == len(bins) - 1:
                    return idx - 1
                left_cost = min_cost(idx - 1)
                right_cost = min_cost(idx)
                return idx - 1 if left_cost <= right_cost else idx

        changed = True
        while changed and len(bins) > 2:
            changed = False
            for idx in range(len(bins)):
                if bins[idx]["count"] / n_samples < min_ratio:
                    merge_idx = _pick_neighbour(idx)
                    bins = _merge_at(bins, merge_idx)
                    changed = True
                    break

        # ── Stage 2: merge to satisfy max_bins ────────────────

        while len(bins) > max_bins and len(bins) > 2:
            best = min(range(len(bins) - 1), key=max_cost)
            bins = _merge_at(bins, best)

        return bins

    # ── ScorecardModel construction ───────────────────────────────

    def to_scorecard_model(self, X, y_binary=None, feature_names=None,
                           n_bins=10, max_bins_per_feature=None,
                           min_bin_ratio=0.0, max_bins_criterion="mse",
                           min_ratio_criterion="lower_count"):
        """Convert this fitted surrogate to a standardised ScorecardModel.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_binary : array-like of shape (n_samples,), optional
            Original binary target. Required when using ``"chi2"``
            criterion.
        feature_names : list of str, optional
        n_bins : int, default=10
            Quantile fallback bin count (only for truly continuous
            contributions, e.g. LinearSurrogate).
        max_bins_per_feature : int, optional
            Maximum bins per feature after pruning.  ``None`` = no limit.
        min_bin_ratio : float, default=0.0
            Minimum proportion of samples per bin.
        max_bins_criterion : {"score_diff", "mse", "chi2"}
            Merge criterion for Stage 2 (max_bins enforcement).
        min_ratio_criterion : {"lower_count", "mse", "chi2"}
            Merge criterion for Stage 1 (min_ratio enforcement).

        Returns
        -------
        ScorecardModel
        """
        from ..scorecard_model import ScorecardModel, FeatureRule, BinRule

        X_arr = np.asarray(X)
        n_samples, n_features = X_arr.shape
        y_arr = (
            np.asarray(y_binary).ravel() if y_binary is not None else None
        )

        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(X.columns)
            else:
                feature_names = [f"Feature_{i}" for i in range(n_features)]

        # Base score = prediction − sum(contributions)
        preds = self.predict(X)
        contribs = self.contributions(X)
        base_score = float(np.mean(preds - contribs.sum(axis=1)))

        # Mask-based bins → ranges → prune → edge-based rules
        feature_bins = self._get_feature_bins(X, n_bins)
        features = []

        for j, bins in feature_bins.items():
            # Mask → range dicts
            ranges = []
            for b in bins:
                mask = b["mask"]
                x_vals = X_arr[mask, j]
                r = {
                    "x_min": float(x_vals.min()),
                    "x_max": float(x_vals.max()),
                    "score": b["score"],
                    "count": int(mask.sum()),
                }
                if y_arr is not None:
                    r["target_count"] = int(y_arr[mask].sum())
                if "lower_edge" in b:
                    r["lower_edge"] = b["lower_edge"]
                if "upper_edge" in b:
                    r["upper_edge"] = b["upper_edge"]
                ranges.append(r)
            ranges.sort(key=lambda r: r["x_min"])

            # Prune
            effective_max = max_bins_per_feature or len(ranges)
            if min_bin_ratio > 0 or effective_max < len(ranges):
                ranges = self._prune_bins(
                    ranges, effective_max, min_bin_ratio, n_samples,
                    max_bins_criterion, min_ratio_criterion,
                )

            if len(ranges) < 2:
                continue

            # Exact edges if available, otherwise x_min of next bin
            rules = []
            for k, r in enumerate(ranges):
                lower = r.get(
                    "lower_edge",
                    -np.inf if k == 0 else ranges[k]["x_min"],
                )
                upper = r.get(
                    "upper_edge",
                    np.inf
                    if k == len(ranges) - 1
                    else ranges[k + 1]["x_min"],
                )
                rules.append(BinRule(lower=lower, upper=upper, score=r["score"]))

            features.append(
                FeatureRule(name=feature_names[j], index=j, bins=rules)
            )

        sm = ScorecardModel(base_score=base_score, features=features)
        sm.fit(X, y_binary)
        return sm

    def scorecard(self, X, y_binary, feature_names=None, n_bins=10,
                  max_bins_per_feature=None, min_bin_ratio=0.0,
                  max_bins_criterion="mse",
                  min_ratio_criterion="lower_count"):
        """Build a Scorecard from this fitted surrogate.

        Shortcut for ``to_scorecard_model(…).scorecard(X, y_binary)``.
        """
        sm = self.to_scorecard_model(
            X, y_binary, feature_names, n_bins,
            max_bins_per_feature, min_bin_ratio,
            max_bins_criterion, min_ratio_criterion,
        )
        return sm.scorecard(X, y_binary)
