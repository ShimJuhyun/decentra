import numpy as np
import lightgbm as lgb
import shap

from .base import BaseSurrogate

_DEPTH_LEAVES = {1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 63}


class TreeSurrogate(BaseSurrogate):
    """Depth-constrained LightGBM surrogate.

    For ``max_depth=1`` (scorecard), ``pred_contrib`` gives exact
    additive contributions identical to SHAP. For deeper trees,
    SHAP is used automatically.

    Parameters
    ----------
    max_depth : int, default=1
        Maximum tree depth. 1 = scorecard (additive).
    n_estimators : int, default=1000
    learning_rate : float, default=0.05
    early_stopping_rounds : int, default=50
    random_state : int, default=317
    monotone_detect_mode : {"auto", "none"}, default="none"
        - ``"auto"``: auto-detect monotone direction from Spearman
          correlation (p-value < 0.05) for features NOT specified
          in ``monotone_constraints``.
        - ``"none"``: no auto-detection. Only ``monotone_constraints``
          is used.
    monotone_constraints : dict, optional
        Per-feature monotone constraints.  Keys are **feature names**
        (str).  Values: 1 (increasing), -1 (decreasing), 0 (none).

        When ``monotone_detect_mode="auto"``, features listed here are
        **excluded from auto-detection** and use the specified value
        directly.  Use ``0`` to explicitly exclude a feature from
        monotone constraints.

        Examples::

            monotone_constraints={"age": -1}          # force decreasing
            monotone_constraints={"DebtRatio": 0}     # exclude from auto
            monotone_constraints={"age": -1, "DebtRatio": 0}
    **lgb_params : dict
        Extra keyword arguments passed to LGBMRegressor.

    Attributes
    ----------
    monotone_constraints_ : list[int] or None
        Resolved per-feature constraints after fit.
    monotone_correlations_ : list[float] or None
        Spearman correlations (only when ``monotone_detect_mode="auto"``).

    Examples
    --------
    >>> surr = TreeSurrogate(monotone_detect_mode="auto")
    >>> surr.fit(X_train, teacher_logit, eval_set=(X_val, val_logit))
    >>> surr.monotone_constraints_
    [1, -1, 1, 0, -1, ...]

    >>> surr = TreeSurrogate(
    ...     monotone_detect_mode="auto",
    ...     monotone_constraints={"DebtRatio": 0, "age": -1},
    ... )
    """

    def __init__(
        self,
        max_depth=1,
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        random_state=317,
        monotone_detect_mode="none",
        monotone_constraints=None,
        verbose=0,
        **lgb_params,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.monotone_detect_mode = monotone_detect_mode
        self.monotone_constraints = monotone_constraints
        self.verbose = verbose
        self.lgb_params = lgb_params
        self.model_ = None
        self.monotone_constraints_ = None
        self.monotone_correlations_ = None

    @property
    def is_additive(self):
        return self.max_depth == 1

    @property
    def feature_importances_(self):
        """LightGBM split-based feature importance."""
        return self.model_.feature_importances_

    # ── Fit ───────────────────────────────────────────────────

    def fit(self, X, y_logit, *, eval_set=None, sample_weight=None):
        num_leaves = _DEPTH_LEAVES.get(self.max_depth, 2 ** self.max_depth - 1)
        params = dict(
            max_depth=self.max_depth,
            num_leaves=num_leaves,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=self.random_state,
            n_jobs=-1,
        )
        params.update(self.lgb_params)

        # Monotone constraints
        mc = self._resolve_monotone(X, y_logit)
        if mc is not None:
            params["monotone_constraints"] = mc

        self.model_ = lgb.LGBMRegressor(**params)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds,
                                   verbose=bool(self.verbose)),
                lgb.log_evaluation(self.verbose),
            ]

        self.model_.fit(X, y_logit, **fit_kwargs)
        self._store_training_stats(X, y_logit)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def contributions(self, X):
        if self.max_depth == 1:
            return self.model_.predict(X, pred_contrib=True)[:, :-1]
        return np.array(shap.TreeExplainer(self.model_).shap_values(X))

    # ── depth-1: threshold 기반 exact binning ─────────────────

    def _get_feature_bins(self, X, n_bins=10):
        """For depth=1, extract exact split thresholds from the tree.

        LightGBM convention: x <= threshold → left leaf.
        BinRule convention:  lower <= x < upper.
        Alignment:  upper = nextafter(threshold) so that x=threshold
        satisfies x < upper, landing in the left bin.
        """
        if self.max_depth != 1:
            return super()._get_feature_bins(X, n_bins)

        X_arr = np.asarray(X)
        contribs = self.contributions(X)

        # Collect every split threshold per feature from all trees
        feat_thresholds = {}
        for tree_info in self.model_.booster_.dump_model()["tree_info"]:
            node = tree_info["tree_structure"]
            if "split_feature" not in node:
                continue
            j = node["split_feature"]
            feat_thresholds.setdefault(j, set()).add(node["threshold"])

        result = {}
        for j, thresholds in feat_thresholds.items():
            # nextafter aligns LightGBM's <= with BinRule's <
            edges = (
                [-np.inf]
                + [np.nextafter(t, np.inf) for t in sorted(thresholds)]
                + [np.inf]
            )

            bins = []
            for k in range(len(edges) - 1):
                mask = (X_arr[:, j] >= edges[k]) & (X_arr[:, j] < edges[k + 1])
                if mask.sum() == 0:
                    continue
                bins.append({
                    "mask": mask,
                    "score": float(np.mean(contribs[mask, j])),
                    "lower_edge": edges[k],
                    "upper_edge": edges[k + 1],
                })

            if len(bins) > 1:
                result[j] = bins

        return result
