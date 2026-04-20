"""Sequential priority-ordered LightGBM surrogate.

Fits a depth-1 LightGBM surrogate stage-by-stage, one feature at a time,
in an order determined by BB TreeSHAP priority. Two fitting modes:

- ``fit_mode='frozen'`` (residual backfitting). Stage k trains a single-
  feature depth-1 LightGBM on the current residual; the booster is then
  appended and its prediction subtracted from the residual. Each feature's
  shape is fixed once learned.

- ``fit_mode='cumulative'`` (curriculum warm-start). Stage k extends the
  active feature set by one, masks inactive columns to their training mean
  (so LightGBM cannot gain from them), and continues boosting from the
  previous stage via ``init_model``. Earlier features' shapes can adjust
  as later features are added.

Priority modes:

- ``priority_method='abs'``: features are ordered by mean |BB SHAP|
  across samples (standard feature-importance ordering).
- ``priority_method='signed_rejected'``: features are ordered by
  signed SHAP sum across *rejected* samples only. Positive sum means
  the feature pushes the rejected cohort further into adverse direction;
  this priority aligns fitting budget with the AdvTop-k target.

The surrogate is post-hoc: it explains an existing BB's output, not the
raw target. BB SHAP values (on training data) must be supplied via
``shap_values=...`` at fit time; alternatively, a trained tree booster
can be passed as ``base_model=...`` and TreeSHAP is computed internally.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import BaseSurrogate


class SequentialPrioritySurrogate(BaseSurrogate):
    """Depth-1 LightGBM surrogate fit in priority-ordered stages.

    Parameters
    ----------
    priority_method : {'abs', 'signed_rejected'}, default='signed_rejected'
    fit_mode : {'frozen', 'cumulative'}, default='cumulative'
    n_estimators_per_stage : int, default=100
        Boosting rounds per stage.
    learning_rate : float, default=0.05
    reject_percentile : float, default=90.0
        Used by 'signed_rejected' priority: samples with
        ``prob >= Q_{reject_percentile}`` are treated as rejected.
    random_state : int, default=42
    """

    def __init__(
        self,
        priority_method: str = "signed_rejected",
        fit_mode: str = "cumulative",
        n_estimators_per_stage: int = 100,
        learning_rate: float = 0.05,
        reject_percentile: float = 90.0,
        random_state: int = 42,
        verbose: int = -1,
    ):
        self.priority_method = priority_method
        self.fit_mode = fit_mode
        self.n_estimators_per_stage = n_estimators_per_stage
        self.learning_rate = learning_rate
        self.reject_percentile = reject_percentile
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.priority_ = None
        self.order_ = None
        self.boosters_ = None          # frozen: list of (j, booster)
        self.booster_ = None           # cumulative: final lgb.Booster
        self.base_score_ = 0.0
        self.feature_names_ = None
        self.n_features_ = None
        self.monotone_constraints_ = None
        self.monotone_correlations_ = None

    # ── Priority computation ─────────────────────────────────────

    def _compute_priority(self, shap_values, bb_prob=None):
        sv = np.asarray(shap_values)
        if self.priority_method == "abs":
            return np.abs(sv).mean(axis=0)
        if self.priority_method == "signed_rejected":
            if bb_prob is None:
                mask = np.ones(sv.shape[0], dtype=bool)
            else:
                cutoff = np.percentile(bb_prob, self.reject_percentile)
                mask = bb_prob >= cutoff
                if mask.sum() == 0:
                    mask = np.ones(sv.shape[0], dtype=bool)
            return sv[mask].sum(axis=0)
        raise ValueError(
            f"Unknown priority_method: {self.priority_method!r}"
        )

    # ── Fit modes ────────────────────────────────────────────────

    def _fit_frozen(self, X_arr, y_logit):
        import lightgbm as lgb

        self.boosters_ = []
        residual = y_logit - y_logit.mean()
        self.base_score_ = float(y_logit.mean())

        params = dict(
            max_depth=1,
            num_leaves=2,
            n_estimators=self.n_estimators_per_stage,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=1.0,
            min_child_samples=20,
            verbose=-1,
            random_state=self.random_state,
            n_jobs=-1,
        )

        for j in self.order_:
            X_j = X_arr[:, [j]]
            reg = lgb.LGBMRegressor(**params)
            reg.fit(X_j, residual)
            residual = residual - reg.predict(X_j)
            self.boosters_.append((int(j), reg))

    def _fit_cumulative(self, X_arr, y_logit):
        import lightgbm as lgb

        n_samples, p = X_arr.shape
        self.col_means_ = X_arr.mean(axis=0)
        self.base_score_ = 0.0

        booster = None
        params = dict(
            objective="regression",
            metric="rmse",
            max_depth=1,
            num_leaves=2,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=1.0,
            min_child_samples=20,
            verbose=-1,
            seed=self.random_state,
            num_threads=-1,
        )

        active = np.zeros(p, dtype=bool)
        for j in self.order_:
            active[j] = True
            X_masked = self._mask_inactive(X_arr, active)
            dataset = lgb.Dataset(
                X_masked, label=y_logit, free_raw_data=False,
            )
            booster = lgb.train(
                params,
                dataset,
                num_boost_round=self.n_estimators_per_stage,
                init_model=booster,
                keep_training_booster=True,
            )
        self.booster_ = booster

    def _mask_inactive(self, X_arr, active):
        X_masked = X_arr.copy()
        for j in range(X_arr.shape[1]):
            if not active[j]:
                X_masked[:, j] = self.col_means_[j]
        return X_masked

    # ── Fit ──────────────────────────────────────────────────────

    def fit(
        self,
        X,
        y_logit,
        *,
        shap_values=None,
        base_model=None,
        bb_prob=None,
        eval_set=None,
        sample_weight=None,
    ):
        X_arr = np.asarray(X, dtype=float)
        self.n_features_ = X_arr.shape[1]

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [
                f"f{i}" for i in range(self.n_features_)
            ]

        # Resolve SHAP values
        if shap_values is None:
            if base_model is None:
                raise ValueError(
                    "SequentialPrioritySurrogate.fit requires either "
                    "base_model or shap_values."
                )
            import shap
            raw = shap.TreeExplainer(base_model).shap_values(X_arr)
            if isinstance(raw, list):
                raw = raw[1] if len(raw) == 2 else raw[-1]
            shap_values = np.asarray(raw)
            if shap_values.ndim == 3:
                shap_values = (
                    shap_values[..., 1]
                    if shap_values.shape[-1] == 2
                    else shap_values[..., -1]
                )
            if shap_values.shape[1] == self.n_features_ + 1:
                shap_values = shap_values[:, : self.n_features_]
        else:
            shap_values = np.asarray(shap_values)

        if shap_values.shape != X_arr.shape:
            raise ValueError(
                f"shap_values shape {shap_values.shape} != "
                f"X shape {X_arr.shape}"
            )

        self.priority_ = self._compute_priority(shap_values, bb_prob)
        self.order_ = np.argsort(-self.priority_)

        y_arr = np.asarray(y_logit, dtype=float).ravel()

        if self.fit_mode == "frozen":
            self._fit_frozen(X_arr, y_arr)
        elif self.fit_mode == "cumulative":
            self._fit_cumulative(X_arr, y_arr)
        else:
            raise ValueError(f"Unknown fit_mode: {self.fit_mode!r}")

        self.model_ = self  # mark fitted
        self._store_training_stats(X, y_logit)
        return self

    # ── Predict / contributions ──────────────────────────────────

    def predict(self, X):
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        if self.fit_mode == "frozen":
            out = np.full(X_arr.shape[0], self.base_score_)
            for j, booster in self.boosters_:
                out = out + booster.predict(X_arr[:, [j]])
            return out
        # cumulative
        X_masked = X_arr.copy()
        return self.booster_.predict(X_masked)

    def contributions(self, X):
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        n = X_arr.shape[0]
        out = np.zeros((n, self.n_features_))

        if self.fit_mode == "frozen":
            for j, booster in self.boosters_:
                shape_contrib = booster.predict(X_arr[:, [j]])
                out[:, j] = shape_contrib - shape_contrib.mean()
            return out

        # cumulative: use LightGBM's pred_contrib (depth-1 → exact SHAP)
        raw = self.booster_.predict(X_arr, pred_contrib=True)
        raw = np.asarray(raw)
        # pred_contrib returns (n_samples, n_features + 1); last col is base
        return raw[:, : self.n_features_] - raw[:, : self.n_features_].mean(
            axis=0
        )

    # ── Introspection ────────────────────────────────────────────

    @property
    def feature_importances_(self):
        if self.priority_ is None:
            raise RuntimeError("Not fitted.")
        return np.abs(self.priority_)

    @property
    def is_additive(self):
        return True
