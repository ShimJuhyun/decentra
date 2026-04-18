from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .stats import TrainingStats


@dataclass
class BinRule:
    """A single bin: lower (inclusive) <= x < upper (exclusive).

    For the last bin, upper = +inf so it catches all x >= lower.
    """

    lower: float
    upper: float
    score: float

    def contains(self, x):
        return (x >= self.lower) & (x < self.upper)


@dataclass
class FeatureRule:
    """Binning rules for one feature."""

    name: str
    index: int
    bins: List[BinRule] = field(default_factory=list)

    def assign_scores(self, x_vals):
        """Map feature values -> bin scores."""
        x = np.asarray(x_vals, dtype=float)
        scores = np.zeros_like(x)
        for b in self.bins:
            scores[b.contains(x)] = b.score
        return scores


class ScorecardModel:
    """Standardised scorecard model: base_score + per-feature bin rules.

    Produced by any surrogate via ``surr.to_scorecard_model(X_train)``.

    Lifecycle
    ---------
    1. **Creation**: ``sm = surr.to_scorecard_model(X_train)``
       Bin structure + scores are fixed. training_stats_ and
       mean_contributions_ are automatically computed.

    2. **fit(X_train, y_binary)**: Compute deployment-ready state.
       Stores training_stats_, mean_contributions_, and population-level
       reason code rankings. Call once before deployment.

    3. **transform(X_new)**: Apply fitted scorecard to new data.
       Returns predictions, centered contributions, adverse features,
       and contribution rankings.

    4. **fit_transform(X_train, y_binary)**: fit + transform on same data.

    Examples
    --------
    >>> sm = surr.to_scorecard_model(X_train)
    >>> sm.fit(X_train, y_binary_train)
    >>> result = sm.transform(X_new)
    >>> result['predictions']    # ndarray
    >>> result['contributions']  # ndarray (centered)
    >>> result['adverse']        # list of ndarray[str]
    >>> result['ranking']        # ndarray of str
    """

    def __init__(self, base_score=0.0, features=None,
                 mean_contributions_=None, training_stats_=None,
                 is_fitted_=False):
        self.base_score = base_score
        self.features = features if features is not None else []
        self.mean_contributions_ = mean_contributions_
        self.training_stats_ = training_stats_
        self.is_fitted_ = is_fitted_

    # ── fit / transform / fit_transform ───────────────────────

    def fit(self, X, y_binary=None):
        """Prepare scorecard for deployment.

        - Compute mean_contributions_ for centering
        - Store training_stats_ (feature distributions)
        - Build display Scorecard if y_binary provided

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (same data used in to_scorecard_model).
        y_binary : array-like of shape (n_samples,), optional
            Binary target. If provided, stores Scorecard display object.

        Returns
        -------
        self
        """
        raw = self._raw_contributions(X)
        self.mean_contributions_ = np.mean(raw, axis=0)
        self.training_stats_ = TrainingStats.from_data(X)

        if y_binary is not None:
            from .scorecard import Scorecard
            self.scorecard_ = Scorecard.from_scorecard_model(
                self, X, np.asarray(y_binary)
            )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply fitted scorecard to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dict with keys:
            predictions : ndarray of shape (n_samples,)
            contributions : ndarray of shape (n_samples, n_features)
                Centered contributions.
            ranking : ndarray of shape (n_samples, n_features), dtype=str
                Feature names ranked by |contribution| descending.
            adverse : list of ndarray[str]
                Adverse features per sample (contrib < 0, sorted).
        """
        if not self.is_fitted_:
            raise RuntimeError("Not fitted. Call fit() first.")
        return {
            'predictions': self.predict(X),
            'contributions': self.contributions(X),
            'ranking': self.contribution_ranking(X),
            'adverse': self.adverse_features(X),
        }

    def fit_transform(self, X, y_binary=None):
        """fit + transform on the same data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_binary : array-like of shape (n_samples,), optional

        Returns
        -------
        dict (same as transform)
        """
        self.fit(X, y_binary)
        return self.transform(X)

    # ── Scoring ───────────────────────────────────────────────

    def predict(self, X):
        """Predict total score using bin rules.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        X_arr = np.asarray(X)
        total = np.full(X_arr.shape[0], self.base_score)
        for feat in self.features:
            total += feat.assign_scores(X_arr[:, feat.index])
        return total

    # ── Contributions & Attribution ──────────────────────────

    def _raw_contributions(self, X):
        """Uncentered bin scores per feature."""
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]
        n_features = X_arr.shape[1]
        contribs = np.zeros((n_samples, n_features))
        for feat in self.features:
            contribs[:, feat.index] = feat.assign_scores(X_arr[:, feat.index])
        return contribs

    def contributions(self, X):
        """Centered per-feature contributions.

        Returns raw - mean_contributions_ if fitted,
        otherwise raw (backward compatible).

        Returns
        -------
        ndarray of shape (n_samples, n_total_features)
        """
        raw = self._raw_contributions(X)
        if self.mean_contributions_ is not None:
            return raw - self.mean_contributions_
        return raw

    def predict_with_contributions(self, X):
        """Predict and return contributions simultaneously.

        Returns
        -------
        pred : ndarray of shape (n_samples,)
        contribs : ndarray of shape (n_samples, n_features)
        """
        return self.predict(X), self.contributions(X)

    def _feature_names(self, X):
        if hasattr(X, 'columns'):
            return np.array(list(X.columns))
        return np.array([f'Feature_{i}' for i in range(np.asarray(X).shape[1])])

    def contribution_ranking(self, X):
        """Feature names ranked by |contribution| descending, per sample.

        Returns
        -------
        ndarray of shape (n_samples, n_features), dtype=str
        """
        contribs = self.contributions(X)
        feature_names = self._feature_names(X)
        order = np.argsort(np.abs(contribs), axis=1)[:, ::-1]
        return feature_names[order]

    def adverse_features(self, X):
        """Features with negative contribution (adverse), per sample.

        Returns
        -------
        list of ndarray[str]
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

    # ── Scorecard display ─────────────────────────────────────

    def scorecard(self, X, y_binary):
        """Build a display Scorecard from this model + observed data.

        Convenience method. Equivalent to fit(X, y_binary).scorecard_
        """
        from .scorecard import Scorecard
        return Scorecard.from_scorecard_model(self, X, y_binary)

    # ── Serialisation ─────────────────────────────────────────

    def to_dict(self):
        return {
            "base_score": self.base_score,
            "features": [
                {
                    "name": f.name,
                    "index": f.index,
                    "bins": [
                        {"lower": b.lower, "upper": b.upper, "score": b.score}
                        for b in f.bins
                    ],
                }
                for f in self.features
            ],
        }

    @classmethod
    def from_dict(cls, d):
        features = []
        for fd in d["features"]:
            bins = [BinRule(**bd) for bd in fd["bins"]]
            features.append(
                FeatureRule(name=fd["name"], index=fd["index"], bins=bins)
            )
        return cls(base_score=d["base_score"], features=features)

    def __repr__(self):
        n_feat = len(self.features)
        n_bins = sum(len(f.bins) for f in self.features)
        fitted = ", fitted" if self.is_fitted_ else ""
        return (
            f"ScorecardModel(base={self.base_score:.4f}, "
            f"{n_feat} features, {n_bins} bins{fitted})"
        )
