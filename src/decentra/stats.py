"""Training data statistics storage.

All surrogates and ScorecardModel store per-feature statistics
from the training data used during fit / scorecard creation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class FeatureStats:
    """Per-feature statistics from training data."""

    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    q25: float = 0.0
    q75: float = 0.0
    iqr: float = 0.0
    n_unique: int = 0
    missing_rate: float = 0.0

    # Histogram (for visualization / density checks)
    hist_counts: Optional[np.ndarray] = field(default=None, repr=False)
    hist_edges: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class TrainingStats:
    """Aggregated training data statistics.

    Stored in surrogate.training_stats_ and ScorecardModel.training_stats_.
    """

    n_samples: int = 0
    n_features: int = 0
    feature_names: list = field(default_factory=list)
    features: Dict[str, FeatureStats] = field(default_factory=dict)

    # Target statistics
    target_mean: float = 0.0
    target_std: float = 0.0

    @classmethod
    def from_data(cls, X, y=None, feature_names=None, n_hist_bins=50):
        """Compute statistics from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), optional
        feature_names : list of str, optional
        n_hist_bins : int, default=50
        """
        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape

        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                feature_names = [f'Feature_{i}' for i in range(n_features)]

        features = {}
        for j, name in enumerate(feature_names):
            col = X_arr[:, j]
            valid = col[~np.isnan(col)]

            if len(valid) == 0:
                features[name] = FeatureStats()
                continue

            q25, median, q75 = np.percentile(valid, [25, 50, 75])
            hist_counts, hist_edges = np.histogram(valid, bins=n_hist_bins)

            features[name] = FeatureStats(
                mean=float(np.mean(valid)),
                std=float(np.std(valid)),
                median=float(median),
                min=float(np.min(valid)),
                max=float(np.max(valid)),
                q25=float(q25),
                q75=float(q75),
                iqr=float(q75 - q25),
                n_unique=int(len(np.unique(valid))),
                missing_rate=float(1 - len(valid) / n_samples),
                hist_counts=hist_counts,
                hist_edges=hist_edges,
            )

        stats = cls(
            n_samples=n_samples,
            n_features=n_features,
            feature_names=feature_names,
            features=features,
        )

        if y is not None:
            y_arr = np.asarray(y, dtype=float)
            stats.target_mean = float(np.mean(y_arr))
            stats.target_std = float(np.std(y_arr))

        return stats

    def get_stds(self):
        """Return {feature_name: std} dict (for interventional-fidelity effort normalization)."""
        return {name: fs.std for name, fs in self.features.items()}

    def summary(self):
        """Return summary DataFrame."""
        import pandas as pd
        rows = []
        for name, fs in self.features.items():
            rows.append({
                'Feature': name,
                'Mean': round(fs.mean, 4),
                'Std': round(fs.std, 4),
                'Min': round(fs.min, 4),
                'Q25': round(fs.q25, 4),
                'Median': round(fs.median, 4),
                'Q75': round(fs.q75, 4),
                'Max': round(fs.max, 4),
                'Unique': fs.n_unique,
                'Missing': f'{fs.missing_rate:.1%}',
            })
        return pd.DataFrame(rows)
