import numpy as np

from .base import BaseSurrogate


class EBMSurrogate(BaseSurrogate):
    """Explainable Boosting Machine (EBM) surrogate.

    Wraps InterpretML's ExplainableBoostingRegressor.
    With ``interactions=0`` this is a GAM (additive).
    With ``interactions>0`` this is a GA²M (includes pairwise terms).

    Parameters
    ----------
    interactions : int, default=0
        Number of pairwise interaction terms.
        0 = pure GAM (additive).
    random_state : int, default=317
    n_jobs : int, default=1
    monotone_detect_mode : {"auto", "none"}, default="none"
    monotone_constraints : dict[str, int], optional

    Examples
    --------
    >>> surr = EBMSurrogate(monotone_detect_mode="auto")
    >>> surr.fit(X_train, teacher_logit)
    >>> surr.is_additive
    True
    """

    def __init__(self, interactions=0, validation_size=0.2,
                 early_stopping_rounds=50, early_stopping_tolerance=1e-4,
                 random_state=317, n_jobs=1,
                 monotone_detect_mode="none", monotone_constraints=None):
        self.interactions = interactions
        self.validation_size = validation_size
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_detect_mode = monotone_detect_mode
        self.monotone_constraints = monotone_constraints
        self.model_ = None
        self.monotone_constraints_ = None
        self.monotone_correlations_ = None

    @property
    def is_additive(self):
        return self.interactions == 0

    @property
    def feature_importances_(self):
        """EBM term importances (mean absolute contribution per term)."""
        return np.array(self.model_.term_importances())

    def fit(self, X, y_logit, *, eval_set=None, sample_weight=None):
        from interpret.glassbox import ExplainableBoostingRegressor

        mc = self._resolve_monotone(X, y_logit)

        self.model_ = ExplainableBoostingRegressor(
            interactions=self.interactions,
            validation_size=self.validation_size,
            early_stopping_rounds=self.early_stopping_rounds,
            early_stopping_tolerance=self.early_stopping_tolerance,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            monotone_constraints=mc,
        )
        self.model_.fit(X, y_logit)
        self._store_training_stats(X, y_logit)
        return self

    def predict(self, X):
        self._check_is_fitted()
        return self.model_.predict(X)

    def contributions(self, X):
        self._check_is_fitted()
        el = self.model_.explain_local(X)
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        n_features = len(el._internal_obj['specific'][0]['scores'])
        return np.array([
            [el._internal_obj['specific'][i]['scores'][j]
             for j in range(n_features)]
            for i in range(n_samples)
        ])
