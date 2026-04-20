"""End-to-end surrogate benchmarking.

One call produces prediction, attribution, and (optionally) interventional
fidelity for every registered surrogate on a single train/test split.

``run_benchmark`` is the building block of the per-fold CV executor used in
Phase 6.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.stats import spearmanr

from ..surrogate.base import BaseSurrogate
from ..metrics.prediction import prediction_fidelity
from ..metrics.named import (
    attribution_fidelity_named,
    random_baseline_advtopk_named,
)


# A surrogate factory: called with feature_names to produce a fresh instance.
SurrogateFactory = Callable[[List[str]], BaseSurrogate]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    surrogates: Dict[str, SurrogateFactory]
    """name → callable(feature_names) → unfitted surrogate."""

    reject_percentile: float = 90.0
    agree_cutoff: float = 0.10
    ks: Sequence[int] = (1, 3, 4)
    adv_ks: Sequence[int] = (1, 4)
    random_state: int = 42
    target_scale: str = "score"
    """'score' if surrogates are trained on bb_score, 'logit' otherwise."""
    compute_interventional: bool = False
    """If True, compute interventional fidelity for every surrogate whose
    ``to_scorecard_model`` is defined (typically only TreeSurrogate(max_depth=1))."""
    missing_policy: str = "zero"
    """align_attributions missing policy."""


@dataclass
class BenchmarkResult:
    rows: List[Dict] = field(default_factory=list)
    contribs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    adverse_contribs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    models: Dict[str, BaseSurrogate] = field(default_factory=dict)
    info: Dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path, *, save_models: bool = False):
        """Save rows as JSON + full object as pickle.

        Parameters
        ----------
        path : str | Path
            Path without extension. ``<path>.json`` and ``<path>.pkl`` are
            written.
        save_models : bool, default=False
            Pickle includes fitted surrogate models. Turn off to keep files
            small when aggregating across many folds.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p.with_suffix(".json"), "w") as f:
            json.dump(
                {"rows": self.rows, "info": self.info},
                f, indent=2, default=_json_default,
            )

        dump = {
            "rows": self.rows,
            "contribs": {k: v for k, v in self.contribs.items()},
            "adverse_contribs": {
                k: v for k, v in self.adverse_contribs.items()
            },
            "predictions": self.predictions,
            "info": self.info,
        }
        if save_models:
            dump["models"] = self.models
        with open(p.with_suffix(".pkl"), "wb") as f:
            pickle.dump(dump, f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def _bb_adverse_from_shap(
    bb_shap: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """BB TreeSHAP is on log-odds scale: SHAP>0 already = adverse."""
    return pd.DataFrame(bb_shap, columns=list(feature_names))


def _extras(y_true, y_prob) -> Dict[str, float]:
    """AUC, Brier, KS."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    brier = float(brier_score_loss(y_true, y_prob))
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted) / max(y_true.sum(), 1)
    cum_neg = np.cumsum(1 - y_sorted) / max((1 - y_true).sum(), 1)
    ks = float(np.max(np.abs(cum_pos - cum_neg)))
    return {"AUC": auc, "Brier": brier, "KS": ks}


def run_benchmark(
    *,
    teacher,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_target: np.ndarray,
    y_test_binary: np.ndarray,
    bb_shap_test: np.ndarray,
    bb_prob_test: np.ndarray,
    bb_score_test: np.ndarray,
    feature_names: Sequence[str],
    config: BenchmarkConfig,
    train_val_split: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> BenchmarkResult:
    """Fit every surrogate and compute metrics against the BB.

    Parameters
    ----------
    teacher : fitted estimator or None
        Needed only if ``config.compute_interventional=True`` (for probing).
    X_train, X_test : DataFrame with column == feature_names
    y_train_target : the surrogate's regression target on train
        (usually teacher's bb_score_train or bb_logit_train).
    y_test_binary : binary default labels on the test set (for AUC/KS/Brier).
    bb_shap_test, bb_prob_test, bb_score_test : teacher outputs on test
    train_val_split : (train_positions, val_positions) in X_train.index space;
        each surrogate uses these for early stopping. None ⇒ no early stop.

    Returns
    -------
    BenchmarkResult
    """
    feature_names = list(feature_names)
    reject = bb_prob_test >= np.percentile(bb_prob_test, config.reject_percentile)
    bb_adverse = _bb_adverse_from_shap(bb_shap_test, feature_names)
    result = BenchmarkResult()

    # Random baselines (computed once, attached to info)
    random_adv = {
        f"Random_AdvTop{k}": random_baseline_advtopk_named(
            bb_adverse, reject, k, len(feature_names)
        )
        for k in config.adv_ks
    }
    result.info.update({
        "n_test": int(len(X_test)),
        "n_reject": int(reject.sum()),
        "n_features": len(feature_names),
        "reject_percentile": config.reject_percentile,
        "random_state": config.random_state,
        "random_baseline": random_adv,
        "feature_names": feature_names,
    })

    # Optional early-stop splits
    eval_set = None
    if train_val_split is not None:
        tr_pos, val_pos = train_val_split
        X_tr = X_train.iloc[tr_pos]
        X_val = X_train.iloc[val_pos]
        y_tr = y_train_target[tr_pos]
        y_val = y_train_target[val_pos]
        eval_set = (X_val, y_val)
        fit_X, fit_y = X_tr, y_tr
    else:
        fit_X, fit_y = X_train, y_train_target

    for name, factory in config.surrogates.items():
        t0 = time.time()
        surr = factory(feature_names)
        try:
            surr.fit(fit_X, fit_y, eval_set=eval_set)
        except TypeError:
            # Some surrogates (e.g. OptBinning) may not accept eval_set
            surr.fit(fit_X, fit_y)

        pred = np.asarray(surr.predict(X_test))
        contribs_arr = np.asarray(surr.contributions(X_test))
        # Adverse contributions (adverse > 0)
        adv_df = surr.adverse_contributions(
            X_test, target_scale=config.target_scale
        )

        # Prediction fidelity
        pred_fid = prediction_fidelity(
            y_logit=bb_score_test,      # both in 'score' scale when target_scale='score'
            surr_pred=pred,
            prob_true=bb_prob_test,
            surr_prob=bb_prob_test,     # placeholder; Agree is noop here
            agree_cutoff=config.agree_cutoff,
        )
        # Replace Agree with teacher-vs-surrogate score cutoff comparison
        th = np.percentile(bb_score_test, 100 - config.reject_percentile)
        pred_fid["Agree"] = float(
            np.mean((bb_score_test <= th) == (pred <= th))
        )

        # Attribution fidelity (named)
        attr_fid = attribution_fidelity_named(
            bb_adverse, adv_df, reject,
            ks=config.ks, adv_ks=config.adv_ks,
            missing=config.missing_policy,
        )

        # Extra performance metrics (y_test_binary vs surrogate prob estimate)
        try:
            from .._utils import sigmoid
            surr_prob = sigmoid(pred)  # only meaningful if trained on logit
        except Exception:
            surr_prob = None
        extras = _extras(y_test_binary, bb_prob_test)  # BB extras as reference
        extras = {f"BB_{k}": v for k, v in extras.items()}

        row = {
            "surrogate": name,
            **pred_fid,
            **attr_fid,
            **extras,
            "fit_seconds": round(time.time() - t0, 2),
        }

        result.rows.append(row)
        result.contribs[name] = pd.DataFrame(
            contribs_arr, columns=feature_names
        )
        result.adverse_contribs[name] = adv_df
        result.predictions[name] = pred
        result.models[name] = surr

    return result
