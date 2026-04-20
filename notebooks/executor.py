"""One-shot executor for the P5 pilot analysis.

Given a fitted teacher and a train/test split, produces every metric that the
N01–N07 notebooks generate (prediction + attribution + calibration +
interventional + cutoff + complexity), and saves the aggregated result to
``<out_dir>/result_<tag>.{pkl,json}``.

This is the unit that `N_CV` loops over, one call per fold.

Typical usage
-------------
    >>> from executor import run_case
    >>> result = run_case(
    ...     teacher=lgb_clf,
    ...     X_train=X_tr, y_train=y_tr,
    ...     X_test=X_te,  y_test=y_te,
    ...     out_dir='outputs/NCV/fold_0',
    ...     tag='fold0',
    ... )
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score

from decentra._utils import transform_logit_to_score
from decentra.surrogate import (
    TreeSurrogate, EBMSurrogate, LinearSurrogate, BinningSurrogate,
)
from decentra.calibration import FeatureCalibrator, BinCalibrator
from decentra.experiments import BenchmarkConfig, run_benchmark
from decentra.metrics.named import attribution_fidelity_named
from decentra.metrics.interventional import (
    extract_bin_structure, interventional_fidelity,
)


# ----- surrogate zoo ---------------------------------------------------------

def default_surrogate_factories(n_jobs_ebm: int = 8) -> Dict:
    """The 8 canonical surrogates + monotone variants used by the pilot."""
    return {
        "Tree-d1":      lambda fn: TreeSurrogate(max_depth=1, verbose=0),
        "Tree-d1-mono": lambda fn: TreeSurrogate(
            max_depth=1, verbose=0, monotone_detect_mode="auto"),
        "Tree-d3":      lambda fn: TreeSurrogate(max_depth=3, verbose=0),
        "Tree-d6":      lambda fn: TreeSurrogate(max_depth=6, verbose=0),
        "EBM":          lambda fn: EBMSurrogate(interactions=0, n_jobs=n_jobs_ebm),
        "EBM-mono":     lambda fn: EBMSurrogate(
            interactions=0, n_jobs=n_jobs_ebm, monotone_detect_mode="auto"),
        "Ridge":        lambda fn: LinearSurrogate(method="ridge", alpha=1.0),
        "OptBin+Ridge": lambda fn: BinningSurrogate(
            method="ridge", alpha=1.0, max_n_bins=10),
    }


@dataclass
class CaseResult:
    """Aggregated result for one case (one fold or one pilot split)."""

    tag: str
    bench_rows: List[Dict] = field(default_factory=list)
    calibration_rows: List[Dict] = field(default_factory=list)
    interventional_rows: List[Dict] = field(default_factory=list)
    cutoff_rows: List[Dict] = field(default_factory=list)
    scorecard_rows: List[Dict] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.bench_rows)

    def save(self, out_dir) -> Tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"result_{self.tag}.json"
        pkl_path = out_dir / f"result_{self.tag}.pkl"

        # JSON: rows + info (human-readable)
        payload = {
            "tag": self.tag,
            "info": self.info,
            "bench_rows": self.bench_rows,
            "calibration_rows": self.calibration_rows,
            "interventional_rows": self.interventional_rows,
            "cutoff_rows": self.cutoff_rows,
            "scorecard_rows": self.scorecard_rows,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

        # Pickle: full object (same payload currently, but extensible)
        with open(pkl_path, "wb") as f:
            pickle.dump(asdict(self), f)
        return pkl_path, json_path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _teacher_outputs(teacher, X_test, X_train, reject_pct=90):
    """Extract BB SHAP, prob, score for test and score for train."""
    prob_te = teacher.predict_proba(X_test)[:, 1]
    prob_tr = teacher.predict_proba(X_train)[:, 1]
    shap_vals = shap.TreeExplainer(teacher).shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.asarray(shap_vals, dtype=np.float32)
    score_te = transform_logit_to_score(prob_te)
    score_tr = transform_logit_to_score(prob_tr)
    reject = prob_te >= np.percentile(prob_te, reject_pct)
    return {
        "prob_te": prob_te, "prob_tr": prob_tr,
        "shap_te": shap_vals,
        "score_te": score_te, "score_tr": score_tr,
        "reject": reject,
    }


def run_case(
    *,
    teacher,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir,
    tag: str,
    reject_percentile: float = 90.0,
    cutoff_pcts: Sequence[float] = (95, 90, 80, 70, 50),
    calibration_lams: Sequence[float] = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
    n_jobs_ebm: int = 8,
    compute_interventional: bool = True,
    compute_cutoff: bool = True,
    compute_calibration: bool = True,
    compute_scorecard: bool = True,
    random_state: int = 42,
) -> CaseResult:
    """Full pilot analysis for a single (teacher, train, test) triple.

    The teacher is assumed to be a fitted classifier with ``predict_proba``
    and compatible with ``shap.TreeExplainer`` (LightGBM / XGBoost / any
    tree ensemble).

    Returns
    -------
    CaseResult with all metric rows, saved to ``out_dir/result_<tag>.{pkl,json}``.
    """
    t_start = time.time()
    feature_names = list(X_train.columns)
    bb = _teacher_outputs(teacher, X_test, X_train, reject_percentile)

    # Internal train/val split for surrogate early stopping
    tr_pos, val_pos = train_test_split(
        np.arange(len(X_train)), test_size=0.2,
        stratify=y_train, random_state=random_state,
    )

    factories = default_surrogate_factories(n_jobs_ebm=n_jobs_ebm)
    cfg = BenchmarkConfig(
        surrogates=factories,
        reject_percentile=reject_percentile,
        target_scale="score",
        ks=(1, 3, 4), adv_ks=(1, 4),
        missing_policy="zero",
        random_state=random_state,
    )
    bench = run_benchmark(
        teacher=None,
        X_train=X_train, X_test=X_test,
        y_train_target=bb["score_tr"],
        y_test_binary=np.asarray(y_test),
        bb_shap_test=bb["shap_te"],
        bb_prob_test=bb["prob_te"],
        bb_score_test=bb["score_te"],
        feature_names=feature_names,
        config=cfg,
        train_val_split=(tr_pos, val_pos),
    )

    result = CaseResult(tag=tag)
    result.info = {
        "tag": tag,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_names),
        "reject_percentile": reject_percentile,
        "random_state": random_state,
        "random_baseline": bench.info["random_baseline"],
        "teacher_AUC_test": float(
            roc_auc_score(np.asarray(y_test), bb["prob_te"])
        ),
        "runtime_seconds_teacher_outputs": round(time.time() - t_start, 1),
    }
    result.bench_rows = bench.rows

    # ------- Calibration ablations (Tree-d1, EBM) ---------------------------
    if compute_calibration:
        bb_adv = pd.DataFrame(bb["shap_te"], columns=feature_names)
        for sname in ["Tree-d1", "EBM"]:
            if sname not in bench.models:
                continue
            surr = bench.models[sname]
            pred = np.asarray(surr.predict(X_test))
            contribs = np.asarray(surr.contributions(X_test))
            contribs_tr = np.asarray(surr.contributions(X_train))
            pred_tr = np.asarray(surr.predict(X_train))

            # Feature calibration (magnitude-preserving fix)
            fc = FeatureCalibrator()
            cal_c, cal_p = fc.fit_transform(contribs, bb["shap_te"], pred)
            cal_adv = pd.DataFrame(-cal_c, columns=feature_names)
            fid_f = attribution_fidelity_named(
                bb_adv, cal_adv, bb["reject"], ks=(1, 3, 4), adv_ks=(1, 4),
            )
            result.calibration_rows.append({
                "surrogate": sname, "method": "Cal-Feature",
                "R2": r2_score(bb["score_te"], cal_p), **fid_f,
            })

            # Bin calibration λ sweep
            for lam in calibration_lams:
                try:
                    bc = BinCalibrator(lam=float(lam), gamma=0.5)
                    # Use train-side SHAP (recompute) for fit
                    shap_tr = shap.TreeExplainer(teacher).shap_values(X_train)
                    if isinstance(shap_tr, list):
                        shap_tr = shap_tr[1]
                    shap_tr = np.asarray(shap_tr, dtype=np.float32)
                    bc.fit(contribs_tr, shap_tr, bb["score_tr"], pred_tr,
                           len(feature_names))
                    c2, p2 = bc.transform(contribs, pred)
                    c2_adv = pd.DataFrame(-c2, columns=feature_names)
                    fid_b = attribution_fidelity_named(
                        bb_adv, c2_adv, bb["reject"], ks=(1, 3, 4), adv_ks=(1, 4),
                    )
                    result.calibration_rows.append({
                        "surrogate": sname, "method": f"Cal-Bin-L{lam}",
                        "R2": r2_score(bb["score_te"], p2), **fid_b,
                    })
                except Exception as e:
                    result.calibration_rows.append({
                        "surrogate": sname, "method": f"Cal-Bin-L{lam}",
                        "error": str(e),
                    })

    # ------- Interventional fidelity ----------------------------------------
    if compute_interventional and "Tree-d1" in bench.models:
        try:
            surr = bench.models["Tree-d1"]
            bin_struct = extract_bin_structure(
                surr.model_, X_train, len(feature_names)
            )
            contribs = np.asarray(surr.contributions(X_test))
            ifi = interventional_fidelity(
                bin_struct, teacher, X_test, bb["reject"], contribs
            )
            result.interventional_rows.append({"surrogate": "Tree-d1", **ifi})
        except Exception as e:
            result.interventional_rows.append({"surrogate": "Tree-d1", "error": str(e)})

    # ------- Cutoff sensitivity (reuse bench.models) ------------------------
    if compute_cutoff:
        for sname in ["Tree-d1", "EBM", "Ridge"]:
            if sname not in bench.models:
                continue
            adv = bench.models[sname].adverse_contributions(
                X_test, target_scale="score"
            )
            bb_adv = pd.DataFrame(bb["shap_te"], columns=feature_names)
            for pct in cutoff_pcts:
                reject_p = bb["prob_te"] >= np.percentile(bb["prob_te"], pct)
                fid = attribution_fidelity_named(
                    bb_adv, adv, reject_p, ks=(1,), adv_ks=(1, 4),
                )
                result.cutoff_rows.append({
                    "surrogate": sname, "pct": float(pct),
                    "n_reject": int(reject_p.sum()),
                    "AdvTop1": fid["AdvTop1"], "AdvTop4": fid["AdvTop4"],
                    "AdvFull_R": fid["AdvFull_R"], "AdvFull_J": fid["AdvFull_J"],
                })

    # ------- Scorecard conversion (Tree-d1, EBM) ----------------------------
    if compute_scorecard:
        bb_adv = pd.DataFrame(bb["shap_te"], columns=feature_names)
        for sname in ["Tree-d1", "EBM"]:
            if sname not in bench.models:
                continue
            try:
                sc = bench.models[sname].to_scorecard_model(
                    X_train, y_binary=np.asarray(y_train),
                    feature_names=feature_names,
                    max_bins_per_feature=5, min_bin_ratio=0.05,
                )
                pred = sc.predict(X_test)
                contribs = sc.contributions(X_test)
                adv = pd.DataFrame(-np.asarray(contribs), columns=feature_names)
                fid = attribution_fidelity_named(
                    bb_adv, adv, bb["reject"], ks=(1, 3, 4), adv_ks=(1, 4),
                )
                result.scorecard_rows.append({
                    "surrogate": f"{sname}→Scorecard",
                    "R2": r2_score(bb["score_te"], pred), **fid,
                })
            except Exception as e:
                result.scorecard_rows.append({
                    "surrogate": f"{sname}→Scorecard", "error": str(e),
                })

    result.info["runtime_total_seconds"] = round(time.time() - t_start, 1)
    pkl_path, json_path = result.save(out_dir)
    print(f"[{tag}] saved {pkl_path} ({result.info['runtime_total_seconds']}s)")
    return result


if __name__ == "__main__":
    # Smoke test with N01/N02 outputs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GMSC")
    parser.add_argument("--out", default="outputs/executor_smoke")
    args = parser.parse_args()

    with open("outputs/N01/datasets.pkl", "rb") as f:
        datasets = pickle.load(f)
    d = datasets[args.dataset]
    teacher = lgb.Booster(model_file=f"outputs/N02/bb_model_{args.dataset}.txt")

    class BoosterClf:
        def __init__(self, b): self.b = b
        def predict_proba(self, X):
            p = self.b.predict(np.asarray(X))
            return np.column_stack([1 - p, p])

    clf = BoosterClf(teacher)
    run_case(
        teacher=clf,
        X_train=d["X_train"], y_train=d["y_train"],
        X_test=d["X_test"], y_test=d["y_test"],
        out_dir=args.out, tag=f"{args.dataset}_smoke",
    )
