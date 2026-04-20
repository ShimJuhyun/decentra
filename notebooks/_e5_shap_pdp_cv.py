"""E5 — D5 (ShapPdp / Choi & Cha 2026) 5-fold stratified CV.

Mirrors _e3_cv_5fold.py exactly on: random state, fold splitter, sparse
binary filter, median imputation, teacher LGBMClassifier config. Replaces
the multi-surrogate bench with a single ShapPdpSurrogate fit so that CV
numbers are directly comparable to the NB03/§6.2.1 table.

Per-fold pipeline:
  1. Stratified split (RS=42, n_splits=5) on concat(X_train, X_test).
  2. Per-fold sparse-binary filter (refit to avoid leakage).
  3. Median imputation.
  4. Fit teacher LightGBM (same HPs as _e3).
  5. Compute BB TreeSHAP on training fold, BB prob/score on test fold.
  6. Fit ShapPdpSurrogate; OLS-calibrate log-odds → BB score scale.
  7. Evaluate {R², Spearman, Top-k, AdvTop-k, AdvFull} on test fold.

Outputs:
  outputs/e5_shap_pdp_cv/{ds}_fold{i}.json
  outputs/e5_shap_pdp_cv/cv_summary.csv  — mean ± 95% CI per metric
"""
from __future__ import annotations

import warnings; warnings.filterwarnings('ignore')

import json
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import sem, spearmanr, t, wilcoxon
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from decentra._utils import information_value
from decentra.metrics.attribution import (
    attribution_fidelity,
    random_baseline_advtopk,
)
from decentra.surrogate import ShapPdpSurrogate

N_SPLITS = 5
RS_CV = 42
ROOT = Path(r'C:/github/decentra')
OUT = ROOT / '.outputs' / 'e5_shap_pdp_cv'
OUT.mkdir(parents=True, exist_ok=True)

DATASETS_PATH = ROOT / '.outputs' / 'NB01' / 'datasets.pkl'
with open(DATASETS_PATH, 'rb') as f:
    datasets = pickle.load(f)


def filter_sparse_binaries(X, y, min_rate=0.01, min_iv=0.02):
    dropped = []
    for col in X.columns:
        vc = X[col].dropna().value_counts(normalize=True)
        if len(vc) != 2:
            continue
        minority = float(vc.min())
        iv = information_value(X[col].values, np.asarray(y))
        if minority < min_rate and iv < min_iv:
            dropped.append(col)
    return X.drop(columns=dropped), dropped


def fit_teacher(X_tr, y_tr, seed=RS_CV):
    X_tr_i, X_val, y_tr_i, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    clf = lgb.LGBMClassifier(
        max_depth=-1, num_leaves=63, n_estimators=1000,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, random_state=seed, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


def transform_logit_to_score(p, pdo=40, score_0=600, odds_0=50):
    """Same score transform used in executor._teacher_outputs."""
    logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    factor = pdo / np.log(2)
    offset = score_0 - factor * np.log(odds_0)
    return offset - factor * logit


def compute_bb_artifacts(teacher, X_train, X_test, reject_pct=90):
    prob_tr = teacher.predict_proba(X_train)[:, 1]
    prob_te = teacher.predict_proba(X_test)[:, 1]
    score_tr = transform_logit_to_score(prob_tr)
    score_te = transform_logit_to_score(prob_te)
    expl = shap.TreeExplainer(teacher)
    sv_tr = expl.shap_values(X_train)
    sv_te = expl.shap_values(X_test)
    for name, arr_container in [('tr', sv_tr), ('te', sv_te)]:
        pass
    def _to2d(raw, nfeat):
        if isinstance(raw, list):
            raw = raw[1] if len(raw) == 2 else raw[-1]
        arr = np.asarray(raw)
        if arr.ndim == 3:
            arr = arr[..., 1] if arr.shape[-1] == 2 else arr[..., -1]
        if arr.shape[1] == nfeat + 1:
            arr = arr[:, :nfeat]
        return arr.astype(np.float32)
    sv_tr = _to2d(sv_tr, X_train.shape[1])
    sv_te = _to2d(sv_te, X_test.shape[1])
    reject = prob_te >= np.percentile(prob_te, reject_pct)
    return dict(prob_tr=prob_tr, prob_te=prob_te, score_tr=score_tr,
                score_te=score_te, shap_tr=sv_tr, shap_te=sv_te,
                reject=reject)


def ols_calibrate(raw_train, target_train):
    x = np.asarray(raw_train, dtype=float)
    y = np.asarray(target_train, dtype=float)
    b = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    a = float(y.mean() - b * x.mean())
    return a, b


def evaluate_row(surr_contribs, surr_pred, bb):
    r2 = round(r2_score(bb['score_te'], surr_pred), 4)
    sp = round(float(spearmanr(bb['score_te'], surr_pred)[0]), 4)
    af = attribution_fidelity(bb['shap_te'], surr_contribs, bb['reject'],
                              bb_sign=1, surr_sign=1)  # D5 log-odds
    n_features = bb['shap_te'].shape[1]
    rb1 = round(random_baseline_advtopk(bb['shap_te'], bb['reject'], 1,
                                        n_features, bb_sign=1), 4)
    rb4 = round(random_baseline_advtopk(bb['shap_te'], bb['reject'], 4,
                                        n_features, bb_sign=1), 4)
    return {'R2': r2, 'Spearman': sp,
            **{k: round(v, 4) for k, v in af.items()},
            'Random_AT1': rb1, 'Random_AT4': rb4}


all_rows = []
for ds_name, d in datasets.items():
    X_all = pd.concat([d['X_train'], d['X_test']])
    y_all = pd.concat([d['y_train'], d['y_test']])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS_CV)

    for fold_i, (tr, te) in enumerate(skf.split(X_all, y_all)):
        fold_path = OUT / f'{ds_name}_fold{fold_i}.json'
        if fold_path.exists():
            with open(fold_path) as fh:
                all_rows.append(json.load(fh))
            print(f'[{ds_name} fold {fold_i}] skipped (cached)', flush=True)
            continue
        t0 = time.time()
        X_tr = X_all.iloc[tr].copy(); y_tr = y_all.iloc[tr].copy()
        X_te = X_all.iloc[te].copy(); y_te = y_all.iloc[te].copy()

        X_tr, dropped = filter_sparse_binaries(X_tr, y_tr)
        X_te = X_te.drop(columns=dropped)
        med = X_tr.median()
        X_tr = X_tr.fillna(med); X_te = X_te.fillna(med)

        teacher = fit_teacher(X_tr, y_tr)
        bb = compute_bb_artifacts(teacher, X_tr, X_te)

        surr = ShapPdpSurrogate(
            binning='optbinning', max_n_bins=10, min_bin_size=0.01,
            smoother_n_estimators=200, smoother_max_depth=2,
            smoother_learning_rate=0.05,
            monotone_detect_mode='none',  # single auto-detection at smoother step
            random_state=RS_CV,
        )
        # BB-pure surrogate: binning_y=None → ContinuousOptimalBinning on
        # BB score (y_logit), matching OptBin+Ridge's default behaviour.
        # Original y is NOT used anywhere in the pipeline — this is a
        # proper post-hoc explanation of BB's output, not a partial refit
        # to the raw binary target.
        surr.fit(X_tr, bb['score_tr'],
                 shap_values=bb['shap_tr'])

        raw_train = surr.predict(X_tr)
        a, b = ols_calibrate(raw_train, bb['score_tr'])
        surr_pred = a + b * surr.predict(X_te)
        surr_contribs = surr.contributions(X_te)

        row = evaluate_row(surr_contribs, surr_pred, bb)
        row.update({
            'dataset': ds_name, 'fold': fold_i, 'surrogate': 'D5-ShapPdp',
            'n_features': int(X_tr.shape[1]), 'n_train': int(len(X_tr)),
            'n_test': int(len(X_te)), 'dropped_sparse': int(len(dropped)),
            'pdo_a': round(a, 3), 'pdo_b': round(b, 3),
            'time_s': round(time.time() - t0, 1),
        })
        all_rows.append(row)
        print(f'[{ds_name} fold {fold_i}] R2={row["R2"]:.3f} '
              f'AT1={row["AdvTop1"]:.3f} AT4={row["AdvTop4"]:.3f} '
              f'AFR={row["AdvFull_R"]:.3f} ({row["time_s"]}s)', flush=True)

        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / f'{ds_name}_fold{fold_i}.json', 'w') as f:
            json.dump(row, f, indent=2, default=float)


df = pd.DataFrame(all_rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / 'cv_all_rows.csv', index=False)


def agg(v):
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return {'mean': float(np.mean(v)) if len(v) == 1 else float('nan'),
                'std': float('nan'), 'ci95': float('nan')}
    return {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)),
            'ci95': float(sem(v) * t.ppf(0.975, len(v) - 1))}


METRICS = ['R2', 'Spearman', 'Top1', 'Top3', 'Top4', 'AdvTop1', 'AdvTop4',
           'AdvFull_R', 'AdvFull_J']
summary = []
for ds, g in df.groupby('dataset'):
    for m in METRICS:
        if m not in g.columns:
            continue
        vals = g[m].dropna().values
        s = agg(vals)
        summary.append({'dataset': ds, 'metric': m,
                        **{k: round(v, 4) if isinstance(v, float) else v
                           for k, v in s.items()},
                        'n': len(vals)})
sum_df = pd.DataFrame(summary)
sum_df.to_csv(OUT / 'cv_summary.csv', index=False)
print('\n=== CV Summary (D5-ShapPdp) ===')
print(sum_df.to_string(index=False))

with open(OUT / 'cv_summary.json', 'w') as f:
    json.dump({'n_splits': N_SPLITS, 'random_state': RS_CV,
               'summary': sum_df.to_dict(orient='records'),
               'rows': all_rows}, f, indent=2, default=float)

print(f'\nALL_DONE. Saved to {OUT}/')
