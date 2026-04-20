"""E6 — Sequential Priority Surrogate 4-case CV ablation.

Four configurations:
  (abs, frozen) — mean |SHAP| priority, residual backfitting
  (abs, cumulative) — mean |SHAP| priority, curriculum warm-start
  (signed_rejected, frozen) — rejected-only signed SHAP, residual backfit
  (signed_rejected, cumulative) — rejected-only signed, curriculum

Mirrors _e5 exactly on RS, splitter, sparse-binary filter, teacher config.
Compares against Tree-d1 baseline (from NB03/§6.2.1) implicitly via the
same protocol.

Outputs:
  .outputs/e6_seq_priority_cv/{ds}_{pm}_{fm}_fold{i}.json
  .outputs/e6_seq_priority_cv/cv_summary.{csv,json}
"""
from __future__ import annotations

import warnings; warnings.filterwarnings('ignore')

import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import sem, spearmanr, t
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from decentra._utils import information_value
from decentra.metrics.attribution import (
    attribution_fidelity,
    random_baseline_advtopk,
)
from decentra.surrogate import SequentialPrioritySurrogate

N_SPLITS = 5
RS = 42
N_EST_PER_STAGE = 100
ROOT = Path(r'C:/github/decentra')
OUT = ROOT / '.outputs' / 'e6_seq_priority_cv'
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
        if float(vc.min()) < min_rate and \
                information_value(X[col].values, np.asarray(y)) < min_iv:
            dropped.append(col)
    return X.drop(columns=dropped), dropped


def fit_teacher(X_tr, y_tr, seed=RS):
    X_tr_i, X_val, y_tr_i, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    clf = lgb.LGBMClassifier(
        max_depth=-1, num_leaves=63, n_estimators=1000, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
        random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


def logit_to_score(p, pdo=40, score_0=600, odds_0=50):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    factor = pdo / np.log(2)
    offset = score_0 - factor * np.log(odds_0)
    return offset - factor * np.log(p / (1 - p))


def to2d(raw, nfeat):
    if isinstance(raw, list):
        raw = raw[1] if len(raw) == 2 else raw[-1]
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[..., 1] if arr.shape[-1] == 2 else arr[..., -1]
    if arr.shape[1] == nfeat + 1:
        arr = arr[:, :nfeat]
    return arr.astype(np.float32)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)


CASES = [
    ('abs', 'frozen'),
    ('abs', 'cumulative'),
    ('signed_rejected', 'frozen'),
    ('signed_rejected', 'cumulative'),
]

all_rows = []
for ds_name, d in datasets.items():
    X_all = pd.concat([d['X_train'], d['X_test']])
    y_all = pd.concat([d['y_train'], d['y_test']])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)

    for fold_i, (tr, te) in enumerate(skf.split(X_all, y_all)):
        X_tr = X_all.iloc[tr].copy(); y_tr = y_all.iloc[tr].copy()
        X_te = X_all.iloc[te].copy(); y_te = y_all.iloc[te].copy()

        X_tr, dropped = filter_sparse_binaries(X_tr, y_tr)
        X_te = X_te.drop(columns=dropped)
        med = X_tr.median()
        X_tr = X_tr.fillna(med); X_te = X_te.fillna(med)

        teacher = fit_teacher(X_tr, y_tr)
        prob_tr = teacher.predict_proba(X_tr)[:, 1]
        prob_te = teacher.predict_proba(X_te)[:, 1]
        score_tr = logit_to_score(prob_tr)
        score_te = logit_to_score(prob_te)
        expl = shap.TreeExplainer(teacher)
        sv_tr = to2d(expl.shap_values(X_tr), X_tr.shape[1])
        sv_te = to2d(expl.shap_values(X_te), X_te.shape[1])
        reject = prob_te >= np.percentile(prob_te, 90)
        nf = sv_te.shape[1]
        rb1 = round(random_baseline_advtopk(sv_te, reject, 1, nf, bb_sign=1), 4)
        rb4 = round(random_baseline_advtopk(sv_te, reject, 4, nf, bb_sign=1), 4)

        for pm, fm in CASES:
            key = f'{pm}_{fm}'
            fold_path = OUT / f'{ds_name}_{key}_fold{fold_i}.json'
            if fold_path.exists():
                with open(fold_path) as fh:
                    all_rows.append(json.load(fh))
                print(f'[{ds_name} {key} fold {fold_i}] cached', flush=True)
                continue
            t0 = time.time()
            surr = SequentialPrioritySurrogate(
                priority_method=pm, fit_mode=fm,
                n_estimators_per_stage=N_EST_PER_STAGE,
                learning_rate=0.05, random_state=RS,
            )
            surr.fit(X_tr, score_tr, shap_values=sv_tr, bb_prob=prob_tr)

            pred = surr.predict(X_te)
            contribs = surr.contributions(X_te)

            # BB SHAP is log-odds (positive=adverse); surrogate contribs are
            # score-space (negative=adverse). Use surr_sign=-1 as with
            # other score-space surrogates.
            r2 = round(r2_score(score_te, pred), 4)
            sp = round(float(spearmanr(score_te, pred)[0]), 4)
            af = attribution_fidelity(
                sv_te, contribs, reject, bb_sign=1, surr_sign=-1,
            )

            row = {
                'dataset': ds_name, 'fold': fold_i,
                'priority': pm, 'fit_mode': fm, 'surrogate': f'Seq-{key}',
                'R2': r2, 'Spearman': sp,
                **{k: round(v, 4) for k, v in af.items()},
                'Random_AT1': rb1, 'Random_AT4': rb4,
                'n_features': int(X_tr.shape[1]),
                'n_train': int(len(X_tr)), 'n_test': int(len(X_te)),
                'dropped_sparse': int(len(dropped)),
                'time_s': round(time.time() - t0, 1),
            }
            all_rows.append(row)
            save_json(fold_path, row)
            print(
                f'[{ds_name} {key:32s} fold {fold_i}] '
                f'R2={r2:.3f} AT1={row["AdvTop1"]:.3f} '
                f'AT4={row["AdvTop4"]:.3f} AFR={row["AdvFull_R"]:.3f} '
                f'({row["time_s"]}s)', flush=True,
            )


df = pd.DataFrame(all_rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / 'cv_all_rows.csv', index=False)


def agg(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return {'mean': float(np.mean(v)) if len(v) == 1 else float('nan'),
                'std': float('nan'), 'ci95': float('nan')}
    return {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)),
            'ci95': float(sem(v) * t.ppf(0.975, len(v) - 1))}


METRICS = ['R2', 'Spearman', 'Top1', 'Top3', 'Top4', 'AdvTop1', 'AdvTop4',
           'AdvFull_R', 'AdvFull_J']
summary = []
for (ds, surr), g in df.groupby(['dataset', 'surrogate']):
    for m in METRICS:
        if m not in g.columns:
            continue
        vals = g[m].dropna().values
        s = agg(vals)
        summary.append({
            'dataset': ds, 'surrogate': surr, 'metric': m,
            **{k: round(float(v), 4) if isinstance(v, (int, float))
                                        and not np.isnan(v) else v
               for k, v in s.items()},
            'n': len(vals),
        })
sum_df = pd.DataFrame(summary)
sum_df.to_csv(OUT / 'cv_summary.csv', index=False)
save_json(OUT / 'cv_summary.json', {
    'n_splits': N_SPLITS, 'random_state': RS,
    'n_est_per_stage': N_EST_PER_STAGE,
    'summary': sum_df.to_dict(orient='records'),
    'rows': all_rows,
})

print('\n=== Sequential Priority 4-case CV Summary ===')
pivot = sum_df.pivot_table(
    index=['dataset', 'surrogate'], columns='metric', values='mean',
).round(4)
print(pivot.to_string())
print(f'\nSaved: {OUT}')
