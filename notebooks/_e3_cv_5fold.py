"""E3 — 5-fold stratified CV via executor.run_case.

Per-fold pipeline:
- Stratified split (RS=42)
- Per-fold sparse-binary filter (refit to avoid leakage)
- Per-fold teacher fit (LGBMClassifier)
- executor.run_case (bench + calibration + scorecard + interventional)
- Aggregate and run Wilcoxon + BH on AT-4 across folds

Outputs:
- outputs/e3_cv/fold_{i}/result_{ds}_fold{i}.{pkl,json}
- outputs/e3_cv/cv_summary.{csv,json}
- outputs/e3_cv/cv_wilcoxon_AT4.{csv,json}
"""
import warnings; warnings.filterwarnings('ignore')
import sys, json, pickle, time, numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import wilcoxon, sem, t
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from executor import run_case

from decentra._utils import information_value

N_SPLITS = 5
RS_CV = 42
OUT = Path('outputs/e3_cv'); OUT.mkdir(parents=True, exist_ok=True)
with open('outputs/N01/datasets.pkl','rb') as f: datasets = pickle.load(f)


def filter_sparse_binaries(X, y, min_rate=0.01, min_iv=0.02):
    dropped = []
    for col in X.columns:
        vc = X[col].dropna().value_counts(normalize=True)
        if len(vc) != 2: continue
        minority = float(vc.min())
        iv = information_value(X[col].values, np.asarray(y))
        if minority < min_rate and iv < min_iv:
            dropped.append(col)
    return X.drop(columns=dropped), dropped


def fit_teacher(X_tr, y_tr, seed=RS_CV):
    X_tr_i, X_val, y_tr_i, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    clf = lgb.LGBMClassifier(max_depth=-1, num_leaves=63, n_estimators=1000,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, min_child_samples=50,
                              random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


all_results = []
for ds_name, d in datasets.items():
    # Reconstruct "full" X, y from N01 split (pilot uses pre-split)
    X_all = pd.concat([d['X_train'], d['X_test']])
    y_all = pd.concat([d['y_train'], d['y_test']])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS_CV)
    for fold_i, (tr, te) in enumerate(skf.split(X_all, y_all)):
        t0 = time.time()
        X_tr_fold = X_all.iloc[tr].copy(); y_tr_fold = y_all.iloc[tr].copy()
        X_te_fold = X_all.iloc[te].copy(); y_te_fold = y_all.iloc[te].copy()

        # Per-fold sparse-binary filter
        X_tr_fold, dropped = filter_sparse_binaries(X_tr_fold, y_tr_fold)
        X_te_fold = X_te_fold.drop(columns=dropped)
        # Per-fold median imputation
        med = X_tr_fold.median()
        X_tr_fold = X_tr_fold.fillna(med); X_te_fold = X_te_fold.fillna(med)

        teacher = fit_teacher(X_tr_fold, y_tr_fold)
        print(f'[{ds_name} fold {fold_i}] teacher AUC={teacher.best_score_["valid_0"].get("binary_logloss", -1):.4f}, '
              f'starting run_case (dropped {len(dropped)} sparse)', flush=True)

        res = run_case(
            teacher=teacher,
            X_train=X_tr_fold, y_train=y_tr_fold,
            X_test=X_te_fold,   y_test=y_te_fold,
            out_dir=OUT / f'fold_{fold_i}',
            tag=f'{ds_name}_fold{fold_i}',
            compute_interventional=True,
            compute_calibration=True,
            compute_scorecard=True,
            n_jobs_ebm=8,
        )
        all_results.append({'dataset': ds_name, 'fold': fold_i, 'result': res,
                             'dropped_n': len(dropped)})
        print(f'[{ds_name} fold {fold_i}] done in {time.time()-t0:.0f}s', flush=True)


# --------- Aggregation ----------
ROWS = []
for it in all_results:
    ds = it['dataset']; fold = it['fold']; r = it['result']
    for row in r.bench_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'bench', **row})
    for row in r.calibration_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'calibration', **row})
    for row in r.interventional_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'interventional', **row})
    for row in r.scorecard_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'scorecard', **row})

df_all = pd.DataFrame(ROWS)
df_all.to_csv(OUT/'cv_all_rows.csv', index=False)

def agg(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2: return {'mean': float(np.mean(v)) if len(v)==1 else float('nan'),
                            'std': float('nan'), 'ci95': float('nan')}
    return {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)),
             'ci95': float(sem(v) * t.ppf(0.975, len(v)-1))}

bench = df_all[df_all['source']=='bench']
METRICS = ['R2','Spearman','Agree','Top1','Top4','AdvTop1','AdvTop4','AdvFull_R','AdvFull_J','coverage_surr']
summary = []
for (ds, surr), g in bench.groupby(['dataset','surrogate']):
    for m in METRICS:
        if m not in g.columns: continue
        vals = g[m].dropna().values
        s = agg(vals)
        summary.append({'dataset': ds, 'surrogate': surr, 'metric': m,
                         **s, 'n': len(vals)})
sum_df = pd.DataFrame(summary)
sum_df.to_csv(OUT/'cv_summary.csv', index=False)

# Wilcoxon on AT-4
wil_rows = []
for ds in bench['dataset'].unique():
    piv = bench[bench['dataset']==ds].pivot(index='fold', columns='surrogate', values='AdvTop4')
    surrs = list(piv.columns)
    for a, b in combinations(surrs, 2):
        va, vb = piv[a].dropna().values, piv[b].dropna().values
        n = min(len(va), len(vb))
        if n < 3:
            continue
        try:
            stat, p = wilcoxon(va[:n], vb[:n])
            wil_rows.append({'dataset': ds, 'a': a, 'b': b,
                              'mean_a': float(np.mean(va)), 'mean_b': float(np.mean(vb)),
                              'W': float(stat), 'p_raw': float(p), 'n': n})
        except ValueError as e:
            wil_rows.append({'dataset': ds, 'a': a, 'b': b, 'error': str(e)})

wil = pd.DataFrame(wil_rows)
def bh(ps):
    ps = np.asarray(ps, dtype=float); m = len(ps)
    order = np.argsort(ps); ranked = np.empty_like(ps)
    for rank, idx in enumerate(order):
        ranked[idx] = ps[idx] * m / (rank + 1)
    return np.minimum.accumulate(ranked[order[::-1]])[::-1]

if 'p_raw' in wil.columns:
    wil['p_bh'] = np.nan
    for ds, g in wil.groupby('dataset'):
        mask = g['p_raw'].notna()
        wil.loc[g.index[mask], 'p_bh'] = bh(g.loc[mask, 'p_raw'].values)

wil.to_csv(OUT/'cv_wilcoxon_AT4.csv', index=False)
with open(OUT/'cv_wilcoxon_AT4.json','w') as f:
    json.dump(wil.to_dict(orient='records'), f, indent=2, default=float)

# Final summary JSON
final = {'n_splits': N_SPLITS, 'random_state': RS_CV,
          'summary': sum_df.to_dict(orient='records'),
          'wilcoxon_AT4': wil.to_dict(orient='records') if 'p_raw' in wil.columns else [],
          'paths': {'rows_csv': str(OUT/'cv_all_rows.csv'),
                     'summary_csv': str(OUT/'cv_summary.csv')}}
with open(OUT/'cv_summary.json','w') as f:
    json.dump(final, f, indent=2, default=float)

print(f'ALL_DONE. Saved to {OUT}/', flush=True)
