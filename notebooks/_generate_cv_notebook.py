"""One-shot generator for N_CV notebook. Run: python _generate_cv_notebook.py"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


cv = nbf.v4.new_notebook()
cv.cells = [
    md("""# N_CV — Cross-Validation Execution

**Purpose**: Run `executor.run_case` across 10 stratified folds of every dataset.
Aggregate per-fold metrics, compute mean ± CI, and write a paper-ready JSON
summary.

**Outputs** (`outputs/N_CV/`):
- `fold_{i}/result_{ds}_fold{i}.{pkl,json}` — per fold, per dataset
- `cv_summary.json` — aggregated mean / std / 95% CI per (dataset, surrogate, metric)
- `cv_wilcoxon.json` — pairwise Wilcoxon signed-rank tests on AdvTop-4

**Design**:
- Same data preprocessing as N01 (including `SparseBinaryFilter`) **per fold**
  to avoid leakage.
- Same teacher hyperparameters as N02.
- `compute_interventional=True` but the BinCalibrator λ-sweep keeps the default 6 points.
- Single-threaded outer loop; EBM uses `n_jobs=8` internally.
"""),
    code("""import warnings; warnings.filterwarnings('ignore')
import sys, json, pickle, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import wilcoxon, sem, t
import lightgbm as lgb

sys.path.insert(0, str(Path.cwd()))    # so executor.py is importable
from executor import run_case

from decentra._utils import information_value

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

N01_PKL = Path('../outputs/N01/datasets.pkl')
OUT = Path('../outputs/N_CV'); OUT.mkdir(parents=True, exist_ok=True)
with open(N01_PKL, 'rb') as f: datasets = pickle.load(f)

N_SPLITS = 10
RS_CV = 42
print('Datasets:', list(datasets))"""),
    md("## 1. Per-fold executor loop"),
    code("""TEACHER_PARAMS = dict(max_depth=-1, num_leaves=63, n_estimators=1000,
                       learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       min_child_samples=50, random_state=RS_CV, n_jobs=-1, verbose=-1)

def _fit_teacher(X_tr, y_tr):
    # internal train/val split for early stopping
    X_tr_i, X_val, y_tr_i, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=RS_CV)
    clf = lgb.LGBMClassifier(**TEACHER_PARAMS)
    clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf

all_results = []
for ds_name, d in datasets.items():
    # Reconstruct a single "master" X, y by concatenating N01 train+test
    X_all = pd.concat([d['X_train'], d['X_test']])
    y_all = pd.concat([d['y_train'], d['y_test']])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS_CV)
    for fold_i, (tr, te) in enumerate(skf.split(X_all, y_all)):
        t0 = time.time()
        X_tr_fold = X_all.iloc[tr].copy(); y_tr_fold = y_all.iloc[tr].copy()
        X_te_fold = X_all.iloc[te].copy(); y_te_fold = y_all.iloc[te].copy()

        # Per-fold sparse-binary filter (refit)
        X_tr_fold, dropped = filter_sparse_binaries(X_tr_fold, y_tr_fold)
        X_te_fold = X_te_fold.drop(columns=dropped)

        # Per-fold imputation
        med = X_tr_fold.median()
        X_tr_fold = X_tr_fold.fillna(med); X_te_fold = X_te_fold.fillna(med)

        teacher = _fit_teacher(X_tr_fold, y_tr_fold)
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
        all_results.append({'dataset': ds_name, 'fold': fold_i, 'result': res})
        print(f'{ds_name} fold {fold_i}: {time.time()-t0:.0f}s total')"""),
    md("## 2. Aggregate across folds"),
    code("""ROWS = []
for it in all_results:
    ds = it['dataset']; fold = it['fold']; r = it['result']
    for row in r.bench_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'bench', **row})
    for row in r.calibration_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'calibration', **row})
    for row in r.interventional_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'interventional', **row})
    for row in r.cutoff_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'cutoff', **row})
    for row in r.scorecard_rows:
        ROWS.append({'dataset': ds, 'fold': fold, 'source': 'scorecard', **row})

df_all = pd.DataFrame(ROWS)
df_all.to_csv(OUT/'cv_all_rows.csv', index=False)

# Main bench aggregate (mean ± 95% CI)
def agg(g, col):
    v = g[col].dropna().values
    if len(v) < 2: return pd.Series({'mean': np.nan, 'std': np.nan, 'ci95': np.nan})
    m, s = float(np.mean(v)), float(np.std(v, ddof=1))
    ci = float(sem(v) * t.ppf(0.975, len(v)-1))
    return pd.Series({'mean': m, 'std': s, 'ci95': ci})

bench = df_all[df_all['source']=='bench']
METRICS = ['R2','Spearman','Agree','Top1','Top4','AdvTop1','AdvTop4','AdvFull_R','AdvFull_J']
summary = []
for (ds, surr), g in bench.groupby(['dataset','surrogate']):
    for m in METRICS:
        if m not in g.columns: continue
        s = agg(g, m)
        summary.append({'dataset': ds, 'surrogate': surr, 'metric': m,
                         **s.to_dict(), 'n_folds': len(g)})
sum_df = pd.DataFrame(summary)
sum_df.to_csv(OUT/'cv_summary.csv', index=False)
print(sum_df.head(20).round(4).to_string(index=False))"""),
    md("## 3. Pairwise Wilcoxon signed-rank on AdvTop-4 (per dataset)"),
    code("""from itertools import combinations
wil_rows = []
for ds in bench['dataset'].unique():
    sub = bench[bench['dataset']==ds].pivot(index='fold', columns='surrogate', values='AdvTop4')
    surrs = list(sub.columns)
    for a, b in combinations(surrs, 2):
        va, vb = sub[a].dropna().values, sub[b].dropna().values
        if len(va) < 6 or len(vb) < 6: continue
        n = min(len(va), len(vb))
        try:
            stat, p = wilcoxon(va[:n], vb[:n])
            wil_rows.append({'dataset': ds, 'a': a, 'b': b,
                              'mean_a': float(np.mean(va)),
                              'mean_b': float(np.mean(vb)),
                              'W': float(stat), 'p_raw': float(p), 'n': n})
        except ValueError as e:
            wil_rows.append({'dataset': ds, 'a': a, 'b': b, 'error': str(e)})

# Benjamini–Hochberg correction per dataset
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

wil.to_json(OUT/'cv_wilcoxon.json', orient='records', indent=2)
print(wil.round(4).to_string(index=False))"""),
    md("## 4. Final summary dump"),
    code("""FINAL = {
    'n_splits': N_SPLITS, 'random_state': RS_CV,
    'summary': sum_df.to_dict(orient='records'),
    'wilcoxon': wil.to_dict(orient='records') if 'p_raw' in wil.columns else [],
    'paths': {
        'rows_csv': str(OUT/'cv_all_rows.csv'),
        'summary_csv': str(OUT/'cv_summary.csv'),
    },
}
with open(OUT/'cv_summary.json','w') as f:
    json.dump(FINAL, f, indent=2, default=float)
print('Wrote', OUT/'cv_summary.json')"""),
]
cv.metadata = {
    'kernelspec': {'display_name': 'Python 3 (decentra)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.x'},
}

with open(HERE/'N_CV_cross_validation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(cv, f)
print('wrote N_CV_cross_validation.ipynb')
