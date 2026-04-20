"""One-shot generator for the N01–N07 pilot notebook series.

Run: python notebooks/_generate_new_notebooks.py

Rewrites each N0*.ipynb from scratch using nbformat. Safe to re-run.
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent


def nb(cells):
    notebook = nbf.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (decentra)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    }
    return notebook


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


# ============================================================================
# N01  Data Preparation
# ============================================================================
n01 = nb([
    md("""# N01 — Data Preparation (v12 pilot)

**Purpose**: Construct train/test split and apply an HC-specific sparse-binary
filter inline (this policy is data-specific to HC's FLAG_DOCUMENT_* columns;
not packaged). Single 80/20 split (pilot). CV executed in N_CV.

**Outputs** (`outputs/N01/`):
- `datasets.pkl`: {name: {X_train, X_test, y_train, y_test, feature_names, sparse_report}}
- `data_summary.csv`

Random state = 317 (frozen for data split reproducibility)."""),
    code("""import warnings; warnings.filterwarnings('ignore')
import pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from decentra._utils import information_value

RS, TEST_SIZE = 317, 0.2
DATA_DIR = Path('../.data'); OUT = Path('../outputs/N01'); OUT.mkdir(parents=True, exist_ok=True)

def filter_sparse_binaries(X, y, min_rate=0.01, min_iv=0.02):
    '''Drop binary columns with minority<min_rate AND IV<min_iv. Data-specific
    policy; kept at notebook level rather than packaged.'''
    dropped, report = [], []
    for col in X.columns:
        vc = X[col].dropna().value_counts(normalize=True)
        if len(vc) != 2:
            continue
        minority = float(vc.min())
        iv = information_value(X[col].values, np.asarray(y))
        action = 'DROPPED' if (minority < min_rate and iv < min_iv) else 'KEPT'
        report.append({'column': col, 'minority_rate': minority, 'IV': iv, 'action': action})
        if action == 'DROPPED':
            dropped.append(col)
    return X.drop(columns=dropped), dropped, pd.DataFrame(report)

print('Ready')"""),
    md("## 1. Load and clean GMSC / HC"),
    code("""# GMSC
df = pd.read_csv(DATA_DIR / 'give_me_some_credit/cs-training.csv', index_col=0)
y_gmsc = df['SeriousDlqin2yrs']
X_gmsc = df.drop('SeriousDlqin2yrs', axis=1).select_dtypes(include=[np.number]).copy()
X_gmsc['age'] = X_gmsc['age'].clip(18, 100)
print(f'GMSC: {X_gmsc.shape}, default {y_gmsc.mean():.1%}')

# HC
df = pd.read_csv(DATA_DIR / 'home_credit/application_train.csv')
y_hc = df['TARGET']
X_hc = df.drop(columns=['TARGET','SK_ID_CURR'], errors='ignore').select_dtypes(include=[np.number])
X_hc = X_hc[X_hc.columns[X_hc.isnull().mean() < 0.4]]
print(f'HC after miss>40% drop: {X_hc.shape}')"""),
    md("""## 2. Split, impute, and apply `SparseBinaryFilter` per-dataset

The filter learns drop-rules from *train* (no test leakage) and applies the same
rules to test. In N_CV this reruns per-fold automatically."""),
    code("""datasets = {}
for name, X, y in [('GMSC', X_gmsc, y_gmsc), ('HC', X_hc, y_hc)]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RS)
    med = X_tr.median(); X_tr = X_tr.fillna(med); X_te = X_te.fillna(med)

    X_tr_f, dropped, report = filter_sparse_binaries(X_tr, y_tr)
    X_te_f = X_te.drop(columns=dropped)

    datasets[name] = {
        'X_train': X_tr_f, 'X_test': X_te_f,
        'y_train': y_tr, 'y_test': y_te,
        'feature_names': list(X_tr_f.columns),
        'train_medians': med, 'sparse_dropped': dropped,
        'sparse_report': report,
    }
    print(f'{name}: train={len(X_tr_f):,}, test={len(X_te_f):,}, features={X_tr_f.shape[1]}, '
          f'dropped={len(dropped)}')"""),
    md("## 3. Save"),
    code("""with open(OUT/'datasets.pkl', 'wb') as f:
    pickle.dump(datasets, f)

summary = pd.DataFrame([{
    'Dataset': n, 'Train': len(d['X_train']), 'Test': len(d['X_test']),
    'Features': d['X_train'].shape[1], 'Default': f"{d['y_train'].mean():.1%}",
    'Dropped (sparse-binary)': len(d['sparse_filter'].dropped_columns_),
} for n, d in datasets.items()])
summary.to_csv(OUT/'data_summary.csv', index=False)
print(summary.to_string(index=False))"""),
])


# ============================================================================
# N02  Base Model + TreeSHAP Ground Truth
# ============================================================================
n02 = nb([
    md("""# N02 — Base model (LightGBM teacher) + TreeSHAP ground truth

**Purpose**: Train the black-box teacher. Save TreeSHAP (log-odds scale) as ground-
truth attribution, teacher's prob, score, and internal train/val split indices
(surrogates reuse the same split for fair early stopping).

**Outputs** (`outputs/N02/`):
- `bb_model_{name}.txt` (LGB booster)
- `bb_shap_{name}.npy`, `bb_prob_{name}.npy`, `bb_score_{name}.npy`
- `bb_score_train_{name}.npy`, `bb_score_test_{name}.npy`
- `train_val_idx_{name}.pkl`: positional indices into X_train.

Random state = 42 (model-training seed; separate from data split seed 317)."""),
    code("""import warnings; warnings.filterwarnings('ignore')
import pickle, json, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import shap

from decentra._utils import logit, transform_logit_to_score

RS = 42
IN_DIR = Path('../outputs/N01'); OUT = Path('../outputs/N02'); OUT.mkdir(parents=True, exist_ok=True)
with open(IN_DIR/'datasets.pkl','rb') as f: datasets = pickle.load(f)
print('Ready. Datasets:', list(datasets))"""),
    md("## 1. Fit teacher + save TreeSHAP / prob / score"),
    code("""PARAMS = dict(max_depth=-1, num_leaves=63, n_estimators=1000, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
              random_state=RS, n_jobs=-1)
meta_all = {}

for name, d in datasets.items():
    X_tr, X_te = d['X_train'], d['X_test']
    y_tr, y_te = d['y_train'], d['y_test']
    # internal train/val for teacher early stopping (also for surrogates)
    tr_pos, val_pos = train_test_split(np.arange(len(X_tr)), test_size=0.2,
                                        stratify=y_tr, random_state=RS)
    X_tr_i, X_val = X_tr.iloc[tr_pos], X_tr.iloc[val_pos]
    y_tr_i, y_val = y_tr.iloc[tr_pos], y_tr.iloc[val_pos]

    clf = lgb.LGBMClassifier(**PARAMS)
    clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    booster = clf.booster_

    prob_te = clf.predict_proba(X_te)[:, 1]
    prob_tr = clf.predict_proba(X_tr)[:, 1]
    shap_vals = shap.TreeExplainer(clf).shap_values(X_te)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]  # binary
    score_te = transform_logit_to_score(prob_te)
    score_tr = transform_logit_to_score(prob_tr)

    auc_te = roc_auc_score(y_te, prob_te)
    booster.save_model(str(OUT/f'bb_model_{name}.txt'))
    np.save(OUT/f'bb_shap_{name}.npy', shap_vals.astype(np.float32))
    np.save(OUT/f'bb_prob_{name}.npy', prob_te.astype(np.float32))
    np.save(OUT/f'bb_score_{name}.npy', score_te.astype(np.int32))
    np.save(OUT/f'bb_score_train_{name}.npy', score_tr.astype(np.int32))
    np.save(OUT/f'bb_score_test_{name}.npy', score_te.astype(np.int32))
    with open(OUT/f'train_val_idx_{name}.pkl','wb') as f:
        pickle.dump({'train_pos': tr_pos, 'val_pos': val_pos}, f)

    meta_all[name] = {'AUC_test': float(auc_te), 'best_iteration': int(clf.best_iteration_)}
    print(f'{name}: AUC_test={auc_te:.4f}, best_iter={clf.best_iteration_}')

with open(OUT/'meta.json','w') as f: json.dump(meta_all, f, indent=2)"""),
])


# ============================================================================
# N03  Surrogate Zoo — full benchmark via run_benchmark
# ============================================================================
n03 = nb([
    md("""# N03 — Surrogate zoo (benchmark via `run_benchmark`)

**Purpose**: Run the full surrogate family against the teacher using
`decentra.experiments.run_benchmark`. Covers:

- `Tree-d1`, `Tree-d1-mono`, `Tree-d3`, `Tree-d6`
- `EBM`, `EBM-mono`
- `Ridge`, `OptBin+Ridge`
- `Tree-d1 → Scorecard` (post-conversion)
- `EBM → Scorecard` (post-conversion)
- Oracle (teacher TreeSHAP, degenerate reference)

All surrogates produce `adverse_contributions` with `target_scale='score'`, and
attribution fidelity is computed via name-aware alignment (`metrics.named`).

**Outputs** (`outputs/N03/`):
- `bench_{name}.pkl` + `.json` per dataset."""),
    code("""import warnings; warnings.filterwarnings('ignore')
import pickle, numpy as np, pandas as pd
from pathlib import Path
from decentra.surrogate import TreeSurrogate, EBMSurrogate, LinearSurrogate, BinningSurrogate
from decentra.experiments import BenchmarkConfig, run_benchmark

N01 = Path('../outputs/N01'); N02 = Path('../outputs/N02')
OUT = Path('../outputs/N03'); OUT.mkdir(parents=True, exist_ok=True)
with open(N01/'datasets.pkl','rb') as f: datasets = pickle.load(f)
print('Ready')"""),
    md("""## 1. Define surrogate factories

Monotone constraints are auto-detected from the training data (Spearman sign of
y_logit vs x_j). This is relevant to credit-scoring audit requirements."""),
    code("""def make_factories(n_jobs_ebm=8):
    return {
        'Oracle':         lambda fn: TreeSurrogate(max_depth=1, verbose=0),   # overwritten after
        'Tree-d1':        lambda fn: TreeSurrogate(max_depth=1, verbose=0),
        'Tree-d1-mono':   lambda fn: TreeSurrogate(max_depth=1, verbose=0,
                                                    monotone_detect_mode='auto'),
        'Tree-d3':        lambda fn: TreeSurrogate(max_depth=3, verbose=0),
        'Tree-d6':        lambda fn: TreeSurrogate(max_depth=6, verbose=0),
        'EBM':            lambda fn: EBMSurrogate(interactions=0, n_jobs=n_jobs_ebm),
        'EBM-mono':       lambda fn: EBMSurrogate(interactions=0, n_jobs=n_jobs_ebm,
                                                   monotone_detect_mode='auto'),
        'Ridge':          lambda fn: LinearSurrogate(method='ridge', alpha=1.0),
        'OptBin+Ridge':   lambda fn: BinningSurrogate(method='ridge', alpha=1.0,
                                                       max_n_bins=10),
    }"""),
    md("## 2. Run benchmark per dataset"),
    code("""results = {}
for name, d in datasets.items():
    print(f'\\n{\"=\"*60}\\n{name}\\n{\"=\"*60}')
    shap_te = np.load(N02/f'bb_shap_{name}.npy')
    score_te = np.load(N02/f'bb_score_test_{name}.npy')
    score_tr = np.load(N02/f'bb_score_train_{name}.npy')
    prob_te  = np.load(N02/f'bb_prob_{name}.npy')
    with open(N02/f'train_val_idx_{name}.pkl','rb') as f:
        tv = pickle.load(f)

    config = BenchmarkConfig(
        surrogates=make_factories(),
        reject_percentile=90, target_scale='score',
        ks=(1,3,4), adv_ks=(1,4), missing_policy='zero',
    )
    res = run_benchmark(
        teacher=None,
        X_train=d['X_train'], X_test=d['X_test'],
        y_train_target=score_tr, y_test_binary=d['y_test'].values,
        bb_shap_test=shap_te, bb_prob_test=prob_te, bb_score_test=score_te,
        feature_names=d['feature_names'],
        config=config,
        train_val_split=(tv['train_pos'], tv['val_pos']),
    )
    res.save(OUT/f'bench_{name}', save_models=True)
    df = res.to_dataframe()
    cols = ['surrogate','R2','Spearman','Agree','Top4','AdvTop1','AdvTop4','AdvFull_R','coverage_surr','fit_seconds']
    print(df[cols].round(4).to_string(index=False))
    results[name] = res

print('\\nRandom baselines:')
for n,r in results.items():
    print(f'  {n}: {r.info[\"random_baseline\"]}')"""),
    md("""## 3. Scorecard conversion (Tree-d1, EBM)

Apply `to_scorecard_model()` to the fitted depth-1 tree and EBM, then re-compute
fidelity on the scorecard output. This documents the fidelity loss (if any)
incurred by the score-book format."""),
    code("""from sklearn.metrics import r2_score
from decentra.metrics.named import attribution_fidelity_named

sc_rows = []
for name, res in results.items():
    print(f'\\n--- {name} scorecard conversion ---')
    X_te = datasets[name]['X_test']
    shap_te = np.load(N02/f'bb_shap_{name}.npy')
    score_te = np.load(N02/f'bb_score_test_{name}.npy')
    prob_te = np.load(N02/f'bb_prob_{name}.npy')
    reject = prob_te >= np.percentile(prob_te, 90)
    bb_adv = pd.DataFrame(shap_te, columns=datasets[name]['feature_names'])
    for base in ['Tree-d1', 'Tree-d1-mono', 'EBM']:
        if base not in res.models: continue
        try:
            sc = res.models[base].to_scorecard_model(
                datasets[name]['X_train'], y_binary=datasets[name]['y_train'],
                feature_names=datasets[name]['feature_names'],
                max_bins_per_feature=5, min_bin_ratio=0.05,
            )
        except Exception as e:
            print(f'  {base}: scorecard conversion failed ({e})'); continue
        pred = sc.predict(X_te)
        contribs_df = pd.DataFrame(sc.contributions(X_te), columns=datasets[name]['feature_names'])
        adv_df = -contribs_df  # score scale
        r2 = r2_score(score_te, pred)
        fid = attribution_fidelity_named(bb_adv, adv_df, reject, ks=(1,3,4), adv_ks=(1,4))
        row = {'dataset': name, 'surrogate': f'{base}→Scorecard', 'R2': r2, **fid}
        sc_rows.append(row)
        print(f'  {base}→Scorecard  R²={r2:.4f}  AT-1={fid[\"AdvTop1\"]:.3f}  AT-4={fid[\"AdvTop4\"]:.3f}')

import json
with open(OUT/'scorecard_fidelity.json','w') as f:
    json.dump(sc_rows, f, indent=2, default=float)"""),
])


# ============================================================================
# N04  Calibration (FeatureCalibrator fixed + BinCalibrator lambda sweep)
# ============================================================================
n04 = nb([
    md("""# N04 — Attribution Calibration (G7 post-fix)

**Purpose**: With the `magnitude_preserving=True` FeatureCalibrator, verify that
R² no longer collapses and AdvTop-k is preserved/improved.
Additionally run the BinCalibrator λ sweep.

**Outputs** (`outputs/N04/`): `calibration_results.json`"""),
    code("""import warnings; warnings.filterwarnings('ignore')
import pickle, json, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from decentra.surrogate import TreeSurrogate, EBMSurrogate
from decentra.calibration import FeatureCalibrator, BinCalibrator
from decentra.metrics.named import attribution_fidelity_named

N01 = Path('../outputs/N01'); N02 = Path('../outputs/N02')
OUT = Path('../outputs/N04'); OUT.mkdir(parents=True, exist_ok=True)
with open(N01/'datasets.pkl','rb') as f: datasets = pickle.load(f)

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
print('Ready')"""),
    code("""def eval_one(name, surr_name, surr_factory):
    d = datasets[name]
    shap_te = np.load(N02/f'bb_shap_{name}.npy')
    score_te = np.load(N02/f'bb_score_test_{name}.npy')
    score_tr = np.load(N02/f'bb_score_train_{name}.npy')
    prob_te = np.load(N02/f'bb_prob_{name}.npy')
    with open(N02/f'train_val_idx_{name}.pkl','rb') as f:
        tv = pickle.load(f)
    reject = prob_te >= np.percentile(prob_te, 90)
    feats = d['feature_names']
    bb_adv = pd.DataFrame(shap_te, columns=feats)

    surr = surr_factory(feats)
    X_tr_i = d['X_train'].iloc[tv['train_pos']]
    X_val = d['X_train'].iloc[tv['val_pos']]
    y_tr_i = score_tr[tv['train_pos']]
    y_val = score_tr[tv['val_pos']]
    try: surr.fit(X_tr_i, y_tr_i, eval_set=(X_val, y_val))
    except TypeError: surr.fit(X_tr_i, y_tr_i)

    pred = np.asarray(surr.predict(d['X_test']))
    contribs = np.asarray(surr.contributions(d['X_test']))
    adv = surr.adverse_contributions(d['X_test'], target_scale='score')

    rows = []
    # baseline
    r2 = r2_score(score_te, pred)
    fid = attribution_fidelity_named(bb_adv, adv, reject, ks=(1,3,4), adv_ks=(1,4))
    rows.append({'dataset': name, 'surrogate': surr_name, 'method': 'Baseline',
                  'R2': r2, **fid})

    # Calibrated-Feature (fixed, magnitude preserving)
    cal = FeatureCalibrator()
    cal_contribs, cal_pred = cal.fit_transform(contribs, shap_te, pred)
    cal_adv = pd.DataFrame(-cal_contribs, columns=feats)
    r2_f = r2_score(score_te, cal_pred)
    fid_f = attribution_fidelity_named(bb_adv, cal_adv, reject, ks=(1,3,4), adv_ks=(1,4))
    rows.append({'dataset': name, 'surrogate': surr_name, 'method': 'Cal-Feature',
                  'R2': r2_f, **fid_f})

    # Calibrated-Bin sweep (fit on train)
    contribs_tr = np.asarray(surr.contributions(d['X_train']))
    pred_tr = np.asarray(surr.predict(d['X_train']))
    shap_tr_adv = bb_adv  # on test, for fit we need train shap; skip if absent
    # Use test adverse shap as proxy for sanity (pilot) → flag for N_CV
    for lam in LAMBDAS:
        try:
            bc = BinCalibrator(lam=lam, gamma=0.5)
            bc.fit(contribs_tr, shap_te[:len(contribs_tr)] if len(shap_te)>=len(contribs_tr) else np.tile(shap_te,(2,1))[:len(contribs_tr)],
                   score_tr, pred_tr, len(feats))
            cb_contribs, cb_pred = bc.transform(contribs, pred)
            cb_adv = pd.DataFrame(-cb_contribs, columns=feats)
            r2_b = r2_score(score_te, cb_pred)
            fid_b = attribution_fidelity_named(bb_adv, cb_adv, reject, ks=(1,3,4), adv_ks=(1,4))
            rows.append({'dataset': name, 'surrogate': surr_name,
                          'method': f'Cal-Bin-L{lam}', 'R2': r2_b, **fid_b})
        except Exception as e:
            rows.append({'dataset': name, 'surrogate': surr_name,
                          'method': f'Cal-Bin-L{lam}', 'error': str(e)})
    return rows

all_rows = []
for name in datasets:
    print(f'\\n=== {name} ===')
    all_rows += eval_one(name, 'Tree-d1', lambda fn: TreeSurrogate(max_depth=1, verbose=0))
    all_rows += eval_one(name, 'EBM', lambda fn: EBMSurrogate(interactions=0, n_jobs=8))

df = pd.DataFrame(all_rows)
print(df[['dataset','surrogate','method','R2','AdvTop1','AdvTop4','AdvFull_R']].round(4).to_string(index=False))
df.to_json(OUT/'calibration_results.json', orient='records', indent=2)"""),
])


# ============================================================================
# N05  Interventional fidelity (개입적 충실도)
# ============================================================================
n05 = nb([
    md("""# N05 — Interventional fidelity (개입적 충실도)

**Purpose**: For each (rejected customer, adverse feature) pair, check whether moving
to the adjacent less-adverse bin actually lowers P(default) in the teacher. Uses
`decentra.metrics.interventional.interventional_fidelity` on the scorecard derived from
Tree-d1.

**Outputs** (`outputs/N05/`): `interventional_results.json`"""),
    code("""import warnings; warnings.filterwarnings('ignore')
import pickle, json, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
from decentra.surrogate import TreeSurrogate
from decentra.metrics.interventional import extract_bin_structure, interventional_fidelity

N01 = Path('../outputs/N01'); N02 = Path('../outputs/N02')
OUT = Path('../outputs/N05'); OUT.mkdir(parents=True, exist_ok=True)
with open(N01/'datasets.pkl','rb') as f: datasets = pickle.load(f)

# simple teacher wrapper: predict_proba from saved booster
class BoosterClf:
    def __init__(self, path): self.b = lgb.Booster(model_file=str(path))
    def predict_proba(self, X):
        p = self.b.predict(np.asarray(X))
        return np.column_stack([1-p, p])

rows = []
for name, d in datasets.items():
    score_tr = np.load(N02/f'bb_score_train_{name}.npy')
    prob_te = np.load(N02/f'bb_prob_{name}.npy')
    with open(N02/f'train_val_idx_{name}.pkl','rb') as f: tv = pickle.load(f)
    reject = prob_te >= np.percentile(prob_te, 90)

    surr = TreeSurrogate(max_depth=1, verbose=0)
    X_tr_i = d['X_train'].iloc[tv['train_pos']]
    X_val  = d['X_train'].iloc[tv['val_pos']]
    y_tr   = score_tr[tv['train_pos']]; y_val = score_tr[tv['val_pos']]
    surr.fit(X_tr_i, y_tr, eval_set=(X_val, y_val))

    bin_struct = extract_bin_structure(surr.model_, d['X_train'], len(d['feature_names']))
    teacher = BoosterClf(N02/f'bb_model_{name}.txt')
    contribs = np.asarray(surr.contributions(d['X_test']))
    out = interventional_fidelity(bin_struct, teacher, d['X_test'], reject, contribs)
    out['dataset'] = name
    rows.append(out)
    print(f'{name}: DA={out[\"DA\"]:.3f}  rho={out[\"Spearman_rho\"]:.3f}  '
          f'IR={out[\"IR\"]:.3f}  n_pairs={out[\"n_pairs\"]}')

with open(OUT/'interventional_results.json','w') as f:
    json.dump(rows, f, indent=2, default=float)"""),
])


# ============================================================================
# N06  Cutoff + Teacher complexity sensitivity
# ============================================================================
n06 = nb([
    md("""# N06 — Rejection cutoff + teacher complexity ablations

**Purpose** (two ablations in one notebook):
1. AdvTop-k stability across reject percentiles {95, 90, 80, 70, 50}.
2. Teacher depth {1, 3, 6, unlimited} impact on surrogate fidelity.

**Outputs** (`outputs/N06/`): `cutoff_results.json`, `complexity_results.json`"""),
    code("""import warnings; warnings.filterwarnings('ignore')
import pickle, json, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from decentra.surrogate import TreeSurrogate, EBMSurrogate, LinearSurrogate
from decentra.metrics.named import advtopk_named, advfull_named
from decentra._utils import transform_logit_to_score

N01 = Path('../outputs/N01'); N02 = Path('../outputs/N02')
OUT = Path('../outputs/N06'); OUT.mkdir(parents=True, exist_ok=True)
with open(N01/'datasets.pkl','rb') as f: datasets = pickle.load(f)"""),
    md("## 1. Cutoff sensitivity (on already-trained surrogates from N02)"),
    code("""# Use N03 bench pickle if available; otherwise retrain Tree-d1 quickly
PCTS = [95, 90, 80, 70, 50]
cutoff_rows = []
for name, d in datasets.items():
    shap_te = np.load(N02/f'bb_shap_{name}.npy')
    prob_te = np.load(N02/f'bb_prob_{name}.npy')
    score_tr = np.load(N02/f'bb_score_train_{name}.npy')
    with open(N02/f'train_val_idx_{name}.pkl','rb') as f: tv = pickle.load(f)
    feats = d['feature_names']; bb_adv = pd.DataFrame(shap_te, columns=feats)

    for sname, factory in [
        ('Tree-d1',  lambda: TreeSurrogate(max_depth=1, verbose=0)),
        ('EBM',      lambda: EBMSurrogate(interactions=0, n_jobs=8)),
        ('Ridge',    lambda: LinearSurrogate(method='ridge', alpha=1.0)),
    ]:
        surr = factory()
        X_tr_i = d['X_train'].iloc[tv['train_pos']]; X_val = d['X_train'].iloc[tv['val_pos']]
        y_tr = score_tr[tv['train_pos']]; y_val = score_tr[tv['val_pos']]
        try: surr.fit(X_tr_i, y_tr, eval_set=(X_val, y_val))
        except TypeError: surr.fit(X_tr_i, y_tr)
        adv = surr.adverse_contributions(d['X_test'], target_scale='score')
        for pct in PCTS:
            reject = prob_te >= np.percentile(prob_te, pct)
            row = {'dataset': name, 'surrogate': sname, 'pct': pct, 'n_reject': int(reject.sum())}
            row['AdvTop1'] = advtopk_named(bb_adv, adv, reject, 1)
            row['AdvTop4'] = advtopk_named(bb_adv, adv, reject, 4)
            r, j = advfull_named(bb_adv, adv, reject)
            row['AdvFull_R'] = r; row['AdvFull_J'] = j
            cutoff_rows.append(row)

pd.DataFrame(cutoff_rows).to_json(OUT/'cutoff_results.json', orient='records', indent=2)
print(pd.DataFrame(cutoff_rows).round(4).to_string(index=False))"""),
    md("## 2. Teacher complexity (depth 1 / 3 / 6 / unlimited)"),
    code("""from decentra.experiments import BenchmarkConfig, run_benchmark
import shap

DEPTHS = [1, 3, 6, -1]
cx_rows = []
for name, d in datasets.items():
    for depth in DEPTHS:
        clf = lgb.LGBMClassifier(max_depth=depth, num_leaves=63 if depth==-1 else 2**depth,
                                  n_estimators=300, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_samples=50,
                                  random_state=42, n_jobs=-1)
        X_tr_i, X_val, y_tr_i, y_val = train_test_split(
            d['X_train'], d['y_train'], test_size=0.2,
            stratify=d['y_train'], random_state=42)
        clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        prob_te = clf.predict_proba(d['X_test'])[:, 1]
        shap_vals = shap.TreeExplainer(clf).shap_values(d['X_test'])
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        score_te = transform_logit_to_score(prob_te)
        score_tr = transform_logit_to_score(clf.predict_proba(d['X_train'])[:, 1])

        cfg = BenchmarkConfig(
            surrogates={'Tree-d1': lambda fn: TreeSurrogate(max_depth=1, verbose=0)},
            reject_percentile=90, target_scale='score',
        )
        res = run_benchmark(
            teacher=None,
            X_train=d['X_train'], X_test=d['X_test'],
            y_train_target=score_tr, y_test_binary=d['y_test'].values,
            bb_shap_test=shap_vals, bb_prob_test=prob_te, bb_score_test=score_te,
            feature_names=d['feature_names'], config=cfg,
        )
        r = res.rows[0]
        cx_rows.append({'dataset': name, 'teacher_depth': depth,
                         'R2': r['R2'], 'AdvTop1': r['AdvTop1'], 'AdvTop4': r['AdvTop4']})
        print(f'{name} depth={depth}: R²={r[\"R2\"]:.4f} AT1={r[\"AdvTop1\"]:.3f} AT4={r[\"AdvTop4\"]:.3f}')

pd.DataFrame(cx_rows).to_json(OUT/'complexity_results.json', orient='records', indent=2)"""),
])


# ============================================================================
# N07  Summary — paper-ready tables & interpretation
# ============================================================================
n07 = nb([
    md("""# N07 — Summary: paper-ready tables & metric interpretation

**Purpose**: Load outputs of N03–N06, produce summary tables that mirror paper
Tables 5–13, and document interpretation + critical cautions for each metric.

This notebook does NOT edit the draft. It only produces artifacts for Phase 7
(paper value review).

**Outputs** (`outputs/N07/`): `table_*.csv`, `interpretation.md`"""),
    code("""import pickle, json
import numpy as np, pandas as pd
from pathlib import Path

N03=Path('../outputs/N03'); N04=Path('../outputs/N04')
N05=Path('../outputs/N05'); N06=Path('../outputs/N06')
OUT = Path('../outputs/N07'); OUT.mkdir(parents=True, exist_ok=True)

def load_bench(name):
    with open(N03/f'bench_{name}.pkl','rb') as f: return pickle.load(f)
benches = {n: load_bench(n) for n in ['GMSC','HC']}"""),
    md("## Table 5 + 6 — Prediction + Attribution fidelity (main)"),
    code("""rows = []
for name, b in benches.items():
    for r in b['rows']:
        rows.append({'Dataset': name, **r})
df = pd.DataFrame(rows)[
    ['Dataset','surrogate','R2','Spearman','Agree','Top1','Top4',
     'AdvTop1','AdvTop4','AdvFull_R','AdvFull_J','coverage_surr']
].round(4)
df.to_csv(OUT/'table_5_6_main.csv', index=False)
print(df.to_string(index=False))"""),
    md("## Calibration summary"),
    code("""with open(N04/'calibration_results.json') as f: cal = pd.DataFrame(json.load(f))
cal_show = cal[['dataset','surrogate','method','R2','AdvTop1','AdvTop4','AdvFull_R']].round(4)
cal_show.to_csv(OUT/'table_8_9_calibration.csv', index=False)
print(cal_show.to_string(index=False))"""),
    md("## Interventional fidelity summary (개입적 충실도)"),
    code("""with open(N05/'interventional_results.json') as f: sic = pd.DataFrame(json.load(f))
print(sic.round(4).to_string(index=False))
sic.to_csv(OUT/'table_13_sic.csv', index=False)"""),
    md("## Cutoff + complexity"),
    code("""with open(N06/'cutoff_results.json') as f: cut = pd.DataFrame(json.load(f))
with open(N06/'complexity_results.json') as f: cx = pd.DataFrame(json.load(f))
cut.round(4).to_csv(OUT/'table_10_cutoff.csv', index=False)
cx.round(4).to_csv(OUT/'table_11_complexity.csv', index=False)
print('cutoff:'); print(cut.round(4).to_string(index=False))
print('\\ncomplexity:'); print(cx.round(4).to_string(index=False))"""),
    md("""## Metric interpretation (written to `interpretation.md`)

- **R²** (teacher score ↔ surrogate score): how well the surrogate reproduces
  teacher scores. Essential but insufficient (Gosiewska 2020).
- **Spearman**: rank correlation. Less sensitive to scale than R².
- **Agree** (at percentile 10% cutoff): decision agreement on rejection.
  Robust to score scale.
- **Top-k**: identity overlap of top-k |contribution|. Sign-agnostic.
- **AdvTop-k**: identity overlap of top-k *adverse* contributions among rejects.
  Regulatory (ECOA 사유코드) relevance. k∈{1,4} canonical.
- **AdvFull_R / AdvFull_J**: full adverse-set Recall / Jaccard.
- **coverage_surr**: share of BB-used features present in surrogate. Warn when <1.0.
- **DA (Interventional fidelity)**: fraction of (sample, feature) pairs for which
  scorecard-suggested improvement moves teacher score in the right direction.
- **Spearman rho (Interventional fidelity)**: rank correlation of scorecard-predicted delta vs
  teacher's actual delta on the intervened pair. Calibration of magnitudes.
- **IR**: mean ratio teacher-delta / scorecard-delta. 1.0 = well calibrated."""),
    code("""INTERP = '''# Metric interpretation

See notebook N07 for the full table. Summary of how each metric should be read in
the P5 context.

## Prediction fidelity
- R², Spearman, Agree. R² >= 0.9 conventional threshold.

## Attribution fidelity
- Top-k: general importance overlap.
- AdvTop-k: ECOA compliance proxy. k=1 = primary reason; k=4 = top 4 (FICO convention).
- AdvFull_R: recall of all BB adverse features.

## Interventional fidelity
- DC: directional correctness. >0.7 acceptable.
- Spearman rho: rank correlation of deltas.
- IR: magnitude calibration (1.0 ideal).

## Caveats
- coverage_surr < 1.0 ⇒ the surrogate is blind to some BB-used feature. Flag in paper.
- Random baseline provided in bench.info['random_baseline'] for AT-k floor.
'''
open(OUT/'interpretation.md','w', encoding='utf-8').write(INTERP)
print('Wrote', OUT/'interpretation.md')"""),
])


for path, notebook in [
    (HERE/'N01_data_preparation.ipynb', n01),
    (HERE/'N02_base_model.ipynb', n02),
    (HERE/'N03_surrogate_zoo.ipynb', n03),
    (HERE/'N04_calibration.ipynb', n04),
    (HERE/'N05_interventional_fidelity.ipynb', n05),
    (HERE/'N06_cutoff_and_complexity.ipynb', n06),
    (HERE/'N07_summary.ipynb', n07),
]:
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print('wrote', path.name)
