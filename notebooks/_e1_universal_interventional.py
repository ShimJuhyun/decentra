"""E1 — Universal interventional fidelity across all surrogates.

Answers the thesis question: "is attribution fidelity (AdvTop-k) a necessary
condition for interventional fidelity?"

For each surrogate on each dataset:
1. Fit and compute adverse contributions.
2. For each rejected sample, take surrogate's top-k adverse features (k=1,3,4)
   and replace their values with training medians.
3. Measure teacher's actual P(default) change.
4. Aggregate DA@k (directional accuracy) across rejected samples.

Hypothesis: surrogates with low AdvTop-k (e.g., Ridge on HC) will also have
low DA@k — i.e., following their advice doesn't actually help.

Outputs: outputs/e1_universal_interventional/e1_{dataset}.json
"""
import warnings; warnings.filterwarnings('ignore')
import pickle, json, time, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from decentra.surrogate import TreeSurrogate, EBMSurrogate, LinearSurrogate, BinningSurrogate
from decentra.metrics.interventional import median_intervention_fidelity
from decentra.metrics.named import attribution_fidelity_named
from decentra._utils import transform_logit_to_score

OUT = Path('outputs/e1_universal_interventional'); OUT.mkdir(parents=True, exist_ok=True)
with open('outputs/N01/datasets.pkl','rb') as f: datasets = pickle.load(f)


def fit_teacher(X_tr, y_tr, seed=42):
    X_tr_i, X_val, y_tr_i, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    clf = lgb.LGBMClassifier(max_depth=-1, num_leaves=63, n_estimators=1000,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, min_child_samples=50,
                              random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf, X_tr_i.index


def make_factories(n_jobs_ebm=8):
    return {
        'Tree-d1':      lambda: TreeSurrogate(max_depth=1, verbose=0),
        'Tree-d1-mono': lambda: TreeSurrogate(max_depth=1, verbose=0,
                                               monotone_detect_mode='auto'),
        'Tree-d3':      lambda: TreeSurrogate(max_depth=3, verbose=0),
        'Tree-d6':      lambda: TreeSurrogate(max_depth=6, verbose=0),
        'EBM':          lambda: EBMSurrogate(interactions=0, n_jobs=n_jobs_ebm),
        'EBM-mono':     lambda: EBMSurrogate(interactions=0, n_jobs=n_jobs_ebm,
                                               monotone_detect_mode='auto'),
        'Ridge':        lambda: LinearSurrogate(method='ridge', alpha=1.0),
        'OptBin+Ridge': lambda: BinningSurrogate(method='ridge', alpha=1.0,
                                                   max_n_bins=10),
    }


for ds_name, d in datasets.items():
    print(f'\n=== {ds_name} ({d["X_train"].shape[1]} features) ===', flush=True)
    teacher, tr_inner_idx = fit_teacher(d['X_train'], d['y_train'])

    prob_te = teacher.predict_proba(d['X_test'])[:, 1]
    score_tr = transform_logit_to_score(teacher.predict_proba(d['X_train'])[:, 1])
    shap_te = shap.TreeExplainer(teacher).shap_values(d['X_test'])
    if isinstance(shap_te, list): shap_te = shap_te[1]
    reject = prob_te >= np.percentile(prob_te, 90)

    feats = d['feature_names']
    bb_adv = pd.DataFrame(shap_te, columns=feats)
    medians = d['X_train'].median()

    # BB reference: DA if we follow BB SHAP's top-k adverse
    bb_dif = median_intervention_fidelity(
        teacher, bb_adv, d['X_test'], medians, reject, k_values=(1, 3, 4))
    print(f'  BB reference DA@1={bb_dif["DA@1"]:.3f} DA@4={bb_dif["DA@4"]:.3f} '
          f'(n={bb_dif["n@1"]})', flush=True)

    rows = [{'surrogate': 'BB_Oracle', 'AT1': 1.0, 'AT4': 1.0, 'R2': 1.0,
             **bb_dif}]

    for sname, factory in make_factories().items():
        t0 = time.time()
        surr = factory()
        # Use train rows (same as executor pipeline)
        X_tr_use = d['X_train'].loc[tr_inner_idx]
        y_tr_use = score_tr[d['X_train'].index.get_indexer(tr_inner_idx)]
        try: surr.fit(X_tr_use, y_tr_use)
        except TypeError: surr.fit(X_tr_use, y_tr_use)

        pred = np.asarray(surr.predict(d['X_test']))
        score_te = transform_logit_to_score(prob_te)
        r2 = r2_score(score_te, pred)
        adv = surr.adverse_contributions(d['X_test'], target_scale='score')
        fid = attribution_fidelity_named(bb_adv, adv, reject, ks=(1,3,4), adv_ks=(1,4))
        dif = median_intervention_fidelity(
            teacher, adv, d['X_test'], medians, reject, k_values=(1, 3, 4))
        row = {'surrogate': sname, 'R2': r2, 'fit_s': round(time.time()-t0, 1),
                'AT1': fid['AdvTop1'], 'AT4': fid['AdvTop4'],
                'AdvFull_R': fid['AdvFull_R'], **dif}
        rows.append(row)
        print(f'  {sname:<14s} R²={r2:.3f} AT-1={row["AT1"]:.3f} AT-4={row["AT4"]:.3f} '
              f'→ DA@1={dif["DA@1"]:.3f} DA@4={dif["DA@4"]:.3f} ({row["fit_s"]:.0f}s)',
              flush=True)

    with open(OUT/f'e1_{ds_name}.json', 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    pd.DataFrame(rows).to_csv(OUT/f'e1_{ds_name}.csv', index=False)

# Combined summary
combined = []
for ds_name in datasets:
    rows = json.load(open(OUT/f'e1_{ds_name}.json'))
    for r in rows:
        combined.append({'dataset': ds_name, **r})
pd.DataFrame(combined).to_csv(OUT/'e1_summary.csv', index=False)
print(f'\nSaved to {OUT}/')
