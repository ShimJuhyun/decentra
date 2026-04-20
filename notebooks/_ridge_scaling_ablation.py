"""Option B — LinearSurrogate(Ridge) scaling ablation on HC.

Compares how Ridge's AT-1 / AT-4 / R² / top-1 adverse distribution change with
different feature-scaling strategies:

- ``scale=True`` (default, StandardScaler)
- ``scale=False`` (no scaling, raw features)
- ``RobustScaler`` applied externally (then surrogate with scale=False)

Motivation: on HC (41 features, RS=317 single split), Ridge top-1 adverse is
FLAG_EMP_PHONE for 94.6% of rejected samples → AT-1 = 0.000 vs BB. The
StandardScaler artifact is suspected — rare-class values in sparse binaries
get inflated z-scores when std is small.
"""
import warnings; warnings.filterwarnings('ignore')
import pickle, json, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
import shap
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from decentra.surrogate import LinearSurrogate
from decentra.metrics.named import attribution_fidelity_named
from decentra._utils import transform_logit_to_score

OUT = Path('outputs/ridge_scaling_ablation'); OUT.mkdir(parents=True, exist_ok=True)
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
    return clf


def eval_ridge(X_tr, X_te, y_score_tr, bb_shap_te, bb_prob_te, bb_score_te,
                feats, label, scale_kind):
    if scale_kind == 'standard':
        surr = LinearSurrogate(method='ridge', alpha=1.0, scale=True)
        X_tr_use, X_te_use = X_tr, X_te
    elif scale_kind == 'none':
        surr = LinearSurrogate(method='ridge', alpha=1.0, scale=False)
        X_tr_use, X_te_use = X_tr, X_te
    elif scale_kind == 'robust':
        rs = RobustScaler().fit(X_tr)
        X_tr_use = pd.DataFrame(rs.transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te_use = pd.DataFrame(rs.transform(X_te), columns=X_te.columns, index=X_te.index)
        surr = LinearSurrogate(method='ridge', alpha=1.0, scale=False)
    else:
        raise ValueError(scale_kind)

    surr.fit(X_tr_use, y_score_tr)
    pred = np.asarray(surr.predict(X_te_use))
    adv = surr.adverse_contributions(X_te_use, target_scale='score')
    r2 = r2_score(bb_score_te, pred)
    reject = bb_prob_te >= np.percentile(bb_prob_te, 90)
    bb_adv = pd.DataFrame(bb_shap_te, columns=feats)
    fid = attribution_fidelity_named(bb_adv, adv, reject, ks=(1,3,4), adv_ks=(1,4))

    # Top-1 adverse distribution on rejected samples
    adv.index = X_te_use.index
    adv_rej = adv.loc[reject[reject].index if hasattr(reject, 'loc') else reject]
    # Simple: rebuild from mask
    adv_rej = pd.DataFrame(adv.values[reject], columns=feats)
    top1 = adv_rej.idxmax(axis=1).value_counts().head(5).to_dict()

    return {'label': label, 'scale': scale_kind, 'R2': round(r2, 4),
             **{k: round(v, 4) for k, v in fid.items()},
             'top1_dist': top1}


rows = []
for ds_name in ['GMSC', 'HC']:
    d = datasets[ds_name]
    teacher = fit_teacher(d['X_train'], d['y_train'])
    prob_te = teacher.predict_proba(d['X_test'])[:, 1]
    score_tr = transform_logit_to_score(teacher.predict_proba(d['X_train'])[:, 1])
    score_te = transform_logit_to_score(prob_te)
    shap_te = shap.TreeExplainer(teacher).shap_values(d['X_test'])
    if isinstance(shap_te, list): shap_te = shap_te[1]

    print(f"\n=== {ds_name} ({d['X_train'].shape[1]} features) ===")
    for kind in ['standard', 'none', 'robust']:
        res = eval_ridge(d['X_train'], d['X_test'], score_tr,
                          shap_te, prob_te, score_te,
                          d['feature_names'], ds_name, kind)
        rows.append({'dataset': ds_name, **res})
        print(f"  Ridge({kind:<8s}) R2={res['R2']:.3f}  "
              f"AT-1={res['AdvTop1']:.3f}  AT-4={res['AdvTop4']:.3f}  "
              f"cov={res['coverage_surr']:.2f}")
        print(f"    top1 adv top-5: {res['top1_dist']}")

with open(OUT/'ridge_scaling_ablation.json', 'w') as f:
    json.dump(rows, f, indent=2, default=str)
pd.DataFrame(rows).drop(columns=['top1_dist']).to_csv(OUT/'ridge_scaling_ablation.csv', index=False)
print(f"\nSaved to {OUT}/")
