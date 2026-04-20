"""E2 — Within-family R² optimization ablation via quantile stratification.

Framing: a natural way to raise surrogate R² is to stratify training by
teacher's predicted quantile and fit per-segment surrogates. This reduces
within-segment variance and boosts R² by construction. If C1 (R²≠AT-k) is
wrong, AT-k should also improve. We test this rigorously.

Variants per dataset × family:
  - Global (Q=1)
  - Quantile-3, Quantile-5, Quantile-10 (stratified per teacher's score quantile)

Outputs:
  outputs/e2_quantile_ablation/e2_{dataset}.csv,json
"""
import warnings; warnings.filterwarnings('ignore')
import pickle, json, time, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from decentra.surrogate import TreeSurrogate, EBMSurrogate
from decentra.metrics.named import attribution_fidelity_named
from decentra.metrics.interventional import median_intervention_fidelity
from decentra._utils import transform_logit_to_score

OUT = Path('outputs/e2_quantile_ablation'); OUT.mkdir(parents=True, exist_ok=True)
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


def _logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def make_surr(kind, n_jobs_ebm=8):
    if kind == 'Tree-d1':
        return TreeSurrogate(max_depth=1, verbose=0)
    if kind == 'EBM':
        return EBMSurrogate(interactions=0, n_jobs=n_jobs_ebm)
    raise ValueError(kind)


def fit_stratified(teacher, X_train, y_score_train, kind, n_segments, feat_names):
    """Stratify by teacher logit quantile on train; fit per-segment surrogate."""
    prob_tr = teacher.predict_proba(X_train)[:, 1]
    logit_tr = _logit(prob_tr)
    edges = np.quantile(logit_tr, np.linspace(0, 1, n_segments + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    surrogates = {}
    sizes = {}
    for q in range(n_segments):
        mask = (logit_tr > edges[q]) & (logit_tr <= edges[q+1])
        n = int(mask.sum())
        sizes[q] = n
        if n < 200:
            continue
        X_q = X_train.iloc[np.where(mask)[0]]
        y_q = y_score_train[mask]
        surr = make_surr(kind)
        try:
            surr.fit(X_q, y_q)
            surrogates[q] = surr
        except Exception as e:
            print(f'  Segment {q} fit failed: {e}')
    return edges, surrogates, sizes


def predict_stratified(teacher, X_test, edges, surrogates, feat_names, target_scale='score'):
    """Route each test sample to its training-edge-defined segment."""
    prob_te = teacher.predict_proba(X_test)[:, 1]
    logit_te = _logit(prob_te)
    n = len(X_test)

    pred = np.full(n, np.nan)
    contribs = np.zeros((n, len(feat_names)))
    adverse = np.zeros((n, len(feat_names)))

    for q, surr in surrogates.items():
        mask = (logit_te > edges[q]) & (logit_te <= edges[q+1])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        X_seg = X_test.iloc[idx]
        pred[idx] = np.asarray(surr.predict(X_seg))
        c_seg = np.asarray(surr.contributions(X_seg))
        contribs[idx] = c_seg
        adv_seg = surr.adverse_contributions(X_seg, target_scale=target_scale)
        adverse[idx] = adv_seg.values

    # Fallback: any unassigned (edge boundary), use nearest
    unassigned = np.isnan(pred)
    if unassigned.any():
        # Assign to nearest segment
        last_surr_q = max(surrogates.keys()) if surrogates else 0
        for i in np.where(unassigned)[0]:
            X_i = X_test.iloc[[i]]
            surr = surrogates[last_surr_q]
            pred[i] = surr.predict(X_i)[0]
            c = np.asarray(surr.contributions(X_i))[0]
            contribs[i] = c
            a = surr.adverse_contributions(X_i, target_scale=target_scale).values[0]
            adverse[i] = a

    return pred, pd.DataFrame(contribs, columns=feat_names, index=X_test.index), \
           pd.DataFrame(adverse, columns=feat_names, index=X_test.index)


rows = []
for ds_name, d in datasets.items():
    print(f'\n=== {ds_name} ({d["X_train"].shape[1]} features) ===', flush=True)
    teacher = fit_teacher(d['X_train'], d['y_train'])
    prob_te = teacher.predict_proba(d['X_test'])[:, 1]
    score_tr = transform_logit_to_score(teacher.predict_proba(d['X_train'])[:, 1])
    score_te = transform_logit_to_score(prob_te)
    shap_te = shap.TreeExplainer(teacher).shap_values(d['X_test'])
    if isinstance(shap_te, list): shap_te = shap_te[1]
    reject = prob_te >= np.percentile(prob_te, 90)
    feats = d['feature_names']
    bb_adv = pd.DataFrame(shap_te, columns=feats)
    medians = d['X_train'].median()

    for kind in ['Tree-d1', 'EBM']:
        print(f' --- {kind} ---', flush=True)
        # Global (Q=1): just fit on full training
        t0 = time.time()
        surr = make_surr(kind)
        surr.fit(d['X_train'], score_tr)
        pred = np.asarray(surr.predict(d['X_test']))
        adv = surr.adverse_contributions(d['X_test'], target_scale='score')
        r2 = r2_score(score_te, pred)
        fid = attribution_fidelity_named(bb_adv, adv, reject, ks=(1,3,4), adv_ks=(1,4))
        dif = median_intervention_fidelity(teacher, adv, d['X_test'], medians, reject,
                                             k_values=(1, 4))
        rows.append({'dataset': ds_name, 'family': kind, 'n_segments': 1,
                      'R2': r2, 'AT1': fid['AdvTop1'], 'AT4': fid['AdvTop4'],
                      'AdvFull_R': fid['AdvFull_R'],
                      'DA@1': dif['DA@1'], 'DA@4': dif['DA@4'],
                      'fit_s': round(time.time()-t0, 1)})
        print(f'  Global     R²={r2:.4f} AT-1={fid["AdvTop1"]:.3f} AT-4={fid["AdvTop4"]:.3f} '
              f'DA@1={dif["DA@1"]:.3f} DA@4={dif["DA@4"]:.3f} ({time.time()-t0:.0f}s)', flush=True)

        for Q in [3, 5, 10]:
            t0 = time.time()
            edges, surrogates, sizes = fit_stratified(
                teacher, d['X_train'], score_tr, kind, Q, feats)
            if len(surrogates) < Q:
                print(f'  Q={Q}: only {len(surrogates)}/{Q} segments fit', flush=True)
            pred, contribs, adv = predict_stratified(
                teacher, d['X_test'], edges, surrogates, feats)
            r2 = r2_score(score_te, pred)
            fid = attribution_fidelity_named(bb_adv, adv, reject, ks=(1,3,4), adv_ks=(1,4))
            dif = median_intervention_fidelity(teacher, adv, d['X_test'], medians, reject,
                                                 k_values=(1, 4))
            rows.append({'dataset': ds_name, 'family': kind, 'n_segments': Q,
                          'R2': r2, 'AT1': fid['AdvTop1'], 'AT4': fid['AdvTop4'],
                          'AdvFull_R': fid['AdvFull_R'],
                          'DA@1': dif['DA@1'], 'DA@4': dif['DA@4'],
                          'fit_s': round(time.time()-t0, 1)})
            print(f'  Quantile-{Q:<2d} R²={r2:.4f} AT-1={fid["AdvTop1"]:.3f} '
                  f'AT-4={fid["AdvTop4"]:.3f} DA@1={dif["DA@1"]:.3f} '
                  f'DA@4={dif["DA@4"]:.3f} ({time.time()-t0:.0f}s)', flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT/'e2_quantile_ablation.csv', index=False)
with open(OUT/'e2_quantile_ablation.json','w') as f:
    json.dump(rows, f, indent=2, default=float)
print(f'\nSaved to {OUT}/')
print()
print(df.to_string(index=False))
