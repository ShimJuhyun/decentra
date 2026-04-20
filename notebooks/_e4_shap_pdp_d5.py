"""E4 — D5 (ShapPdp / Choi & Cha 2026) surrogate zoo add-on.

Adds the Choi & Cha (2026) LightGBM-SHAP-PDP scorecard pipeline as D5
to the NB03 surrogate zoo, using the same GMSC/HC splits, teacher model
(BB = LightGBM), evaluation metrics (R², Spearman, Top-k, AdvTop-k,
AdvFull) and random baselines.

Pipeline re-cap (ShapPdpSurrogate):
  1. OptBinning per feature (same binning axis as D4=OptBin+Ridge).
  2. BB TreeSHAP on the training data (recomputed here).
  3. Per-bin mean SHAP (PDP).
  4. Monotone LightGBM regressor on (bin_center -> SHAP mean).
  5. Lookup bin -> f_j(bin_center) at inference; sum across features.

Outputs (to outputs/NB03/):
  - contribs_ShapPdp_GMSC.npy, contribs_ShapPdp_HC.npy
  - results_D5.json   (per-dataset metrics, append-style)

Dependencies: NB01 datasets.pkl, NB02 bb_model_*.txt, bb_score_test_*.npy,
bb_prob_*.npy, train_val_idx_*.pkl. No NB03 re-run required.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import json
import pickle
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
import shap
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from decentra.surrogate import ShapPdpSurrogate
from decentra.metrics.attribution import (
    attribution_fidelity,
    random_baseline_advtopk,
)

NB01_DIR = Path('outputs/NB01')
NB02_DIR = Path('outputs/NB02')
OUT_DIR = Path('outputs/NB03')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(surr_contribs, surr_pred, bb_shap, bb_score, bb_prob, n_features,
             surr_sign=-1):
    """surr_sign=-1 for score-space surrogates (Tree/EBM/Ridge);
    surr_sign=+1 for log-odds-space surrogates (ShapPdp/D5 preserves BB
    SHAP's sign: positive contribution = increases default = adverse)."""
    r2 = round(r2_score(bb_score, surr_pred), 4)
    sp = round(float(spearmanr(bb_score, surr_pred)[0]), 4)
    reject = bb_prob >= np.percentile(bb_prob, 90)
    af = attribution_fidelity(bb_shap, surr_contribs, reject,
                              bb_sign=1, surr_sign=surr_sign)
    rb1 = round(random_baseline_advtopk(bb_shap, reject, 1, n_features,
                                        bb_sign=1), 4)
    rb4 = round(random_baseline_advtopk(bb_shap, reject, 4, n_features,
                                        bb_sign=1), 4)
    return {'R2': r2, 'Spearman': sp,
            **{k: round(v, 4) for k, v in af.items()},
            'Random_AT1': rb1, 'Random_AT4': rb4}


def ols_calibrate(raw_train, target_train):
    """Linear map a + b * raw that mimics Choi & Cha Step 6 (PDO).

    Choi & Cha produce log-odds sums; our BB comparator `bb_score` is a
    PDO-scaled score. A single-variable OLS on training data recovers the
    same linear scale mapping the paper applies post-hoc.
    Returns (a, b). Apply as ``a + b * raw``.
    """
    x = np.asarray(raw_train, dtype=float)
    y = np.asarray(target_train, dtype=float)
    b = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    a = float(y.mean() - b * x.mean())
    return a, b


def main():
    with open(NB01_DIR / 'datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)

    all_results = {}
    for name, d in datasets.items():
        print(f'\n{"="*70}\n  {name}: D5 (ShapPdp / Choi & Cha 2026)\n{"="*70}')

        X_train = d['X_train']
        X_test = d['X_test']
        y_binary_train = np.asarray(d['y_train'])
        feature_names = d['feature_names']
        n_features = len(feature_names)

        bb_score_train = np.load(NB02_DIR / f'bb_score_train_{name}.npy')
        bb_score_test = np.load(NB02_DIR / f'bb_score_test_{name}.npy')
        bb_shap_test = np.load(NB02_DIR / f'bb_shap_{name}.npy')
        bb_prob_test = np.load(NB02_DIR / f'bb_prob_{name}.npy')

        booster = lgb.Booster(model_file=str(NB02_DIR / f'bb_model_{name}.txt'))

        t0 = time.time()
        print('  Computing BB TreeSHAP on training data ...', flush=True)
        explainer = shap.TreeExplainer(booster)
        raw = explainer.shap_values(X_train)
        if isinstance(raw, list):
            raw = raw[1] if len(raw) == 2 else raw[-1]
        bb_shap_train = np.asarray(raw)
        # TreeExplainer on a binary booster can return shape
        # (n, p, 2) or (n, p+1, 2); collapse to the positive-class p features.
        if bb_shap_train.ndim == 3:
            bb_shap_train = bb_shap_train[..., 1] \
                if bb_shap_train.shape[-1] == 2 else bb_shap_train[..., -1]
        if bb_shap_train.shape[1] == n_features + 1:
            bb_shap_train = bb_shap_train[:, :n_features]
        assert bb_shap_train.shape == X_train.shape, (
            f'train SHAP shape {bb_shap_train.shape} != X_train '
            f'{X_train.shape}')
        print(f'    done ({time.time()-t0:.1f}s), '
              f'shape={bb_shap_train.shape}')

        t1 = time.time()
        # Choi & Cha apply monotone only to the SHAP smoother, not to
        # OptBinning. Disable pre-binning auto-detection (which would be
        # derived vs. bb_score and pick the opposite sign of the binary
        # WoE target, forcing OptBinning into median fallback).
        surr = ShapPdpSurrogate(
            binning='optbinning',
            max_n_bins=10,
            min_bin_size=0.01,
            smoother_n_estimators=200,
            smoother_max_depth=2,
            smoother_learning_rate=0.05,
            monotone_detect_mode='none',
            random_state=42,
        )
        surr.fit(
            X_train, bb_score_train,
            shap_values=bb_shap_train,
            binning_y=y_binary_train,
        )

        raw_train = surr.predict(X_train)
        a, b = ols_calibrate(raw_train, bb_score_train)
        surr_pred = a + b * surr.predict(X_test)
        surr_contribs = surr.contributions(X_test)  # log-odds scale

        # BB SHAP is log-odds (positive = adverse). D5 contribs are also
        # log-odds with the same sign convention, so surr_sign=+1.
        metrics = evaluate(surr_contribs, surr_pred,
                           bb_shap_test, bb_score_test, bb_prob_test,
                           n_features, surr_sign=1)
        metrics['time_s'] = round(time.time() - t1, 1)
        metrics['pdo_a'] = round(a, 3)
        metrics['pdo_b'] = round(b, 3)
        metrics['shap_train_time_s'] = round(t1 - t0, 1)
        metrics['n_features_fitted'] = int(len(surr.bin_scores_))

        # Scorecard conversion (unpruned + pruned) to mirror §6 in NB03
        for prune_name, prune_params in [
            ('unpruned', dict(max_bins_per_feature=None, min_bin_ratio=0.0)),
            ('pruned', dict(max_bins_per_feature=5, min_bin_ratio=0.05)),
        ]:
            sm = surr.to_scorecard_model(
                X_train, y_binary=y_binary_train,
                feature_names=feature_names, **prune_params,
            )
            sc_raw_train = sm.predict(X_train)
            a_sc, b_sc = ols_calibrate(sc_raw_train, bb_score_train)
            sc_pred = a_sc + b_sc * sm.predict(X_test)
            sc_contribs = sm.contributions(X_test)
            m_sc = evaluate(sc_contribs, sc_pred, bb_shap_test,
                            bb_score_test, bb_prob_test, n_features,
                            surr_sign=1)
            m_sc['n_bins'] = int(sum(len(f.bins) for f in sm.features))
            m_sc['n_active'] = int(len(sm.features))
            mono_count = 0
            for feat in sm.features:
                scores = [b.score for b in feat.bins]
                if len(scores) >= 2:
                    diffs = np.diff(scores)
                    if np.all(diffs >= -1e-8) or np.all(diffs <= 1e-8):
                        mono_count += 1
            m_sc['mono_pct'] = round(
                mono_count / max(m_sc['n_active'], 1) * 100, 1)
            metrics[f'sc_{prune_name}'] = m_sc

        all_results[name] = metrics

        np.save(OUT_DIR / f'contribs_ShapPdp_{name}.npy',
                surr_contribs.astype(np.float32))

        sc_u = metrics['sc_unpruned']
        sc_p = metrics['sc_pruned']
        print(f'  ShapPdp (raw)           R2={metrics["R2"]:.3f}  '
              f'AT1={metrics["AdvTop1"]:.3f}  AT4={metrics["AdvTop4"]:.3f}  '
              f'AFR={metrics["AdvFull_R"]:.3f}  '
              f'({metrics["time_s"]}s fit / {metrics["shap_train_time_s"]}s train-SHAP)')
        print(f'  ShapPdp (sc_unpruned)   R2={sc_u["R2"]:.3f}  '
              f'AT4={sc_u["AdvTop4"]:.3f}  bins={sc_u["n_bins"]}  '
              f'feat={sc_u["n_active"]}  mono={sc_u["mono_pct"]}%')
        print(f'  ShapPdp (sc_pruned)     R2={sc_p["R2"]:.3f}  '
              f'AT4={sc_p["AdvTop4"]:.3f}  bins={sc_p["n_bins"]}  '
              f'feat={sc_p["n_active"]}  mono={sc_p["mono_pct"]}%')

    with open(OUT_DIR / 'results_D5.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {OUT_DIR / "results_D5.json"}')


if __name__ == '__main__':
    main()
