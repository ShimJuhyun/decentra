# Decentra — Specification

> This document summarizes **what this codebase does** by reading the current
> `src/decentra/`, `notebooks/`, and `tests/` trees. It is the baseline for
> the accompanying PRD and TDD.

---

## 1. One-line summary

> **A research-and-production framework that mimics black-box credit-scoring
> models (e.g. LightGBM) with interpretable surrogates, converts those
> surrogates into deployable scorecards, and benchmarks them along three
> fidelity axes: prediction, adverse-reason attribution, and intervention.**

- **Research output** — metrics for the P5 paper line ("LightGBM Scorecard based on SHAP Values").
- **Production output** — `ScorecardModel` (deployable) plus `Scorecard` (display table with bins, weights, and reason codes).

---

## 2. Problem

Black-box credit-risk models are accurate but
- hard to satisfy regulation (Adverse Action Notice, model governance), and
- not in a *score + reason-code* form that underwriting and customer-notice systems can consume.

A faithful surrogate therefore has to answer four questions:

| Fidelity axis | Question |
|---|---|
| Predictive | Does the surrogate's logit/score match the teacher's? |
| Attribution | Does the surrogate's SHAP pattern, especially for rejected customers, match the teacher's? |
| Interventional | Does "if I change this feature, the score moves *this way*" agree with the teacher? |
| Deployment | Can all of the above survive compression into a scorecard table? |

Decentra defines, measures, compares, and serializes those four axes.

---

## 3. End-to-end data flow

```
Raw (X, y_binary)
        │
        ▼
[Black-box Teacher]   (external LightGBM-like model)
   predict_proba → bb_prob
   TreeSHAP       → bb_shap     (log-odds space, >0 = adverse)
   logit_to_score → bb_score    (score space, higher = safer)
        │
        ▼
[Surrogate]   y_target = bb_score or bb_logit
   .fit(X, y_target [, eval_set, base_model, shap_values])
   .predict(X)                → surr_pred
   .contributions(X)          → centered per-feature contributions (n × p)
   .adverse_contributions(X)  → DataFrame oriented so adverse > 0
        │
        ▼
[Calibration  (optional)]
   FeatureCalibrator  — per-feature α rescale (magnitude_preserving option)
   BinCalibrator      — refit bin scores under (1−λ)·L_pred + λ·L_attr + γ·∥s−s_old∥²
        │
        ▼
[Metrics]
   prediction_fidelity                → {R², Agree, Spearman}
   attribution_fidelity / *_named     → {Top-k, AdvTop-k, AdvFull_R/J, coverage}
   interventional_fidelity            → {DA, IR, Spearman_ρ, n_pairs}
   median_intervention_fidelity       → {DA@k, mean_delta@k, …}
        │
        ▼
[Scorecard artefacts]
   surr.to_scorecard_model(X, y_binary)         → ScorecardModel
       ├── base_score
       └── FeatureRule(name, index, [BinRule(lower, upper, score), …])
   ScorecardModel.scorecard(X, y_binary)        → Scorecard
       └── to_dataframe()  →  evaluation item / bin / weight / reason code (P###, N###)
```

---

## 4. Module inventory

### `src/decentra/`

| Path | Role |
|---|---|
| `__init__.py` | Public API: `Scorecard`, `ScorecardModel`, `TrainingStats`, `FeatureStats` |
| `_utils.py` | `information_value`, `logit`, `sigmoid`, `transform_logit_to_score` (PDO/anchor) |
| `stats.py` | `FeatureStats` / `TrainingStats` (training-data distribution snapshot — used for effort normalization, monitoring) |
| `scorecard_model.py` | `BinRule`, `FeatureRule`, `ScorecardModel` (deployable representation + fit/transform/predict) |
| `scorecard.py` | `Scorecard` (display only — reason codes P###/N###, target rate, composition) |
| `surrogate/base.py` | `BaseSurrogate` abstract class. Auto monotone detection, bin pruning (MSE / Chi² / score_diff), shared `to_scorecard_model` logic |
| `surrogate/tree.py` | `TreeSurrogate` — depth-1 → exact additive SHAP via `pred_contrib`, deeper → SHAP. LightGBM `<=` vs `BinRule.contains` resolved via `nextafter` |
| `surrogate/linear.py` | `LinearSurrogate` (OLS/Ridge/Lasso/EN with sign-flip monotone enforcement), `BinningSurrogate` (OptBinning + WoE/Dummy + linear), alias `OptBinningSurrogate` |
| `surrogate/ebm.py` | `EBMSurrogate` (InterpretML `ExplainableBoostingRegressor`, GAM or GA²M with interactions) |
| `surrogate/shap_pdp.py` | `ShapPdpSurrogate` — Choi & Cha (2026) D5 baseline. Does **not** regress on y_logit; bins → mean BB SHAP → monotone LGBM smoother |
| `surrogate/sequential.py` | `SequentialPrioritySurrogate` — stage-wise depth-1 LGBM ordered by feature priority (`abs` or `signed_rejected`), `frozen` or `cumulative` fit mode |
| `calibration/feature.py` | `FeatureCalibrator` — per-feature α; `magnitude_preserving=True` prevents R² collapse; optional `sign_align` |
| `calibration/bin.py` | `BinCalibrator` — refits bin scores with `(1-λ)·L_pred + λ·L_attr + γ·∥s-s_old∥²` |
| `metrics/prediction.py` | `prediction_fidelity` → {R², Agree, Spearman} |
| `metrics/attribution.py` | Positional: `topk`, `advtopk`, `advfull`, `attribution_fidelity`, `random_baseline_advtopk` (uses `bb_sign` / `surr_sign`) |
| `metrics/named.py` | Name-aligned: `align_attributions` (with `AlignmentInfo`, `missing="zero/drop/raise"`), `*_named`, `attribution_fidelity_named` (DataFrame / dict / (array, names)) |
| `metrics/interventional.py` | `extract_bin_structure`, `interventional_fidelity` (DA / IR / Spearman ρ), `median_intervention_fidelity` (model-agnostic via median substitution), legacy alias `compute_sic_sc` |
| `experiments/benchmark.py` | `BenchmarkConfig`, `BenchmarkResult`, `run_benchmark` — runs every surrogate factory on one train/test split and aggregates the full metric panel |

### `tests/`

| Path | Role |
|---|---|
| `test_feature_calibrator.py` | FeatureCalibrator: R² preservation under scale mismatch, magnitude preservation, deprecation warning, relative shape match |
| `test_shap_pdp.py` | ShapPdpSurrogate: fit with `base_model` or precomputed SHAP, centered contributions, scorecard conversion, input validation |
| `local_scorecard_explainer.py` | **Not a test.** Local row-level scorecard explanation + lift recommendation prototype (2025-05-29). Defines `LocalScorecardExplainer`, `PriorFetureInfo` (typo), `LocalExplainerState` |

### Outside the package but actually the main entry point

| Path | Role |
|---|---|
| `notebooks/executor.py` | `run_case` — for (teacher, train, test) it produces bench + calibration + interventional + cutoff + scorecard rows and saves them. In practice this *is* the library's main API |
| `notebooks/_e1`…`_e6.py` | CV / quantile ablation / SHAP-PDP / Sequential Priority experiment drivers |
| `notebooks/_pilot_run.py` | Pilot smoke test |
| `notebooks/N*.ipynb`, `NB*.ipynb` | Presentation and paper notebooks |

---

## 5. Domain conventions (easy to forget)

1. **Two sign conventions coexist.**
   - BB SHAP — log-odds space, `>0` is adverse.
   - Surrogate contributions — score space, `<0` is adverse.
   - Metrics either accept `bb_sign=+1` / `surr_sign=-1`, or call `adverse_contributions(target_scale="score"|"logit")` to unify the convention to `>0 = adverse`.

2. **Reject definition** — `bb_prob >= percentile(bb_prob, reject_percentile)`, default 90%.

3. **Centering convention** — `surrogate.contributions()` always returns centered values (`raw - mean_contributions_`). `_raw_contributions()` is internal.

4. **Bin rule convention** — `BinRule.contains` uses `lower <= x < upper`; the last bin has `upper = +inf`. `TreeSurrogate._get_feature_bins` aligns LightGBM's `<= threshold` to this convention with `nextafter`, preserving boundary-valued samples.

5. **Monotone auto-detection** — Spearman correlation with p-value < 0.05. A user-supplied dict overrides the auto value per feature; unlisted features stay auto.

6. **`ScorecardModel.fit` does two things.**
   (a) Caches `mean_contributions_` and `training_stats_`,
   (b) if `build_display=True` and `y_binary` is given, also builds the display `Scorecard` (opt-in).

7. **Surrogate `fit` target.** The parameter is named `y_logit` for legacy reasons, but in practice surrogates accept either `bb_score` (score space) or `bb_logit` (log-odds space). `target_scale` only changes the sign of `adverse_contributions`.

---

## 6. Current structural pain points (refactor motivation)

| # | Issue | Impact |
|---|---|---|
| 1 | **The real entry point lives outside `src`.** `notebooks/executor.py::run_case` is the main API in practice but is imported via `sys.path.insert`. Same for `_e*.py`. | Not reusable, not testable, not importable for outside users. |
| 2 | **`tests/` is separated from `src/`**, and `tests/local_scorecard_explainer.py` is not a test — it's a production-shaped prototype. | The user asked to merge tests into `src`; this prototype needs a real home. |
| 3 | **Two parallel metric tracks** — positional (`attribution.py`) vs named (`named.py`). `executor.py` uses named, `_e4_shap_pdp_d5.py` uses positional. | Confusing, inconsistent, "which is canonical?" |
| 4 | `OptBinningSurrogate = BinningSurrogate` only exists as a backward-compat alias. | Either deprecate explicitly or unify. |
| 5 | `compute_sic_sc` and similar legacy aliases linger. | Cleanup. |
| 6 | `Scorecard.to_dataframe()` uses Korean column names hard-coded. | Encoding/internationalization risk. |
| 7 | `PriorFetureInfo` typo (Feture → Feature). | Public API readability. |
| 8 | `experiments/benchmark.py` is single-split only; cross-validation logic lives in `_e3_cv_5fold.py`, `_e5_*`, `_e6_*`. | CV is not a first-class abstraction. |

---

## 7. Output destinations

- **Research outputs** — `.outputs/e*_*/{cv_summary, cv_wilcoxon_AT4, fold_*/result_*}.{csv,json,pkl}` → paper tables and figures.
- **Operational outputs** — `ScorecardModel` (serializable via `to_dict` / `from_dict`) and `Scorecard.to_dataframe()` (deliverable for underwriting / customer-notice systems).

---

## 8. Synthesis — why this code exists

> **To systematically measure, for opaque credit-scoring models (LightGBM and friends),
> how (1) prediction fidelity, (2) adverse-reason fidelity, (3) interventional fidelity,
> and (4) scorecard-deployment stability trade off when the model is mimicked by various
> interpretable surrogates and converted into a scorecard.**

This is at once a research codebase (driving the P5 paper line) and a library producing
artefacts that an operational credit-scoring system can consume.
