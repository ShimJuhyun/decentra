# Source Purpose: Decentra

## 1. What This Codebase Is For

Decentra is a research-oriented Python package for converting black-box credit-risk models into interpretable surrogate scorecards.

The code appears to have been written to bridge three needs that usually conflict in credit modeling:

- predictive accuracy from black-box models
- scorecard-style transparency for business and review processes
- evidence that explanations and suggested actions are faithful to the original model

The package is therefore closer to a decision-oriented model interpretation framework than a generic explainability library.

## 2. Inferred Original Motivation

The author likely wanted to test whether a LightGBM-style credit model can be approximated by simpler, auditable structures while preserving the parts that matter operationally:

- applicant score or default-risk ranking
- feature-level adverse reasons
- reason-code ordering
- bin-based scorecard tables
- intervention guidance for rejected applicants
- stability of bins and scores under cross-validation or bootstrap sampling

The notebooks and experiment modules suggest an iterative research workflow: train a teacher model, fit multiple surrogates, evaluate them, calibrate attributions, and compare scorecard variants.

## 3. Main Concepts

### Teacher Model

The teacher is the original black-box model, usually a tree model such as LightGBM. It produces probability, logit, or score outputs.

### Surrogate Model

A surrogate is a simpler model trained to mimic the teacher output. Decentra implements several surrogate families, including tree, linear, binning, EBM, SHAP-PDP, and sequential-priority variants.

### Contributions

Contributions are per-feature additive values explaining a prediction. They are used for ranking features, generating adverse reasons, and building scorecards.

### Scorecard

A scorecard is a bin-rule representation of a model:

```text
prediction = base_score + sum(feature_bin_score)
```

### Reason Codes

Reason codes rank the feature-bin effects that explain a score. Positive and negative code conventions are used to distinguish favorable and adverse factors.

### Interventional Fidelity

Interventional fidelity checks whether changing features suggested by a surrogate actually improves the teacher model outcome.

## 4. Current Code Organization

The reusable package code lives under `src/decentra`.

Important modules:

- `surrogate`: surrogate implementations
- `scorecard_model`: deployable scorecard model
- `scorecard`: display-oriented scorecard table
- `calibration`: feature and bin calibration against black-box SHAP
- `metrics`: prediction, attribution, named attribution, and intervention metrics
- `experiments`: benchmark orchestration

The `tests` directory currently contains both real tests and a local scorecard explainer prototype. That prototype should be moved into `src`.

## 5. What Should Change During Restructuring

- Move production-like local explanation code from `tests` to `src/decentra/explain`.
- Keep tests focused on behavior verification.
- Make scorecard and explanation APIs explicit and stable.
- Repair encoding-damaged Korean labels and comments where they affect public output.
- Keep notebook code as research workflow unless the logic is reusable enough to become package code.

## 6. One-Sentence Purpose

Decentra exists to make black-box credit-risk decisions explainable, scorecard-compatible, and testably faithful to the original model.

