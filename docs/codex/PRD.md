# PRD: Decentra Source Restructuring

## 1. Product Intent

Decentra is a Python library for turning black-box credit-risk models into interpretable, decision-oriented scorecards.

The project appears to have been built to answer a practical model-risk question:

> Can we keep the predictive strength of a black-box model while producing scorecard-like explanations, adverse-action reasons, and actionable improvement guidance?

This is not a general-purpose XAI toolkit. The source is organized around credit scoring concepts: default risk, score/logit outputs, rejected customers, adverse features, reason codes, scorecard bins, calibration, and interventional fidelity.

## 2. Problem

Black-box models such as LightGBM can perform well in credit-risk prediction, but business use often requires:

- transparent feature-level explanations
- stable scorecard artifacts
- adverse-action reason codes
- validation that explanations agree with the black-box model
- validation that suggested feature changes actually improve black-box outcomes

Existing explainability methods can show feature importance, but they do not automatically produce deployable scorecards or evaluate whether an explanation is actionable.

## 3. Users

- Credit-risk data scientists building challenger or surrogate models.
- Model-risk reviewers checking explanation fidelity.
- Strategy analysts who need scorecard-style tables and reason codes.
- Researchers comparing surrogate modeling methods.

## 4. Goals

- Provide a consistent API for multiple surrogate model families.
- Convert fitted surrogates into standardized scorecard models.
- Generate global scorecards and local row-level explanations.
- Compare surrogates using prediction, attribution, and intervention metrics.
- Move prototype production logic out of `tests` into `src`.
- Keep `tests` as executable specifications for package behavior.

## 5. Non-Goals

- Building a full model-serving platform.
- Replacing upstream black-box model training pipelines.
- Providing a regulatory compliance guarantee.
- Building a UI or report-generation product in this phase.
- Supporting arbitrary unstructured model types.

## 6. Current Capabilities

The current source already contains these major building blocks:

- `decentra.surrogate`
  - `TreeSurrogate`
  - `LinearSurrogate`
  - `BinningSurrogate`
  - `EBMSurrogate`
  - `ShapPdpSurrogate`
  - `SequentialPrioritySurrogate`
- `decentra.scorecard_model`
  - deployable additive scorecard representation
- `decentra.scorecard`
  - display scorecard with bins, counts, target rates, and reason codes
- `decentra.calibration`
  - feature-level and bin-level calibration to black-box SHAP
- `decentra.metrics`
  - prediction, attribution, named attribution, and interventional fidelity
- `decentra.experiments`
  - benchmark orchestration
- `tests/local_scorecard_explainer.py`
  - local scorecard explanation prototype that should move into `src`

## 7. User Stories

1. As a data scientist, I can fit a surrogate to a black-box model's score or logit output.
2. As a reviewer, I can compare surrogate predictions to the black-box decisions.
3. As a reviewer, I can compare surrogate adverse reasons to black-box SHAP reasons.
4. As a strategy analyst, I can export a scorecard table with feature bins, bin scores, target rates, and reason codes.
5. As a case reviewer, I can explain one applicant row and see which feature-bin changes would improve the score.
6. As a researcher, I can benchmark surrogate methods across folds and datasets.

## 8. Functional Requirements

### Surrogates

- Each surrogate should expose `fit`, `predict`, `contributions`, `transform`, and `predict_with_contributions`.
- Each surrogate should preserve feature-name ordering where possible.
- Each surrogate should support contribution ranking and adverse feature extraction.
- Additive surrogates should support conversion to `ScorecardModel`.

### Scorecards

- `ScorecardModel` should represent `base_score + sum(feature_bin_score)`.
- Scorecard bins should have deterministic boundary behavior.
- Scorecard display should include:
  - feature name
  - bin range
  - bin score
  - sample count
  - target count
  - target rate
  - reason code

### Local Explanation

- Local scorecard explanation should move from `tests/local_scorecard_explainer.py` into `src/decentra/explain`.
- It should generate:
  - row-level current recode/bin table
  - reason-code ranking
  - score-lift candidates
  - optional grade transitions when the model provides `get_grade`

### Metrics

- Prediction fidelity should compare teacher and surrogate outputs.
- Attribution fidelity should compare top-k and adverse top-k reasons.
- Named attribution metrics should align by feature name, not only position.
- Interventional fidelity should test whether suggested changes improve teacher output.

## 9. Quality Requirements

- Public APIs should remain simple and sklearn-like.
- Tests should cover behavior and data contracts, not only smoke execution.
- Optional dependencies should remain optional where possible.
- Korean business labels should be encoding-safe and documented.
- Research notebooks should call package APIs rather than contain duplicated core logic.

## 10. Success Criteria

- `tests/local_scorecard_explainer.py` no longer contains production logic.
- `src/decentra/explain/local_scorecard.py` exposes the local explanation API.
- Existing tests pass after restructuring.
- New tests cover local explanation, scorecard conversion, and named attribution alignment.
- `docs/codex/PRD.md` and `docs/codex/TDD.md` are the canonical restructuring references.

## 11. Open Questions

- Should the canonical surrogate target be logit scale or credit-score scale?
- Should public adverse contributions always use `value > 0 = adverse`?
- Should reason-code ranking be based on signed score, absolute score, relative score, or a business priority field?
- Should local lift recommendations allow multi-feature combinations by default?
- Which notebook experiments should become formal experiment modules?

