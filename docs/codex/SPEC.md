# Specification: Decentra Restructuring

## 1. System Overview

Decentra provides a pipeline for model replacement and explanation in credit scoring:

```text
training data
  -> black-box teacher model
  -> teacher score/logit and SHAP values
  -> surrogate model
  -> contributions
  -> scorecard model
  -> scorecard display, reason codes, local lift recommendations
  -> fidelity metrics
```

The refactoring should preserve this pipeline while clarifying package boundaries.

## 2. Module Responsibilities

### `decentra.surrogate`

Owns surrogate models that approximate teacher outputs.

Required behavior:

- fit to teacher output
- predict teacher-like scores/logits
- produce per-feature contributions
- expose feature rankings and adverse features
- convert additive forms to `ScorecardModel`

### `decentra.scorecard_model`

Owns the deployable scorecard representation.

Required behavior:

- hold `base_score`, `FeatureRule`, and `BinRule`
- predict with additive bin scores
- produce centered contributions after fitting
- serialize and deserialize to dictionaries

### `decentra.scorecard`

Owns display/reporting scorecards.

Required behavior:

- build from `ScorecardModel`
- compute bin counts and target rates
- assign reason codes
- return a stable pandas `DataFrame`

### `decentra.explain`

New module for local explanation features.

Required behavior:

- explain one row against scorecard prior information
- identify current recodes/bins
- compute candidate scores per recode
- rank reason codes
- generate lift recommendations

### `decentra.calibration`

Owns calibration of surrogate contributions against black-box SHAP.

Required behavior:

- feature-level magnitude-preserving calibration
- optional sign alignment
- bin-level calibration for discrete contribution values

### `decentra.metrics`

Owns model comparison metrics.

Required behavior:

- prediction fidelity
- attribution fidelity
- feature-name-based attribution alignment
- interventional fidelity

### `decentra.experiments`

Owns reusable benchmark orchestration.

Required behavior:

- run configured surrogate factories
- compute common metrics
- collect outputs and fitted models
- save benchmark results

## 3. Public Data Types

### `BinRule`

Fields:

- `lower: float`
- `upper: float`
- `score: float`

Boundary convention:

```python
lower <= x < upper
```

### `FeatureRule`

Fields:

- `name: str`
- `index: int`
- `bins: list[BinRule]`

### `ScorecardModel`

Fields:

- `base_score: float`
- `features: list[FeatureRule]`
- `mean_contributions_`
- `training_stats_`
- `is_fitted_`

### `PriorFeatureInfo`

Fields:

- `name: str`
- `description: str`
- `ranges: list[tuple[float, float]]`
- `recodes: list[int]`
- `representative_values: list[float] | None`

## 4. Public Method Contracts

### Surrogate `fit`

```python
fit(X, y_teacher, *, eval_set=None, sample_weight=None, **kwargs) -> self
```

`y_teacher` may be teacher logit, score, or another documented continuous target.

### Surrogate `predict`

```python
predict(X) -> ndarray shape (n_samples,)
```

### Surrogate `contributions`

```python
contributions(X) -> ndarray shape (n_samples, n_features)
```

### Surrogate `transform`

```python
transform(X) -> dict
```

Required keys:

- `predictions`
- `contributions`
- `ranking`
- `adverse`

### Local Explanation

```python
LocalScorecardExplainer(model, scorecard_prior).explain(row, output="all")
```

Accepted output values:

- `reasons`
- `lift`
- `all`

## 5. Sign Conventions

The code currently uses both score-scale and logit-scale conventions.

Recommended public boundary:

- `value > 0` means adverse in public adverse-contribution tables.
- Internal contribution sign may differ by model target, but conversion must happen before metric comparison.

## 6. Scorecard Requirements

- Scorecard predictions must equal `base_score + sum(bin_scores)`.
- `to_dict` and `from_dict` must preserve prediction behavior.
- Boundary behavior must be deterministic and tested.
- Display output must not depend on corrupted encoding.
- Reason-code ranking rules must be documented.

## 7. Local Explanation Requirements

- A row must map to exactly one current recode per feature.
- Current recode scaled score must be zero.
- Candidate recodes must compute scores by changing only the candidate feature unless generating combinations.
- Lift recommendations must include from/to feature values and from/to scores.
- If the model has `get_grade`, lift output must include from/to grades.
- Duplicate-feature combinations must be filtered.

## 8. Testing Requirements

Minimum test groups:

- feature calibration
- SHAP-PDP surrogate
- scorecard model prediction and serialization
- scorecard display schema
- local scorecard explanation
- named attribution alignment
- adverse sign convention

Optional dependency tests should be skippable when the dependency is unavailable.

## 9. Documentation Requirements

The canonical restructuring docs are:

- `source-purpose.md`
- `SPEC.md`
- `PRD.md`
- `TDD.md`

Each English document must have a Korean counterpart with `_kor.md`.

