# TDD: Decentra Source Restructuring

## 1. Objective

Restructure Decentra so that package code, prototype logic, tests, and research notebooks have clear boundaries.

The immediate restructuring target is to integrate useful logic currently under `tests` into `src`, especially the local scorecard explainer prototype.

## 2. Current Source Interpretation

The codebase has three layers mixed together:

- Library layer: reusable package modules under `src/decentra`.
- Research layer: notebooks and experiment scripts under `notebooks`.
- Test/prototype layer: pytest tests plus `local_scorecard_explainer.py` under `tests`.

The library layer is already organized around surrogate modeling, scorecard conversion, calibration, and fidelity metrics. The main structural issue is that local row-level scorecard explanation exists as a prototype under `tests`, while it is actually product logic.

## 3. Target Structure

```text
src/decentra/
  __init__.py
  scorecard.py
  scorecard_model.py
  stats.py
  _utils.py
  explain/
    __init__.py
    local_scorecard.py
  surrogate/
    __init__.py
    base.py
    tree.py
    linear.py
    ebm.py
    shap_pdp.py
    sequential.py
  calibration/
    __init__.py
    feature.py
    bin.py
  metrics/
    __init__.py
    prediction.py
    attribution.py
    named.py
    interventional.py
  experiments/
    __init__.py
    benchmark.py
tests/
  test_feature_calibrator.py
  test_shap_pdp.py
  test_local_scorecard.py
  test_scorecard_model.py
  test_named_metrics.py
```

## 4. Public API Proposal

### Surrogate API

All surrogates should conform to:

```python
surr.fit(X_train, y_teacher, **kwargs)
pred = surr.predict(X)
contribs = surr.contributions(X)
result = surr.transform(X)
sm = surr.to_scorecard_model(X_train, y_binary=y_train)
```

`transform(X)` should return:

- `predictions`
- `contributions`
- `ranking`
- `adverse`

### Scorecard API

`ScorecardModel` is the deployable representation:

```python
sm.predict(X)
sm.contributions(X)
sm.transform(X)
sm.scorecard(X, y_binary)
sm.to_dict()
ScorecardModel.from_dict(payload)
```

### Local Explanation API

New module:

```python
from decentra.explain import LocalScorecardExplainer, PriorFeatureInfo
```

Expected use:

```python
features = [PriorFeatureInfo(...)]
explainer = LocalScorecardExplainer(model, features)
reasons, lifts = explainer.explain(row, output="all")
```

Supported outputs:

- `output="reasons"`
- `output="lift"`
- `output="all"`

Temporary backward-compatible aliases:

- `PriorFetureInfo = PriorFeatureInfo`
- `_type="rc"` maps to `output="reasons"`
- `_type="lift"` maps to `output="lift"`
- `_type="all"` maps to `output="all"`

## 5. Data Contracts

### `PriorFeatureInfo`

Fields:

- `name: str`
- `description: str = ""`
- `ranges: list[tuple[float, float]]`
- `recodes: list[int]`
- `representative_values: list[float] | None = None`

Validation:

- `ranges` and `recodes` must have equal length.
- `representative_values`, if provided, must have equal length.
- Ranges must be sorted.
- Ranges should not overlap.

### `LocalExplainerState`

Fields:

- feature prior fields
- `scores`
- `score_mean`
- `relative_scores`
- `scaled_scores`
- `recode`

Behavior:

- `relative_scores = candidate_score - feature_mean_score`
- `scaled_scores = candidate_score - current_recode_score`
- current recode's scaled score must be `0`

### `BinRule`

Current boundary convention:

```python
lower <= x < upper
```

Any LightGBM threshold conversion must preserve LightGBM's `x <= threshold` behavior by adjusting upper boundaries where needed.

## 6. Implementation Plan

1. Create `src/decentra/explain/`.
2. Move `tests/local_scorecard_explainer.py` into `src/decentra/explain/local_scorecard.py`.
3. Rename `PriorFetureInfo` to `PriorFeatureInfo`.
4. Keep compatibility aliases for one transition period.
5. Replace `_type` with `output`.
6. Add dataclass validation.
7. Add tests for local explanation behavior.
8. Repair `Scorecard.to_dataframe()` encoding/syntax issues.
9. Add scorecard roundtrip and boundary tests.
10. Keep notebook changes separate from package restructuring.

## 7. Test Plan

### Existing Tests

Keep and maintain:

- `test_feature_calibrator.py`
- `test_shap_pdp.py`

### New Local Explanation Tests

- `test_prior_feature_info_validates_lengths`
- `test_local_reasons_marks_current_recode`
- `test_local_scaled_score_current_recode_is_zero`
- `test_local_lift_recomputes_score`
- `test_local_lift_filters_duplicate_feature_combinations`
- `test_local_lift_includes_grade_when_model_supports_get_grade`

### New Scorecard Tests

- `test_scorecard_model_predict_roundtrip_dict`
- `test_scorecard_transform_requires_fit`
- `test_scorecard_bin_boundary_alignment`
- `test_scorecard_to_dataframe_schema`

### New Metrics Tests

- `test_named_alignment_zero_fills_missing_features`
- `test_named_alignment_raises_on_missing_when_configured`
- `test_adverse_contributions_positive_means_adverse`

## 8. Migration Strategy

- Move code first with minimal behavior changes.
- Add tests around existing behavior before changing semantics.
- Keep aliases for misspelled and legacy names.
- Document sign conventions before enforcing broad API changes.
- Do not fold notebook experiment scripts into `src` unless they are reusable.

## 9. Known Risks

- Some Korean comments and labels are mojibake-corrupted.
- `Scorecard.to_dataframe()` appears damaged and should be fixed before use.
- Optional dependencies such as `optbinning` and `interpret` may not exist in all test environments.
- SHAP and LightGBM tests can be slow; fixtures should stay small.
- Local lift combinations can grow quickly with many features and bins.

## 10. Definition of Done

- `docs/codex/` contains the intended English documents and their `_kor.md` Korean counterparts.
- Production local explanation code lives under `src/decentra/explain`.
- `tests/` contains tests, fixtures, and test utilities only.
- Public imports work from `decentra.explain`.
- Current and newly added tests pass.
- The README can point to the new API without referencing test files as product code.
