# TDD — Decentra Restructure

> Technical Design Document for the restructure described in [PRD.md](./PRD.md).
> "TDD" here means *Technical Design Document* — the test strategy section
> happens to follow a TDD-style "tests before behavior changes" plan.
>
> See [SPEC.md](./SPEC.md) for the current state.

---

## 1. Scope

1. **Promote** core orchestration code (`notebooks/executor.py::run_case`, CV drivers) from `notebooks/` into `src/decentra/experiments/`.
2. **Promote** the local-explainer prototype from `tests/local_scorecard_explainer.py` into `src/decentra/explain/local_scorecard.py`.
3. **Integrate** the `tests/` tree into `src/decentra/tests/`.
4. **Unify** the metric public surface (named-first), the surrogate alias scheme, and the legacy aliases.
5. **Verify** by behavior tests (output equivalence, round-trip identity, schema stability), not just smoke tests.

Out of scope: new surrogate families, model-serving APIs, training-pipeline replacement.

---

## 2. Target package layout

```text
src/decentra/
  __init__.py                  # Scorecard, ScorecardModel, TrainingStats, FeatureStats
  _utils.py                    # information_value, logit, sigmoid, transform_logit_to_score
  stats.py                     # FeatureStats, TrainingStats
  scorecard_model.py           # BinRule, FeatureRule, ScorecardModel
  scorecard.py                 # Scorecard (display)

  surrogate/
    __init__.py                # BaseSurrogate + all concrete surrogates
    base.py                    # BaseSurrogate, monotone detection, bin pruning, to_scorecard_model
    tree.py                    # TreeSurrogate
    linear.py                  # LinearSurrogate, BinningSurrogate (+ deprecated OptBinningSurrogate alias)
    ebm.py                     # EBMSurrogate
    shap_pdp.py                # ShapPdpSurrogate
    sequential.py              # SequentialPrioritySurrogate

  calibration/
    __init__.py                # FeatureCalibrator, BinCalibrator
    feature.py
    bin.py

  metrics/
    __init__.py                # canonical = named; positional re-exports retained
    prediction.py
    attribution.py             # positional (kept; "_positional" suffix optional)
    named.py                   # canonical: attribution_fidelity_named, align_attributions
    interventional.py          # interventional_fidelity, median_intervention_fidelity, extract_bin_structure

  explain/                     # NEW
    __init__.py                # LocalScorecardExplainer, PriorFeatureInfo, local_scorecard_explain
    local_scorecard.py         # moved + cleaned from tests/local_scorecard_explainer.py
    prior.py                   # PriorFeatureInfo, LocalExplainerState

  experiments/                 # PROMOTED FROM notebooks/
    __init__.py                # run_benchmark, run_case, run_cv, BenchmarkConfig, ...
    benchmark.py               # current run_benchmark
    case.py                    # MOVED from notebooks/executor.py (run_case, default_surrogate_factories)
    cv.py                      # NEW: run_cv with stratified splitter + Wilcoxon/BH aggregator
    aggregate.py               # NEW: cv-summary, wilcoxon, BH adjustment helpers

  tests/                       # MOVED FROM repo-root tests/ (Option B in PRD §6)
    __init__.py
    conftest.py                # shared fixtures: tiny_dataset, tiny_teacher, tiny_surrogate
    test_scorecard_model.py
    test_scorecard.py
    test_surrogate_tree.py
    test_surrogate_linear_binning.py
    test_surrogate_ebm.py
    test_surrogate_shap_pdp.py
    test_surrogate_sequential.py
    test_calibration_feature.py
    test_calibration_bin.py
    test_metrics_prediction.py
    test_metrics_attribution.py
    test_metrics_named.py
    test_metrics_interventional.py
    test_experiments_benchmark.py
    test_experiments_case.py
    test_experiments_cv.py
    test_explain_local.py
```

**Notebooks (`notebooks/`)** keep their place; their imports update to
`from decentra.experiments import run_case, run_cv` etc. `_e*.py` become thin
drivers that pass dataset-specific glue into `run_cv`.

Pytest discovers tests via `pytest src/decentra/tests` (or with
`tool.pytest.ini_options.testpaths = ["src/decentra/tests"]` in `pyproject.toml`).

---

## 3. Public API surface (target)

### 3.1 `decentra` top-level

```python
from decentra import (
    Scorecard, ScorecardModel,
    TrainingStats, FeatureStats,
)
```

### 3.2 Surrogates

```python
from decentra.surrogate import (
    BaseSurrogate,
    TreeSurrogate, LinearSurrogate, BinningSurrogate,
    EBMSurrogate, ShapPdpSurrogate, SequentialPrioritySurrogate,
)
# Deprecated alias kept for one minor version:
from decentra.surrogate import OptBinningSurrogate   # → BinningSurrogate
```

All surrogates implement (see PRD §8.1):

```python
fit(X, y_target, *, eval_set=None, sample_weight=None, **opts) -> self
predict(X) -> ndarray
contributions(X) -> ndarray                    # centered
adverse_contributions(X, target_scale="score"|"logit") -> DataFrame  # adverse > 0
transform(X) -> dict
fit_transform(X, y, **opts) -> dict
predict_with_contributions(X) -> (pred, contribs)
to_scorecard_model(X, y_binary=None, **bin_opts) -> ScorecardModel
feature_importances_ : ndarray                 # property
is_additive : bool                             # property
```

### 3.3 Scorecard

`ScorecardModel` and `Scorecard` keep their current shape.
Additions:

- `ScorecardModel.to_dict()` / `from_dict()` — round-trip identity guarantee.
- `Scorecard.to_dataframe(locale="ko"|"en")` — Korean stays default.

### 3.4 Local explanation

```python
from decentra.explain import (
    LocalScorecardExplainer,
    PriorFeatureInfo,
    local_scorecard_explain,
)
# Deprecated alias for one minor version:
from decentra.explain import PriorFetureInfo  # → PriorFeatureInfo
```

```python
explainer = LocalScorecardExplainer(model, prior_feature_info_list)
reasons_df, lift_df = explainer.explain(row, output="all")   # "reasons" / "lift" / "all"

# Convenience for ScorecardModel-backed models:
explainer = LocalScorecardExplainer.from_scorecard_model(sm, priors=...)
```

`_type=` is accepted but emits DeprecationWarning.

### 3.5 Metrics

```python
from decentra.metrics import (
    prediction_fidelity,
    attribution_fidelity_named,          # canonical
    align_attributions, AlignmentInfo,
    interventional_fidelity, median_intervention_fidelity,
    extract_bin_structure,
)
# Positional path retained (one minor version), but not the canonical import:
from decentra.metrics.attribution import attribution_fidelity  # legacy / positional
```

`compute_sic_sc` is removed.

### 3.6 Experiments

```python
from decentra.experiments import (
    BenchmarkConfig, BenchmarkResult, run_benchmark,
    CaseResult, run_case, default_surrogate_factories,
    CVResult, run_cv,
)
```

`run_cv` signature (proposed):

```python
def run_cv(
    *,
    datasets: dict[str, dict],            # {"GMSC": {"X": ..., "y": ...}, ...}
    teacher_factory: Callable,            # (X_train, y_train, seed) -> fitted teacher
    surrogate_factories: dict[str, SurrogateFactory],
    splitter,                             # e.g. StratifiedKFold(n_splits=5, random_state=42)
    out_dir: str | Path,
    target_metric: str = "AdvTop4",       # for Wilcoxon
    case_kwargs: dict | None = None,      # forwarded to run_case
) -> CVResult
```

`CVResult` exposes:

- `summary_df` — per (dataset, surrogate, metric) → mean / std / CI95 / n
- `wilcoxon_df` — pairwise Wilcoxon on `target_metric` with BH-adjusted p-values
- `fold_results : dict[(dataset, fold) -> CaseResult]`
- `save(path)` — writes the same files `_e3_cv_5fold.py` writes today.

---

## 4. Data contracts

### 4.1 Surrogate `contributions(X)`
- Shape `(n_samples, n_features)`.
- **Centered** on the training mean (`raw - mean_contributions_`).
- Column order is the order of `X.columns` (or the order at fit time when the input is an ndarray).

### 4.2 `BinRule`
- `contains(x)` uses `lower <= x < upper`, last bin has `upper = +inf`.
- LightGBM `<= threshold` → translated to `BinRule` boundary via `np.nextafter(t, +inf)` (already in `TreeSurrogate`).

### 4.3 `PriorFeatureInfo`
- `name: str`
- `description: str`
- `ranges: list[tuple[float, float]]` — sorted, non-overlapping
- `recodes: list[int]` — `len(recodes) == len(ranges)`
- `representative_values: list[float] | None` — if provided, length must match `ranges`
- Validation in `__post_init__`: raise `ValueError` on length mismatch or overlapping ranges.

### 4.4 `LocalExplainerState`
- Inherits `PriorFeatureInfo`.
- Holds `scores`, `score_mean`, `relative_scores`, `scaled_scores`, `recode`.
- Invariant: after `update()`, `scaled_scores[recodes.index(recode)] == 0`.

### 4.5 `ScorecardModel.to_dict() / from_dict()`
- `to_dict()` is JSON-safe (`float`, `int`, `str` only).
- `from_dict(to_dict(sm)).predict(X) == sm.predict(X)` element-wise.

### 4.6 Adverse-sign convention at metric boundaries
- Inputs to `attribution_fidelity_named` are assumed to encode **`value > 0 = adverse`** on both sides.
- BB SHAP in log-odds space already satisfies this. Surrogates produce it via `adverse_contributions(target_scale=...)`.

---

## 5. Refactoring steps (ordered, each independently shippable)

1. **Scaffolding**
   - Create `src/decentra/explain/`, `src/decentra/experiments/case.py`, `cv.py`, `aggregate.py`, `src/decentra/tests/`.
   - Add `[tool.pytest.ini_options] testpaths = ["src/decentra/tests"]` to `pyproject.toml`.
2. **Move tests (Option B)**
   - `git mv tests/test_*.py src/decentra/tests/`.
   - Add `conftest.py` with shared fixtures.
   - Verify `pytest` discovers and runs them unchanged.
3. **Promote `local_scorecard_explainer.py`**
   - `git mv tests/local_scorecard_explainer.py src/decentra/explain/local_scorecard.py`.
   - Split out `PriorFeatureInfo` / `LocalExplainerState` into `src/decentra/explain/prior.py`.
   - Add deprecated alias `PriorFetureInfo`.
   - Rename `_type` → `output` with deprecation alias.
   - Add input validation (range overlap, length checks).
4. **Promote `notebooks/executor.py`**
   - `git mv notebooks/executor.py src/decentra/experiments/case.py`.
   - Update imports inside `case.py`: `from decentra.surrogate import ...` etc. (no `sys.path` hacks).
   - Keep `default_surrogate_factories`, `run_case`, `CaseResult` exports.
   - Add a shim `notebooks/executor.py` that re-exports from `decentra.experiments` for backward compat (deprecation warning).
5. **Add `run_cv`**
   - Extract the CV / Wilcoxon / BH logic from `_e3_cv_5fold.py` into `decentra.experiments.cv:run_cv` and `aggregate.py`.
   - Make sparse-binary filter / median imputation pluggable preprocessing hooks (`preprocess_fold` callable in `run_cv`).
   - Add `CVResult` dataclass.
6. **Update `_e*.py`** to call `run_cv` / `run_case` instead of inlining the loop. Keep them as thin drivers under `notebooks/`.
7. **Metric unification**
   - Make `decentra.metrics` re-export the named functions by default.
   - Tag positional `attribution_fidelity` with a deprecation note in docstring (keep the symbol).
   - Remove `compute_sic_sc`.
8. **Cleanup aliases**
   - `OptBinningSurrogate = BinningSurrogate` becomes a deprecated re-export with `DeprecationWarning` on import.
9. **`Scorecard.to_dataframe(locale=...)`**
   - Keep current Korean columns as `locale="ko"` (default).
   - Add `locale="en"` column set: `"feature", "bin", "weight", "count", "target_count", "share", "target_rate", "reason_code", "reason_desc"`.
10. **Documentation**
    - Update `README.md` to point at `docs/claude/{SPEC, PRD, TDD}.md` and show one end-to-end example.

Each step ends with `pytest` green.

---

## 6. Test strategy

### 6.1 Tests to keep (already exist)

- `test_feature_calibrator.py` (4 tests).
- `test_shap_pdp.py` (5 tests).

These pass today and must keep passing after every step in §5.

### 6.2 New tests by area

**ScorecardModel**

- `test_scorecard_model_to_dict_roundtrip`
  `from_dict(to_dict(sm)).predict(X) == sm.predict(X)` exact equality.
- `test_scorecard_model_predict_matches_surrogate_within_tolerance`
  For `TreeSurrogate(max_depth=1)`, `sm.predict(X)` vs `surr.predict(X)` differ by at most `tolerance` (define in test).
- `test_scorecard_model_centered_contributions`
  `contributions(X_train).mean(axis=0) ≈ 0`.

**Scorecard**

- `test_scorecard_to_dataframe_ko_columns`
  default `locale="ko"` returns exact Korean column names currently used.
- `test_scorecard_to_dataframe_en_columns`
  `locale="en"` returns the documented English column set.
- `test_scorecard_reason_codes_assigned_by_abs_score_desc`
  P001 is the highest |score| among positive bins, N001 the highest |score| among negative bins.

**Surrogate (representative — one per family)**

- `test_tree_surrogate_depth1_bin_boundary_alignment`
  A value equal to a LightGBM split threshold lands in the **left** bin of `BinRule` (per the `nextafter` alignment).
- `test_linear_surrogate_centered_contribs_sum_matches_predict`
  `surr.predict(X) ≈ surr.contributions(X).sum(axis=1) + intercept_term`.
- `test_binning_surrogate_woe_signs_respect_monotone_constraint`
  With `monotone_constraints={"x_pos": 1}`, fitted coefficient on `x_pos` is ≥ 0 (sign-flip enforced).
- `test_ebm_surrogate_is_additive_when_interactions_zero`.
- `test_shap_pdp_requires_base_model_or_shap`  (already present).
- `test_sequential_priority_order_matches_priority_method`
  With `priority_method="abs"`, `surr.order_` equals `argsort(-abs(SHAP).mean(0))`.

**Calibration**

- `test_feature_calibrator_preserves_r2_under_scale_mismatch` (already exists).
- `test_bin_calibrator_lambda_zero_recovers_prediction_only_solution`
  With `lam=0`, the new bin scores reduce to OLS on the centered target (within tolerance).
- `test_bin_calibrator_lambda_one_drives_attribution`
  With `lam=1`, after calibration `attribution_fidelity_named` improves vs raw on synthetic mismatched data.

**Metrics**

- `test_named_attribution_align_zero_missing_policy`
  Missing features filled with 0; coverage is reported correctly.
- `test_named_attribution_align_raise_missing_policy`
  Mismatched columns raise `ValueError` with informative message.
- `test_advtopk_named_no_rejects_returns_zero`
  Empty reject mask → return 0.0 (not NaN, not crash).
- `test_advtopk_named_sign_convention`
  Passing log-odds-space attribution (>0 adverse) on both sides gives same result as score-space (<0 adverse, sign-flipped before call).
- `test_interventional_fidelity_no_pairs_returns_nan_n_pairs_zero`
  When no rejected sample has a less-adverse neighbor bin, return `{DA: nan, ..., n_pairs: 0}` (no crash).
- `test_median_intervention_fidelity_da_at_k_consistent_with_manual`
  For a tiny constructed dataset, `DA@1` matches a hand-computed value.

**Experiments**

- `test_run_benchmark_returns_one_row_per_surrogate`
  Length of `result.rows` equals number of factories.
- `test_run_case_writes_expected_files`
  `out_dir/result_<tag>.json` and `.pkl` exist after `run_case`.
- `test_run_cv_summary_columns_present`
  `summary_df` has columns `{dataset, surrogate, metric, mean, std, ci95, n}` for every metric in `METRICS`.
- `test_run_cv_matches_e3_for_one_dataset_within_tolerance`
  Pin a fixed dataset / seed / surrogate set and check that `run_cv` produces fold-level `AdvTop4` within `tolerance` of a stored expected value (regression fixture).

**Local explanation**

- `test_local_scorecard_reasons_current_bin_marked_once`
  For each feature, exactly one row in the reason table has `is_real == "T"`.
- `test_local_scorecard_scaled_score_zero_at_current_recode`
  At the current row's recode, scaled score is exactly 0; better recodes have positive `scaled`.
- `test_local_scorecard_lift_no_duplicate_feature_in_combination`
  Multi-feature combinations never include duplicate `feature` values within one row group.
- `test_local_scorecard_lift_changes_value_to_nearest_boundary`
  After lift, the row's value equals `min` or `max` of the target bin, whichever is closer to the original value.
- `test_local_scorecard_grade_columns_only_when_model_has_get_grade`
  Model with `get_grade` → DataFrame contains `From Grade` / `To Grade`. Model without → those columns are *absent*, not NaN.
- `test_local_scorecard_from_scorecard_model_helper`
  `LocalScorecardExplainer.from_scorecard_model(sm)` followed by `.explain(row)` returns the same `reasons_df` schema.
- `test_prior_feature_info_alias_deprecation`
  Importing `PriorFetureInfo` emits `DeprecationWarning`.

### 6.3 Regression fixtures

- A tiny seeded dataset (`n=300`, `p=5`) lives in `src/decentra/tests/fixtures/tiny.py` and is shared via `conftest.py`.
- A "golden" expected-metrics JSON (`fixtures/golden_cv.json`) records the values `run_cv` must produce on the tiny dataset under a fixed seed. `test_run_cv_matches_golden` compares per-fold values with `np.testing.assert_allclose(..., rtol=1e-6)`.

### 6.4 Snapshot tests (optional, for `Scorecard.to_dataframe`)

- Pre-compute a tiny scorecard's expected DataFrame and pickle it under `fixtures/scorecard_snapshot.pkl`. Compare row-by-row.

---

## 7. Migration / compatibility

- **Old name → new name** maintained for one minor version with `DeprecationWarning`:

| Old | New |
|---|---|
| `OptBinningSurrogate` | `BinningSurrogate` |
| `compute_sic_sc` | `interventional_fidelity` (removed, not aliased) |
| `PriorFetureInfo` | `PriorFeatureInfo` |
| `LocalScorecardExplainer.explain(..., _type=...)` | `output=...` |
| `from executor import run_case` (in notebooks) | `from decentra.experiments import run_case` (shim re-export with warning) |

- **No mathematical behavior change** during the move. Any change to a metric formula must come **after** the move and have an explicit test.
- **Encoding hygiene**: when touching a file, repair mojibake in comments / docstrings; never introduce new mojibake.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Moving `executor.py` breaks pickled `CaseResult` files under `.outputs/`. | `CaseResult` dataclass keeps the same name and field layout; if the module path matters for unpickling, ship a thin compatibility module under `notebooks/executor.py` that re-exports. |
| `Scorecard.to_dataframe()` Korean columns are load-bearing for downstream consumers. | Keep them as default (`locale="ko"`); add English as opt-in. |
| Optional dependencies (`optbinning`, `interpret`) make CI flaky. | Mark dependent tests with `pytest.importorskip`. |
| `run_cv` produces slightly different numerics on a different machine. | Use `assert_allclose(rtol=1e-6, atol=1e-8)` and seed all randomness. For E2E "matches `_e3_cv_5fold.py`" test, use `rtol=1e-3` (LightGBM determinism caveat). |
| `tests/` move under `src/` upsets external CI assumptions. | Document the change in README and `pyproject.toml`. If the user prefers the conservative path (Option A only), skip step §5.2; everything else still applies. |
| `_e*.py` are currently fold-of-truth for the paper experiments. | Keep them runnable. After step §5.6 they just call `run_cv`, but a frozen snapshot of one is committed for reference. |

---

## 9. Definition of Done

- Layout from §2 exists; `pytest src/decentra/tests` passes.
- No `sys.path.insert` in `notebooks/_e*.py` (drivers import from `decentra.experiments`).
- `tests/local_scorecard_explainer.py` removed; its replacement lives under `src/decentra/explain/` and is unit-tested.
- `ScorecardModel.from_dict(sm.to_dict()).predict(X)` is element-wise identical to `sm.predict(X)` in the test fixture.
- `run_cv` on the tiny fixture matches `fixtures/golden_cv.json` within `rtol=1e-6`.
- README points at `docs/claude/{SPEC, PRD, TDD}.md` and shows a 15-line end-to-end snippet.
- All deprecated aliases emit `DeprecationWarning` and are listed in a CHANGELOG.
