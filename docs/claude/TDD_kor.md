# TDD — Decentra 재구성

> [PRD_kor.md](./PRD_kor.md) 에서 정의된 재구성을 위한 Technical Design Document.
> 여기서 "TDD" 는 *Technical Design Document* 의 의미이며, 테스트 전략 절은
> "행동 변경 전에 테스트를 둔다" 라는 TDD 스타일 계획을 따른다.
>
> 현재 상태는 [SPEC_kor.md](./SPEC_kor.md) 참고.

---

## 1. Scope

1. **승격**: `notebooks/executor.py::run_case` 와 CV driver 의 핵심 로직을 `notebooks/` → `src/decentra/experiments/` 로 이동.
2. **승격**: `tests/local_scorecard_explainer.py` prototype 을 `src/decentra/explain/local_scorecard.py` 로 이동.
3. **통합**: `tests/` 트리를 `src/decentra/tests/` 로 통합.
4. **통일**: metric public surface (name-aligned 우선), surrogate alias 정책, legacy alias 정리.
5. **검증**: 동작 테스트(출력 동치, round-trip 동일성, schema 안정성)로 — smoke 가 아니다.

Out of scope: 새 surrogate family, 모델 서빙 API, 학습 파이프라인 교체.

---

## 2. 목표 패키지 레이아웃

```text
src/decentra/
  __init__.py                  # Scorecard, ScorecardModel, TrainingStats, FeatureStats
  _utils.py                    # information_value, logit, sigmoid, transform_logit_to_score
  stats.py                     # FeatureStats, TrainingStats
  scorecard_model.py           # BinRule, FeatureRule, ScorecardModel
  scorecard.py                 # Scorecard (display)

  surrogate/
    __init__.py                # BaseSurrogate + 모든 구체 surrogate
    base.py                    # BaseSurrogate, monotone 감지, bin pruning, to_scorecard_model
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
    __init__.py                # canonical = named; positional 재노출 유지
    prediction.py
    attribution.py             # positional (유지, 옵션으로 "_positional" suffix)
    named.py                   # canonical: attribution_fidelity_named, align_attributions
    interventional.py          # interventional_fidelity, median_intervention_fidelity, extract_bin_structure

  explain/                     # NEW
    __init__.py                # LocalScorecardExplainer, PriorFeatureInfo, local_scorecard_explain
    local_scorecard.py         # tests/local_scorecard_explainer.py 에서 이동 + 정리
    prior.py                   # PriorFeatureInfo, LocalExplainerState

  experiments/                 # notebooks/ 에서 승격
    __init__.py                # run_benchmark, run_case, run_cv, BenchmarkConfig, ...
    benchmark.py               # 현재 run_benchmark
    case.py                    # notebooks/executor.py 에서 이동 (run_case, default_surrogate_factories)
    cv.py                      # NEW: StratifiedKFold + Wilcoxon/BH 집계 포함 run_cv
    aggregate.py               # NEW: cv-summary, wilcoxon, BH adjustment 헬퍼

  tests/                       # repo-root tests/ → 여기로 이동 (PRD §6 의 Option B)
    __init__.py
    conftest.py                # 공유 fixture: tiny_dataset, tiny_teacher, tiny_surrogate
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

**`notebooks/`** 는 그대로 유지하되 import 가 `from decentra.experiments import run_case, run_cv` 등으로 바뀐다. `_e*.py` 는 데이터셋별 glue 만 가진 얇은 driver 로 줄어들고, `run_cv` 를 호출한다.

`pyproject.toml` 에 `[tool.pytest.ini_options] testpaths = ["src/decentra/tests"]` 추가하면 `pytest` 가 자동 발견한다.

---

## 3. Public API surface (목표)

### 3.1 `decentra` 최상위

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
# 한 minor 버전 동안 deprecated alias 유지:
from decentra.surrogate import OptBinningSurrogate   # → BinningSurrogate
```

모든 surrogate 가 구현 (PRD §8.1 참고):

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

`ScorecardModel`, `Scorecard` 의 현재 형태 유지. 추가 사항:

- `ScorecardModel.to_dict()` / `from_dict()` — round-trip identity 보장.
- `Scorecard.to_dataframe(locale="ko"|"en")` — 한국어가 default.

### 3.4 Local explanation

```python
from decentra.explain import (
    LocalScorecardExplainer,
    PriorFeatureInfo,
    local_scorecard_explain,
)
# 한 minor 버전 동안 deprecated alias:
from decentra.explain import PriorFetureInfo  # → PriorFeatureInfo
```

```python
explainer = LocalScorecardExplainer(model, prior_feature_info_list)
reasons_df, lift_df = explainer.explain(row, output="all")   # "reasons" / "lift" / "all"

# ScorecardModel 기반 모델을 위한 편의:
explainer = LocalScorecardExplainer.from_scorecard_model(sm, priors=...)
```

`_type=` 는 수용하지만 `DeprecationWarning` 발생.

### 3.5 Metrics

```python
from decentra.metrics import (
    prediction_fidelity,
    attribution_fidelity_named,          # canonical
    align_attributions, AlignmentInfo,
    interventional_fidelity, median_intervention_fidelity,
    extract_bin_structure,
)
# positional 경로 한 minor 버전 동안 유지, 그러나 canonical import 는 아님:
from decentra.metrics.attribution import attribution_fidelity  # legacy / positional
```

`compute_sic_sc` 제거.

### 3.6 Experiments

```python
from decentra.experiments import (
    BenchmarkConfig, BenchmarkResult, run_benchmark,
    CaseResult, run_case, default_surrogate_factories,
    CVResult, run_cv,
)
```

`run_cv` 시그니처 (제안):

```python
def run_cv(
    *,
    datasets: dict[str, dict],            # {"GMSC": {"X": ..., "y": ...}, ...}
    teacher_factory: Callable,            # (X_train, y_train, seed) -> fitted teacher
    surrogate_factories: dict[str, SurrogateFactory],
    splitter,                             # 예: StratifiedKFold(n_splits=5, random_state=42)
    out_dir: str | Path,
    target_metric: str = "AdvTop4",       # Wilcoxon 기준
    case_kwargs: dict | None = None,      # run_case 로 전달
) -> CVResult
```

`CVResult` 공개 필드:

- `summary_df` — (dataset, surrogate, metric) → mean / std / CI95 / n
- `wilcoxon_df` — `target_metric` 에 대한 쌍별 Wilcoxon + BH-보정 p-value
- `fold_results : dict[(dataset, fold) -> CaseResult]`
- `save(path)` — `_e3_cv_5fold.py` 가 오늘 만드는 것과 같은 파일들을 기록.

---

## 4. 데이터 컨트랙트

### 4.1 Surrogate `contributions(X)`
- 형태 `(n_samples, n_features)`.
- **중심화** 됨 (`raw - mean_contributions_`).
- 컬럼 순서는 `X.columns` 의 순서 (ndarray 입력일 때는 fit 시점의 순서).

### 4.2 `BinRule`
- `contains(x)` 는 `lower <= x < upper`, 마지막 bin 은 `upper = +inf`.
- LightGBM 의 `<= threshold` → `np.nextafter(t, +inf)` 로 `BinRule` 경계 정렬 (현재 `TreeSurrogate` 가 이미 구현).

### 4.3 `PriorFeatureInfo`
- `name: str`
- `description: str`
- `ranges: list[tuple[float, float]]` — 정렬됨, 비중첩
- `recodes: list[int]` — `len(recodes) == len(ranges)`
- `representative_values: list[float] | None` — 제공 시 `ranges` 길이와 일치
- `__post_init__` 검증: 길이 불일치·범위 중첩 시 `ValueError`.

### 4.4 `LocalExplainerState`
- `PriorFeatureInfo` 상속.
- `scores`, `score_mean`, `relative_scores`, `scaled_scores`, `recode` 보유.
- 불변식: `update()` 호출 후 `scaled_scores[recodes.index(recode)] == 0`.

### 4.5 `ScorecardModel.to_dict() / from_dict()`
- `to_dict()` 는 JSON-safe (`float`, `int`, `str` 만).
- `from_dict(to_dict(sm)).predict(X) == sm.predict(X)` element-wise.

### 4.6 메트릭 경계의 adverse 부호 약속
- `attribution_fidelity_named` 입력은 양쪽 모두 **`value > 0 = adverse`** 가정.
- BB SHAP 는 log-odds 공간에서 이미 만족. surrogate 는 `adverse_contributions(target_scale=...)` 로 변환.

---

## 5. 재구성 단계 (순차, 각 단계가 독립적으로 ship 가능)

1. **스캐폴딩**
   - `src/decentra/explain/`, `src/decentra/experiments/case.py`, `cv.py`, `aggregate.py`, `src/decentra/tests/` 생성.
   - `pyproject.toml` 에 `[tool.pytest.ini_options] testpaths = ["src/decentra/tests"]` 추가.
2. **테스트 이동 (Option B)**
   - `git mv tests/test_*.py src/decentra/tests/`.
   - 공유 fixture 의 `conftest.py` 추가.
   - `pytest` 가 변경 없이 발견·통과하는지 확인.
3. **`local_scorecard_explainer.py` 승격**
   - `git mv tests/local_scorecard_explainer.py src/decentra/explain/local_scorecard.py`.
   - `PriorFeatureInfo` / `LocalExplainerState` 를 `src/decentra/explain/prior.py` 로 분리.
   - `PriorFetureInfo` deprecated alias 추가.
   - `_type` → `output` (deprecation alias).
   - 입력 검증(범위 중첩, 길이) 추가.
4. **`notebooks/executor.py` 승격**
   - `git mv notebooks/executor.py src/decentra/experiments/case.py`.
   - `case.py` 내부 import 를 `from decentra.surrogate import ...` 등으로 (`sys.path` 해킹 제거).
   - `default_surrogate_factories`, `run_case`, `CaseResult` export 유지.
   - 후방호환을 위해 `notebooks/executor.py` 를 `decentra.experiments` 재노출 + DeprecationWarning shim 으로 남김.
5. **`run_cv` 추가**
   - `_e3_cv_5fold.py` 의 CV / Wilcoxon / BH 로직을 `decentra.experiments.cv:run_cv` 와 `aggregate.py` 로 추출.
   - sparse-binary 필터 / median 임퓨테이션을 `run_cv` 의 `preprocess_fold` hook 로 plug-in 가능하게.
   - `CVResult` dataclass 추가.
6. **`_e*.py` 업데이트** — loop 인라인을 제거하고 `run_cv` / `run_case` 호출로. `notebooks/` 에 얇은 driver 로 유지.
7. **Metric 통일**
   - `decentra.metrics` 의 default re-export 를 named 로.
   - positional `attribution_fidelity` 의 docstring 에 deprecation note (symbol 은 유지).
   - `compute_sic_sc` 제거.
8. **Alias 정리**
   - `OptBinningSurrogate = BinningSurrogate` 를 import 시 `DeprecationWarning` 를 띄우는 deprecated re-export 로 전환.
9. **`Scorecard.to_dataframe(locale=...)`**
   - 현재 한글 컬럼 → `locale="ko"` (default).
   - `locale="en"` 컬럼 세트 추가: `"feature", "bin", "weight", "count", "target_count", "share", "target_rate", "reason_code", "reason_desc"`.
10. **문서**
    - `README.md` 가 `docs/claude/{SPEC, PRD, TDD}.md` 를 가리키고 end-to-end 예제 1개 보여주도록 업데이트.

각 단계 종료 시 `pytest` green.

---

## 6. 테스트 전략

### 6.1 유지되는 테스트 (이미 존재)

- `test_feature_calibrator.py` (4 tests).
- `test_shap_pdp.py` (5 tests).

오늘 통과하며, §5 의 모든 단계 후에도 통과해야 한다.

### 6.2 영역별 신규 테스트

**ScorecardModel**

- `test_scorecard_model_to_dict_roundtrip`
  `from_dict(to_dict(sm)).predict(X) == sm.predict(X)` 정확히 일치.
- `test_scorecard_model_predict_matches_surrogate_within_tolerance`
  `TreeSurrogate(max_depth=1)` 에서 `sm.predict(X)` vs `surr.predict(X)` 차이가 `tolerance` 이내.
- `test_scorecard_model_centered_contributions`
  `contributions(X_train).mean(axis=0) ≈ 0`.

**Scorecard**

- `test_scorecard_to_dataframe_ko_columns`
  default `locale="ko"` 가 현재 사용 중인 한글 컬럼명을 정확히 반환.
- `test_scorecard_to_dataframe_en_columns`
  `locale="en"` 이 문서화된 영어 컬럼 세트를 반환.
- `test_scorecard_reason_codes_assigned_by_abs_score_desc`
  P001 은 양수 bin 중 `|score|` 최대, N001 은 음수 bin 중 `|score|` 최대.

**Surrogate (대표 — family 당 1개)**

- `test_tree_surrogate_depth1_bin_boundary_alignment`
  LightGBM split threshold 와 정확히 같은 값이 `BinRule` 의 **왼쪽** bin 에 들어간다 (nextafter 정렬).
- `test_linear_surrogate_centered_contribs_sum_matches_predict`
  `surr.predict(X) ≈ surr.contributions(X).sum(axis=1) + intercept_term`.
- `test_binning_surrogate_woe_signs_respect_monotone_constraint`
  `monotone_constraints={"x_pos": 1}` 일 때 `x_pos` 의 fit 된 계수가 ≥ 0 (sign-flip 강제).
- `test_ebm_surrogate_is_additive_when_interactions_zero`.
- `test_shap_pdp_requires_base_model_or_shap`  (이미 존재).
- `test_sequential_priority_order_matches_priority_method`
  `priority_method="abs"` 에서 `surr.order_ == argsort(-abs(SHAP).mean(0))`.

**Calibration**

- `test_feature_calibrator_preserves_r2_under_scale_mismatch` (이미 존재).
- `test_bin_calibrator_lambda_zero_recovers_prediction_only_solution`
  `lam=0` 에서 새 bin score 가 centered target 의 OLS 해와 일치 (허용오차 안).
- `test_bin_calibrator_lambda_one_drives_attribution`
  `lam=1` 에서 합성 mismatched 데이터에 대해 `attribution_fidelity_named` 가 raw 대비 개선.

**Metrics**

- `test_named_attribution_align_zero_missing_policy`
  missing feature 가 0 으로 채워지고, coverage 가 올바르게 보고됨.
- `test_named_attribution_align_raise_missing_policy`
  컬럼 불일치 시 정보성 메시지와 함께 `ValueError`.
- `test_advtopk_named_no_rejects_returns_zero`
  빈 reject mask → 0.0 반환 (NaN 아님, crash 아님).
- `test_advtopk_named_sign_convention`
  log-odds 공간 attribution (>0=adverse) 양쪽 입력 결과 == score 공간(<0=adverse, 사전 flip) 결과.
- `test_interventional_fidelity_no_pairs_returns_nan_n_pairs_zero`
  less-adverse 이웃 bin 이 없는 경우 `{DA: nan, ..., n_pairs: 0}` 반환 (crash 없음).
- `test_median_intervention_fidelity_da_at_k_consistent_with_manual`
  tiny 구성 데이터에서 `DA@1` 이 수동 계산값과 일치.

**Experiments**

- `test_run_benchmark_returns_one_row_per_surrogate`
  `result.rows` 길이가 factory 수와 같음.
- `test_run_case_writes_expected_files`
  `run_case` 후 `out_dir/result_<tag>.json`, `.pkl` 존재.
- `test_run_cv_summary_columns_present`
  `summary_df` 에 `{dataset, surrogate, metric, mean, std, ci95, n}` 컬럼이 모든 metric 에 대해 존재.
- `test_run_cv_matches_e3_for_one_dataset_within_tolerance`
  고정 데이터셋 / seed / surrogate set 에서, `run_cv` 의 fold-level `AdvTop4` 가 저장된 expected 값과 `tolerance` 안에서 일치 (regression fixture).

**Local explanation**

- `test_local_scorecard_reasons_current_bin_marked_once`
  각 feature 별 reason 표에서 `is_real == "T"` 인 row 가 정확히 하나.
- `test_local_scorecard_scaled_score_zero_at_current_recode`
  현재 행의 recode 에서 scaled 가 정확히 0; 더 좋은 recode 는 양의 `scaled`.
- `test_local_scorecard_lift_no_duplicate_feature_in_combination`
  multi-feature 조합 안에서 같은 `feature` 값이 중복되지 않음.
- `test_local_scorecard_lift_changes_value_to_nearest_boundary`
  lift 후 row 값이 target bin 의 `min` 또는 `max` 중 원본 값에 가까운 쪽과 일치.
- `test_local_scorecard_grade_columns_only_when_model_has_get_grade`
  `get_grade` 있는 모델 → `From Grade` / `To Grade` 컬럼 존재. 없으면 두 컬럼이 *부재* (NaN 아님).
- `test_local_scorecard_from_scorecard_model_helper`
  `LocalScorecardExplainer.from_scorecard_model(sm)` 후 `.explain(row)` 가 동일한 `reasons_df` schema 반환.
- `test_prior_feature_info_alias_deprecation`
  `PriorFetureInfo` import 시 `DeprecationWarning`.

### 6.3 Regression fixtures

- 작은 seeded 데이터셋(`n=300`, `p=5`)이 `src/decentra/tests/fixtures/tiny.py` 에 있고 `conftest.py` 로 공유.
- "golden" expected-metrics JSON (`fixtures/golden_cv.json`) 이 tiny 데이터셋 + 고정 seed 에서 `run_cv` 가 만들어야 할 값을 기록. `test_run_cv_matches_golden` 가 fold 별 값을 `np.testing.assert_allclose(..., rtol=1e-6)` 으로 비교.

### 6.4 Snapshot 테스트 (선택, `Scorecard.to_dataframe` 용)

- 작은 scorecard 의 기대 DataFrame 을 미리 계산하여 `fixtures/scorecard_snapshot.pkl` 로 저장. row-by-row 비교.

---

## 7. 마이그레이션 / 호환성

- **옛 이름 → 새 이름** 한 minor 버전 동안 `DeprecationWarning` 와 함께 유지:

| Old | New |
|---|---|
| `OptBinningSurrogate` | `BinningSurrogate` |
| `compute_sic_sc` | `interventional_fidelity` (alias 없이 제거) |
| `PriorFetureInfo` | `PriorFeatureInfo` |
| `LocalScorecardExplainer.explain(..., _type=...)` | `output=...` |
| `from executor import run_case` (notebooks) | `from decentra.experiments import run_case` (shim 재노출 + warning) |

- **이동 중 수치 동작 변화 금지.** metric 공식 변경은 이동 *이후* 에만, 명시적 테스트와 함께.
- **인코딩 위생**: 파일을 만질 때 주석·docstring 의 mojibake 수정; 새로 만들지 않는다.

---

## 8. 위험과 완화

| 위험 | 완화 |
|---|---|
| `executor.py` 이동이 `.outputs/` 의 pickle 된 `CaseResult` 파일을 깨뜨림. | `CaseResult` dataclass 의 이름·필드 구성을 유지; unpickle 에 module path 가 필요하면 `notebooks/executor.py` 를 얇은 compat 모듈로 남김 (재노출). |
| `Scorecard.to_dataframe()` 의 한글 컬럼이 downstream 에서 load-bearing. | default 로 한글 유지 (`locale="ko"`), 영어는 opt-in. |
| 옵션 의존성(`optbinning`, `interpret`) 이 CI 를 flaky 하게 만듦. | 해당 테스트에 `pytest.importorskip`. |
| `run_cv` 가 다른 머신에서 약간 다른 수치를 만듦. | `assert_allclose(rtol=1e-6, atol=1e-8)`, 모든 난수 seed 고정. "_e3_cv_5fold.py 와 일치" 테스트는 `rtol=1e-3` (LightGBM 결정성 caveat). |
| `tests/` 의 `src/` 이동이 외부 CI 가정을 깨뜨림. | README, `pyproject.toml` 에 변경 명시. 사용자가 보수적 경로(Option A 만) 를 원하면 §5.2 를 생략; 나머지는 동일 적용. |
| `_e*.py` 가 현재 논문 실험의 source of truth. | 계속 동작 가능. §5.6 이후 `run_cv` 만 호출하지만, 참고용으로 frozen snapshot 하나를 커밋. |

---

## 9. Definition of Done

- §2 의 레이아웃 존재, `pytest src/decentra/tests` 통과.
- `notebooks/_e*.py` 에 `sys.path.insert` 0건 (driver 가 `decentra.experiments` 에서 import).
- `tests/local_scorecard_explainer.py` 제거; 대체본이 `src/decentra/explain/` 아래에서 unit-test 됨.
- tiny fixture 에서 `ScorecardModel.from_dict(sm.to_dict()).predict(X)` 가 `sm.predict(X)` 와 element-wise 동일.
- tiny fixture 에서 `run_cv` 가 `fixtures/golden_cv.json` 과 `rtol=1e-6` 안에서 일치.
- README 가 `docs/claude/{SPEC, PRD, TDD}.md` 를 가리키고 15줄 end-to-end 스니펫을 보여줌.
- 모든 deprecated alias 가 `DeprecationWarning` 를 발생시키고 CHANGELOG 에 기록됨.
