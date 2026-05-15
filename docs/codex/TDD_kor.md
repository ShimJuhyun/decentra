# TDD: Decentra 소스 재구성

## 1. 목적

Decentra를 재구성하여 package code, prototype logic, tests, research notebooks의 경계를 명확히 합니다.

즉시 처리할 대상은 현재 `tests` 아래에 있는 유용한 로직을 `src`로 통합하는 것이며, 특히 local scorecard explainer prototype이 핵심입니다.

## 2. 현재 소스 해석

코드베이스에는 세 계층이 섞여 있습니다.

- Library layer: `src/decentra` 아래의 재사용 가능한 package module
- Research layer: `notebooks` 아래의 notebook과 experiment script
- Test/prototype layer: pytest test와 `tests/local_scorecard_explainer.py`

Library layer는 이미 surrogate modeling, scorecard conversion, calibration, fidelity metric 중심으로 구성되어 있습니다. 주요 구조적 문제는 local row-level scorecard explanation이 실제 product logic임에도 `tests` 아래 prototype으로 존재한다는 점입니다.

## 3. 목표 구조

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

## 4. 공개 API 제안

### Surrogate API

모든 surrogate는 다음 형식을 따라야 합니다.

```python
surr.fit(X_train, y_teacher, **kwargs)
pred = surr.predict(X)
contribs = surr.contributions(X)
result = surr.transform(X)
sm = surr.to_scorecard_model(X_train, y_binary=y_train)
```

`transform(X)`는 다음 key를 반환해야 합니다.

- `predictions`
- `contributions`
- `ranking`
- `adverse`

### Scorecard API

`ScorecardModel`은 배포 가능한 표현입니다.

```python
sm.predict(X)
sm.contributions(X)
sm.transform(X)
sm.scorecard(X, y_binary)
sm.to_dict()
ScorecardModel.from_dict(payload)
```

### Local Explanation API

새 모듈:

```python
from decentra.explain import LocalScorecardExplainer, PriorFeatureInfo
```

사용 예:

```python
features = [PriorFeatureInfo(...)]
explainer = LocalScorecardExplainer(model, features)
reasons, lifts = explainer.explain(row, output="all")
```

지원 출력:

- `output="reasons"`
- `output="lift"`
- `output="all"`

임시 하위 호환 alias:

- `PriorFetureInfo = PriorFeatureInfo`
- `_type="rc"`는 `output="reasons"`로 매핑
- `_type="lift"`는 `output="lift"`로 매핑
- `_type="all"`은 `output="all"`로 매핑

## 5. 데이터 계약

### `PriorFeatureInfo`

필드:

- `name: str`
- `description: str = ""`
- `ranges: list[tuple[float, float]]`
- `recodes: list[int]`
- `representative_values: list[float] | None = None`

검증:

- `ranges`와 `recodes` 길이가 같아야 합니다.
- `representative_values`가 제공되면 길이가 같아야 합니다.
- range는 정렬되어야 합니다.
- range는 겹치지 않아야 합니다.

### `LocalExplainerState`

필드:

- feature prior fields
- `scores`
- `score_mean`
- `relative_scores`
- `scaled_scores`
- `recode`

동작:

- `relative_scores = candidate_score - feature_mean_score`
- `scaled_scores = candidate_score - current_recode_score`
- 현재 recode의 scaled score는 반드시 `0`이어야 합니다.

### `BinRule`

현재 boundary convention:

```python
lower <= x < upper
```

LightGBM threshold 변환은 LightGBM의 `x <= threshold` 동작을 보존하도록 upper boundary를 조정해야 합니다.

## 6. 구현 계획

1. `src/decentra/explain/`을 생성합니다.
2. `tests/local_scorecard_explainer.py`를 `src/decentra/explain/local_scorecard.py`로 이동합니다.
3. `PriorFetureInfo`를 `PriorFeatureInfo`로 변경합니다.
4. 한 transition 기간 동안 compatibility alias를 유지합니다.
5. `_type`을 `output`으로 교체합니다.
6. dataclass validation을 추가합니다.
7. local explanation 동작 테스트를 추가합니다.
8. `Scorecard.to_dataframe()`의 encoding/syntax 문제를 수정합니다.
9. scorecard roundtrip 및 boundary 테스트를 추가합니다.
10. notebook 변경은 package restructuring과 분리합니다.

## 7. 테스트 계획

### 기존 테스트

유지:

- `test_feature_calibrator.py`
- `test_shap_pdp.py`

### 새 Local Explanation 테스트

- `test_prior_feature_info_validates_lengths`
- `test_local_reasons_marks_current_recode`
- `test_local_scaled_score_current_recode_is_zero`
- `test_local_lift_recomputes_score`
- `test_local_lift_filters_duplicate_feature_combinations`
- `test_local_lift_includes_grade_when_model_supports_get_grade`

### 새 Scorecard 테스트

- `test_scorecard_model_predict_roundtrip_dict`
- `test_scorecard_transform_requires_fit`
- `test_scorecard_bin_boundary_alignment`
- `test_scorecard_to_dataframe_schema`

### 새 Metrics 테스트

- `test_named_alignment_zero_fills_missing_features`
- `test_named_alignment_raises_on_missing_when_configured`
- `test_adverse_contributions_positive_means_adverse`

## 8. 마이그레이션 전략

- 먼저 동작 변경을 최소화하여 코드를 이동합니다.
- 의미 변경 전에 기존 동작을 테스트로 고정합니다.
- 오탈자 이름과 legacy 이름에 대한 alias를 유지합니다.
- sign convention은 광범위한 API 변경 전에 문서화합니다.
- notebook experiment script는 재사용 가능할 때만 `src`로 편입합니다.

## 9. 알려진 위험

- 일부 한글 주석과 라벨이 mojibake로 깨져 있습니다.
- `Scorecard.to_dataframe()`은 손상된 것으로 보이며 사용 전 수정이 필요합니다.
- `optbinning`, `interpret` 같은 optional dependency가 모든 테스트 환경에 없을 수 있습니다.
- SHAP과 LightGBM 테스트는 느릴 수 있으므로 fixture는 작게 유지해야 합니다.
- local lift combination은 feature와 bin 수가 많을 때 빠르게 커질 수 있습니다.

## 10. 완료 기준

- `docs/codex/`에는 의도한 영어 문서와 해당 `_kor.md` 한국어 문서가 있습니다.
- production local explanation code가 `src/decentra/explain` 아래에 있습니다.
- `tests/`에는 test, fixture, test utility만 있습니다.
- public import가 `decentra.explain`에서 동작합니다.
- 현재 테스트와 새 테스트가 통과합니다.
- README가 test file을 product code로 참조하지 않고 새 API를 안내할 수 있습니다.

