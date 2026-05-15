# 명세서: Decentra 재구성

## 1. 시스템 개요

Decentra는 신용평가에서 model replacement와 explanation을 위한 파이프라인을 제공합니다.

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

재구성은 이 파이프라인을 유지하면서 package boundary를 명확히 해야 합니다.

## 2. 모듈 책임

### `decentra.surrogate`

teacher output을 근사하는 surrogate model을 담당합니다.

필수 동작:

- teacher output에 fit
- teacher-like score/logit 예측
- feature별 contribution 생성
- feature ranking과 adverse feature 제공
- additive form을 `ScorecardModel`로 변환

### `decentra.scorecard_model`

배포 가능한 scorecard 표현을 담당합니다.

필수 동작:

- `base_score`, `FeatureRule`, `BinRule` 보유
- additive bin score로 예측
- fit 이후 centered contribution 생성
- dict로 serialize/deserialize

### `decentra.scorecard`

표시 및 리포팅용 scorecard를 담당합니다.

필수 동작:

- `ScorecardModel`에서 생성
- bin count와 target rate 계산
- reason code 부여
- 안정적인 pandas `DataFrame` 반환

### `decentra.explain`

local explanation 기능을 위한 새 모듈입니다.

필수 동작:

- scorecard prior information 기준으로 한 row 설명
- 현재 recode/bin 식별
- recode별 candidate score 계산
- reason code ranking
- lift recommendation 생성

### `decentra.calibration`

black-box SHAP에 대한 surrogate contribution 보정을 담당합니다.

필수 동작:

- feature-level magnitude-preserving calibration
- optional sign alignment
- discrete contribution value에 대한 bin-level calibration

### `decentra.metrics`

모델 비교 metric을 담당합니다.

필수 동작:

- prediction fidelity
- attribution fidelity
- feature-name 기반 attribution alignment
- interventional fidelity

### `decentra.experiments`

재사용 가능한 benchmark orchestration을 담당합니다.

필수 동작:

- 설정된 surrogate factory 실행
- 공통 metric 계산
- output과 fitted model 수집
- benchmark result 저장

## 3. 공개 데이터 타입

### `BinRule`

필드:

- `lower: float`
- `upper: float`
- `score: float`

Boundary convention:

```python
lower <= x < upper
```

### `FeatureRule`

필드:

- `name: str`
- `index: int`
- `bins: list[BinRule]`

### `ScorecardModel`

필드:

- `base_score: float`
- `features: list[FeatureRule]`
- `mean_contributions_`
- `training_stats_`
- `is_fitted_`

### `PriorFeatureInfo`

필드:

- `name: str`
- `description: str`
- `ranges: list[tuple[float, float]]`
- `recodes: list[int]`
- `representative_values: list[float] | None`

## 4. 공개 메서드 계약

### Surrogate `fit`

```python
fit(X, y_teacher, *, eval_set=None, sample_weight=None, **kwargs) -> self
```

`y_teacher`는 teacher logit, score 또는 문서화된 다른 continuous target일 수 있습니다.

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

필수 key:

- `predictions`
- `contributions`
- `ranking`
- `adverse`

### Local Explanation

```python
LocalScorecardExplainer(model, scorecard_prior).explain(row, output="all")
```

허용되는 output:

- `reasons`
- `lift`
- `all`

## 5. Sign Convention

현재 코드는 score-scale과 logit-scale convention을 모두 사용합니다.

권장 public boundary:

- public adverse-contribution table에서는 `value > 0`이 adverse를 의미합니다.
- 내부 contribution sign은 model target에 따라 다를 수 있지만, metric 비교 전 변환되어야 합니다.

## 6. Scorecard 요구사항

- scorecard prediction은 `base_score + sum(bin_scores)`와 같아야 합니다.
- `to_dict`와 `from_dict`는 prediction behavior를 보존해야 합니다.
- boundary behavior는 결정적이고 테스트되어야 합니다.
- display output은 깨진 encoding에 의존하지 않아야 합니다.
- reason-code ranking rule은 문서화되어야 합니다.

## 7. Local Explanation 요구사항

- 한 row는 feature마다 정확히 하나의 current recode에 매핑되어야 합니다.
- current recode scaled score는 0이어야 합니다.
- candidate recode는 combination 생성이 아닌 경우 candidate feature만 변경해 score를 계산해야 합니다.
- lift recommendation은 from/to feature value와 from/to score를 포함해야 합니다.
- 모델에 `get_grade`가 있으면 lift output은 from/to grade를 포함해야 합니다.
- duplicate-feature combination은 필터링해야 합니다.

## 8. 테스트 요구사항

최소 테스트 그룹:

- feature calibration
- SHAP-PDP surrogate
- scorecard model prediction and serialization
- scorecard display schema
- local scorecard explanation
- named attribution alignment
- adverse sign convention

optional dependency test는 dependency가 없을 때 skip 가능해야 합니다.

## 9. 문서 요구사항

재구성 기준 문서는 다음입니다.

- `source-purpose.md`
- `SPEC.md`
- `PRD.md`
- `TDD.md`

각 영어 문서는 `_kor.md` 한국어 대응 문서를 가져야 합니다.

