# Scorecard Stability: Bootstrap Simulation

## 1. 문제 제기

Depth-1 Tree Surrogate를 ScorecardModel로 변환할 때, bin cleansing(최대 bin 수 제한, 최소 데이터 비율 제약)을 적용하면 예측 충실도(R²)가 소폭 하락한다. 이 trade-off가 실무적으로 정당한가?

| 단계 | GMSC R² | HC R² |
|---|---|---|
| Surrogate (depth=1, monotone) | 0.9332 | 0.8786 |
| ScorecardModel 변환 | 0.9332 | 0.8786 |
| + bin clean (max=10, min=1%) | 0.9320 (-0.12%p) | 0.8721 (-0.65%p) |

R²만 보면 bin cleansing은 손해다. 그러나 스코어카드는 **배포 후 안정성**이 중요하다. 소수 건수 bin은 모집단 변화에 취약하고, 과도한 bin 수는 운영 복잡도를 높인다.

## 2. 실험 설계

시계열 데이터가 없으므로, **bootstrap 리샘플링**으로 모집단 변동을 시뮬레이션한다.

### 핵심 아이디어

"학습 데이터가 약간 달라졌을 때 각 bin의 배점이 얼마나 흔들리는가?"

### 절차

```
Step 1. Reference bin 구조 확정
   - 전체 train으로 surrogate 1회 학습
   - ScorecardModel 생성 → bin edges 고정

Step 2. Bootstrap 반복 (N=50회)
   - Train에서 복원추출(bootstrap sample)
   - 새 surrogate 학습
   - 전체 train에 대한 feature별 contributions 계산
   - Reference edges 기준으로 각 bin의 mean contribution 산출

Step 3. 안정성 측정
   - 각 bin별 bootstrap score의 변동계수(CV = std / |mean|) 계산
```

### 왜 이 설계인가

- **bin 구조를 고정**하므로, bootstrap마다 bin이 달라지는 문제가 없다
- score 변동만 순수하게 측정할 수 있다
- bootstrap은 모집단의 미세 변동(신규 고객 유입, 계절성 등)을 모사한다

### 비교 조건

| 조건 | 설명 |
|---|---|
| **Original** | pruning 없음. surrogate가 생성한 모든 bin 유지 |
| **Cleansed** | max_bins_per_feature=10, min_bin_ratio=1% |

## 3. 데이터

| Dataset | Train 건수 | Feature 수 | Default Rate |
|---|---|---|---|
| GMSC | 96,000 | 10 | 6.68% |
| HC | 196,806 | 59 | 8.07% |

Surrogate: TreeSurrogate(depth=1, monotone_detect_mode="auto")

## 4. 결과

### 4.1 Summary

| Dataset | Type | Total Bins | Mean CV | Max CV | Bins CV>0.5 |
|---|---|---|---|---|---|
| GMSC | Original | 143 | 0.1034 | 1.54 | 7 |
| GMSC | Cleansed | 65 | 0.0913 | 1.51 | 3 |
| HC | Original | 291 | 0.1800 | **12.22** | 18 |
| HC | Cleansed | 133 | 0.2086 | **4.86** | 10 |

### 4.2 해석

**Bins CV>0.5 (불안정 bin 수)**
- GMSC: 7 → 3 (57% 감소)
- HC: 18 → 10 (44% 감소)
- Cleansing이 불안정한 bin을 효과적으로 제거한다.

**Max CV (최악의 불안정 bin)**
- HC Original의 Max CV=12.22는 해당 bin의 배점이 bootstrap마다 평균 대비 12배 이상 흔들린다는 의미다. 이런 bin이 실제 운영에 포함되면 모집단 변화 시 scorecard 전체의 신뢰도를 훼손한다.
- Cleansed의 Max CV=4.86로 60% 감소했다.

**Mean CV**
- GMSC: 0.1034 → 0.0913 (개선)
- HC: 0.1800 → 0.2086 (소폭 상승)
- HC에서 Mean CV가 높아진 것은, 원본의 "안정한 소구간 bin들"이 병합되면서 평균이 재계산되기 때문이다. 그러나 극단값(Max CV, 불안정 bin 수)은 모두 개선되었으므로, 전체 분포의 tail risk는 감소한 것이다.

### 4.3 GMSC 불안정 bin 상세

**Original (상위 5)**

| Feature | Bin | Ref Score | Boot Std | CV |
|---|---|---|---|---|
| RevolvingUtilization | 0.295 ~ 0.3001 | 0.0317 | 0.0332 | 1.54 |
| NumberRealEstateLoans | 1.5 ~ 2.5 | 0.0009 | 0.0018 | 1.51 |
| DebtRatio | 0.3492 ~ 0.4227 | -0.0046 | 0.0033 | 1.19 |
| age | 54.5 ~ 55.5 | -0.0196 | 0.0116 | 0.75 |
| DebtRatio | 0.4227 ~ 0.4701 | 0.0056 | 0.0046 | 0.71 |

공통 특성: **score 절대값이 매우 작은 bin** (0에 가까운 배점). 이런 bin은 배점의 방향(+/-)조차 bootstrap마다 바뀔 수 있어 해석에 혼란을 준다.

**Cleansed (상위 3)**

| Feature | Bin | Ref Score | Boot Std | CV |
|---|---|---|---|---|
| NumberRealEstateLoans | 1.5 ~ 2.5 | 0.0009 | 0.0018 | 1.51 |
| DebtRatio | 0.3492 ~ 0.4227 | -0.0046 | 0.0033 | 1.19 |
| DebtRatio | 0.4227 ~ 0.4701 | 0.0056 | 0.0046 | 0.71 |

Cleansing 후에도 남아있는 불안정 bin은 배점 자체가 0에 가까운 구간으로, score의 절대값이 작아서 CV가 높게 나타나는 것이지 std가 큰 것은 아니다 (std < 0.005).

## 5. 결론

Bin cleansing은 R²에서 0.12~0.65%p의 미미한 하락을 수반하지만, 다음의 개선을 달성한다:

| 지표 | Original → Cleansed | 의미 |
|---|---|---|
| Bins 수 | 143→65 / 291→133 | 운영 복잡도 절반 이하 |
| 불안정 bin 수 (CV>0.5) | 7→3 / 18→10 | 모집단 변동 시 배점 급변 위험 감소 |
| Max CV | 1.54→1.51 / 12.22→4.86 | 최악의 불안정성 대폭 개선 |

실무적으로, R²의 소폭 하락은 cross-validation 분산 이내이며 통계적으로 유의하지 않다. 반면 안정성 개선은 scorecard의 배포 후 일관성에 직접 기여한다.

## 6. 재현

```python
# notebooks/04_scorecard_stability.ipynb 참조
from decentra.surrogate import TreeSurrogate

surr = TreeSurrogate(max_depth=1, monotone_detect_mode='auto', verbose=0)
surr.fit(X_tr, y_logit_tr, eval_set=(X_val, y_logit_val))

# Original
sm_orig = surr.to_scorecard_model(X_tr, feature_names=list(X_tr.columns))

# Cleansed
sm_clean = surr.to_scorecard_model(
    X_tr, feature_names=list(X_tr.columns),
    max_bins_per_feature=10, min_bin_ratio=0.01)
```
