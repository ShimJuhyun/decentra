# 소스 목적: Decentra

## 1. 이 코드베이스의 목적

Decentra는 블랙박스 신용위험 모델을 해석 가능한 대체 모델과 스코어카드로 변환하기 위한 연구 지향 Python 패키지입니다.

이 코드는 일반적인 설명가능 AI 도구라기보다, 신용평가 업무에서 자주 충돌하는 세 가지 요구를 연결하려는 목적에 가깝습니다.

- 블랙박스 모델의 예측 성능
- 업무와 검토 절차에 필요한 스코어카드 형태의 투명성
- 설명과 개선 제안이 원래 모델에 충실하다는 근거

## 2. 추정되는 최초 개발 동기

작성자는 LightGBM과 같은 신용평가 모델을 더 단순하고 감사 가능한 구조로 근사하면서도, 운영상 중요한 요소를 유지할 수 있는지 검증하려 한 것으로 보입니다.

- 신청자 점수 또는 부도위험 순위
- 피처 단위의 불리한 사유
- 사유코드 순서
- 구간 기반 스코어카드 표
- 거절 고객에 대한 개선 방향
- 교차검증 또는 부트스트랩에서의 구간과 점수 안정성

노트북과 실험 모듈은 반복적인 연구 흐름을 보여줍니다. 교사 모델을 학습하고, 여러 surrogate를 맞추고, 평가하고, attribution을 보정하며, 스코어카드 변형을 비교하는 흐름입니다.

## 3. 주요 개념

### 교사 모델

교사 모델은 원래의 블랙박스 모델입니다. 보통 LightGBM 같은 트리 모델이며 확률, logit, score를 출력합니다.

### Surrogate 모델

Surrogate는 교사 모델의 출력을 모방하도록 학습한 더 단순한 모델입니다. Decentra는 tree, linear, binning, EBM, SHAP-PDP, sequential-priority 계열을 포함합니다.

### Contributions

Contributions는 예측을 설명하는 피처별 가산 값입니다. 피처 순위, adverse reason, 스코어카드 생성에 사용됩니다.

### Scorecard

Scorecard는 모델을 구간 규칙으로 표현한 것입니다.

```text
prediction = base_score + sum(feature_bin_score)
```

### Reason Codes

Reason code는 점수를 설명하는 피처-구간 효과를 순위화한 코드입니다. 유리한 요인과 불리한 요인을 구분하기 위한 코드 체계가 사용됩니다.

### Interventional Fidelity

Interventional fidelity는 surrogate가 제안한 피처 변경이 실제 교사 모델의 결과를 개선하는지 확인합니다.

## 4. 현재 코드 구조

재사용 가능한 패키지 코드는 `src/decentra` 아래에 있습니다.

주요 모듈:

- `surrogate`: surrogate 구현체
- `scorecard_model`: 배포 가능한 scorecard 모델
- `scorecard`: 표시용 scorecard 표
- `calibration`: 블랙박스 SHAP에 대한 feature/bin 보정
- `metrics`: prediction, attribution, named attribution, intervention metric
- `experiments`: benchmark 실행 구조

현재 `tests` 디렉터리에는 실제 테스트와 local scorecard explainer 프로토타입이 함께 있습니다. 이 프로토타입은 `src`로 이동해야 합니다.

## 5. 재구성 시 변경해야 할 점

- `tests`의 local explanation 코드를 `src/decentra/explain`으로 이동합니다.
- `tests`는 동작 검증에 집중합니다.
- scorecard와 explanation API를 명확하고 안정적으로 만듭니다.
- 공개 출력에 영향을 주는 깨진 한글 라벨과 주석을 복구합니다.
- 노트북 코드는 재사용 가능한 로직일 때만 패키지 코드로 승격합니다.

## 6. 한 문장 요약

Decentra는 블랙박스 신용위험 의사결정을 설명 가능하고, 스코어카드로 표현 가능하며, 원래 모델에 대한 충실도를 검증 가능하게 만들기 위한 코드베이스입니다.

