# PRD: Decentra 소스 재구성

## 1. 제품 의도

Decentra는 블랙박스 신용위험 모델을 해석 가능하고 의사결정에 사용할 수 있는 스코어카드로 변환하기 위한 Python 라이브러리입니다.

이 프로젝트는 다음 질문에 답하기 위해 만들어진 것으로 보입니다.

> 블랙박스 모델의 예측력을 유지하면서도 스코어카드식 설명, adverse-action 사유, 실행 가능한 개선 방향을 제공할 수 있는가?

이는 범용 XAI 도구가 아닙니다. 소스는 부도위험, score/logit 출력, 거절 고객, adverse feature, reason code, scorecard bin, calibration, interventional fidelity 같은 신용평가 개념을 중심으로 구성되어 있습니다.

## 2. 문제

LightGBM 같은 블랙박스 모델은 신용위험 예측에서 성능이 좋을 수 있지만, 업무 활용에는 다음이 필요합니다.

- 투명한 피처 단위 설명
- 안정적인 스코어카드 산출물
- adverse-action reason code
- 설명이 블랙박스 모델과 일치한다는 검증
- 제안된 피처 변경이 실제 블랙박스 결과를 개선한다는 검증

기존 설명 기법은 feature importance를 보여줄 수 있지만, 배포 가능한 스코어카드를 자동 생성하거나 설명의 실행 가능성을 평가하지는 않습니다.

## 3. 사용자

- challenger 또는 surrogate 모델을 만드는 신용위험 데이터 사이언티스트
- 설명 충실도를 검토하는 모델 리스크 담당자
- 스코어카드 표와 reason code가 필요한 전략 분석가
- surrogate 모델링 방법을 비교하는 연구자

## 4. 목표

- 여러 surrogate 모델 계열에 일관된 API를 제공합니다.
- 학습된 surrogate를 표준화된 scorecard model로 변환합니다.
- 전역 scorecard와 로컬 row-level explanation을 생성합니다.
- prediction, attribution, intervention metric으로 surrogate를 비교합니다.
- `tests`에 있는 prototype production logic을 `src`로 이동합니다.
- `tests`는 패키지 동작의 실행 가능한 명세로 유지합니다.

## 5. 비목표

- 완전한 model-serving 플랫폼 구축
- 상위 블랙박스 모델 학습 파이프라인 대체
- 규제 준수 자체를 보장
- 이번 단계에서 UI 또는 리포트 생성 제품 구축
- 임의의 비정형 모델 타입 지원

## 6. 현재 기능

현재 소스에는 다음 구성요소가 있습니다.

- `decentra.surrogate`
  - `TreeSurrogate`
  - `LinearSurrogate`
  - `BinningSurrogate`
  - `EBMSurrogate`
  - `ShapPdpSurrogate`
  - `SequentialPrioritySurrogate`
- `decentra.scorecard_model`
  - 배포 가능한 additive scorecard 표현
- `decentra.scorecard`
  - bin, count, target rate, reason code가 포함된 표시용 scorecard
- `decentra.calibration`
  - 블랙박스 SHAP에 대한 feature/bin 보정
- `decentra.metrics`
  - prediction, attribution, named attribution, interventional fidelity
- `decentra.experiments`
  - benchmark orchestration
- `tests/local_scorecard_explainer.py`
  - `src`로 이동해야 하는 local scorecard explanation prototype

## 7. 사용자 스토리

1. 데이터 사이언티스트로서 블랙박스 모델의 score 또는 logit 출력에 surrogate를 맞출 수 있다.
2. 리뷰어로서 surrogate 예측과 블랙박스 의사결정을 비교할 수 있다.
3. 리뷰어로서 surrogate adverse reason과 블랙박스 SHAP reason을 비교할 수 있다.
4. 전략 분석가로서 feature bin, bin score, target rate, reason code가 포함된 scorecard 표를 내보낼 수 있다.
5. 케이스 리뷰어로서 한 신청자 row를 설명하고 어떤 feature-bin 변경이 score를 개선하는지 확인할 수 있다.
6. 연구자로서 여러 fold와 dataset에서 surrogate 방법을 benchmark할 수 있다.

## 8. 기능 요구사항

### Surrogates

- 각 surrogate는 `fit`, `predict`, `contributions`, `transform`, `predict_with_contributions`를 제공해야 합니다.
- 가능한 경우 feature-name 순서를 보존해야 합니다.
- contribution ranking과 adverse feature 추출을 지원해야 합니다.
- additive surrogate는 `ScorecardModel` 변환을 지원해야 합니다.

### Scorecards

- `ScorecardModel`은 `base_score + sum(feature_bin_score)`를 표현해야 합니다.
- scorecard bin은 결정적인 boundary 동작을 가져야 합니다.
- scorecard display는 다음을 포함해야 합니다.
  - feature name
  - bin range
  - bin score
  - sample count
  - target count
  - target rate
  - reason code

### Local Explanation

- local scorecard explanation은 `tests/local_scorecard_explainer.py`에서 `src/decentra/explain`으로 이동해야 합니다.
- 다음을 생성해야 합니다.
  - row-level current recode/bin table
  - reason-code ranking
  - score-lift candidates
  - 모델이 `get_grade`를 제공할 때 optional grade transition

### Metrics

- prediction fidelity는 teacher와 surrogate 출력을 비교해야 합니다.
- attribution fidelity는 top-k와 adverse top-k reason을 비교해야 합니다.
- named attribution metric은 위치가 아니라 feature name 기준으로 정렬해야 합니다.
- interventional fidelity는 제안된 변경이 teacher output을 개선하는지 검증해야 합니다.

## 9. 품질 요구사항

- 공개 API는 단순하고 sklearn-like해야 합니다.
- 테스트는 smoke 실행뿐 아니라 동작과 data contract를 검증해야 합니다.
- optional dependency는 가능한 한 optional로 유지해야 합니다.
- 한글 업무 라벨은 encoding-safe하고 문서화되어야 합니다.
- 연구 노트북은 중복 core logic을 갖기보다 package API를 호출해야 합니다.

## 10. 성공 기준

- `tests/local_scorecard_explainer.py`가 더 이상 production logic을 포함하지 않습니다.
- `src/decentra/explain/local_scorecard.py`가 local explanation API를 제공합니다.
- 재구성 후 기존 테스트가 통과합니다.
- 새 테스트가 local explanation, scorecard conversion, named attribution alignment를 검증합니다.
- `docs/codex/PRD.md`와 `docs/codex/TDD.md`가 재구성 기준 문서로 사용됩니다.

## 11. 열린 질문

- canonical surrogate target은 logit scale이어야 하는가, credit-score scale이어야 하는가?
- public adverse contribution은 항상 `value > 0 = adverse`로 통일해야 하는가?
- reason-code ranking은 signed score, absolute score, relative score, business priority 중 무엇을 기준으로 해야 하는가?
- local lift recommendation은 기본적으로 multi-feature combination을 허용해야 하는가?
- 어떤 notebook experiment를 정식 experiment module로 승격해야 하는가?

