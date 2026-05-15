# Decentra — Specification

> 본 문서는 현재 `src/decentra/`, `notebooks/`, `tests/` 의 코드를 읽고
> **이 프로젝트가 무엇을 하는 코드인지** 명세서 형태로 정리한 것이다.
> 이후 PRD / TDD 의 기준선이 된다.

---

## 1. 한 줄 요약

> **블랙박스 신용평가 모델(LightGBM 등)을 "해석 가능한 surrogate" 로 모사하고,
> 모사 결과를 운영 가능한 scorecard 로 변환한 뒤, 그 충실도를
> 세 축(예측 / 거절사유 / 개입)으로 비교 평가하는 연구·운영 통합 프레임워크.**

- 학술적 산출물: P5 *"LightGBM Scorecard based on SHAP Values"* 계열 실험 결과
- 운영 산출물: `ScorecardModel`(배포용) + `Scorecard`(평가 항목 / 구간 / 가중치 / 사유코드 표)

---

## 2. 문제 정의

신용평가 black-box 모델은 예측력은 높지만,
- 규제(Adverse Action Notice, 모형검증) 요건을 만족하기 어렵고,
- 운영 시스템(심사역·고객 통지문)이 이해 가능한 *score / 사유코드* 형식이 아니다.

따라서 다음을 모두 만족하는 surrogate 가 필요하다.

| 충실도 축 | 질문 |
|---|---|
| Predictive fidelity | BB 의 logit/score 를 잘 흉내내는가? |
| Attribution fidelity | BB 의 SHAP 패턴, 특히 거절고객의 adverse feature 를 잘 흉내내는가? |
| Interventional fidelity | "이 값을 바꾸면 점수가 어떻게 바뀌는가" 를 BB 와 같은 방향으로 답하는가? |
| Deployment fidelity | 위 셋을 만족하면서 scorecard 표(구간/가중치/사유코드) 로 표현 가능한가? |

decentra 는 위 네 축을 정의·측정·비교·산출한다.

---

## 3. 데이터 흐름 한눈에

```
원본 (X, y_binary)
        │
        ▼
[Black-box Teacher]   (외부 LightGBM 등)
   predict_proba → bb_prob
   TreeSHAP       → bb_shap     (log-odds 공간, >0 = adverse)
   logit_to_score → bb_score    (score 공간, 높을수록 우량)
        │
        ▼
[Surrogate]   y_target = bb_score 또는 bb_logit
   .fit(X, y_target [, eval_set, base_model, shap_values])
   .predict(X)                → surr_pred
   .contributions(X)          → 중심화된 feature contribution (n×p)
   .adverse_contributions(X)  → adverse>0 으로 부호 정렬된 DataFrame
        │
        ▼
[Calibration  (선택)]
   FeatureCalibrator  — feature 별 α 보정 (magnitude_preserving 옵션)
   BinCalibrator      — bin score 자체를 (1−λ)·L_pred + λ·L_attr + γ·∥s−s_old∥²
        │
        ▼
[Metrics]
   prediction_fidelity                → {R², Agree, Spearman}
   attribution_fidelity / *_named     → {Top-k, AdvTop-k, AdvFull_R/J, coverage}
   interventional_fidelity            → {DA, IR, Spearman_ρ, n_pairs}
   median_intervention_fidelity       → {DA@k, mean_delta@k, …}
        │
        ▼
[Scorecard 산출]
   surr.to_scorecard_model(X, y_binary)         → ScorecardModel
       ├── base_score
       └── FeatureRule(name, index, [BinRule(lower, upper, score), …])
   ScorecardModel.scorecard(X, y_binary)        → Scorecard
       └── to_dataframe()  →  평가 항목 / 구간 / 가중치 / 사유코드 (P###, N###)
```

---

## 4. 모듈 인벤토리

### `src/decentra/`

| 경로 | 역할 |
|---|---|
| `__init__.py` | 공개 API: `Scorecard`, `ScorecardModel`, `TrainingStats`, `FeatureStats` |
| `_utils.py` | `information_value`, `logit`, `sigmoid`, `transform_logit_to_score` (PDO/anchor) |
| `stats.py` | `FeatureStats` / `TrainingStats` (학습데이터 분포 저장 - effort 정규화, 모니터링용) |
| `scorecard_model.py` | `BinRule`, `FeatureRule`, `ScorecardModel` (배포 표현 + fit/transform/predict) |
| `scorecard.py` | `Scorecard` (display 전용 — 사유코드 P###/N###, target rate, 구성비) |
| `surrogate/base.py` | `BaseSurrogate` 추상클래스. monotone 자동 감지, bin pruning(MSE/Chi²/score_diff), ScorecardModel 변환 공통 로직 |
| `surrogate/tree.py` | `TreeSurrogate` — depth-1 → exact additive SHAP, depth>1 → SHAP. LightGBM `<=` ↔ `BinRule.contains` 의 nextafter alignment 포함 |
| `surrogate/linear.py` | `LinearSurrogate` (OLS/Ridge/Lasso/EN, sign-flip 기반 monotone), `BinningSurrogate`(OptBinning+WoE/Dummy+선형), alias `OptBinningSurrogate` |
| `surrogate/ebm.py` | `EBMSurrogate` (InterpretML ExplainableBoostingRegressor, GAM/GA²M) |
| `surrogate/shap_pdp.py` | `ShapPdpSurrogate` — Choi & Cha (2026) D5. y_logit 회귀 X, BB SHAP 를 PDP 로 binning → monotone LGBM smoother |
| `surrogate/sequential.py` | `SequentialPrioritySurrogate` — feature priority(`abs` / `signed_rejected`) 순으로 stage-wise depth-1 LGBM (frozen / cumulative) |
| `calibration/feature.py` | `FeatureCalibrator` — α 보정, magnitude_preserving 옵션(R² 붕괴 방지), sign_align 옵션 |
| `calibration/bin.py` | `BinCalibrator` — bin score 를 `(1-λ)·L_pred + λ·L_attr + γ·∥s-s_old∥²` 으로 재학습 |
| `metrics/prediction.py` | `prediction_fidelity` → {R², Agree, Spearman} |
| `metrics/attribution.py` | positional: `topk`, `advtopk`, `advfull`, `attribution_fidelity`, `random_baseline_advtopk` |
| `metrics/named.py` | name-aligned: `align_attributions`(AlignmentInfo, missing="zero/drop/raise"), `*_named`, `attribution_fidelity_named` |
| `metrics/interventional.py` | `extract_bin_structure`, `interventional_fidelity` (DA/IR/ρ), `median_intervention_fidelity` (median 치환), legacy alias `compute_sic_sc` |
| `experiments/benchmark.py` | `BenchmarkConfig`, `BenchmarkResult`, `run_benchmark` — surrogate factory dict × 한 split 에 대해 전 메트릭 한 번에 |

### `tests/`

| 경로 | 역할 |
|---|---|
| `test_feature_calibrator.py` | FeatureCalibrator: scale mismatch 하에서 R² 보존, magnitude 보존, deprecation, shape match |
| `test_shap_pdp.py` | ShapPdpSurrogate: base_model / shap_values fit, centered contribution, scorecard 변환, 입력 검증 |
| `local_scorecard_explainer.py` | **테스트가 아님** — local row-level scorecard 설명 + lift 추천 prototype (2025-05-29 작성). `LocalScorecardExplainer`, `PriorFetureInfo`(오타), `LocalExplainerState` |

### 패키지 밖에 있지만 실제로는 핵심 사용 인프라

| 경로 | 역할 |
|---|---|
| `notebooks/executor.py` | `run_case` — teacher + 한 split → bench + calibration + interventional + cutoff + scorecard 변환 결과를 묶어 저장. **실질적으로 decentra 의 main 진입점** |
| `notebooks/_e1`~`_e6.py` | CV / quantile ablation / SHAP-PDP / Sequential Priority 실험 스크립트 |
| `notebooks/_pilot_run.py` | 파일럿 스모크 |
| `notebooks/N*.ipynb`, `NB*.ipynb` | 발표 / 논문용 분석 노트북 |

---

## 5. 도메인 컨벤션 (잊기 쉬운 약속)

1. **부호 약속이 두 가지 공존**
   - BB SHAP: *log-odds* 공간, `>0` = adverse
   - Surrogate contrib: *score* 공간, `<0` = adverse
   - → metrics 는 `bb_sign=+1`, `surr_sign=-1` 을 받거나, `adverse_contributions(target_scale="score"|"logit")` 으로 통일해 `>0=adverse` 로 맞춘다.

2. **Reject 정의**: `bb_prob >= percentile(bb_prob, reject_percentile)` (기본 90%).

3. **Centering 약속**: surrogate `.contributions()` 는 항상 **중심화** (`raw - mean_contributions_`). `_raw_contributions()` 는 internal.

4. **Bin rule 약속**: `BinRule.contains` 는 `lower <= x < upper`, 마지막 bin 만 `upper=+inf`. `TreeSurrogate._get_feature_bins` 가 LightGBM `<=` convention 을 `nextafter` 로 정렬해 boundary 값을 잃지 않는다.

5. **Monotone 자동 감지**: Spearman + p-value<0.05. 사용자가 dict 로 일부 feature 만 고정하면 그 값이 우선, 나머지는 자동.

6. **ScorecardModel.fit 의 두 가지 일**:
   (a) `mean_contributions_` 와 `training_stats_` 캐시,
   (b) `build_display=True` & `y_binary` 면 display `Scorecard` 까지 빌드 (opt-in).

7. **Surrogate fit 의 입력**: `y_logit` 은 conceptual 이름이며, 실제로는 `bb_score` (score 공간) 또는 `bb_logit` (log-odds 공간) 둘 다 받는다. `target_scale` 은 `adverse_contributions` 의 부호 결정에만 영향.

---

## 6. 현재 구조의 문제점 (재구성 동기)

| # | 문제 | 영향 |
|---|---|---|
| 1 | **핵심 진입점이 src 밖** — `notebooks/executor.py::run_case` 가 사실상 main 인데 `sys.path.insert` 로 import. `_e*.py` 도 같음 | 재사용·테스트 불가, 외부 사용자가 import 못 함 |
| 2 | **tests/ 가 src 와 분리** + `tests/local_scorecard_explainer.py` 는 테스트가 아니라 prototype | 사용자 요청대로 src 통합 필요 |
| 3 | **메트릭 두 갈래 공존** — positional(`attribution.py`) ↔ named(`named.py`). `executor.py` 는 named, `_e4.py` 는 positional | 일관성 부족, 어느 쪽이 canonical 인지 모호 |
| 4 | `OptBinningSurrogate = BinningSurrogate` alias 만 살아있음 | deprecation 또는 통일 필요 |
| 5 | `compute_sic_sc` 등 legacy alias 잔존 | 정리 가능 |
| 6 | `Scorecard.to_dataframe()` 의 한글 컬럼 (`평가 항목`/`구간`/…) | 인코딩·국제화 이슈 가능 |
| 7 | `PriorFetureInfo` 오타 (Feature → Feture) | API 가독성 |
| 8 | `experiments/benchmark.py` 는 1-split 전용. 실제 CV / fold 로직은 `_e3_cv_5fold.py` 등에 흩어져 있음 | CV가 first-class 가 아님 |

---

## 7. 산출물 종착지

- **연구 결과**: `.outputs/e*_*/{cv_summary, cv_wilcoxon_AT4, fold_*/result_*}.{csv,json,pkl}` → 논문 표 / 그림
- **운영 산출물**: `ScorecardModel` (`to_dict` / `from_dict` 직렬화 가능) + `Scorecard.to_dataframe()` (현업 통지문 데이터)

---

## 8. "이 코드를 만든 목적" 종합

> **불투명한 신용평가 모델(LightGBM 등)의 출력을 다양한 해석가능 surrogate 로 모사하고,
> 그것을 운영용 scorecard 로 변환했을 때
> (1) 예측 충실도, (2) 거절 사유 충실도, (3) 개입 충실도, (4) 배포 안정성이
> 어떻게 trade-off 되는가** 를 systematic 하게 측정하는, **연구–운영 통합 비교 프레임워크.**
