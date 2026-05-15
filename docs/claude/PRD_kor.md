# PRD — Decentra 재구성

> `decentra` 를 일관된 end-to-end 해석가능 surrogate 프레임워크로 재구성하기 위한
> Product Requirements Document. 사용자의 요청대로 `tests/` (그리고 거기 잘못
> 놓여있는 prototype 코드) 를 `src/` 로 통합하는 작업을 포함한다.
>
> 현재 상태 인벤토리 및 컨벤션은 [SPEC_kor.md](./SPEC_kor.md) 참고.

---

## 1. 제품 한 줄 요약

decentra 는 불투명한 신용평가 모델(LightGBM 등)을 **해석 가능한 surrogate
scorecard** 로 변환하고, 그 결과를 **네 가지 충실도 축**(예측, 거절사유 attribution,
개입, scorecard 배포)으로 평가하여 **연구 결과**와 **운영 산출물** 을 동시에 생성한다.

이 재구성은 현재 "노트북이 사실상 API 인 연구 코드베이스" 를 **canonical workflow 가
`src/decentra/` 안에 있는 라이브러리** 로 전환하며, 테스트를 패키지와 함께 두고,
public surface 를 일관되게 만든다.

---

## 2. 문제 정의

현재 상태:

- 모든 논문 실험이 사용하는 실제 진입점인 `notebooks/executor.py::run_case` 가 **패키지 밖**에 있고 `sys.path` 해킹에 의존한다.
- **row 레벨 local scorecard 설명** prototype 이 `tests/local_scorecard_explainer.py` 에 위치 — import 불가, 테스트 불가.
- positional 메트릭과 named 메트릭 두 가지 API 가 공존하며, 실험 스크립트마다 다른 것을 사용한다.
- P5 논문에서 가장 많이 쓰이는 단위인 **교차검증(CV)** 이 first-class 추상이 아니다. `_e3_cv_5fold.py`, `_e5_*`, `_e6_*` 에 흩어져 있다.
- `tests/` 가 `src/` 와 분리되어 있고, 그 안에 진짜 테스트와 비-테스트 prototype 코드가 섞여 있다.

목표 레이아웃:

- 사용자가 필요한 모든 것을 `from decentra import ...` 로 가져온다.
- 테스트가 패키지와 같이 있고 실제 public contract 를 검증한다.
- CV / benchmark / local-explanation 경로가 모두 first-class 라이브러리 API 다.

---

## 3. 사용자

- **신용평가 모델 개발자** — black-box 의 해석 가능한 대체 모델을 비교
- **모형검증·컴플라이언스** — adverse-action 사유 품질 감사
- **데이터 사이언티스트** — surrogate ablation 을 CV 로 돌리고 논문 표 추출
- **전략·심사 분석가** — `Scorecard.to_dataframe()` 와 row-level local explanation 소비
- **연구자 (P5 논문 라인)** — E1–E6 실험 재현

---

## 4. 목표 (Goals)

1. **핵심 워크플로우를 패키지 안으로 승격.** `notebooks/executor.py::run_case` 와 `_e*.py` 의 재사용 가능 부분을 `src/decentra/` 로 이동.
2. **CV 를 first-class API 로.** `decentra.experiments.run_cv` (또는 동등) 추가 — 논문 실험이 문서화된 함수 호출로 끝나야 한다.
3. **Local-scorecard prototype 승격.** `tests/local_scorecard_explainer.py` → `src/decentra/explain/local_scorecard.py` (이름·검증 정리).
4. **사용자 요청: tests 를 src 와 통합** — 채택 레이아웃은 §6 참고.
5. **Public surface 표준화.** canonical attribution metric API 하나, public 경계에서 adverse 부호 약속 하나, surrogate alias 정책 하나.
6. **수치 동작 보존.** 이동·재구성 과정에서 fold-level 허용오차 범위 안에서 메트릭 값이 변하지 않아야 한다.
7. **노트북 호환성 유지.** 노트북이 package API 를 호출하도록 바뀌되, 사용자가 모든 노트북을 다시 쓰지 않아도 되어야 한다.

---

## 5. Non-goals

- Teacher 모델 학습 파이프라인 대체 (LightGBM 학습은 사용자 영역).
- `ScorecardModel` 의 서빙·REST API 구축.
- 새로운 surrogate family 추가.
- `Scorecard` display 의 본격적 국제화 — 인코딩 이슈만 해결. 한글 컬럼은 기본 유지, 영어 컬럼은 옵션으로 추가.
- 생성된 scorecard 의 규제 인증.
- non-tabular 모델 지원.

---

## 6. tests 와 src 통합 (사용자 요청)

사용자가 명시적으로 "src 에 tests 통합" 을 요청했다. 두 가지 해석이 가능하며,
이 재구성은 **둘 다** 채택한다.

| 행동 | 해결 대상 |
|---|---|
| **A.** `tests/` 안의 비-테스트 prototype 코드(`local_scorecard_explainer.py`)를 `src/decentra/explain/` 으로 승격. | 사실 테스트가 아닌데 tests 폴더에 있어 import 불가. |
| **B.** `tests/` 폴더를 패키지 안으로 옮긴다: `src/decentra/tests/`. | `pyproject.toml` 의 `decentra*` glob 이 이미 잡으므로 추가 설정 없이 자동 포함. `pytest src/decentra/tests` 로 실행. |

**B 의 트레이드오프:**

| 장점 | 단점 |
|---|---|
| 테스트가 코드와 함께 패키지에 들어가서, `decentra` 를 import 한 사용자도 찾을 수 있다. | 휠 크기가 약간 증가. 외부에 tests 를 두는 라이브러리 관례와 어긋남. |
| `decentra*` 하나로 통일된 패키지 트리, 평행 top-level 디렉토리 없음. | 일부 CI / coverage 도구가 top-level `tests/` 를 기본 가정. |
| `pyproject.toml` 의 `decentra*` 가 이미 포함하므로 설정 변경 불필요. | 일부 팀은 "pip install 로 사용자가 받는 것에 테스트는 없는 게 낫다" 는 정책을 둠. |

사용자가 보수적 해석(A 만 — `tests/` 를 repo root 에 유지)을 원하면 실행 전에 확인이 필요하다.
**이 PRD 의 default 는 A + B (둘 다 적용)** 이다.

---

## 7. 핵심 사용자 스토리

1. **연구자** 로서, `decentra.experiments.run_cv(...)` 를 호출하면 오늘 `_e3_cv_5fold.py` 가 만드는 `cv_summary.csv` / `cv_wilcoxon_AT4.csv` 와 동일한 결과를 얻는다.
2. **모델러** 로서, `TreeSurrogate` 를 fit 하고 `to_scorecard_model(...)` 을 호출하면, 그 `ScorecardModel.predict(X_new)` 가 surrogate 의 예측을 문서화된 허용오차 안에서 재현한다.
3. **모형검증자** 로서, `attribution_fidelity_named(...)` 한 가지 일관된 name-aligned API 로 두 surrogate 의 adverse-reason 충실도를 비교한다.
4. **분석가** 로서, `LocalScorecardExplainer(model, prior).explain(row)` 를 호출해 행 단위 사유코드 표와 lift 추천 표를 받는다.
5. **CI 담당** 으로서, `pytest` 를 돌리면 옮겨진 local-explainer, scorecard round-trip, 메트릭 edge case (이름 불일치, reject 없음 등) 가 모두 커버된다.

---

## 8. 기능 요구사항

### 8.1 Surrogate API

모든 surrogate 는 다음을 **반드시** 구현:

```python
surr.fit(X, y_target, *, eval_set=None, sample_weight=None, **opts)
surr.predict(X)              # ndarray (n,)
surr.contributions(X)        # ndarray (n, p), centered
surr.adverse_contributions(X, target_scale="score"|"logit")  # DataFrame, adverse > 0
surr.transform(X)            # dict: predictions, contributions, ranking, adverse
surr.fit_transform(X, y, ...)
surr.predict_with_contributions(X)
surr.to_scorecard_model(X, y_binary=None, **bin_opts)
surr.feature_importances_    # property
surr.is_additive             # property
```

BB SHAP 을 소비하는 surrogate(`ShapPdpSurrogate`, `SequentialPrioritySurrogate`) 는 `base_model=` 또는 `shap_values=` 키워드를 받는다.

### 8.2 ScorecardModel API

```python
sm = surr.to_scorecard_model(X_train, y_binary=y_train)  # 이미 fitted
sm.fit(X, y_binary=None, build_display=False)            # idempotent
sm.predict(X) / sm.contributions(X) / sm.transform(X)
sm.scorecard(X, y_binary) → Scorecard
sm.to_dict() / ScorecardModel.from_dict(d)               # round-trip
```

`from_dict(to_dict)` 의 round-trip 결과 `predict(X)` 가 원본과 정확히 일치해야 한다.

### 8.3 Scorecard API

```python
sc = sm.scorecard(X, y_binary)        # 또는 Scorecard.from_scorecard_model(...)
sc.to_dataframe()                     # 사유코드 포함 display 표
```

`to_dataframe()` 은 안정적인 internal column(영어 키) + 선택적 localized 표시명을 갖는다. 현재 한글 컬럼은 기존 호출자를 위해 계속 동작하지만 더 이상 *유일한* 형태가 아니다.

### 8.4 Local explanation API (tests → src 승격)

```python
from decentra.explain import LocalScorecardExplainer, local_scorecard_explain

explainer = LocalScorecardExplainer(model, prior_feature_info)
reasons_df, lift_df = explainer.explain(row, output="all")  # "reasons" / "lift"
```

- `PriorFetureInfo` → `PriorFeatureInfo` (오타 수정), `PriorFetureInfo` 는 한 minor 버전 동안 deprecated alias 유지.
- `_type="rc"|"lift"|"all"` → `output="reasons"|"lift"|"all"`, `_type` 은 deprecated alias.
- Lift 출력의 조합은 같은 feature 를 중복 포함할 수 없다.
- model 에 `get_grade` 가 있으면 lift 출력에 `From Grade` / `To Grade` 컬럼이 추가되고, 없으면 해당 컬럼은 아예 존재하지 않는다 (NaN 컬럼 X).

### 8.5 Calibration API

`FeatureCalibrator`, `BinCalibrator` 의 현재 동작 유지. 입력·출력의 부호·스케일 약속을 docstring 으로 명시.

### 8.6 Metrics API

- canonical attribution 경로 = **name-aligned**. positional 경로는 `decentra.metrics.attribution_positional` (또는 현재 이름 + deprecation note) 로 유지하되, 권장 import 는 `from decentra.metrics import attribution_fidelity_named`.
- metric 경계의 adverse 부호 약속: **`value > 0 = adverse`**. 호출자는 사전에 정렬된 attribution 을 전달한다 (`surr.adverse_contributions` 사용, BB 쪽은 log-odds SHAP 그대로).
- `compute_sic_sc` 제거, `interventional_fidelity` 사용.

### 8.7 Benchmark / CV API

- `BenchmarkConfig`, `BenchmarkResult`, `run_benchmark` 는 형태 유지.
- `decentra.experiments.run_cv(datasets, surrogate_factories, splitter, ...)` 추가 — fold-level `BenchmarkResult` 와 집계된 `CVResult` (Wilcoxon / BH on 설정 가능한 target metric, default `AdvTop4`) 반환.
- `run_case` (현재 `notebooks/executor.py` 본체) → `decentra.experiments.run_case`, 반환 형태 동일.

### 8.8 노트북 호환성

- 모든 노트북은 계속 동작해야 한다. import 만 `from executor import run_case` → `from decentra.experiments import run_case` 로 바뀐다.
- 단순히 package API 를 orchestration 하는 helper (예: surrogate factory 리스트) 는 패키지로 이동. 노트북은 데이터셋별 glue 와 시각화만 보유.

---

## 9. 품질 요구사항

- **API 스타일** — sklearn 스타일: `fit / predict / transform / fit_transform`, 학습 상태는 `..._` 속성, 숨겨진 전역 상태 없음.
- **DataFrame 친화성** — 사용자가 만지는 모든 입출력은 pandas in / pandas out.
- **이름 우선 정렬** — feature name 이 canonical identifier, positional 은 fallback.
- **행동 테스트** — 테스트는 "그냥 돈다" 가 아니라 *출력값* 을 검증.
- **인코딩 위생** — 이동 과정에서 만지는 주석·docstring 의 mojibake 수정. 새로 만들지 않는다.
- **Deprecation 규율** — 옛 이름들(`OptBinningSurrogate`, `compute_sic_sc`, `PriorFetureInfo`, `_type`)은 한 minor 버전 동안 `DeprecationWarning` 후 제거.

---

## 10. 성공 기준

- 재구성 후 `pytest` 가 새 레이아웃에서 통과한다.
- 사용자가 surrogate fit, scorecard 생성, local explanation / lift 생성, CV 실험을 오로지 `decentra.*` public import 만으로 할 수 있다. `sys.path` 해킹 0건.
- 새 `run_cv` 가 만드는 `cv_summary.csv` 가 적어도 한 회귀 데이터셋에서 기존 `_e3_cv_5fold.py` 결과와 fold-level 수치 노이즈 안에서 일치한다.
- `ScorecardModel.from_dict(sm.to_dict()).predict(X) == sm.predict(X)` 정확히.
- `tests/local_scorecard_explainer.py` 가 production 코드로서는 사라지고, 그 기능이 `src/decentra/explain/` 아래에서 unit-test 된다.
- README (또는 `docs/claude/`) 가 SPEC / PRD / TDD 를 링크하고, end-to-end 스니펫 하나를 보여준다.

---

## 11. Open questions

| # | 질문 | 답변 없을 때 기본값 |
|---|---|---|
| 1 | `tests/` 를 `src/decentra/tests/` 로 옮기는가(§6 B), 아니면 repo root 유지(A 만)? | A + B (`src` 안으로 이동). |
| 2 | surrogate 학습 target 의 canonical 은 `bb_score` 인가 `bb_logit` 인가? | `bb_score` (현재 `executor.run_case` 처럼). |
| 3 | metric 경계의 adverse 부호 약속을 강제(원시 SHAP 입력 시 raise)할 것인가? | 강제하지 않고, 두 가지 입력을 받되 기대를 문서화. |
| 4 | deprecated alias 의 수명? | 한 minor 버전 (0.1 → 0.2 유지, 0.3 제거). |
| 5 | `Scorecard.to_dataframe()` 기본 컬럼은 영어 + `locale="ko"` 스위치? 아니면 한글 우선? | 한글 우선, 영어는 플래그. |
| 6 | `LocalScorecardExplainer` 가 `ScorecardModel` 을 native 로 이해할 것인가 (사용자가 custom `TempModel` 대신 fitted scorecard 전달)? | 예 — `from_scorecard_model(sm)` helper 제공. |
| 7 | `_e*.py` 의 종착지? `decentra.experiments.eN` 으로 promote? 아니면 `run_cv` 만 호출하는 얇은 driver 로 `notebooks/` 에 유지? | 얇은 driver 는 `notebooks/`, 재사용 가능 부분은 `decentra.experiments`. |
