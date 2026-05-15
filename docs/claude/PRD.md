# PRD — Decentra Restructure

> Product Requirements Document for restructuring `decentra` into a coherent,
> deployable, end-to-end interpretable-surrogate framework, including the
> integration of `tests/` (and prototype code currently parked there) into `src/`.
>
> See [SPEC.md](./SPEC.md) for the current-state inventory and conventions.

---

## 1. Product summary

Decentra converts opaque credit-risk models (LightGBM and friends) into
interpretable surrogate scorecards, evaluates them along four fidelity axes
(prediction, adverse-reason attribution, intervention, scorecard deployment),
and produces both research metrics and operational scorecard artefacts.

This refactor turns the current "research codebase with notebooks-as-API" into
a library where the canonical workflow lives inside `src/decentra/`, tests are
co-located with the package, and the public surface is consistent.

---

## 2. Problem statement

Today:

- The actual entry point used by every paper experiment, `notebooks/executor.py::run_case`, is **outside** the package and depends on `sys.path` hacks.
- A prototype that does **local row-level scorecard explanation** lives at `tests/local_scorecard_explainer.py`, where it cannot be imported and is not tested.
- Two parallel metric APIs (positional vs named) coexist, and different experiment scripts pick different ones.
- Cross-validation, the most-used unit of work for the P5 paper, has no first-class abstraction — it lives in `_e3_cv_5fold.py`, `_e5_*`, `_e6_*`.
- `tests/` is separated from `src/` and contains a mix of real tests and non-test prototype code.

We want a layout where:

- A user can `from decentra import ...` everything they need.
- Tests sit next to the package and verify real public contracts.
- The CV / benchmark / local-explanation paths are all first-class library APIs.

---

## 3. Target users

- **Credit-risk modellers** comparing interpretable replacements for a black-box model.
- **Model-risk / compliance reviewers** auditing adverse-action reason quality.
- **Data scientists** running cross-validated surrogate ablations and exporting paper tables.
- **Strategy / underwriting analysts** consuming `Scorecard.to_dataframe()` and per-row local explanations.
- **Researchers (P5 paper line)** running E1–E6 experiments reproducibly.

---

## 4. Goals

1. **Promote the core workflow into the package.** Move `notebooks/executor.py::run_case` and the parts of `_e*.py` that are reusable into `src/decentra/`.
2. **Make CV a first-class API.** Add `decentra.experiments.run_cv` (or equivalent) so the paper experiments call a documented function, not a 100-line notebook script.
3. **Promote the local-scorecard prototype.** Move `tests/local_scorecard_explainer.py` into `src/decentra/explain/local_scorecard.py` with cleaned-up naming and validation.
4. **Integrate tests with the source tree** per the user's explicit request — see §6 for the chosen layout.
5. **Standardize the public surface.** One canonical attribution metric API, one adverse-sign convention at public boundaries, one canonical surrogate alias scheme.
6. **Preserve numerical behavior during the move.** Re-shuffling code must not change metric values within fold-level tolerances.
7. **Keep notebooks runnable.** Notebooks should now call package APIs rather than local helpers; the user does not want to rewrite every notebook.

---

## 5. Non-goals

- Replacing the teacher-model training pipeline (LightGBM training stays user-side).
- Building a serving / REST API for `ScorecardModel`.
- Adding new surrogate families.
- Internationalizing the `Scorecard` display table beyond fixing encoding issues. Korean labels can remain as the default; English labels are an additive option.
- Regulatory certification of the produced scorecards.
- Supporting non-tabular models.

---

## 6. Integration of `tests/` with `src/` (user-requested)

The user asked for tests to be integrated into `src/`. Two interpretations are
on the table; this refactor adopts **both**:

| Action | What it solves |
|---|---|
| **A.** Promote non-test prototype code out of `tests/` into `src/decentra/explain/` and `src/decentra/experiments/`. | `tests/local_scorecard_explainer.py` is not a test; it should live in the package. |
| **B.** Move the `tests/` folder under the package: `src/decentra/tests/`. | Tests ship next to the code they verify; the package `decentra*` glob in `pyproject.toml` already picks them up. They remain runnable with `pytest src/decentra/tests`. |

**Trade-offs (B):**

| Pros | Cons |
|---|---|
| Tests are bundled with the package; users importing `decentra` can find them. | Slightly larger wheel; non-standard for libraries that prefer keeping tests external. |
| One unified `decentra*` package tree, no parallel top-level dirs. | Some CI / coverage tools assume top-level `tests/`. |
| `pyproject.toml` already includes `decentra*`, so no config change needed. | Some teams prefer separating tests for "what users get on pip install" reasons. |

If the user prefers the more conservative interpretation (only **A**, keep
`tests/` at repo root), that should be confirmed before execution — `tests/`
under `src/` is the *default* in this PRD.

---

## 7. Core user stories

1. **As a researcher**, I run `decentra.experiments.run_cv(...)` and get the same `cv_summary.csv` / `cv_wilcoxon_AT4.csv` that `_e3_cv_5fold.py` produces today.
2. **As a credit-risk modeller**, I fit a `TreeSurrogate`, call `to_scorecard_model(...)`, and get a `ScorecardModel` whose `.predict(X_new)` matches the surrogate's predictions within a documented tolerance.
3. **As a reviewer**, I compare two surrogates' adverse-reason fidelity via `attribution_fidelity_named(...)` — one consistent name-aligned API, regardless of which surrogate is on which side.
4. **As an analyst**, I call `LocalScorecardExplainer(model, prior).explain(row)` and get a row-level reason-code table plus a lift-recommendation table.
5. **As a CI maintainer**, I run `pytest` and tests cover the moved local-explainer, the scorecard round-trip, and the metric edge cases (name mismatch, missing reject set, no rejected samples, etc.).

---

## 8. Functional requirements

### 8.1 Surrogate API

Every surrogate **must** implement:

```python
surr.fit(X, y_target, *, eval_set=None, sample_weight=None, **opts)
surr.predict(X)              # ndarray (n,)
surr.contributions(X)        # ndarray (n, p), centered
surr.adverse_contributions(X, target_scale="score"|"logit")  # DataFrame, adverse > 0
surr.transform(X)            # dict: predictions, contributions, ranking, adverse
surr.fit_transform(X, y, ...)
surr.predict_with_contributions(X)
surr.to_scorecard_model(X, y_binary=None, **bin_opts)  # ScorecardModel
surr.feature_importances_    # property (ndarray) where applicable
surr.is_additive             # property (bool)
```

Optional surrogates that consume BB SHAP (`ShapPdpSurrogate`, `SequentialPrioritySurrogate`) accept `base_model=` or `shap_values=` keyword args.

### 8.2 ScorecardModel API

```python
sm = surr.to_scorecard_model(X_train, y_binary=y_train)  # already fitted
sm.fit(X, y_binary=None, build_display=False)            # idempotent
sm.predict(X) / sm.contributions(X) / sm.transform(X)
sm.scorecard(X, y_binary) → Scorecard
sm.to_dict() / ScorecardModel.from_dict(d)               # round-trip
```

The round-trip `from_dict(to_dict)` must produce identical `predict(X)` output.

### 8.3 Scorecard API

```python
sc = sm.scorecard(X, y_binary)        # or Scorecard.from_scorecard_model(...)
sc.to_dataframe()                     # display table with reason codes
```

`to_dataframe()` returns a DataFrame with stable internal columns (English keys
plus optional localized display labels). The current Korean labels must keep
working for existing callers but no longer be the only option.

### 8.4 Local explanation API (promoted from `tests/`)

```python
from decentra.explain import LocalScorecardExplainer, local_scorecard_explain

explainer = LocalScorecardExplainer(model, prior_feature_info)
reasons_df, lift_df = explainer.explain(row, output="all")  # or "reasons" / "lift"
```

- `PriorFetureInfo` → `PriorFeatureInfo` (typo fix), with `PriorFetureInfo` kept as a deprecated alias for one minor version.
- `_type="rc"|"lift"|"all"` → `output="reasons"|"lift"|"all"`, with `_type` accepted as a deprecated alias.
- Lift output must not include duplicate features in a combination.
- If the model has `get_grade`, lift output includes `From Grade` / `To Grade`; otherwise those columns are absent (not NaN columns).

### 8.5 Calibration API

`FeatureCalibrator` and `BinCalibrator` keep their current behavior. Public sign / scale conventions for their *inputs* and *outputs* are documented.

### 8.6 Metrics API

- One canonical attribution path: **name-aligned**. The positional path is retained as `decentra.metrics.attribution_positional` (or kept under the current name with a deprecation note), but the canonical import is `from decentra.metrics import attribution_fidelity_named`.
- Public adverse-sign convention at the metric boundary: **`value > 0 = adverse`**. Callers are expected to pass already-aligned attributions (use `surr.adverse_contributions` and BB log-odds SHAP directly).
- `compute_sic_sc` removed in favor of `interventional_fidelity`.

### 8.7 Benchmark / CV API

- `BenchmarkConfig`, `BenchmarkResult`, `run_benchmark` keep their shape.
- Add `decentra.experiments.run_cv(datasets, surrogate_factories, splitter, ...)` that produces fold-level `BenchmarkResult` plus an aggregated `CVResult` with Wilcoxon / BH on a configurable target metric (default: `AdvTop4`).
- `run_case` (the `notebooks/executor.py` body) becomes `decentra.experiments.run_case`, with the same return shape.

### 8.8 Notebook compatibility

- All notebooks must continue to run, but their imports change from `from executor import run_case` to `from decentra.experiments import run_case`.
- Notebook helpers that purely orchestrate package APIs (e.g. surrogate factory lists) move into the package; notebooks keep dataset-specific glue and plotting.

---

## 9. Quality requirements

- **API style** — sklearn-like: `fit / predict / transform / fit_transform`, training state on `..._` attributes, no hidden global state.
- **DataFrame friendliness** — pandas in, pandas out for everything user-facing.
- **Name-first alignment** — feature names are the canonical identifier; positional matching is fallback only.
- **Behavioral tests** — tests verify *outputs*, not just that code runs.
- **Encoding hygiene** — fix any mojibake in comments / docstrings touched during the move. Do not introduce new mojibake.
- **Deprecation discipline** — old names (`OptBinningSurrogate`, `compute_sic_sc`, `PriorFetureInfo`, `_type`) emit DeprecationWarning for one minor version, then are removed.

---

## 10. Success criteria

- `pytest` passes after the refactor with the new layout.
- A user can fit a surrogate, build a scorecard, generate local explanation / lift output, and run a CV experiment using only public `decentra.*` imports — no `sys.path` hacks.
- `cv_summary.csv` produced via the new `run_cv` matches the existing `_e3_cv_5fold.py` output within fold-level numerical noise on at least one regression dataset.
- `ScorecardModel.from_dict(sm.to_dict()).predict(X)` matches `sm.predict(X)` exactly.
- `tests/local_scorecard_explainer.py` no longer exists as production code under `tests/`; its functionality is under `src/decentra/explain/` and is unit-tested.
- README (or `docs/claude/`) links to SPEC / PRD / TDD and shows one end-to-end snippet.

---

## 11. Open questions

| # | Question | Default if not answered |
|---|---|---|
| 1 | Should `tests/` move under `src/decentra/tests/` (B in §6), or stay at repo root (only A)? | Both A and B (move under `src/`). |
| 2 | Is `bb_score` (score scale) or `bb_logit` (log-odds) the canonical surrogate training target going forward? | `bb_score`, as in `executor.run_case` today. |
| 3 | Should the public adverse convention at the metric boundary be enforced (raise on raw SHAP being passed without conversion)? | No — accept either, document the expectation. |
| 4 | How long should deprecated aliases live? | One minor version (e.g., 0.1 → 0.2 keeps, 0.3 drops). |
| 5 | Should `Scorecard.to_dataframe()` default to English columns with a `locale="ko"` switch, or stay Korean-first? | Korean-first, English available via a flag. |
| 6 | Should `LocalScorecardExplainer` understand `ScorecardModel` natively (so users can pass a fitted scorecard instead of a custom `TempModel`)? | Yes — provide `from_scorecard_model(sm)` helper. |
| 7 | Where do `_e*.py` go? Promote each into `decentra.experiments.eN`, or keep them in `notebooks/` as thin drivers calling `run_cv`? | Thin drivers in `notebooks/`, reusable parts in `decentra.experiments`. |
