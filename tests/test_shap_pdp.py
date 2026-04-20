"""Smoke tests for ShapPdpSurrogate (Choi & Cha 2026 baseline)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "age": rng.uniform(20, 70, n),
        "income": rng.uniform(10, 200, n),
        "debt": rng.uniform(0, 50, n),
    })
    # teacher logit loosely related to features
    logit = (
        -0.05 * X["age"]
        + 0.02 * X["income"]
        - 0.1 * X["debt"]
        + rng.normal(0, 0.1, n)
    )
    y_binary = (logit > logit.median()).astype(int)
    return X, logit.to_numpy(), y_binary.to_numpy()


@pytest.fixture
def base_model(toy_data):
    import lightgbm as lgb
    X, _, y_binary = toy_data
    m = lgb.LGBMClassifier(
        n_estimators=50, max_depth=3, random_state=0, verbose=-1,
    )
    m.fit(X, y_binary)
    return m


def test_fit_with_base_model(toy_data, base_model):
    from decentra.surrogate import ShapPdpSurrogate

    X, y_logit, y_binary = toy_data
    surr = ShapPdpSurrogate(max_n_bins=5, smoother_n_estimators=50)
    surr.fit(X, y_logit, base_model=base_model, binning_y=y_binary)

    assert surr.bin_scores_ is not None
    assert set(surr.bin_scores_.keys()) <= {0, 1, 2}
    for j, scores in surr.bin_scores_.items():
        assert scores.ndim == 1
        assert len(scores) >= 2


def test_fit_with_precomputed_shap(toy_data, base_model):
    import shap
    from decentra.surrogate import ShapPdpSurrogate

    X, y_logit, y_binary = toy_data
    explainer = shap.TreeExplainer(base_model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]

    surr = ShapPdpSurrogate(max_n_bins=5, smoother_n_estimators=50)
    surr.fit(X, y_logit, shap_values=sv, binning_y=y_binary)

    contribs = surr.contributions(X)
    assert contribs.shape == X.shape
    # contributions are centered → column means ≈ 0
    np.testing.assert_allclose(contribs.mean(axis=0), 0.0, atol=1e-8)


def test_predict_and_contributions(toy_data, base_model):
    from decentra.surrogate import ShapPdpSurrogate

    X, y_logit, y_binary = toy_data
    surr = ShapPdpSurrogate(max_n_bins=5, smoother_n_estimators=50).fit(
        X, y_logit, base_model=base_model, binning_y=y_binary,
    )

    preds = surr.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()

    ranks = surr.contribution_ranking(X)
    assert ranks.shape == X.shape


def test_scorecard_conversion(toy_data, base_model):
    from decentra.surrogate import ShapPdpSurrogate

    X, y_logit, y_binary = toy_data
    surr = ShapPdpSurrogate(max_n_bins=5, smoother_n_estimators=50).fit(
        X, y_logit, base_model=base_model, binning_y=y_binary,
    )

    sm = surr.to_scorecard_model(X, y_binary=y_binary)
    assert len(sm.features) >= 1

    contribs = surr.contributions(X)
    sm_preds = sm.predict(X)
    assert sm_preds.shape == (len(X),)

    # feature importances well-defined
    imp = surr.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert (imp >= 0).all()


def test_requires_shap_or_base_model(toy_data):
    from decentra.surrogate import ShapPdpSurrogate

    X, y_logit, y_binary = toy_data
    surr = ShapPdpSurrogate()
    with pytest.raises(ValueError, match="base_model or shap_values"):
        surr.fit(X, y_logit, binning_y=y_binary)
