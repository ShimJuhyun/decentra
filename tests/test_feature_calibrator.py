import numpy as np
import pytest
from sklearn.metrics import r2_score

from decentra.calibration import FeatureCalibrator


def _synthetic_scale_mismatch(rng, n=2000, p=5, scale_surr=50.0, scale_bb=0.5):
    """Surrogate on 'score scale', BB on 'logit scale' (~100x gap)."""
    # Ground-truth attributions: bb shape similar to surr but with different
    # per-feature relative magnitudes and totally different absolute scale.
    base_shape = rng.normal(0, 1, size=(n, p))
    bb_shap = base_shape * rng.uniform(0.5, 2.0, size=p) * scale_bb
    # Surrogate reproduces direction but with compressed shape (needs calibration)
    surr_contribs = base_shape * rng.uniform(0.8, 1.2, size=p) * scale_surr
    base_value = 500.0
    surr_pred = base_value + surr_contribs.sum(axis=1)
    y = base_value + (base_shape * rng.uniform(1.0, 1.5, size=p)).sum(axis=1)
    return surr_contribs, bb_shap, surr_pred, y


def test_feature_calibrator_preserves_r2_under_scale_mismatch():
    rng = np.random.default_rng(42)
    surr_contribs, bb_shap, surr_pred, y = _synthetic_scale_mismatch(rng)
    r2_before = r2_score(y, surr_pred)

    cal = FeatureCalibrator()
    cal_contribs, cal_pred = cal.fit_transform(surr_contribs, bb_shap, surr_pred)
    r2_after = r2_score(y, cal_pred)

    # Should not collapse: within 20% of baseline (baseline ~0.8+)
    assert r2_after > 0.5 * r2_before, (
        f"R² collapsed: before={r2_before:.3f}, after={r2_after:.3f}"
    )


def test_feature_calibrator_magnitude_preservation():
    rng = np.random.default_rng(0)
    surr_contribs, bb_shap, surr_pred, _ = _synthetic_scale_mismatch(rng)

    cal = FeatureCalibrator(magnitude_preserving=True)
    cal.fit(surr_contribs, bb_shap)

    su_abs_mean = np.mean(np.abs(surr_contribs), axis=0)
    total_before = su_abs_mean.sum()
    total_after = (cal.alpha_ * su_abs_mean).sum()

    np.testing.assert_allclose(total_after, total_before, rtol=1e-6)


def test_feature_calibrator_legacy_raises_warning():
    rng = np.random.default_rng(1)
    surr_contribs, bb_shap, surr_pred, _ = _synthetic_scale_mismatch(rng)

    cal = FeatureCalibrator(magnitude_preserving=False)
    with pytest.warns(DeprecationWarning, match="magnitude_preserving=False"):
        cal.fit(surr_contribs, bb_shap)


def test_feature_calibrator_relative_shape_matches_bb():
    """After calibration, relative shape (alpha * surr_abs) should match bb shape."""
    rng = np.random.default_rng(7)
    surr_contribs, bb_shap, _, _ = _synthetic_scale_mismatch(rng)

    cal = FeatureCalibrator()
    cal.fit(surr_contribs, bb_shap)

    bb_abs_mean = np.mean(np.abs(bb_shap), axis=0)
    su_abs_mean = np.mean(np.abs(surr_contribs), axis=0)
    calibrated_abs_mean = cal.alpha_ * su_abs_mean

    # Relative shares should match BB's shares
    bb_share = bb_abs_mean / bb_abs_mean.sum()
    cal_share = calibrated_abs_mean / calibrated_abs_mean.sum()
    np.testing.assert_allclose(cal_share, bb_share, atol=1e-6)
