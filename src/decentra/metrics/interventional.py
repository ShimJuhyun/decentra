import numpy as np
from scipy.stats import spearmanr

from .._utils import logit


def extract_bin_structure(model, X_sample, n_features):
    """Extract per-feature bin structure from a depth-1 LightGBM model.

    Uses ``pred_contrib`` unique values as bin scores, then maps each
    score back to the feature-value range that produces it.

    Parameters
    ----------
    model : LGBMRegressor (fitted, max_depth=1)
    X_sample : array-like
        Training data used to discover all bins.
    n_features : int

    Returns
    -------
    dict : {feature_idx: [{'score', 'x_min', 'x_max', 'x_mid', 'count'}, ...]}
    """
    contribs = model.predict(X_sample, pred_contrib=True)[:, :-1]
    x_arr = np.asarray(X_sample)
    bins = {}

    for j in range(n_features):
        unique_scores = np.unique(np.round(contribs[:, j], 8))
        if len(unique_scores) < 2:
            continue
        score_vals = np.round(contribs[:, j], 8)
        x_vals = x_arr[:, j]
        bin_info = []
        for us in sorted(unique_scores):
            mask = np.abs(score_vals - us) < 1e-7
            bin_info.append({
                "score": float(us),
                "x_min": float(x_vals[mask].min()),
                "x_max": float(x_vals[mask].max()),
                "x_mid": float(np.median(x_vals[mask])),
                "count": int(mask.sum()),
            })
        bins[j] = sorted(bin_info, key=lambda b: b["x_mid"])
    return bins


def compute_sic_sc(bin_structure, teacher, X_eval, reject, contribs):
    """SIC-SC: Scorecard-based interventional fidelity.

    For each (rejected customer, adverse feature) pair, finds the
    neighbouring improved bin and compares the scorecard's predicted
    improvement with the teacher's actual improvement.

    Parameters
    ----------
    bin_structure : dict
        Output of ``extract_bin_structure``.
    teacher : classifier with ``predict_proba``
        The black-box teacher model.
    X_eval : array-like of shape (n_samples, n_features)
    reject : array-like of bool, shape (n_samples,)
    contribs : ndarray of shape (n_samples, n_features)

    Returns
    -------
    dict with keys: DC, Spearman_rho, IR, n_pairs
    """
    X_arr = np.asarray(X_eval, dtype=float)
    base_logit = logit(teacher.predict_proba(X_arr)[:, 1])

    delta_s_list = []
    delta_b_list = []

    for j, bins in bin_structure.items():
        if len(bins) < 2:
            continue
        for i in range(len(X_arr)):
            if not reject[i]:
                continue
            if contribs[i, j] <= 0:
                continue

            x_val = X_arr[i, j]

            # Find current bin
            current_bin = None
            for b_idx, b in enumerate(bins):
                if x_val <= b["x_max"] + 1e-8:
                    current_bin = b_idx
                    break
            if current_bin is None:
                current_bin = len(bins) - 1

            current_score = bins[current_bin]["score"]

            # Adjacent bins with lower score (= improvement)
            candidates = []
            if current_bin > 0:
                candidates.append(current_bin - 1)
            if current_bin < len(bins) - 1:
                candidates.append(current_bin + 1)

            better = [c for c in candidates if bins[c]["score"] < current_score]
            if not better:
                continue

            best_idx = min(better, key=lambda c: current_score - bins[c]["score"])
            target_bin = bins[best_idx]
            delta_s = current_score - target_bin["score"]
            if delta_s <= 1e-8:
                continue

            # Measure actual improvement in teacher
            x_mod = X_arr[i].copy()
            x_mod[j] = target_bin["x_mid"]
            new_logit = logit(
                teacher.predict_proba(x_mod.reshape(1, -1))[:, 1]
            )[0]
            delta_b = base_logit[i] - new_logit

            delta_s_list.append(delta_s)
            delta_b_list.append(delta_b)

    if len(delta_s_list) < 3:
        return {
            "DA": np.nan, "Spearman_rho": np.nan,
            "IR": np.nan, "n_pairs": len(delta_s_list),
        }

    ds = np.array(delta_s_list)
    db = np.array(delta_b_list)
    return {
        "DA": float(np.mean(db > 0)),
        "Spearman_rho": float(spearmanr(ds, db)[0]),
        "IR": float(np.mean(db / ds)),
        "n_pairs": len(ds),
    }
