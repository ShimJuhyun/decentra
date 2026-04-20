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


def interventional_fidelity(bin_structure, teacher, X_eval, reject, contribs,
                             adverse_sign=-1):
    """Scorecard-based interventional fidelity (개입적 충실도).

    For each (rejected customer, adverse feature) pair, find the neighbouring
    *less-adverse* bin and compare the scorecard's predicted adversity
    reduction with the teacher's actual log-odds reduction.

    Parameters
    ----------
    bin_structure : dict
        Output of :func:`extract_bin_structure`.
    teacher : classifier with ``predict_proba``
    X_eval : array-like of shape (n_samples, n_features)
    reject : array-like of bool, shape (n_samples,)
    contribs : ndarray of shape (n_samples, n_features)
        Raw surrogate contributions on ``X_eval``.
    adverse_sign : {+1, -1}, default=-1
        +1 if ``contribs`` is logit-scaled (positive = adverse).
        -1 if ``contribs`` is score-scaled (negative = adverse, the credit
        scoring convention used elsewhere in this package).

    Returns
    -------
    dict with keys: DA, Spearman_rho, IR, n_pairs
    """
    X_arr = np.asarray(X_eval, dtype=float)
    base_logit = logit(teacher.predict_proba(X_arr)[:, 1])
    adv_contribs = adverse_sign * np.asarray(contribs)

    delta_s_list, delta_b_list = [], []

    for j, bins in bin_structure.items():
        if len(bins) < 2:
            continue
        # Per-bin adverse score (higher = more adverse for this feature)
        adv_bin_score = [adverse_sign * b["score"] for b in bins]

        for i in range(len(X_arr)):
            if not reject[i]:
                continue
            if adv_contribs[i, j] <= 0:
                continue

            x_val = X_arr[i, j]
            current_bin = None
            for b_idx, b in enumerate(bins):
                if x_val <= b["x_max"] + 1e-8:
                    current_bin = b_idx
                    break
            if current_bin is None:
                current_bin = len(bins) - 1

            current_adv = adv_bin_score[current_bin]

            candidates = []
            if current_bin > 0:
                candidates.append(current_bin - 1)
            if current_bin < len(bins) - 1:
                candidates.append(current_bin + 1)

            # "less adverse" = smaller adv_bin_score
            better = [c for c in candidates if adv_bin_score[c] < current_adv]
            if not better:
                continue
            best_idx = min(better, key=lambda c: current_adv - adv_bin_score[c])
            target_bin = bins[best_idx]
            delta_s = current_adv - adv_bin_score[best_idx]
            if delta_s <= 1e-8:
                continue

            x_mod = X_arr[i].copy()
            x_mod[j] = target_bin["x_mid"]
            new_logit = logit(
                teacher.predict_proba(x_mod.reshape(1, -1))[:, 1]
            )[0]
            delta_b = base_logit[i] - new_logit   # >0 if teacher agrees

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


# Deprecated legacy name kept to avoid breaking stale imports. Emits no
# warning so existing pickles/notebooks continue to work; new code should
# import ``interventional_fidelity`` directly.
compute_sic_sc = interventional_fidelity


def median_intervention_fidelity(
    teacher,
    adverse_df,
    X_test,
    train_medians,
    reject,
    k_values=(1, 4),
):
    """Surrogate-agnostic interventional fidelity via median intervention.

    For each rejected sample, take the surrogate's top-k adverse features
    (those with the largest positive values in ``adverse_df``), replace
    their values with ``train_medians``, and measure the teacher's
    probability change.

    This enables comparison across surrogates with different internal
    structures (tree, linear, binning, etc.), answering: "if the consumer
    followed the surrogate's adverse-action advice, would the teacher
    actually have granted the loan?"

    Parameters
    ----------
    teacher : classifier with ``predict_proba``
    adverse_df : pandas.DataFrame of shape (n_samples, n_features)
        Adverse contributions. Columns = feature names. Value > 0 = adverse.
        Typically from :meth:`BaseSurrogate.adverse_contributions`.
    X_test : pandas.DataFrame of shape (n_samples, n_features)
        Raw test features (same columns as ``adverse_df``).
    train_medians : pandas.Series
        Per-feature training-data medians used as neutral intervention
        value. Index = feature names.
    reject : array-like of bool
    k_values : sequence of int

    Returns
    -------
    dict with keys ``DA@k``, ``mean_delta@k``, ``median_delta@k``, ``n@k``
    for each k in ``k_values``. ``DA@k`` is the fraction of rejected
    samples for which the teacher's probability of default decreased after
    the top-k intervention.
    """
    X_test = X_test.copy()
    reject = np.asarray(reject, dtype=bool)
    reject_pos = np.where(reject)[0]
    prob_orig = teacher.predict_proba(X_test)[:, 1]
    feat_names = list(adverse_df.columns)
    missing_feats = [f for f in feat_names if f not in X_test.columns]
    if missing_feats:
        raise ValueError(
            f"adverse_df has features not in X_test: {missing_feats[:5]}"
        )

    results = {}
    for k in k_values:
        X_mod = X_test.iloc[reject].copy().reset_index(drop=True)
        adv_rej = adverse_df.iloc[reject].reset_index(drop=True)

        valid_mask = np.zeros(len(X_mod), dtype=bool)
        for i in range(len(X_mod)):
            row = adv_rej.iloc[i]
            top_k = row[row > 0].nlargest(k)
            if len(top_k) == 0:
                continue
            for feat in top_k.index:
                X_mod.iat[i, X_mod.columns.get_loc(feat)] = train_medians[feat]
            valid_mask[i] = True

        if valid_mask.sum() == 0:
            results[f"DA@{k}"] = float("nan")
            results[f"mean_delta@{k}"] = float("nan")
            results[f"median_delta@{k}"] = float("nan")
            results[f"n@{k}"] = 0
            continue

        X_mod_v = X_mod.loc[valid_mask]
        prob_mod = teacher.predict_proba(X_mod_v)[:, 1]
        prob_orig_rej = prob_orig[reject_pos][valid_mask]
        deltas = prob_orig_rej - prob_mod   # >0 if intervention improved

        results[f"DA@{k}"] = float(np.mean(deltas > 0))
        results[f"mean_delta@{k}"] = float(np.mean(deltas))
        results[f"median_delta@{k}"] = float(np.median(deltas))
        results[f"n@{k}"] = int(len(deltas))

    return results
