import numpy as np


def topk(bb_shap, surr_shap, k):
    """Top-k overlap of absolute contributions (full sample).

    Parameters
    ----------
    bb_shap : ndarray of shape (n_samples, n_features)
    surr_shap : ndarray of shape (n_samples, n_features)
    k : int

    Returns
    -------
    float : mean overlap ratio in [0, 1].
    """
    overlaps = []
    for i in range(len(bb_shap)):
        bb_top = set(np.argsort(np.abs(bb_shap[i]))[-k:])
        su_top = set(np.argsort(np.abs(surr_shap[i]))[-k:])
        overlaps.append(len(bb_top & su_top) / k)
    return float(np.mean(overlaps))


def advtopk(bb_shap, surr_shap, reject, k):
    """AdvTop-k: adverse (positive) contribution overlap among rejected samples.

    Parameters
    ----------
    bb_shap, surr_shap : ndarray of shape (n_samples, n_features)
    reject : array-like of bool, shape (n_samples,)
        True for rejected (high-risk) samples.
    k : int

    Returns
    -------
    float : mean overlap ratio in [0, 1].
    """
    overlaps = []
    for i in range(len(bb_shap)):
        if not reject[i]:
            continue
        bb_pos = np.where(bb_shap[i] > 0)[0]
        su_pos = np.where(surr_shap[i] > 0)[0]
        if len(bb_pos) == 0 or len(su_pos) == 0:
            continue
        ke = min(k, len(bb_pos), len(su_pos))
        bb_top = set(bb_pos[np.argsort(bb_shap[i][bb_pos])[-ke:]])
        su_top = set(su_pos[np.argsort(surr_shap[i][su_pos])[-ke:]])
        overlaps.append(len(bb_top & su_top) / ke)
    return float(np.mean(overlaps)) if overlaps else 0.0


def advfull(bb_shap, surr_shap, reject):
    """AdvFull: full adverse-set Recall and Jaccard among rejected samples.

    Returns
    -------
    tuple of (recall, jaccard)
    """
    recalls, jaccards = [], []
    for i in range(len(bb_shap)):
        if not reject[i]:
            continue
        bb_set = set(np.where(bb_shap[i] > 0)[0])
        su_set = set(np.where(surr_shap[i] > 0)[0])
        if len(bb_set) == 0:
            continue
        inter = len(bb_set & su_set)
        recalls.append(inter / len(bb_set))
        union = len(bb_set | su_set)
        jaccards.append(inter / union if union > 0 else 0.0)
    return (
        float(np.mean(recalls)) if recalls else 0.0,
        float(np.mean(jaccards)) if jaccards else 0.0,
    )


def random_baseline_advtopk(bb_shap, reject, k, p):
    """Expected AdvTop-k under random feature attribution.

    Parameters
    ----------
    bb_shap : ndarray of shape (n_samples, n_features)
    reject : array-like of bool
    k : int
    p : int
        Total number of features.
    """
    a_counts = []
    for i in range(len(bb_shap)):
        if not reject[i]:
            continue
        a_counts.append(np.sum(bb_shap[i] > 0))
    if not a_counts:
        return 0.0
    return float(np.mean([min(k, a) for a in a_counts])) / p


def attribution_fidelity(bb_shap, surr_shap, reject):
    """All attribution fidelity metrics in one call.

    Returns
    -------
    dict with keys: Top1, Top4, AdvTop1, AdvTop4, AdvFull_R, AdvFull_J
    """
    af_r, af_j = advfull(bb_shap, surr_shap, reject)
    return {
        "Top1": topk(bb_shap, surr_shap, 1),
        "Top4": topk(bb_shap, surr_shap, 4),
        "AdvTop1": advtopk(bb_shap, surr_shap, reject, 1),
        "AdvTop4": advtopk(bb_shap, surr_shap, reject, 4),
        "AdvFull_R": af_r,
        "AdvFull_J": af_j,
    }
