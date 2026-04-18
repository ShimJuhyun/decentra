"""Attribution fidelity metrics.

Sign convention
---------------
BB SHAP is in **log-odds** space: ``>0`` = increases default = adverse.
Surrogate contribs are in **score** space: ``<0`` = decreases score = adverse.

All functions accept ``bb_sign`` and ``surr_sign`` parameters to handle this:
- ``bb_sign=1`` (default): BB values ``> 0`` are adverse.
- ``surr_sign=-1``: Surrogate values ``< 0`` are adverse.

The sign parameter multiplies the array before the ``> 0`` check,
so ``surr_sign=-1`` flips the convention to match BB.
"""

import numpy as np


def topk(bb_shap, surr_shap, k):
    """Top-k overlap of absolute contributions (full sample, sign-agnostic).

    Parameters
    ----------
    bb_shap, surr_shap : ndarray of shape (n_samples, n_features)
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


def advtopk(bb_shap, surr_shap, reject, k, bb_sign=1, surr_sign=-1):
    """AdvTop-k: adverse contribution overlap among rejected samples.

    Parameters
    ----------
    bb_shap, surr_shap : ndarray of shape (n_samples, n_features)
    reject : array-like of bool, shape (n_samples,)
    k : int
    bb_sign : int, default=1
        1 if ``bb > 0`` means adverse (log-odds convention).
    surr_sign : int, default=-1
        -1 if ``surr < 0`` means adverse (score convention).

    Returns
    -------
    float : mean overlap ratio in [0, 1].
    """
    overlaps = []
    for i in range(len(bb_shap)):
        if not reject[i]:
            continue
        bb_adv = bb_sign * bb_shap[i]
        su_adv = surr_sign * surr_shap[i]
        bb_pos = np.where(bb_adv > 0)[0]
        su_pos = np.where(su_adv > 0)[0]
        if len(bb_pos) == 0 or len(su_pos) == 0:
            continue
        ke = min(k, len(bb_pos), len(su_pos))
        bb_top = set(bb_pos[np.argsort(bb_adv[bb_pos])[-ke:]])
        su_top = set(su_pos[np.argsort(su_adv[su_pos])[-ke:]])
        overlaps.append(len(bb_top & su_top) / ke)
    return float(np.mean(overlaps)) if overlaps else 0.0


def advfull(bb_shap, surr_shap, reject, bb_sign=1, surr_sign=-1):
    """AdvFull: full adverse-set Recall and Jaccard among rejected samples.

    Parameters
    ----------
    bb_sign : int, default=1
    surr_sign : int, default=-1

    Returns
    -------
    tuple of (recall, jaccard)
    """
    recalls, jaccards = [], []
    for i in range(len(bb_shap)):
        if not reject[i]:
            continue
        bb_set = set(np.where(bb_sign * bb_shap[i] > 0)[0])
        su_set = set(np.where(surr_sign * surr_shap[i] > 0)[0])
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


def random_baseline_advtopk(bb_shap, reject, k, p, bb_sign=1):
    """Expected AdvTop-k under random feature attribution.

    Parameters
    ----------
    bb_shap : ndarray of shape (n_samples, n_features)
    reject : array-like of bool
    k, p : int
    bb_sign : int, default=1
    """
    a_counts = []
    for i in range(len(bb_shap)):
        if not reject[i]:
            continue
        a_counts.append(np.sum(bb_sign * bb_shap[i] > 0))
    if not a_counts:
        return 0.0
    return float(np.mean([min(k, a) for a in a_counts])) / p


def attribution_fidelity(bb_shap, surr_shap, reject, bb_sign=1, surr_sign=-1):
    """All attribution fidelity metrics in one call.

    Parameters
    ----------
    bb_sign : int, default=1
    surr_sign : int, default=-1

    Returns
    -------
    dict
    """
    af_r, af_j = advfull(bb_shap, surr_shap, reject, bb_sign, surr_sign)
    return {
        "Top1": topk(bb_shap, surr_shap, 1),
        "Top3": topk(bb_shap, surr_shap, 3),
        "AdvTop1": advtopk(bb_shap, surr_shap, reject, 1, bb_sign, surr_sign),
        "AdvTop4": advtopk(bb_shap, surr_shap, reject, 4, bb_sign, surr_sign),
        "AdvFull_R": af_r,
        "AdvFull_J": af_j,
    }
