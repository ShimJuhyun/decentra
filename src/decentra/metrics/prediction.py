import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def prediction_fidelity(y_logit, surr_pred, prob_true, surr_prob,
                        agree_cutoff=0.10):
    """Prediction fidelity metrics between black-box and surrogate.

    Parameters
    ----------
    y_logit : array-like
        Teacher's log-odds on test set.
    surr_pred : array-like
        Surrogate's log-odds predictions.
    prob_true : array-like
        Teacher's probability predictions.
    surr_prob : array-like
        Surrogate's probability predictions.
    agree_cutoff : float, default=0.10
        Absolute probability threshold for Agree metric.

    Returns
    -------
    dict with keys: R2, Agree, Spearman
    """
    r2 = r2_score(y_logit, surr_pred)
    agree = float(np.mean(
        (prob_true >= agree_cutoff) == (surr_prob >= agree_cutoff)
    ))
    sp = float(spearmanr(y_logit, surr_pred)[0])
    return {"R2": r2, "Agree": agree, "Spearman": sp}
