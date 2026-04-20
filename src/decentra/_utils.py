import numpy as np


def information_value(x, y, eps=1e-8):
    """Information Value (IV) for a binary feature.

    IV = sum_k (p_good_k - p_bad_k) * log(p_good_k / p_bad_k).
    Returns 0.0 if ``x`` does not have exactly 2 non-null values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    mask = ~np.isnan(x)
    if not mask.any():
        return 0.0
    vals = np.unique(x[mask])
    if len(vals) != 2:
        return 0.0
    n_good = int((y == 0).sum())
    n_bad = int((y == 1).sum())
    iv = 0.0
    for v in vals:
        m = x == v
        g = int(((y == 0) & m).sum())
        b = int(((y == 1) & m).sum())
        dr_g = (g + eps) / (n_good + eps)
        dr_b = (b + eps) / (n_bad + eps)
        iv += (dr_g - dr_b) * np.log(dr_g / dr_b)
    return float(iv)


def logit(p, eps=1e-7):
    """Probability to log-odds."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    """Log-odds to probability."""
    return 1.0 / (1.0 + np.exp(-x))


def transform_logit_to_score(p, pdo=40, anchor=500, reverse_prob=True, eps=1e-7):
    """Convert probability to credit score.

    Standard credit scoring formula:
        score = anchor + factor × log-odds

    Parameters
    ----------
    p : float or array-like
        Probability (e.g. P(default)).
    pdo : int, default=40
        Points to Double the Odds.
    anchor : int, default=500
        Base score at 50/50 odds (logit=0).
    reverse_prob : bool, default=True
        If True, higher probability → lower score (standard credit scoring:
        high default prob = low score).
    eps : float, default=1e-7

    Returns
    -------
    int or ndarray[int]
    """
    log_odds = logit(p, eps)
    sign = -1 if reverse_prob else 1
    score = anchor + sign * pdo / np.log(2) * log_odds
    return np.round(score).astype(int)