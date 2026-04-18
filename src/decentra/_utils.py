import numpy as np


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