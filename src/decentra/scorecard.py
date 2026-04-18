import numpy as np
import pandas as pd

from .scorecard_model import ScorecardModel


class Scorecard:
    """Display-oriented scorecard built from a ScorecardModel + observed data.

    Per-bin reason codes:
        score >= 0  →  P prefix (긍정), ranked P001, P002, ...
        score <  0  →  N prefix (부정), ranked N001, N002, ...

    Examples
    --------
    >>> sm = surr.to_scorecard_model(X_train)
    >>> sc = sm.scorecard(X_train, y_binary)
    >>> sc.to_dataframe()
    """

    def __init__(self):
        self.cards_ = None

    # ── Construction ──────────────────────────────────────────

    @classmethod
    def from_scorecard_model(cls, model: ScorecardModel, X, y_binary):
        """Build a display Scorecard from a ScorecardModel.

        Parameters
        ----------
        model : ScorecardModel
        X : array-like of shape (n_samples, n_features)
        y_binary : array-like of shape (n_samples,)
        """
        sc = cls()
        sc._build(model, X, y_binary)
        return sc

    @classmethod
    def from_surrogate(cls, surrogate, X, y_binary, feature_names=None,
                       n_bins=10):
        """Convenience: surrogate → ScorecardModel → Scorecard."""
        sm = surrogate.to_scorecard_model(X, feature_names, n_bins)
        return cls.from_scorecard_model(sm, X, y_binary)

    # ── Internal build ────────────────────────────────────────

    def _build(self, model, X, y_binary):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y_binary).ravel()
        n_samples = X_arr.shape[0]

        cards = []
        for feat in model.features:
            j = feat.index
            raw_bins = []
            for br in feat.bins:
                mask = br.contains(X_arr[:, j])
                count = int(mask.sum())
                if count == 0:
                    continue
                target_count = int(y_arr[mask].sum())
                raw_bins.append({
                    "lower": br.lower,
                    "upper": br.upper,
                    "score": br.score,
                    "count": count,
                    "target_count": target_count,
                    "proportion": count / n_samples,
                    "target_rate": target_count / count,
                    "label": self._bin_label(br.lower, br.upper),
                })

            cards.append({
                "feature": feat.name,
                "feature_idx": j,
                "bins": raw_bins,
            })

        # ── Per-bin reason codes ──────────────────────────────
        positive, negative = [], []
        for ci, card in enumerate(cards):
            for bi, b in enumerate(card["bins"]):
                entry = (ci, bi, b["score"])
                (positive if b["score"] >= 0 else negative).append(entry)

        positive.sort(key=lambda x: abs(x[2]), reverse=True)
        negative.sort(key=lambda x: abs(x[2]), reverse=True)

        for rank, (ci, bi, _) in enumerate(positive, 1):
            cards[ci]["bins"][bi]["reason_code"] = f"P{rank:03d}"
            cards[ci]["bins"][bi]["reason_desc"] = f"긍정 요인 {rank}순위"

        for rank, (ci, bi, _) in enumerate(negative, 1):
            cards[ci]["bins"][bi]["reason_code"] = f"N{rank:03d}"
            cards[ci]["bins"][bi]["reason_desc"] = f"부정 요인 {rank}순위"

        self.cards_ = cards

    # ── Output ────────────────────────────────────────────────

    def to_dataframe(self):
        """Return the scorecard as a pandas DataFrame.

        Columns: 평가 항목, 구간, 가중치 점수, 건수, Target 건수,
                 구성비, Target Rate, 사유코드, 사유코드 설명
        """
        rows = []
        for card in self.cards_:
            for k, b in enumerate(card["bins"]):
                rows.append({
                    "평가 항목": card["feature"] if k == 0 else "",
                    "구간": b["label"],
                    "가중치 점수": round(b["score"], 4),
                    "건수": b["count"],
                    "Target 건수": b["target_count"],
                    "구성비": b["proportion"],
                    "Target Rate": b["target_rate"],
                    "사유코드": b["reason_code"],
                    "사유코드 설명": b["reason_desc"],
                })
        df = pd.DataFrame(rows)
        df["구성비"] = df["구성비"].map("{:.2%}".format)
        df["Target Rate"] = df["Target Rate"].map("{:.2%}".format)
        return df

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _bin_label(lower, upper):
        if np.isinf(lower) and lower < 0:
            return f"< {upper:.4g}"
        if np.isinf(upper):
            return f">= {lower:.4g}"
        return f"{lower:.4g} ~ {upper:.4g}"

    def __repr__(self):
        if self.cards_ is None:
            return "Scorecard(not built)"
        n_features = len(self.cards_)
        n_bins = sum(len(c["bins"]) for c in self.cards_)
        return f"Scorecard({n_features} features, {n_bins} bins)"
