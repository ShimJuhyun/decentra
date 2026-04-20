"""Name-based attribution metrics.

All functions accept attributions as either:
- pandas.DataFrame  — columns are feature names (preferred)
- dict[str, ndarray]  — maps name → (n_samples,)-shaped contributions
- (ndarray, list[str])  — (n_samples, n_features) paired with names

The attribution convention here is unified: ``value > 0`` means the feature
pushed the sample toward rejection (adverse). Callers should produce such
attributions via :meth:`BaseSurrogate.adverse_contributions` (for the
surrogate side) and by applying the appropriate sign convention to BB SHAP
on the teacher side (log-odds TreeSHAP already satisfies this).

The family complements :mod:`decentra.metrics.attribution`, which uses the
legacy ``bb_sign`` / ``surr_sign`` interface and positional column matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

NamedAttr = Union[
    pd.DataFrame, Mapping[str, np.ndarray], Tuple[np.ndarray, Sequence[str]]
]


def _to_dataframe(a: NamedAttr) -> pd.DataFrame:
    """Coerce supported inputs to a DataFrame with string column names."""
    if isinstance(a, pd.DataFrame):
        return a
    if isinstance(a, Mapping):
        return pd.DataFrame({str(k): np.asarray(v) for k, v in a.items()})
    if isinstance(a, tuple) and len(a) == 2:
        arr, names = a
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != len(names):
            raise ValueError(
                f"Array shape {arr.shape} does not match names of "
                f"length {len(names)}."
            )
        return pd.DataFrame(arr, columns=[str(n) for n in names])
    raise TypeError(
        f"Unsupported attribution type: {type(a).__name__}. Expected "
        "DataFrame, dict, or (ndarray, names) tuple."
    )


@dataclass
class AlignmentInfo:
    common: List[str]
    only_a: List[str]
    only_b: List[str]
    n_common: int
    n_only_a: int
    n_only_b: int
    coverage_a: float = field(default=0.0)  # share of A columns in common
    coverage_b: float = field(default=0.0)  # share of B columns in common
    missing_policy: str = "zero"

    def to_dict(self) -> Dict:
        return {
            "common": list(self.common),
            "only_a": list(self.only_a),
            "only_b": list(self.only_b),
            "n_common": self.n_common,
            "n_only_a": self.n_only_a,
            "n_only_b": self.n_only_b,
            "coverage_a": self.coverage_a,
            "coverage_b": self.coverage_b,
            "missing_policy": self.missing_policy,
        }


def align_attributions(
    a: NamedAttr,
    b: NamedAttr,
    *,
    missing: str = "zero",
    strict: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, AlignmentInfo]:
    """Align two named-attribution tables on their common feature columns.

    Parameters
    ----------
    a, b : DataFrame | dict | (ndarray, names)
    missing : {"zero", "drop", "raise"}, default="zero"
        "zero": keep A∪B columns, fill missing with 0.
        "drop": restrict to A∩B columns.
        "raise": require A==B columns, else ValueError.
    strict : bool, default=False
        If True, symmetric difference ≠ ∅ is an error regardless of
        ``missing``.

    Returns
    -------
    a_aligned, b_aligned : DataFrame
        Same columns, same row count.
    info : AlignmentInfo
    """
    da = _to_dataframe(a)
    db = _to_dataframe(b)

    if len(da) != len(db):
        raise ValueError(
            f"Row count mismatch: a has {len(da)}, b has {len(db)}"
        )

    cols_a = list(da.columns)
    cols_b = list(db.columns)
    set_a, set_b = set(cols_a), set(cols_b)
    common = [c for c in cols_a if c in set_b]
    only_a = [c for c in cols_a if c not in set_b]
    only_b = [c for c in cols_b if c not in set_a]

    info = AlignmentInfo(
        common=common,
        only_a=only_a,
        only_b=only_b,
        n_common=len(common),
        n_only_a=len(only_a),
        n_only_b=len(only_b),
        coverage_a=len(common) / len(cols_a) if cols_a else 0.0,
        coverage_b=len(common) / len(cols_b) if cols_b else 0.0,
        missing_policy=missing,
    )

    if strict and (only_a or only_b):
        raise ValueError(
            f"strict=True but columns differ: only_a={only_a}, only_b={only_b}"
        )

    if missing == "raise":
        if only_a or only_b:
            raise ValueError(
                f"Column mismatch: only_a={only_a}, only_b={only_b}"
            )
        union = common
        da_out, db_out = da[union], db[union]
    elif missing == "drop":
        da_out, db_out = da[common], db[common]
    elif missing == "zero":
        union = common + only_a + only_b
        da_out = da.reindex(columns=union, fill_value=0.0)
        db_out = db.reindex(columns=union, fill_value=0.0)
    else:
        raise ValueError(f"Unknown missing policy: {missing!r}")

    return da_out, db_out, info


# -------- metrics on aligned, adverse-signed named attributions -----------

def _select_reject(df: pd.DataFrame, reject: np.ndarray) -> pd.DataFrame:
    reject = np.asarray(reject, dtype=bool)
    if len(reject) != len(df):
        raise ValueError(
            f"reject length {len(reject)} != n_samples {len(df)}"
        )
    return df.loc[reject]


def topk_named(bb: NamedAttr, surr: NamedAttr, k: int, **align_kw) -> float:
    """Top-k overlap of |contributions|, sign-agnostic, all samples.

    Align by name; ranking is computed on the aligned column set.
    """
    a, b, _ = align_attributions(bb, surr, **align_kw)
    bb_abs = a.abs().to_numpy()
    surr_abs = b.abs().to_numpy()
    overlaps = []
    for i in range(len(a)):
        top_a = set(np.argsort(bb_abs[i])[-k:])
        top_b = set(np.argsort(surr_abs[i])[-k:])
        overlaps.append(len(top_a & top_b) / k)
    return float(np.mean(overlaps)) if overlaps else 0.0


def advtopk_named(
    bb: NamedAttr,
    surr: NamedAttr,
    reject: np.ndarray,
    k: int,
    **align_kw,
) -> float:
    """Adverse Top-k overlap on rejected samples.

    Inputs are assumed to already encode "value > 0 = adverse". Use
    :meth:`BaseSurrogate.adverse_contributions` on the surrogate side.
    """
    a, b, _ = align_attributions(bb, surr, **align_kw)
    ar = _select_reject(a, reject).to_numpy()
    br = _select_reject(b, reject).to_numpy()
    overlaps = []
    for i in range(len(ar)):
        a_pos = np.where(ar[i] > 0)[0]
        b_pos = np.where(br[i] > 0)[0]
        if len(a_pos) == 0 or len(b_pos) == 0:
            continue
        ke = min(k, len(a_pos), len(b_pos))
        top_a = set(a_pos[np.argsort(ar[i, a_pos])[-ke:]])
        top_b = set(b_pos[np.argsort(br[i, b_pos])[-ke:]])
        overlaps.append(len(top_a & top_b) / ke)
    return float(np.mean(overlaps)) if overlaps else 0.0


def advfull_named(
    bb: NamedAttr,
    surr: NamedAttr,
    reject: np.ndarray,
    **align_kw,
) -> Tuple[float, float]:
    """Full adverse-set Recall and Jaccard on rejected samples."""
    a, b, _ = align_attributions(bb, surr, **align_kw)
    ar = _select_reject(a, reject).to_numpy()
    br = _select_reject(b, reject).to_numpy()
    recalls, jaccards = [], []
    for i in range(len(ar)):
        sa = set(np.where(ar[i] > 0)[0])
        sb = set(np.where(br[i] > 0)[0])
        if not sa:
            continue
        inter = len(sa & sb)
        recalls.append(inter / len(sa))
        union = len(sa | sb)
        jaccards.append(inter / union if union else 0.0)
    return (
        float(np.mean(recalls)) if recalls else 0.0,
        float(np.mean(jaccards)) if jaccards else 0.0,
    )


def random_baseline_advtopk_named(
    bb: NamedAttr, reject: np.ndarray, k: int, p: int
) -> float:
    """Expected AdvTop-k under random attribution, among rejects."""
    a = _to_dataframe(bb)
    ar = _select_reject(a, reject).to_numpy()
    if len(ar) == 0:
        return 0.0
    a_counts = (ar > 0).sum(axis=1)
    return float(np.mean(np.minimum(k, a_counts))) / p


def attribution_fidelity_named(
    bb: NamedAttr,
    surr: NamedAttr,
    reject: np.ndarray,
    *,
    ks: Sequence[int] = (1, 3, 4),
    adv_ks: Sequence[int] = (1, 4),
    missing: str = "zero",
    strict: bool = False,
    return_info: bool = False,
) -> Dict:
    """Compute a full slate of named attribution fidelity metrics.

    Returns a dict with keys ``Top{k}``, ``AdvTop{k}``, ``AdvFull_R``,
    ``AdvFull_J``, and ``coverage_surr`` (share of BB columns present in
    the surrogate). If ``return_info=True``, includes the raw
    :class:`AlignmentInfo`.
    """
    a, b, info = align_attributions(bb, surr, missing=missing, strict=strict)
    out: Dict = {}
    for k in ks:
        out[f"Top{k}"] = topk_named(a, b, k, missing=missing, strict=strict)
    for k in adv_ks:
        out[f"AdvTop{k}"] = advtopk_named(
            a, b, reject, k, missing=missing, strict=strict
        )
    r, j = advfull_named(a, b, reject, missing=missing, strict=strict)
    out["AdvFull_R"] = r
    out["AdvFull_J"] = j
    out["coverage_surr"] = info.coverage_a  # share of BB kept
    if return_info:
        out["_info"] = info.to_dict()
    return out
