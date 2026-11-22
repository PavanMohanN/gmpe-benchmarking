"""Nonconformity scores and (weighted) conformal quantiles.

Scores are computed on ``cal`` only. Masked targets (SA beyond T_max = 1/LUF)
are NaN and simply drop out, so every IM has its own effective n_cal(T).

Weighted split-conformal quantile (Tibshirani et al., 2019): for calibration
scores s_i with weights w_i and a test point with weight w_x, the threshold is
the smallest s_(j) with  sum_{i<=j} w_(i) >= (1 - alpha) (sum_i w_i + w_x),
and +inf if no such score exists. With all weights 1 this is the usual
ceil((n + 1)(1 - alpha))-th smallest score, the finite-sample guarantee.
"""

from __future__ import annotations

import numpy as np


def normalized_scores(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """|y - mu| / sigma; NaN where any input is missing or sigma <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.abs(y - mu) / sigma
    return np.where(np.isfinite(s) & (sigma > 0), s, np.nan)


def abs_scores(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """|y - mu| (unnormalised ablation)."""
    s = np.abs(y - mu)
    return np.where(np.isfinite(s), s, np.nan)


def effective_n(scores: np.ndarray) -> np.ndarray:
    """Usable calibration scores per IM (column)."""
    return np.isfinite(np.atleast_2d(scores)).sum(axis=0)


def conformal_level(n: int | np.ndarray, alpha: float) -> np.ndarray:
    """Finite-sample adjusted level ceil((n + 1)(1 - alpha)) / n (> 1 means +inf)."""
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, np.ceil((n + 1) * (1 - alpha)) / n, np.inf)


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Unweighted split-conformal threshold; +inf if n_cal is too small."""
    s = np.sort(scores[np.isfinite(scores)])
    k = int(np.ceil((s.size + 1) * (1 - alpha)))
    return float(s[k - 1]) if 1 <= k <= s.size else float("inf")


def weighted_conformal_quantile(scores: np.ndarray, w_cal: np.ndarray, w_test: np.ndarray | float,
                                alpha: float) -> np.ndarray:
    """Per-test-point weighted threshold (vectorised over test points).

    Returns an array shaped like ``w_test``; +inf where the calibration mass is insufficient.
    """
    ok = np.isfinite(scores) & np.isfinite(w_cal) & (w_cal > 0)
    s, w = scores[ok], w_cal[ok]
    wt = np.atleast_1d(np.asarray(w_test, dtype=float))
    if s.size == 0:
        return np.full(wt.shape, np.inf)
    order = np.argsort(s, kind="stable")
    s, cw = s[order], np.cumsum(w[order])
    target = (1 - alpha) * (cw[-1] + wt)
    idx = np.searchsorted(cw, target * (1 - 1e-12), side="left")
    q = np.where(idx < s.size, s[np.minimum(idx, s.size - 1)], np.inf)
    return q
