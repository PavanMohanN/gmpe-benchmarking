"""Covariate-shift density ratios from a domain classifier (no test labels).

A regularised logistic regression separates calibration inputs (label 0)
from test inputs (label 1) on physical predictors only:
Mw, ln R_rup, ln VS30, ln Z1.0, aftershock flag (CL2) and region one-hot.

    w(x) = p(test | x) / p(cal | x) * n_cal / n_test   ~   p_test(x) / p_cal(x)

Support rule: the ratio is undefined where the test domain has no calibration
support. An indicator group (aftershock flag; region one-hot) is dropped as a whole when
any level present in ``test`` occurs in fewer than ``min_support`` calibration
records (dropping only that level would still let the other levels identify
the calibration set). So a
held-out region (``spatial_block``) or class (``cl1_to_cl2``) is matched on its
continuous physics (Mw, distance, site) instead of being weighted to infinity.

Weights are clipped to [w_min, w_max] for stability. Only feature columns are
read; target columns are never passed in.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gmpe_bench.splits.group_splits import REGION_BOXES, REGION_OTHER, assign_region

SHIFT_REGIONS: tuple[str, ...] = tuple(b[0] for b in REGION_BOXES) + (REGION_OTHER,)
N_CONTINUOUS = 4  # Mw, ln Rrup, ln VS30, ln Z1.0; the rest are 0/1 indicators


def shift_features(df: pd.DataFrame) -> np.ndarray:
    """Physical predictors for the domain classifier; never reads targets."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) if c in df \
        else np.full(len(df), np.nan)  # noqa: E731
    rrup, vs30, z1 = num("rrup"), num("vs30"), num("z1")
    cols = [num("mw"), np.log(np.clip(rrup, 0.1, None)), np.log(vs30), np.log(np.clip(z1, 1.0, None))]
    cl2 = (df["cl_class"].astype("string") == "CL2").fillna(False).to_numpy(dtype=float) \
        if "cl_class" in df else np.zeros(len(df))
    region = assign_region(df).to_numpy()
    onehot = [(region == r).astype(float) for r in SHIFT_REGIONS]
    X = np.column_stack(cols + [cl2] + onehot)
    return X


def indicator_groups(n_cols: int, n_continuous: int = N_CONTINUOUS) -> list[list[int]]:
    """Column groups of the indicators in :func:`shift_features`: [CL2], [regions...]."""
    if n_cols <= n_continuous:
        return []
    groups = [[n_continuous]]
    if n_cols > n_continuous + 1:
        groups.append(list(range(n_continuous + 1, n_cols)))
    return groups


def supported_columns(X_cal: np.ndarray, X_test: np.ndarray, n_continuous: int = N_CONTINUOUS,
                      min_support: int = 5) -> np.ndarray:
    """Usable columns: continuous always; an indicator group only if every test level has cal support."""
    keep = np.ones(X_cal.shape[1], dtype=bool)
    for group in indicator_groups(X_cal.shape[1], n_continuous):
        for j in group:
            if np.nansum(X_test[:, j]) > 0 and np.nansum(X_cal[:, j]) < min_support:
                keep[group] = False
                break
    return keep


def density_ratio_weights(X_cal: np.ndarray, X_test: np.ndarray, C: float = 1.0,
                          w_min: float = 0.05, w_max: float = 20.0, n_continuous: int = N_CONTINUOUS,
                          min_support: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Clipped density-ratio weights ``(w_cal, w_test)`` from a logistic domain classifier."""
    if len(X_cal) == 0 or len(X_test) == 0:
        return np.ones(len(X_cal)), np.ones(len(X_test))
    cols = supported_columns(X_cal, X_test, min(n_continuous, X_cal.shape[1]), min_support)
    X = np.vstack([X_cal[:, cols], X_test[:, cols]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing column -> 0
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    keep = X.std(axis=0) > 1e-12
    if not keep.any():
        return np.ones(len(X_cal)), np.ones(len(X_test))
    X = StandardScaler().fit_transform(X[:, keep])
    y = np.r_[np.zeros(len(X_cal)), np.ones(len(X_test))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p) * (len(X_cal) / len(X_test))
    w = np.clip(w, w_min, w_max)
    return w[: len(X_cal)], w[len(X_cal):]


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size of calibration weights."""
    w = w[np.isfinite(w)]
    return float(w.sum() ** 2 / np.sum(w**2)) if w.size else 0.0
