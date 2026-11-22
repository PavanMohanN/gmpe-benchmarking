"""Calibration hold-out carved from a training fold.

Whole groups (events or stations) move to the calibration set until it holds
at least ``cal_frac`` of the training records. With no group key (the random
scheme) records are sampled individually.
"""

from __future__ import annotations

import numpy as np

from gmpe_bench.data import GMDataset


def carve_calibration(ds: GMDataset, train_idx: np.ndarray, group_key: str | None,
                      cal_frac: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split ``train_idx`` into ``(train_fit, cal)``; groups never straddle the two."""
    if not 0.0 < cal_frac < 1.0:
        raise ValueError(f"cal_frac must be in (0, 1), got {cal_frac}")
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    if train_idx.size == 0:
        return train_idx, train_idx
    rng = np.random.default_rng(seed)
    target = cal_frac * train_idx.size

    if group_key is None:
        n_cal = int(round(target))
        cal = np.sort(rng.choice(train_idx, size=n_cal, replace=False))
        return np.setdiff1d(train_idx, cal), cal

    keys = ds.df[group_key].to_numpy()[train_idx]
    groups, counts = np.unique(keys, return_counts=True)
    order = rng.permutation(groups.size)
    cum = np.cumsum(counts[order])
    n_take = int(np.searchsorted(cum, target) + 1)
    n_take = min(n_take, groups.size - 1) if groups.size > 1 else 0
    cal_groups = groups[order[:n_take]]
    in_cal = np.isin(keys, cal_groups)
    return train_idx[~in_cal], train_idx[in_cal]
