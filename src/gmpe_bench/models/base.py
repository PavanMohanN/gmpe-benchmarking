"""Common interface for multi-output (spectral) ML ground-motion models.

All 27 IMs are predicted at once:

    model.fit(df_train_fit)
    mu, sigma = model.predict(df)        # each shape (n_samples, 27), natural-log units

An optional ``offset`` (n, 27) is subtracted from the targets before fitting and
added back to predictions: this is how the residual hybrid models reuse a
physics median. Targets are ln SA [g], ln PGV [cm/s]; NaN where masked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from gmpe_bench.data.targets import get_targets
from gmpe_bench.schema import IM_NAMES, period_of

N_IMS = len(IM_NAMES)


def target_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n, 27) ln targets for the horizontal component; NaN where masked."""
    return get_targets(df, "H").to_numpy(dtype=float)


def sa_columns() -> np.ndarray:
    """Positions of the SA IMs in IM_NAMES (PGA and PGV excluded)."""
    return np.array([i for i, n in enumerate(IM_NAMES) if period_of(n) is not None], dtype=int)


def grouped_holdout(groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask: whole groups (events) held out for internal validation."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * uniq.size))) if uniq.size > 1 else 0
    val_groups = rng.choice(uniq, size=n_val, replace=False)
    return np.isin(groups, val_groups)


class BaseSpectralModel(ABC):
    """Abstract multi-output GMM."""

    name: str = "base"
    ims: list[str] = list(IM_NAMES)

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> BaseSpectralModel:
        """Fit on ``train_fit`` records only."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mu, sigma)``, each of shape ``(len(df), 27)``."""

    @staticmethod
    def _targets(df: pd.DataFrame, offset: np.ndarray | None) -> np.ndarray:
        y = target_matrix(df)
        return y if offset is None else y - offset

    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
