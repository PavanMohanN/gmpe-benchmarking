"""Common interface for every ground-motion model in the benchmark.

A model is fitted for one intensity measure (IM) at a time:

    model.fit(df_train, "SA_1s")
    mu_ln, sigma_ln = model.predict(df_test)

Outputs are natural-log units: ln(g) for PGA and SA, ln(cm/s) for PGV.
``tau`` and ``phi`` are the between- and within-event standard deviations
when the model provides them, otherwise ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from gmpe_bench.schema import IM_NAMES, im_column


def target(df: pd.DataFrame, im: str, component: str = "H") -> np.ndarray:
    """ln observation for one IM; NaN where masked or non-positive."""
    if im not in IM_NAMES:
        raise KeyError(f"Unknown IM {im!r}")
    v = pd.to_numeric(df[im_column(component, im)], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.log(v), np.nan)


class BaseGMM(ABC):
    """Abstract ground-motion model."""

    name: str = "base"

    def __init__(self) -> None:
        self.im: str | None = None
        self.tau: float | None = None
        self.phi: float | None = None

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, im: str) -> BaseGMM:
        """Fit (or calibrate) on ``train_fit`` records for one IM."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mu_ln, sigma_ln)`` arrays of length ``len(df)``."""

    def _check_fitted(self) -> str:
        if self.im is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        return self.im

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, im={self.im!r})"
