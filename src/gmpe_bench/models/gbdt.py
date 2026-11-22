"""XGBoostSpectralGMM: one booster for the median across all IMs, one for the scale.

Long format: each (record, IM) pair is a row, with ln T and PGA/PGV flags as
extra inputs, so the trees share strength across periods and masked
long-period values are simply absent rows.

* Median: ``reg:squarederror``; the tree count is set by early stopping on an
  event-grouped internal validation split of ``train_fit``.
* Scale: event-grouped K-fold *inside* ``train_fit`` gives out-of-fold residuals;
  a second booster fits ``ln r^2``. Since E[ln r^2] = ln sigma^2 - 1.2704 for
  Gaussian r, sigma = sqrt(exp(prediction + 1.2704)).

``device="cuda"`` is used when XGBoost was built with CUDA and a GPU is visible;
any GPU failure falls back to CPU.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from gmpe_bench.models.base import N_IMS, BaseSpectralModel, grouped_holdout
from gmpe_bench.models.features import FeaturePipeline
from gmpe_bench.schema import IM_NAMES, period_of

LN_R2_SHIFT = 1.2703628454614782  # -(digamma(1/2) + ln 2)
SIGMA_MIN, SIGMA_MAX = 0.05, 2.5


def _im_features() -> np.ndarray:
    """(27, 3): ln T (PGA at 0.005 s, PGV at 1 s), is_pga, is_pgv."""
    rows = []
    for name in IM_NAMES:
        p = period_of(name)
        if name == "PGA":
            rows.append([np.log(0.005), 1.0, 0.0])
        elif name == "PGV":
            rows.append([np.log(1.0), 0.0, 1.0])
        else:
            rows.append([np.log(p), 0.0, 0.0])
    return np.asarray(rows, dtype=np.float32)


IM_FEATURES = _im_features()


def to_long(X: np.ndarray, Y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Stack records x IMs. Returns (X_long, y_long, record_index, im_index); NaN targets dropped."""
    n = X.shape[0]
    rec = np.repeat(np.arange(n), N_IMS)
    imi = np.tile(np.arange(N_IMS), n)
    Xl = np.column_stack([X[rec], IM_FEATURES[imi]]).astype(np.float32)
    if Y is None:
        return Xl, None, rec, imi
    y = Y.reshape(-1)
    keep = np.isfinite(y)
    return Xl[keep], y[keep].astype(np.float32), rec[keep], imi[keep]


def resolve_device(preferred: str = "auto") -> str:
    if preferred in ("cpu", "cuda"):
        return preferred
    try:
        if not xgb.build_info().get("USE_CUDA", False):
            return "cpu"
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


@dataclass
class XGBConfig:
    n_estimators: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 8
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 100
    val_frac: float = 0.15
    scale_folds: int = 3
    scale_estimators: int = 400
    scale_max_depth: int = 5
    device: str = "auto"
    extra: dict[str, object] = field(default_factory=dict)


class XGBoostSpectralGMM(BaseSpectralModel):
    name = "XGBoost"

    def __init__(self, config: XGBConfig | None = None, station_encoding: bool = True, seed: int = 0) -> None:
        super().__init__(seed)
        self.config = config or XGBConfig()
        self.pipeline = FeaturePipeline(station_encoding=station_encoding, seed=seed)
        self.device = resolve_device(self.config.device)
        self.mean_model: xgb.XGBRegressor | None = None
        self.scale_model: xgb.XGBRegressor | None = None
        self.best_iteration: int | None = None

    def _regressor(self, n_estimators: int, max_depth: int, early: int | None) -> xgb.XGBRegressor:
        c = self.config
        return xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=c.learning_rate, max_depth=max_depth,
            min_child_weight=c.min_child_weight, subsample=c.subsample, colsample_bytree=c.colsample_bytree,
            reg_lambda=c.reg_lambda, tree_method="hist", device=self.device, random_state=self.seed,
            early_stopping_rounds=early, n_jobs=-1, **c.extra,
        )

    def _fit_reg(self, n_estimators: int, max_depth: int, X: np.ndarray, y: np.ndarray,
                 eval_set: tuple[np.ndarray, np.ndarray] | None = None) -> xgb.XGBRegressor:
        """Fit, retrying on CPU if the GPU path fails."""
        early = self.config.early_stopping_rounds if eval_set is not None else None
        for attempt in range(2):
            model = self._regressor(n_estimators, max_depth, early)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X, y, eval_set=[eval_set] if eval_set else None, verbose=False)
                return model
            except xgb.core.XGBoostError:
                if self.device == "cpu" or attempt:
                    raise
                self.device = "cpu"
        raise RuntimeError("unreachable")

    def fit(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> XGBoostSpectralGMM:
        c = self.config
        Y = self._targets(df, offset)
        X = self.pipeline.fit_transform(df)
        eq = df["eqid"].to_numpy()

        # 1. tree count by early stopping on held-out events of train_fit
        val = grouped_holdout(eq, c.val_frac, self.seed)
        Xa, ya, _, _ = to_long(X[~val], Y[~val])
        Xv, yv, _, _ = to_long(X[val], Y[val])
        probe = self._fit_reg(c.n_estimators, c.max_depth, Xa, ya, (Xv, yv))
        self.best_iteration = int(getattr(probe, "best_iteration", c.n_estimators - 1)) + 1

        # 2. final median model on all of train_fit
        Xl, yl, _, _ = to_long(X, Y)
        self.mean_model = self._fit_reg(self.best_iteration, c.max_depth, Xl, yl)

        # 3. out-of-fold residuals inside train_fit -> scale model
        events = np.unique(eq)
        rng = np.random.default_rng(self.seed + 1)
        fold = pd.Series(rng.permutation(events.size) % c.scale_folds, index=events).reindex(eq).to_numpy()
        oof = np.full_like(Y, np.nan)
        for k in range(c.scale_folds):
            inner, outer = fold != k, fold == k
            Xi, yi, _, _ = to_long(X[inner], Y[inner])
            m = self._fit_reg(self.best_iteration, c.max_depth, Xi, yi)
            Xo, _, _, _ = to_long(X[outer])
            oof[outer] = m.predict(Xo).reshape(-1, N_IMS)
        Xs, ls, _, _ = to_long(X, np.log((Y - oof) ** 2 + 1e-8))
        self.scale_model = self._fit_reg(c.scale_estimators, c.scale_max_depth, Xs, ls)
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._check()
        assert self.mean_model is not None and self.scale_model is not None
        Xl, _, _, _ = to_long(self.pipeline.transform(df))
        mu = self.mean_model.predict(Xl).reshape(-1, N_IMS).astype(float)
        ln_r2 = self.scale_model.predict(Xl).reshape(-1, N_IMS).astype(float)
        sigma = np.clip(np.sqrt(np.exp(ln_r2 + LN_R2_SHIFT)), SIGMA_MIN, SIGMA_MAX)
        if offset is not None:
            mu = mu + offset
        return mu, sigma
