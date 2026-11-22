"""ResidualHybridGMM: physics median from an NGA-West2 model + learned residual correction.

    mu(T) = mu_phys(T) + delta_mu(T),   sigma(T) from the corrector

``mu_phys`` is read from the Phase 3 pygmm cache (or computed on the fly).
The corrector (SpectralMLP or XGBoost) is trained on y - mu_phys using
``train_fit`` only. Magnitude and distance scaling therefore come from the
published model; the corrector learns site, basin and regional departures.

``max_correction`` caps |delta_mu| in ln units (default 2.0 = a factor of about 7.4),
so a corrector extrapolating badly at extreme Mw or distance cannot overturn
the physics median.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gmpe_bench.baselines.ngawest2 import MODEL_CLASSES, cached_predict_table
from gmpe_bench.models.base import BaseSpectralModel
from gmpe_bench.models.gbdt import XGBConfig, XGBoostSpectralGMM
from gmpe_bench.models.spectral_mlp import MLPConfig, SpectralMLP
from gmpe_bench.schema import IM_NAMES


class ResidualHybridGMM(BaseSpectralModel):
    def __init__(self, physics_model: str = "ASK14", corrector: str = "mlp",
                 cache_dir: Path | None = None, station_encoding: bool = True,
                 max_correction: float | None = 2.0, mlp_config: MLPConfig | None = None,
                 xgb_config: XGBConfig | None = None, seed: int = 0) -> None:
        super().__init__(seed)
        if physics_model not in MODEL_CLASSES:
            raise KeyError(f"Unknown physics model {physics_model!r}; choose from {list(MODEL_CLASSES)}")
        self.physics_model = physics_model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_correction = max_correction
        if corrector == "mlp":
            self.corrector: BaseSpectralModel = SpectralMLP(mlp_config, station_encoding, seed)
        elif corrector == "xgb":
            self.corrector = XGBoostSpectralGMM(xgb_config, station_encoding, seed)
        else:
            raise ValueError("corrector must be 'mlp' or 'xgb'")
        self.name = f"Hybrid_{physics_model}_{corrector.upper()}"

    def physics_median(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 27) ln median of the physics model, in IM_NAMES order."""
        table = cached_predict_table(df, self.physics_model, self.cache_dir)
        return table[[f"mu_{n}" for n in IM_NAMES]].to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> ResidualHybridGMM:
        phys = self.physics_median(df)
        self.corrector.fit(df, offset=phys if offset is None else phys + offset)
        self.fitted = True
        return self

    def correction(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Learned ``(delta_mu, sigma)`` before adding the physics median."""
        self._check()
        delta, sigma = self.corrector.predict(df)
        if self.max_correction is not None:
            delta = np.clip(delta, -self.max_correction, self.max_correction)
        return delta, sigma

    def predict(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        delta, sigma = self.correction(df)
        mu = self.physics_median(df) + delta
        if offset is not None:
            mu = mu + offset
        return mu, sigma
