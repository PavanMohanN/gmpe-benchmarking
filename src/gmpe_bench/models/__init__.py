"""ML and deep-learning spectral GMMs (all 27 IMs at once)."""

from __future__ import annotations

from gmpe_bench.models.base import BaseSpectralModel, target_matrix
from gmpe_bench.models.evaluate import evaluate_ml_models, make_ml_model
from gmpe_bench.models.features import FeaturePipeline
from gmpe_bench.models.gbdt import XGBConfig, XGBoostSpectralGMM
from gmpe_bench.models.hybrid import ResidualHybridGMM
from gmpe_bench.models.spectral_mlp import MLPConfig, SpectralMLP

__all__ = [
    "BaseSpectralModel",
    "FeaturePipeline",
    "MLPConfig",
    "ResidualHybridGMM",
    "SpectralMLP",
    "XGBConfig",
    "XGBoostSpectralGMM",
    "evaluate_ml_models",
    "make_ml_model",
    "target_matrix",
]
