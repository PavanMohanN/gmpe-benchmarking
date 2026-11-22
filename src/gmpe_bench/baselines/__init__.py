"""Classical GMM baselines and leakage-free residual decomposition."""

from __future__ import annotations

from gmpe_bench.baselines.base import BaseGMM, target
from gmpe_bench.baselines.evaluate import evaluate_baselines, make_model, score
from gmpe_bench.baselines.mixed_effects import MixedEffectsGMM
from gmpe_bench.baselines.ngawest2 import MODEL_CLASSES, NGAWest2Baseline, predict_table, scenario_kwargs
from gmpe_bench.baselines.residuals import ResidualDecomposition, apply_site_terms, decompose_residuals

__all__ = [
    "MODEL_CLASSES",
    "BaseGMM",
    "MixedEffectsGMM",
    "NGAWest2Baseline",
    "ResidualDecomposition",
    "apply_site_terms",
    "decompose_residuals",
    "evaluate_baselines",
    "make_model",
    "predict_table",
    "scenario_kwargs",
    "score",
    "target",
]
