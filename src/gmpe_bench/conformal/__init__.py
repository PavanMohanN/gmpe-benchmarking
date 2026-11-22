"""Uncertainty calibration: from uncalibrated Gaussian bands to HSCP."""

from __future__ import annotations

from gmpe_bench.conformal.evaluate import classical_predictions, evaluate_conformal
from gmpe_bench.conformal.hierarchy import CrossedTerms, crossed_terms
from gmpe_bench.conformal.methods import (
    HSCP,
    CalSet,
    ClusteredEventConformal,
    ConformalCalibrator,
    CovariateShiftConformal,
    HierarchicalSpectralConformal,
    Intervals,
    NaiveSplitConformal,
    SplitConformal,
    TestInputs,
    UncalibratedGaussian,
    WeightedShiftConformal,
    default_methods,
)
from gmpe_bench.conformal.scores import (
    abs_scores,
    conformal_level,
    effective_n,
    normalized_scores,
    split_conformal_quantile,
    weighted_conformal_quantile,
)
from gmpe_bench.conformal.shift import density_ratio_weights, effective_sample_size, shift_features

__all__ = [
    "HSCP",
    "CalSet",
    "ClusteredEventConformal",
    "ConformalCalibrator",
    "CovariateShiftConformal",
    "CrossedTerms",
    "HierarchicalSpectralConformal",
    "Intervals",
    "NaiveSplitConformal",
    "SplitConformal",
    "TestInputs",
    "UncalibratedGaussian",
    "WeightedShiftConformal",
    "abs_scores",
    "classical_predictions",
    "conformal_level",
    "crossed_terms",
    "default_methods",
    "density_ratio_weights",
    "effective_n",
    "effective_sample_size",
    "evaluate_conformal",
    "normalized_scores",
    "shift_features",
    "split_conformal_quantile",
    "weighted_conformal_quantile",
]
