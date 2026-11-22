"""Benchmark splits, calibration hold-outs and the leakage detector."""

from __future__ import annotations

from gmpe_bench.splits.calibration import carve_calibration
from gmpe_bench.splits.freeze import (
    freeze_all,
    load_splits,
    save_splits,
    write_split_manifest,
    write_summary,
)
from gmpe_bench.splits.group_splits import (
    SCHEMES,
    SplitFold,
    SplitScheme,
    get_scheme,
    make_splits,
)
from gmpe_bench.splits.leakage import (
    LeakageReport,
    assert_fit_transform_clean,
    audit_splits,
    leakage_table,
)

__all__ = [
    "SCHEMES",
    "LeakageReport",
    "SplitFold",
    "SplitScheme",
    "assert_fit_transform_clean",
    "audit_splits",
    "carve_calibration",
    "freeze_all",
    "get_scheme",
    "leakage_table",
    "load_splits",
    "make_splits",
    "save_splits",
    "write_split_manifest",
    "write_summary",
]
