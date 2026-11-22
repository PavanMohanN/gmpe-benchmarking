"""GMPE-Bench: Leakage-free evaluation and hierarchical spectral conformal prediction for GMMs.

Quick start (after building the NGA-West2 profile with ``gmpe-bench build``)::

    import gmpe_bench as gb
    report = gb.evaluate(gb.LinearGMM(), profile="full")
    print(report.summary())

Check an installation without the PEER files with ``gmpe-bench selftest``.

The high-level API lives in :mod:`gmpe_bench.api` and is imported lazily, so ``import gmpe_bench``
and the command-line tool stay fast.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

_API = ("BaseGMM", "BenchmarkReport", "DataNotFoundError", "FoldData", "LeakageError", "LinearGMM",
        "PROFILES", "evaluate", "evaluate_predictions", "iter_folds", "load_benchmark", "load_reference",
        "ALL_SCHEMES", "LEAKAGE_FREE_SCHEMES")
__all__ = ["ALL_SCHEMES", "IM_NAMES", "LEAKAGE_FREE_SCHEMES", "PROFILES", "BaseGMM", "BenchmarkReport",
           "DataNotFoundError", "FoldData", "LeakageError", "LinearGMM", "__version__", "evaluate",
           "evaluate_predictions", "iter_folds", "load_benchmark", "load_reference"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy exports
    if name in _API:
        from gmpe_bench import api

        return getattr(api, name)
    if name == "IM_NAMES":
        from gmpe_bench.schema import IM_NAMES

        return IM_NAMES
    raise AttributeError(f"module 'gmpe_bench' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
