"""Publication tables, figures, captions, prose numbers and the reproducibility manifest."""

from __future__ import annotations

from gmpe_bench.reporting.figures import build_all_figures, write_captions
from gmpe_bench.reporting.inputs import MissingInput, ReportConfig, ReportInputs
from gmpe_bench.reporting.numbers import NumberBook, build_numbers, write_numbers
from gmpe_bench.reporting.paper import PaperArtifacts, build_paper_artifacts
from gmpe_bench.reporting.tables import build_all_tables, to_booktabs

__all__ = [
    "MissingInput",
    "NumberBook",
    "PaperArtifacts",
    "ReportConfig",
    "ReportInputs",
    "build_all_figures",
    "build_all_tables",
    "build_numbers",
    "build_paper_artifacts",
    "to_booktabs",
    "write_captions",
    "write_numbers",
]
