"""One call builds every paper artifact and a SHA-256 manifest of inputs and outputs."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib
import pandas as pd

from gmpe_bench import io
from gmpe_bench.reporting.figures import Figure, build_all_figures, write_captions
from gmpe_bench.reporting.inputs import ReportConfig, ReportInputs
from gmpe_bench.reporting.numbers import Claim, build_numbers, claims_summary, write_numbers
from gmpe_bench.reporting.summary import build_summary, write_summary
from gmpe_bench.reporting.tables import build_all_tables


@dataclass
class PaperArtifacts:
    tables: dict[str, list[Path]]
    figures: dict[str, Figure]
    captions: list[Path]
    manifest: Path
    skipped: dict[str, str]
    numbers: list[Path] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    summary: Path | None = None


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("gmpe-bench")
    except Exception:
        return "unknown"


def build_paper_artifacts(cfg: ReportConfig, log: Callable[[str], None] = print) -> PaperArtifacts:
    """Tables, figures, captions, prose numbers, claim checks, bootstrap intervals,
    ``paper_summary_numbers.json`` and ``paper_manifest.json``."""
    inp = ReportInputs(cfg, log=log)
    tables, skip_t, table_objs = build_all_tables(inp, log)
    figures, skip_f = build_all_figures(inp, log)
    captions = write_captions(figures, table_objs, cfg.paper_dir)
    book = build_numbers(inp, log)
    numbers = write_numbers(book, cfg.paper_dir, inp)
    log(f"numbers: {len(book.text)} values, {len(book.missing)} missing; claims: {claims_summary(book)}")
    summary = write_summary(build_summary(inp, table_objs, book), cfg.paper_dir)
    log(f"summary: {summary}")
    outputs = ([p for ps in tables.values() for p in ps] + [p for f in figures.values() for p in f.paths]
               + captions + numbers + [summary])
    root = Path(cfg.root).resolve()

    def rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(root)).replace("\\", "/")
        except ValueError:
            return str(p)

    manifest = {
        "package": "gmpe-bench", "version": _version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "config": {"profile": cfg.profile, "compare_profile": cfg.compare_profile, "alpha": cfg.alpha,
                   "accuracy_scheme": cfg.accuracy_scheme, "example_model": cfg.example_model,
                   "example_schemes": list(cfg.example_schemes), "site_models": list(cfg.site_models),
                   "bootstrap": {"b": cfg.bootstrap_b, "seed": cfg.bootstrap_seed, "level": cfg.bootstrap_level,
                                 "cluster": "eqid", "pair_models": list(cfg.pair_models),
                                 "station_models": list(cfg.station_models)}},
        "inputs": {rel(p): io.sha256_file(p) for p in sorted(inp.used)},
        "outputs": {rel(p): io.sha256_file(p) for p in sorted(outputs)},
        "skipped": {**skip_t, **skip_f, **{f"numbers:{k}": v for k, v in book.skipped.items()},
                    **{f"bootstrap:{k}": v for k, v in inp.bootstrap.skipped.items()}},
        "bootstrap_checks": {k: v for k, v in inp.bootstrap.meta.items() if k.endswith(("table3", "table4"))},
        "claims": {c.cid: c.status for c in book.claims},
    }
    path = cfg.paper_dir / "paper_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PaperArtifacts(tables, figures, captions, path, {**skip_t, **skip_f}, numbers, book.claims, summary)
