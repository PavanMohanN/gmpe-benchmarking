"""Command line: ``gmpe-bench selftest | info | build | split | baselines | train | conformal | paper``.

``selftest`` also runs an installation smoke test on a tiny in-memory synthetic fixture (not NGA-West2).
"""

from __future__ import annotations

import importlib
import platform
import sys
import tempfile
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="GMPE benchmark: data build, splits and audits.")

DATA_DIR = Path("data") / "processed"
CORE_DEPS = ("numpy", "pandas", "pyarrow", "openpyxl", "sklearn", "typer")
MIN_PY = (3, 10)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("gmpe-bench")
    except Exception:
        return "unknown"


@app.command()
def selftest(
    smoke: bool = typer.Option(True, "--smoke/--no-smoke",
                               help="Also run the installation smoke test on a tiny in-memory synthetic fixture."),
) -> None:
    """Check Python, core dependencies and write access, then run an installation smoke test.

    The smoke test uses a tiny synthetic fixture held in memory (no PEER files needed). It only checks
    that the leakage audit and the conformal calibrators execute; its numbers are not NGA-West2 results.
    """
    rows: list[tuple[str, bool, str]] = []
    ok_py = sys.version_info[:2] >= MIN_PY
    rows.append(("python", ok_py, platform.python_version()))
    for mod in CORE_DEPS:
        try:
            m = importlib.import_module(mod)
            rows.append((mod, True, getattr(m, "__version__", "?")))
        except ImportError as exc:
            rows.append((mod, False, str(exc)))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "probe.txt").write_text("ok", encoding="utf-8")
        rows.append(("write access", True, "temp folder"))
    except OSError as exc:
        rows.append(("write access", False, str(exc)))
    rows.append(("platform", True, platform.platform()))

    for name, ok, detail in rows:
        typer.echo(f"{'PASS' if ok else 'FAIL'}  {name:<13} {detail}")
    if not all(ok for _, ok, _ in rows):
        raise typer.Exit(code=1)
    if smoke and not _smoke_test():
        raise typer.Exit(code=1)
    typer.echo("selftest passed")


def _smoke_test() -> bool:
    """Installation smoke test on the in-memory fixture; prints a banner and PASS/FAIL lines."""
    import time

    from gmpe_bench._smoke import SMOKE_BANNER, run_smoke_test

    bar = "=" * 78
    typer.echo(bar)
    typer.echo(SMOKE_BANNER)
    typer.echo(bar)
    t0 = time.perf_counter()
    try:
        r = run_smoke_test(log=lambda s: None)
    except Exception as exc:  # report any failure as a clean FAIL line
        typer.echo(f"FAIL  smoke test    {type(exc).__name__}: {exc}")
        return False
    cov = r["coverage"]
    checks = [
        ("fixture", True, f"{r['records']} records, {r['events']} synthetic earthquakes (in memory)"),
        ("leakage audit", bool(r["audit_passed"]) and r["event_split_shared_events"] == 0,
         f"{r['folds_audited']} folds, no earthquake shared across Event-split parts"),
        ("model fit", r["rmse"] == r["rmse"], f"LinearGMM, fixture RMSE {r['rmse']:.2f}"),
        ("conformal", all(0.5 < c <= 1.0 for c in cov.values()),
         ", ".join(f"{m} cov {c:.2f}" for m, c in cov.items())),
    ]
    for name, ok, detail in checks:
        typer.echo(f"{'PASS' if ok else 'FAIL'}  {name:<13} {detail}")
    typer.echo(f"smoke test finished in {time.perf_counter() - t0:.1f} s (fixture values are not benchmark results)")
    return all(ok for _, ok, _ in checks)


@app.command()
def info() -> None:
    """Version, platform and default data folder."""
    typer.echo(f"gmpe-bench {_version()}")
    typer.echo(f"python     {platform.python_version()} ({platform.platform()})")
    typer.echo(f"data dir   {DATA_DIR.resolve()}")


@app.command()
def build(
    h: Path = typer.Option(Path("nga_w_h.xlsx"), "--h", help="Horizontal flatfile (.xlsx or .csv)."),
    v: Path = typer.Option(Path("nga_w_v.xlsx"), "--v", help="Vertical flatfile (.xlsx or .csv)."),
    out: Path = typer.Option(DATA_DIR, "--out", help="Output folder."),
    profile: str = typer.Option("both", "--profile", help="full, sp23 or both."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-read the Excel files."),
) -> None:
    """Build the cleaned NGA-West2 dataset (rules R1-R12)."""
    from gmpe_bench.data import build_dataset

    res = build_dataset(h, v, out, profile=profile, cache=not no_cache)
    for prof, c in res.counts.items():
        typer.echo(f"{prof:<5} {c['records']:>6,} records  {c['events']:>4} events  {c['stations']:>5,} stations")
    typer.echo(f"Wrote {', '.join(p.name for p in res.outputs.values())} and {res.manifest_path.name} to {out}")


@app.command()
def split(
    data: Path = typer.Option(DATA_DIR, "--data", help="Folder with full.parquet and sp23.parquet."),
    profiles: str = typer.Option("full,sp23", "--profiles", help="Comma-separated profiles."),
    seed: int = typer.Option(42, "--seed"),
    cal_frac: float = typer.Option(0.25, "--cal-frac", help="Calibration share of each training fold."),
) -> None:
    """Make, audit and freeze all split schemes."""
    from gmpe_bench.splits import freeze_all, leakage_table

    profs = tuple(p.strip() for p in profiles.split(",") if p.strip())
    reports, files = freeze_all(data, profs, seed=seed, cal_frac=cal_frac)
    df = leakage_table(reports)
    summary = df.groupby(["profile", "scheme"], sort=False).agg(
        folds=("fold", "size"), test_mean=("n_test", "mean"),
        seen_event=("test_records_seen_event", "mean"),
        seen_station=("test_records_seen_station", "mean"), passed=("passed", "all"))
    typer.echo(summary.round(3).to_string())
    typer.echo(f"Wrote {', '.join(f.name for f in files.values())} to {data}")
    if not df["passed"].all():
        typer.echo("Leakage audit FAILED; see split_summary.csv", err=True)
        raise typer.Exit(code=1)


@app.command()
def baselines(
    data: Path = typer.Option(DATA_DIR, "--data", help="Folder with profiles and frozen splits."),
    out: Path = typer.Option(Path("results") / "baselines", "--out", help="Output folder."),
    profiles: str = typer.Option("full,sp23", "--profiles"),
    schemes: str = typer.Option("event_grouped,station_grouped,spatial_block,temporal,"
                                "magnitude_holdout,cl1_to_cl2", "--schemes"),
    models: str = typer.Option("ASK14,BSSA14,CB14,CY14,MixedEffects", "--models"),
    ims: str = typer.Option("", "--ims", help="Comma-separated IMs; default all."),
    no_bias: bool = typer.Option(False, "--no-bias", help="Raw NGA-West2 medians, no bias term."),
) -> None:
    """Evaluate classical GMM baselines out of fold on frozen splits."""
    from gmpe_bench.baselines import evaluate_baselines

    def parts(s: str) -> tuple[str, ...]:
        return tuple(x.strip() for x in s.split(",") if x.strip())

    metrics, oof = evaluate_baselines(data, out, profiles=parts(profiles), schemes=parts(schemes),
                                      models=parts(models), ims=list(parts(ims)) or None,
                                      calibrate_bias=not no_bias, log=typer.echo)
    summary = (metrics[metrics["mode"] == "ergodic"]
               .groupby(["profile", "scheme", "model"], sort=False)[["rmse", "bias", "cov90"]].mean())
    typer.echo(summary.round(3).to_string())
    typer.echo(f"Wrote baseline_metrics.csv ({len(metrics):,} rows) and {oof.name} to {out}")


@app.command()
def train(
    data: Path = typer.Option(DATA_DIR, "--data", help="Folder with profiles and frozen splits."),
    out: Path = typer.Option(Path("results") / "ml", "--out", help="Output folder."),
    physics_cache: Path = typer.Option(Path("results") / "baselines" / "cache", "--physics-cache",
                                       help="Phase 3 pygmm cache used by hybrid models."),
    profiles: str = typer.Option("full,sp23", "--profiles"),
    schemes: str = typer.Option("event_grouped,station_grouped,spatial_block,temporal,"
                                "magnitude_holdout,cl1_to_cl2", "--schemes"),
    models: str = typer.Option("XGBoost,SpectralMLP,Hybrid_ASK14_MLP,Hybrid_BSSA14_XGB", "--models"),
    fast: bool = typer.Option(False, "--fast", help="Small models, few epochs/trees."),
    no_station_encoding: bool = typer.Option(False, "--no-station-encoding", help="Ergodic features only."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Train ML/DL spectral GMMs on frozen splits; save test and cal predictions."""
    from gmpe_bench.models import evaluate_ml_models

    def parts(s: str) -> tuple[str, ...]:
        return tuple(x.strip() for x in s.split(",") if x.strip())

    metrics, oof, cal = evaluate_ml_models(data, out, profiles=parts(profiles), schemes=parts(schemes),
                                           models=parts(models), physics_cache=physics_cache,
                                           station_encoding=not no_station_encoding, fast=fast, seed=seed,
                                           log=typer.echo)
    test = metrics[metrics["part"] == "test"]
    typer.echo(test.groupby(["profile", "scheme", "model"], sort=False)[["rmse", "nll", "cov90"]]
               .mean().round(3).to_string())
    typer.echo(f"Wrote ml_metrics.csv, {oof.name} and {cal.name} to {out}")


@app.command()
def conformal(
    data: Path = typer.Option(DATA_DIR, "--data", help="Folder with profiles and frozen splits."),
    out: Path = typer.Option(Path("results") / "conformal", "--out", help="Output folder."),
    ml_dir: Path = typer.Option(Path("results") / "ml", "--ml-dir", help="Phase 4 predictions."),
    baseline_cache: Path = typer.Option(Path("results") / "baselines" / "cache", "--baseline-cache",
                                        help="Phase 3 pygmm cache."),
    profiles: str = typer.Option("full,sp23", "--profiles"),
    schemes: str = typer.Option("event_grouped,station_grouped,spatial_block,temporal,"
                                "magnitude_holdout,cl1_to_cl2", "--schemes"),
    models: str = typer.Option("", "--models", help="Comma-separated; default all found."),
    ablations: bool = typer.Option(False, "--ablations", help="Also run HSCP component ablations."),
    event_counts_only: bool = typer.Option(
        False, "--event-counts-only",
        help="Write only results/conformal/event_counts.parquet (per-earthquake coverage counts for the "
             "bootstrap); the existing Phase 5 CSVs are not touched and no model is refitted."),
) -> None:
    """Calibrate uncertainty bands (Gaussian, split, clustered, shift-weighted, HSCP)."""
    from gmpe_bench.conformal import default_methods, evaluate_conformal

    def parts(s: str) -> tuple[str, ...]:
        return tuple(x.strip() for x in s.split(",") if x.strip())

    metrics, _, _ = evaluate_conformal(data, out, ml_dir=ml_dir, baseline_cache=baseline_cache,
                                       profiles=parts(profiles), schemes=parts(schemes),
                                       models=parts(models) or None, methods=default_methods(ablations),
                                       log=typer.echo, event_counts=True, metrics=not event_counts_only)
    if event_counts_only:
        typer.echo(f"Wrote event_counts.parquet to {out}")
        return
    allim = metrics[(metrics["im"] == "ALL") & (metrics["alpha"] == 0.10)]
    typer.echo(allim.groupby(["scheme", "method"], sort=False)[["coverage", "width"]].mean().round(3).to_string())
    typer.echo("Wrote conformal_metrics.csv, conditional_coverage.csv, conformal_intervals_summary.parquet, "
               f"event_counts.parquet to {out}")


@app.command()
def paper(
    root: Path = typer.Option(Path("."), "--root", help="Repository folder (contains data/ and results/)."),
    out: Path = typer.Option(None, "--out", help="Default: <root>/results/paper."),
    profile: str = typer.Option("full", "--profile"),
    compare_profile: str = typer.Option("sp23", "--compare-profile"),
    alpha: float = typer.Option(0.10, "--alpha"),
    example_model: str = typer.Option("Hybrid_ASK14_MLP", "--example-model"),
    bootstrap_b: int = typer.Option(1000, "--bootstrap-b", help="Event-clustered bootstrap replicates."),
    bootstrap_seed: int = typer.Option(20261122, "--bootstrap-seed"),
) -> None:
    """Build paper tables, figures, captions, numbers.tex, claim checks, bootstrap CIs,
    paper_summary_numbers.json and the manifest."""
    from gmpe_bench.reporting import ReportConfig, build_paper_artifacts
    from gmpe_bench.reporting.figures import MAIN_FIGURES
    from gmpe_bench.reporting.tables import MAIN_TABLES

    cfg = ReportConfig(root=root, profile=profile, compare_profile=compare_profile, alpha=alpha,
                       example_model=example_model, bootstrap_b=bootstrap_b, bootstrap_seed=bootstrap_seed,
                       out_dir=out)
    art = build_paper_artifacts(cfg, log=typer.echo)
    typer.echo(f"{len(art.tables)} tables, {len(art.figures)} figures; manifest {art.manifest}")
    for c in art.claims:
        if c.status != "PASS":
            typer.echo(f"claim {c.cid} {c.status}: {c.statement} [{c.detail}]")
    if any(n in MAIN_TABLES or n in MAIN_FIGURES for n in art.skipped):
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
