"""Phase 5 gate: calibrate uncertainty bands for every model and print the ablation table.

Usage (from the repository root):
    python scripts/04_run_conformal.py
    python scripts/04_run_conformal.py --ablations                  # + HSCP component ablations
    python scripts/04_run_conformal.py --profiles full --models Hybrid_ASK14_MLP,ASK14_cal
    python scripts/04_run_conformal.py --event-counts-only --profiles full   # add bootstrap counts only

Reads data/processed (profiles, splits), results/ml (Phase 4 test + cal predictions) and
results/baselines/cache (Phase 3). Writes results/conformal/:
conformal_metrics.csv, conditional_coverage.csv, conformal_intervals_summary.parquet and
event_counts.parquet (covered / usable counts per test earthquake, for the event-clustered bootstrap).
--event-counts-only recomputes the intervals from the stored predictions and writes only
event_counts.parquet; the other three files and every model are left untouched.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from gmpe_bench.conformal import default_methods, evaluate_conformal
from gmpe_bench.conformal.evaluate import DEFAULT_SCHEMES

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["UncalibratedGaussian", "NaiveSplitConformal", "ClusteredEventConformal",
         "WeightedShiftConformal", "HSCP", "HSCP_noSite", "HSCP_noShift", "HSCP_noSmooth", "HSCP_abs"]


def _ordered(cols: pd.Index) -> list[str]:
    return [m for m in ORDER if m in cols] + [m for m in cols if m not in ORDER]


def report(metrics: pd.DataFrame, cond: pd.DataFrame) -> None:
    pd.set_option("display.width", 240, "display.max_rows", 400, "display.max_columns", 40)
    for alpha in sorted(metrics["alpha"].unique(), reverse=True):
        a = metrics[(metrics["alpha"] == alpha) & (metrics["im"] == "ALL")]
        nominal = 1 - alpha
        cov = a.pivot_table(index="scheme", columns="method", values="coverage", aggfunc="mean", sort=False)
        wid = a.pivot_table(index="scheme", columns="method", values="width", aggfunc="mean", sort=False)
        print(f"\n=== Ablation: mean coverage over models, folds and IMs (nominal {nominal:.0%}) ===")
        print(cov[_ordered(cov.columns)].round(3).to_string())
        print(f"\n=== Mean interval width, ln units (nominal {nominal:.0%}) ===")
        print(wid[_ordered(wid.columns)].round(3).to_string())

    j = metrics[(metrics["alpha"] == 0.10) & metrics["im"].isin(["JOINT_MARGINAL", "JOINT_BAND"])]
    if len(j):
        print("\n=== Spectrum-wide (joint) coverage at 90%: marginal bands vs HSCP simultaneous band ===")
        jt = j.pivot_table(index="scheme", columns=["method", "im"], values="coverage", aggfunc="mean", sort=False)
        print(jt.round(3).to_string())

    hs = metrics[(metrics["alpha"] == 0.10) & (metrics["im"] == "ALL") & (metrics["method"] == "HSCP")]
    if len(hs):
        print("\n=== HSCP at 90% by model (coverage / width / Winkler) ===")
        print(hs.groupby(["scheme", "model"], sort=False)[["coverage", "width", "winkler"]].mean()
              .round(3).to_string())

    if len(cond):
        c = cond[(cond["alpha"] == 0.10) & cond["method"].isin(["NaiveSplitConformal", "WeightedShiftConformal", "HSCP"])]
        g = c.groupby(["stratum", "bin", "method"], sort=False).apply(
            lambda d: d["covered"].sum() / d["n"].sum(), include_groups=False).unstack("method")
        print("\n=== Conditional coverage at 90% (pooled over schemes, models, folds) ===")
        print(g[_ordered(g.columns)].round(3).to_string())

    r = metrics[(metrics["alpha"] == 0.10) & (metrics["im"] == "ALL")].groupby("method")["width_roughness"].mean()
    print("\n=== Spectral roughness of ln half-width (lower = smoother) ===")
    print(r.reindex(_ordered(r.index)).round(4).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "conformal")
    ap.add_argument("--ml-dir", type=Path, default=ROOT / "results" / "ml")
    ap.add_argument("--baseline-cache", type=Path, default=ROOT / "results" / "baselines" / "cache")
    ap.add_argument("--profiles", default="full,sp23")
    ap.add_argument("--schemes", default=",".join(DEFAULT_SCHEMES))
    ap.add_argument("--models", default="", help="Comma-separated; default: all found")
    ap.add_argument("--ablations", action="store_true", help="Add HSCP_noSite/noShift/noSmooth/abs")
    ap.add_argument("--event-counts-only", action="store_true",
                    help="Write only event_counts.parquet; leave the Phase 5 CSVs untouched")
    a = ap.parse_args()

    parts = lambda s: tuple(x.strip() for x in s.split(",") if x.strip())  # noqa: E731
    t0 = time.perf_counter()
    metrics, cond, summary = evaluate_conformal(
        a.data, a.out, ml_dir=a.ml_dir, baseline_cache=a.baseline_cache, profiles=parts(a.profiles),
        schemes=parts(a.schemes), models=parts(a.models) or None, methods=default_methods(a.ablations),
        event_counts=True, metrics=not a.event_counts_only,
    )
    if a.event_counts_only:
        print(f"Wrote {a.out / 'event_counts.parquet'} in {time.perf_counter() - t0:.0f} s")
        return 0
    report(metrics, cond)
    print(f"\nWrote conformal_metrics.csv ({len(metrics):,} rows), conditional_coverage.csv, {summary.name},"
          " event_counts.parquet")
    print(f"Total time {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
