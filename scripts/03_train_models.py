"""Phase 4 gate: train ML/DL spectral GMMs on every frozen split.

Usage (from the repository root):
    python scripts/03_train_models.py
    python scripts/03_train_models.py --fast                          # quick check, small models
    python scripts/03_train_models.py --profiles full --schemes event_grouped --models SpectralMLP

Reads data/processed/{profile}.parquet and splits_{profile}.json; hybrid models read
physics medians from results/baselines/cache (Phase 3). Writes to results/ml/:
ml_metrics.csv, oof_ml_predictions.parquet (test) and cal_ml_predictions.parquet (cal).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from gmpe_bench.models import evaluate_ml_models
from gmpe_bench.models.evaluate import DEFAULT_MODELS, DEFAULT_SCHEMES

ROOT = Path(__file__).resolve().parents[1]


def summarise(m: pd.DataFrame, baseline_csv: Path) -> None:
    pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 30)
    test = m[m["part"] == "test"]
    agg = test.groupby(["profile", "scheme", "model"], sort=False).agg(
        rmse=("rmse", "mean"), bias=("bias", "mean"), nll=("nll", "mean"), cov90=("cov90", "mean"),
        width90=("width90", "mean"), tau=("tau_test", "mean"), phi_s2s=("phi_s2s_test", "mean"),
        phi_0=("phi_0_test", "mean"), fit_s=("fit_seconds", "mean"),
    ).round(3).reset_index()
    print("=== Test metrics, mean over folds and IMs ===")
    print(agg.to_string(index=False))

    if baseline_csv.exists():
        b = pd.read_csv(baseline_csv)
        b = b[b["mode"] == "ergodic"].groupby(["profile", "scheme", "model"], sort=False)[["rmse", "cov90"]].mean()
        ml = test.groupby(["profile", "scheme", "model"], sort=False)[["rmse", "cov90"]].mean()
        both = pd.concat([b, ml]).reset_index()
        best = both.loc[both.groupby(["profile", "scheme"])["rmse"].idxmin()]
        print("\n=== Lowest test RMSE per scheme (Phase 3 ergodic baselines + ML) ===")
        print(best.round(3).to_string(index=False))

    print("\n=== 90% Gaussian coverage on cal vs test (gap motivates conformal) ===")
    cov = m.groupby(["scheme", "model", "part"])["cov90"].mean().unstack("part").round(3)
    print(cov.to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ml")
    ap.add_argument("--physics-cache", type=Path, default=ROOT / "results" / "baselines" / "cache")
    ap.add_argument("--profiles", default="full,sp23")
    ap.add_argument("--schemes", default=",".join(DEFAULT_SCHEMES))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--fast", action="store_true", help="Small models and few epochs/trees")
    ap.add_argument("--no-station-encoding", action="store_true", help="Ergodic features only")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    parts = lambda s: tuple(x.strip() for x in s.split(",") if x.strip())  # noqa: E731
    t0 = time.perf_counter()
    metrics, oof, cal = evaluate_ml_models(
        a.data, a.out, profiles=parts(a.profiles), schemes=parts(a.schemes), models=parts(a.models),
        physics_cache=a.physics_cache, station_encoding=not a.no_station_encoding, fast=a.fast, seed=a.seed,
    )
    summarise(metrics, ROOT / "results" / "baselines" / "baseline_metrics.csv")
    print(f"\nWrote ml_metrics.csv ({len(metrics):,} rows), {oof.name}, {cal.name} to {a.out}")
    print(f"Total time {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
