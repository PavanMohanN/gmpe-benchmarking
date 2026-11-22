"""Phase 3 gate: classical baselines on every frozen split.

Usage (from the repository root):
    python scripts/02_run_baselines.py
    python scripts/02_run_baselines.py --profiles full --schemes event_grouped --ims PGA,SA_1s

Reads data/processed/{full,sp23}.parquet and splits_{profile}.json.
Writes results/baselines/baseline_metrics.csv and oof_predictions.parquet.
The first run evaluates the four NGA-West2 models once per record (about a minute);
later runs reuse results/baselines/cache/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gmpe_bench.baselines import evaluate_baselines
from gmpe_bench.baselines.evaluate import DEFAULT_MODELS, DEFAULT_SCHEMES

ROOT = Path(__file__).resolve().parents[1]
GATE_IMS = ["PGA", "PGV", "SA_0.2s", "SA_1s", "SA_3s"]


def summarise(m: pd.DataFrame) -> None:
    pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 30)
    keys = ["profile", "scheme", "model", "mode"]
    agg = m.groupby(keys, sort=False).agg(
        rmse=("rmse", "mean"), bias=("bias", "mean"), nll=("nll", "mean"),
        cov90=("cov90", "mean"), width90=("width90", "mean"),
        tau=("tau_test", "mean"), phi_s2s=("phi_s2s_test", "mean"), phi_0=("phi_0_test", "mean"),
    ).round(3).reset_index()
    print("=== Mean over folds and IMs ===")
    print(agg.to_string(index=False))

    g = m[(m["scheme"] == "event_grouped") & (m["mode"] == "ergodic") & m["im"].isin(GATE_IMS)]
    if len(g):
        print("\n=== Gate check: event_grouped, ergodic, selected IMs (fold means) ===")
        piv = g.groupby(["profile", "model", "im"], sort=False)[["bias", "rmse", "tau_test", "phi_s2s_test",
                                                                 "phi_0_test", "cov90"]].mean()
        print(piv.round(3).to_string())

    worst = (m[m["mode"] == "ergodic"].groupby(["scheme", "model"])["cov90"].mean()
             .sort_values().head(8).round(3))
    print("\n=== Lowest ergodic 90% coverage (scheme, model) ===")
    print(worst.to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baselines")
    ap.add_argument("--profiles", default="full,sp23")
    ap.add_argument("--schemes", default=",".join(DEFAULT_SCHEMES))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--ims", default="", help="Comma-separated IMs; default all 27")
    ap.add_argument("--no-bias", action="store_true", help="Raw NGA-West2 medians (no train_fit bias term)")
    a = ap.parse_args()

    split = lambda s: tuple(x.strip() for x in s.split(",") if x.strip())  # noqa: E731
    metrics, oof = evaluate_baselines(
        a.data, a.out, profiles=split(a.profiles), schemes=split(a.schemes), models=split(a.models),
        ims=list(split(a.ims)) or None, calibrate_bias=not a.no_bias,
    )
    summarise(metrics)
    print(f"\nWrote {a.out / 'baseline_metrics.csv'} ({len(metrics):,} rows) and {oof.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
