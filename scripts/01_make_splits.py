"""Phase 2 gate: make, audit and freeze every split scheme for both profiles.

Usage (from the repository root):
    python scripts/01_make_splits.py
    python scripts/01_make_splits.py --data data/processed --seed 42 --cal-frac 0.25

Writes splits_full.json, splits_sp23.json, split_summary.csv and
splits_manifest.json to the data folder. Exits with code 1 if any fold fails the audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gmpe_bench.splits import freeze_all, leakage_table

ROOT = Path(__file__).resolve().parents[1]


def table2(df: pd.DataFrame) -> pd.DataFrame:
    """One row per profile x scheme: fold means for counts, max for leakage rates."""
    g = df.groupby(["profile", "scheme"], sort=False)
    return pd.DataFrame({
        "folds": g.size(),
        "test_records": g["n_test"].mean().round(0).astype(int),
        "test_events": g["events_test"].mean().round(1),
        "test_stations": g["stations_test"].mean().round(1),
        "cal_records": g["n_cal"].mean().round(0).astype(int),
        "excluded": g["n_excluded"].max(),
        "seen_event_%": (g["test_records_seen_event"].mean() * 100).round(1),
        "seen_station_%": (g["test_records_seen_station"].mean() * 100).round(1),
        "passed": g["passed"].all(),
    }).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cal-frac", type=float, default=0.25)
    a = ap.parse_args()

    reports, files = freeze_all(a.data, seed=a.seed, cal_frac=a.cal_frac)
    df = leakage_table(reports)
    pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_columns", 30)

    cols = ["profile", "scheme", "fold", "n_train_fit", "n_cal", "n_test", "n_excluded",
            "events_test", "stations_test", "shared_events", "shared_stations",
            "test_records_seen_event", "test_records_seen_station", "passed"]
    print("=== Leakage audit, every fold ===")
    print(df[cols].to_string(index=False))

    print("\n=== Table 2: split summary ===")
    print(table2(df).to_string(index=False))

    failed = df[~df["passed"]]
    if len(failed):
        print("\n=== FAILED folds ===")
        print(failed[["profile", "scheme", "fold", "issues"]].to_string(index=False))

    print(f"\nWrote to {a.data}:")
    for f in files.values():
        print(f"  {f.name}")
    print(f"\nAudit: {len(df) - len(failed)}/{len(df)} folds passed")
    return 1 if len(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
