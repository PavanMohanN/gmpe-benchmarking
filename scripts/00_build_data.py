"""Phase 1 gate: build both profiles and print the numbers we check.

Usage (from the repository root):
    python scripts/00_build_data.py
    python scripts/00_build_data.py --h path/to/nga_w_h.xlsx --v path/to/nga_w_v.xlsx --out data/processed

Defaults: flatfiles in the repository root, outputs in ``data/processed``.
The first run reads the Excel files (a few minutes); later runs use a cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gmpe_bench import io
from gmpe_bench.data import GMDataset, build_dataset
from gmpe_bench.schema import _key

ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "ancheta2014": {"records": 21_336, "events": 599},
    "sp23": {"records": 14_518, "events": 285, "stations": 2_347},
    "sp23_flatfile_events": 607,
}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    by_key = {_key(c): c for c in df.columns}
    return df[by_key[name]]


def event_count_audit(h_path: Path, cache_dir: Path) -> None:
    """Why might we see 599 events where Sedaghati & Pezeshk report 607?"""
    raw = io.read_flatfile(h_path, cache_dir)
    eqid = pd.to_numeric(_col(raw, "EQID"), errors="coerce")
    name = _col(raw, "Earthquake Name").astype("string").str.strip()
    when = (_col(raw, "YEAR").astype("string") + "-" + _col(raw, "MODY").astype("string")
            + "-" + _col(raw, "HRMN").astype("string"))
    d = pd.DataFrame({"eqid": eqid, "name": name, "when": when})

    print("\n=== Event count audit (raw H flatfile) ===")
    print(f"rows                         : {len(d):,}")
    print(f"unique EQID (valid)          : {d['eqid'].dropna().nunique():,}")
    print(f"rows with missing EQID       : {int(d['eqid'].isna().sum()):,}")
    print(f"unique earthquake names      : {d['name'].nunique():,}")
    print(f"unique (name, date-time)     : {d[['name', 'when']].drop_duplicates().shape[0]:,}")
    multi_name = d.groupby("eqid")["name"].nunique()
    multi_eq = d.groupby("name")["eqid"].nunique()
    print(f"EQIDs with >1 name           : {int((multi_name > 1).sum())}")
    print(f"names with >1 EQID           : {int((multi_eq > 1).sum())}")
    if (multi_eq > 1).any():
        print("  examples:", ", ".join(multi_eq[multi_eq > 1].index[:8].astype(str)))
    print(f"Sedaghati & Pezeshk report   : {TARGETS['sp23_flatfile_events']}")


def gate_report(res, out_dir: Path) -> None:
    pd.set_option("display.width", 200, "display.max_colwidth", 90, "display.max_rows", 200)
    cols = ["rule", "description", "n_before", "n_after", "removed", "n_events", "n_stations"]
    for prof, g in res.attrition.groupby("profile", sort=False):
        print(f"\n=== Attrition: {prof} ===")
        print(g[cols].to_string(index=False))

    print("\n=== Rule notes (check these assumptions) ===")
    notes = res.attrition.drop_duplicates("rule")[["rule", "note"]]
    for _, r in notes[notes["note"].astype(bool)].iterrows():
        print(f"{r['rule']:>4}: {r['note']}")

    full = GMDataset.from_parquet(res.outputs["full"])
    usable = full.describe()["records_with_usable"]
    print("\n=== Gate 1: raw counts vs Ancheta et al. (2014) ===")
    first = res.attrition.iloc[0]
    print(f"flatfile rows                : {first['n_after']:,}  (Ancheta: {TARGETS['ancheta2014']['records']:,})")
    print(f"flatfile events              : {first['n_events']:,}  (Ancheta: {TARGETS['ancheta2014']['events']})")
    print(f"full: records / events / sta : {len(full):,} / {full.df['eqid'].nunique()} / {full.df['ssn'].nunique():,}")
    print(f"full: usable H / V / V-H     : {usable['H']:,} / {usable['V']:,} / {usable['VH']:,}")

    sp = res.counts.get("sp23")
    if sp:
        t = TARGETS["sp23"]
        print("\n=== Gate 2: sp23 profile vs Sedaghati & Pezeshk (2023) ===")
        for k in ("records", "events", "stations"):
            diff = sp[k] - t[k]
            print(f"{k:<9}: {sp[k]:>6,}  target {t[k]:>6,}  diff {diff:+,} ({diff / t[k]:+.1%})")

    print("\n=== Usable records per IM (full, H) ===")
    per = pd.Series(full.describe()["usable_per_im"]["H"])
    print(per.to_string())
    print(f"\nOutputs in {out_dir}")
    for k, p in res.outputs.items():
        print(f"  {k:<9} {p.name}")
    print(f"  manifest  {res.manifest_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=Path, default=ROOT / "nga_w_h.xlsx")
    ap.add_argument("--v", type=Path, default=ROOT / "nga_w_v.xlsx")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    res = build_dataset(a.h, a.v, a.out, profile="both", cache=not a.no_cache)
    gate_report(res, a.out)
    event_count_audit(a.h, a.out / "_cache")


if __name__ == "__main__":
    main()
