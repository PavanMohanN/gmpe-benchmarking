"""Build the ML-ready NGA-West2 dataset: rules R1-R12, one call."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from gmpe_bench import io
from gmpe_bench.data import rules
from gmpe_bench.data.rules import AttritionStep, make_step
from gmpe_bench.schema import IM_NAMES, normalise_columns, validate_schema

PROFILES: tuple[str, ...] = ("full", "sp23")


@dataclass
class BuildResult:
    outputs: dict[str, Path]
    attrition: pd.DataFrame
    manifest_path: Path
    counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("gmpe-bench")
    except Exception:  # not installed in editable mode yet
        return "unknown"


def load_flatfiles(h_path: Path, v_path: Path, cache_dir: Path | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, list[AttritionStep]]:
    """R1: read both flatfiles, map to the canonical schema and validate."""
    h_raw = io.read_flatfile(h_path, cache_dir)
    v_raw = io.read_flatfile(v_path, cache_dir)
    h = normalise_columns(h_raw, "H")
    v = normalise_columns(v_raw, "V")
    validate_schema(h, "H")
    validate_schema(v, "V")
    step = make_step("R1", "Read H and V flatfiles", len(h), h,
                     f"H {h_raw.shape[1]} cols, V {v_raw.shape[1]} cols, {len(v)} V rows")
    return h, v, [step]


def run_rules(h: pd.DataFrame, v: pd.DataFrame) -> tuple[pd.DataFrame, list[AttritionStep]]:
    """R2-R8, R10, R11 on canonical frames; returns the base table before profiles."""
    steps: list[AttritionStep] = []
    h, s_h = rules.replace_sentinels(h)
    v, s_v = rules.replace_sentinels(v)
    steps.append(AttritionStep("R2", s_h.description, s_h.n_before, s_h.n_after,
                               s_h.n_events, s_h.n_stations, f"H: {s_h.note}; V: {s_v.note}"))
    df = h
    for fn in (lambda d: rules.join_hv(d, v), rules.lock_keys, rules.mask_unusable_periods,
               rules.apply_quality_flags, rules.drop_missing_keys, rules.assign_cl1_cl2,
               rules.add_vh_ratios, rules.fill_site_source):
        df, step = fn(df)
        steps.append(step)
    return df, steps


def apply_profile(df: pd.DataFrame, profile: str) -> tuple[pd.DataFrame, list[AttritionStep]]:
    if profile not in rules.PROFILES:
        raise ValueError(f"profile must be one of {list(rules.PROFILES)}, got {profile!r}")
    return rules.PROFILES[profile](df)


def attrition_table(steps: Iterable[AttritionStep], profile: str) -> pd.DataFrame:
    rows = [{"profile": profile, **s.as_dict()} for s in steps]
    cols = ["profile", "rule", "description", "n_before", "n_after", "removed",
            "n_events", "n_stations", "note"]
    return pd.DataFrame(rows)[cols]


def build_dataset(h_path: Path, v_path: Path, out_dir: Path, profile: str = "both",
                  cache: bool = True) -> BuildResult:
    """Run R1-R12 and write ``<profile>.parquet``, ``attrition.csv`` and ``manifest.json``."""
    h_path, v_path, out_dir = Path(h_path), Path(v_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = PROFILES if profile == "both" else (profile,)

    h, v, steps = load_flatfiles(h_path, v_path, out_dir / "_cache" if cache else None)
    base, rule_steps = run_rules(h, v)
    steps += rule_steps

    outputs: dict[str, Path] = {}
    tables: list[pd.DataFrame] = []
    counts: dict[str, dict[str, int]] = {}
    for prof in profiles:
        df, prof_steps = apply_profile(base, prof)
        path = io.write_parquet(df, out_dir / f"{prof}.parquet")
        final = make_step("R12", f"Freeze {prof}.parquet", len(df), df)
        tables.append(attrition_table(steps + prof_steps + [final], prof))
        outputs[prof] = path
        counts[prof] = {"records": len(df), "events": int(df["eqid"].nunique()),
                        "stations": int(df["ssn"].nunique())}

    attrition = pd.concat(tables, ignore_index=True)
    att_path = out_dir / "attrition.csv"
    attrition.to_csv(att_path, index=False)
    outputs["attrition"] = att_path

    manifest = {
        "package": "gmpe-bench",
        "version": _version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "inputs": {p.name: {"sha256": io.sha256_file(p)} for p in (h_path, v_path)},
        "outputs": {k: {"file": p.name, "sha256": io.sha256_file(p)} for k, p in outputs.items()},
        "counts": counts,
        "ims": list(IM_NAMES),
        "rules": [s.as_dict() for s in steps],
    }
    manifest_path = io.write_manifest(manifest, out_dir / "manifest.json")
    return BuildResult(outputs, attrition, manifest_path, counts)
