"""Data layer of the gmpe-bench app: read-only views of the published benchmark results.

Nothing here needs the PEER flatfiles, ``full.parquet`` or Streamlit. Sources, in order:

1. the repository's ``results/paper`` folder (``paper_summary_numbers.json`` and ``tables/*.csv``) and
   the per-IM metric files ``results/{baselines,ml,conformal}/*_metrics.csv``;
2. the copies packaged in ``gmpe_bench/app/assets`` (used by ``pip install gmpe-bench[app]`` and by hosted
   deployments that do not carry ``results/``).

The packaged copies are exact copies of the paper outputs, except ``period_*.csv.gz``, which are fold-
(and, for calibration, model-) averages of the per-IM metric files, built by :func:`build_period_tables`.
A test keeps all of them in sync with the repository.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ASSETS = Path(__file__).with_name("assets")
PAPER_JSON = Path("results") / "paper" / "paper_summary_numbers.json"
PAPER_TABLES = Path("results") / "paper" / "tables"

SCHEMES: tuple[str, ...] = ("event_grouped", "station_grouped", "spatial_block", "temporal",
                            "magnitude_holdout", "cl1_to_cl2")
SCHEME_LABELS: dict[str, str] = {"random": "Random (leaky)", "event_grouped": "Event", "station_grouped": "Station",
                                 "spatial_block": "Region", "temporal": "Temporal", "magnitude_holdout": "Mw ≥ 7",
                                 "cl1_to_cl2": "CL1 → CL2"}
SCHEME_QUESTIONS: dict[str, str] = {
    "random": "Leaky reference: records of one earthquake on both sides",
    "event_grouped": "New earthquakes", "station_grouped": "New sites", "spatial_block": "New region",
    "temporal": "Later years (train < 2008)", "magnitude_holdout": "Large-magnitude extrapolation",
    "cl1_to_cl2": "Mainshock → aftershock transfer"}
MODEL_LABELS: dict[str, str] = {
    "ASK14_cal": "ASK14", "BSSA14_cal": "BSSA14", "CB14_cal": "CB14", "CY14_cal": "CY14",
    "ASK14_site": "ASK14 (site)", "BSSA14_site": "BSSA14 (site)", "CB14_site": "CB14 (site)",
    "CY14_site": "CY14 (site)", "MixedEffects": "Mixed effects", "MixedEffects_site": "Mixed effects (site)",
    "XGBoost": "XGBoost", "SpectralMLP": "Spectral MLP", "Hybrid_ASK14_MLP": "Hybrid ASK14+MLP",
    "Hybrid_BSSA14_XGB": "Hybrid BSSA14+XGB"}
MODEL_FAMILY: dict[str, str] = {
    "ASK14_cal": "Published NGA-West2", "BSSA14_cal": "Published NGA-West2", "CB14_cal": "Published NGA-West2",
    "CY14_cal": "Published NGA-West2", "ASK14_site": "Published + station terms",
    "BSSA14_site": "Published + station terms", "MixedEffects": "Statistical", "XGBoost": "Machine learning",
    "SpectralMLP": "Machine learning", "Hybrid_ASK14_MLP": "Hybrid", "Hybrid_BSSA14_XGB": "Hybrid"}
METHOD_LABELS: dict[str, str] = {
    "UncalibratedGaussian": "Gaussian", "NaiveSplitConformal": "Split CP", "ClusteredEventConformal":
    "Event-clustered CP", "WeightedShiftConformal": "Shift-weighted CP", "HSCP": "HSCP",
    "HSCP_noSite": "HSCP w/o site", "HSCP_noShift": "HSCP w/o shift", "HSCP_noSmooth": "HSCP w/o smoothing",
    "HSCP_abs": "HSCP unnormalized"}
MAIN_METHODS: tuple[str, ...] = ("UncalibratedGaussian", "NaiveSplitConformal", "ClusteredEventConformal",
                                 "WeightedShiftConformal", "HSCP")
TABLE_NAMES: tuple[str, ...] = ("table1_attrition", "table2_splits", "table3_accuracy", "table4_conformal_ablation",
                                "table5_conditional_coverage", "tableS1_per_im_metrics", "tableS2_sp23_comparison",
                                "tableS3_bootstrap_cis", "tableS4_station_rmse")
PERIOD_FILES = ("period_accuracy.csv.gz", "period_calibration.csv.gz")


def label(key: str) -> str:
    return MODEL_LABELS.get(key, METHOD_LABELS.get(key, SCHEME_LABELS.get(key, key)))


# ------------------------------------------------------------------ locating sources
def repo_roots() -> list[Path]:
    """Candidate repository roots: ``GMPE_BENCH_ROOT``, the working folder, and the source checkout."""
    roots = []
    if os.environ.get("GMPE_BENCH_ROOT"):
        roots.append(Path(os.environ["GMPE_BENCH_ROOT"]))
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parents[3])   # <root>/src/gmpe_bench/app/data.py
    out: list[Path] = []
    for r in roots:
        r = r.resolve()
        if r not in out:
            out.append(r)
    return out


def find_repo_root() -> Path | None:
    for r in repo_roots():
        if (r / PAPER_JSON).exists() and (r / PAPER_TABLES).is_dir():
            return r
    return None


def data_dir() -> Path:
    """Folder that holds ``full.parquet`` and the frozen splits (``GMPE_BENCH_DATA`` overrides)."""
    if os.environ.get("GMPE_BENCH_DATA"):
        return Path(os.environ["GMPE_BENCH_DATA"])
    for r in repo_roots():
        if (r / "data" / "processed").is_dir():
            return r / "data" / "processed"
    return Path("data") / "processed"


def local_profile_available(profile: str = "full", folder: Path | None = None) -> bool:
    folder = folder or data_dir()
    return (folder / f"{profile}.parquet").exists() and (folder / f"splits_{profile}.json").exists()


# ------------------------------------------------------------------ period extract
def build_period_tables(root: Path, profile: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-IM accuracy and calibration from the repository metric files (fold means).

    Accuracy: one row per (scheme, model, IM): RMSE, Gaussian cov90 and width90 of the model's own sigma,
    and the test-residual variance components. Calibration: one row per (scheme, model, method, nominal, IM).
    """
    root = Path(root)
    base = pd.read_csv(root / "results" / "baselines" / "baseline_metrics.csv")
    base = base[base["profile"] == profile].copy()
    site = base["mode"] == "site_corrected"
    base.loc[site, "model"] = base.loc[site, "model"].str.replace("_cal", "_site", regex=False)
    base.loc[site & (base["model"] == "MixedEffects"), "model"] = "MixedEffects_site"
    ml = pd.read_csv(root / "results" / "ml" / "ml_metrics.csv")
    ml = ml[(ml["profile"] == profile) & (ml["part"] == "test")]
    cols = ["rmse", "cov90", "width90", "tau_test", "phi_s2s_test", "phi_0_test"]
    acc = pd.concat([base[["scheme", "model", "im", "period", *cols]], ml[["scheme", "model", "im", "period", *cols]]])
    acc = (acc.groupby(["scheme", "model", "im"], sort=False)
           .agg(period=("period", "first"), **{c: (c, "mean") for c in cols}).reset_index())
    acc = acc.rename(columns={"tau_test": "tau", "phi_s2s_test": "phi_s2s", "phi_0_test": "phi_0"})
    # residual variance components are computed on the ergodic residuals for every mode; keep them only
    # for models whose residuals they describe
    acc.loc[acc["model"].str.endswith("_site"), ["tau", "phi_s2s", "phi_0"]] = np.nan

    conf = pd.read_csv(root / "results" / "conformal" / "conformal_metrics.csv",
                       usecols=["profile", "scheme", "model", "method", "nominal", "im", "period", "coverage", "width"])
    conf = conf[(conf["profile"] == profile) & ~conf["im"].isin(["ALL", "JOINT_MARGINAL", "JOINT_BAND"])]
    cal = (conf.groupby(["scheme", "model", "method", "nominal", "im"], sort=False)
           .agg(period=("period", "first"), coverage=("coverage", "mean"), width=("width", "mean")).reset_index())
    return acc.round(6), cal.round(6)


def write_period_assets(root: Path, out: Path = ASSETS) -> list[Path]:
    acc, cal = build_period_tables(root)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / PERIOD_FILES[0], out / PERIOD_FILES[1]]
    acc.to_csv(paths[0], index=False, compression={"method": "gzip", "mtime": 0})
    cal.to_csv(paths[1], index=False, compression={"method": "gzip", "mtime": 0})
    return paths


def write_assets(root: Path, out: Path = ASSETS) -> list[Path]:
    """Refresh the packaged copies from a repository checkout (run after ``gmpe-bench paper``)."""
    import shutil

    root = Path(root)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    written = [Path(shutil.copy2(root / PAPER_JSON, out / PAPER_JSON.name))]
    for name in TABLE_NAMES:
        written.append(Path(shutil.copy2(root / PAPER_TABLES / f"{name}.csv", out / "tables" / f"{name}.csv")))
    return written + write_period_assets(root, out)


# ------------------------------------------------------------------ loading
@dataclass
class PaperData:
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    period_accuracy: pd.DataFrame
    period_calibration: pd.DataFrame
    source: str
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- tidy views
    def leakage(self) -> pd.DataFrame:
        rows = [{"model": k, "label": label(k), "family": MODEL_FAMILY.get(k, ""), "rmse_random": v["rmse_random"],
                 "rmse_event": v["rmse_event"], "inflation_pct": v["inflation_pct"]}
                for k, v in self.summary.get("leakage", {}).items()]
        return pd.DataFrame(rows).sort_values("inflation_pct").reset_index(drop=True)

    def accuracy(self) -> pd.DataFrame:
        rows = []
        for scheme, models in self.summary.get("accuracy", {}).items():
            for m, v in models.items():
                ci = v.get("rmse_ci") or {}
                rows.append({"scheme": scheme, "regime": label(scheme), "model": m, "label": label(m),
                             "family": MODEL_FAMILY.get(m, ""), "rmse": v.get("rmse"), "lo": ci.get("lo"),
                             "hi": ci.get("hi"), "cov90": v.get("cov90"), "width90": v.get("width90"),
                             "tau": v.get("tau_test"), "phi_s2s": v.get("phi_s2s_test"), "phi_0": v.get("phi_0_test"),
                             "n_events": ci.get("n_events")})
        df = pd.DataFrame(rows)
        df["regime"] = pd.Categorical(df["regime"], [label(s) for s in SCHEMES], ordered=True)
        return df.sort_values(["regime", "rmse"]).reset_index(drop=True)

    def differences(self) -> pd.DataFrame:
        df = pd.DataFrame(self.summary.get("rmse_differences", []))
        if df.empty:
            return df
        df["regime"] = df["scheme"].map(label)
        df["comparison"] = df["model_a"].map(label) + " − " + df["model_b"].map(label)
        df["resolved"] = (df["lo"] > 0) | (df["hi"] < 0)
        return df

    def calibration(self, nominal: str = "0.90") -> pd.DataFrame:
        rows = []
        for scheme, methods in self.summary.get("conformal", {}).get(nominal, {}).items():
            for meth, v in methods.items():
                ci = v.get("coverage_ci") or {}
                rows.append({"scheme": scheme, "regime": label(scheme), "method": meth, "label": label(meth),
                             "main": meth in MAIN_METHODS, "coverage": v.get("coverage"), "lo": ci.get("lo"),
                             "hi": ci.get("hi"), "width": v.get("width"), "joint_marginal": v.get("joint_marginal_coverage"),
                             "joint_band": v.get("joint_band_coverage"), "joint_band_width": v.get("joint_band_width"),
                             "roughness": v.get("width_roughness"), "ess": v.get("ess"), "n_inf": v.get("n_inf")})
        return pd.DataFrame(rows)

    def mace(self, nominal: str = "0.90") -> pd.DataFrame:
        cal = self.calibration(nominal)
        target = float(nominal)
        cal = cal[cal["scheme"].isin(SCHEMES)]
        out = (cal.assign(err=(cal["coverage"] - target).abs()).groupby(["method", "label"], sort=False)["err"]
               .mean().rename("mace").reset_index())
        return out.sort_values("mace").reset_index(drop=True)

    def station(self) -> pd.DataFrame:
        df = pd.DataFrame(self.summary.get("station_rmse", []))
        if not df.empty:
            df["regime"], df["label"] = df["scheme"].map(label), df["model"].map(label)
        return df

    def splits(self) -> pd.DataFrame:
        rows = [{"scheme": s, "regime": label(s), "question": SCHEME_QUESTIONS.get(s, ""), **v}
                for s, v in self.summary.get("splits", {}).items()]
        return pd.DataFrame(rows)


def _read_period(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(folder / PERIOD_FILES[0]), pd.read_csv(folder / PERIOD_FILES[1])


def load_paper_data(root: Path | None = None, prefer_packaged: bool = False) -> PaperData:
    """Benchmark results for the app: the repository copy when present, else the packaged copy."""
    root = None if prefer_packaged else (Path(root) if root is not None else find_repo_root())
    notes: list[str] = []
    if root is not None and (root / PAPER_JSON).exists():
        summary = json.loads((root / PAPER_JSON).read_text(encoding="utf-8"))
        tables = {n: pd.read_csv(root / PAPER_TABLES / f"{n}.csv") for n in TABLE_NAMES
                  if (root / PAPER_TABLES / f"{n}.csv").exists()}
        source = f"repository ({root / 'results' / 'paper'})"
        metric_files = [root / "results" / d / f for d, f in (("baselines", "baseline_metrics.csv"),
                                                               ("ml", "ml_metrics.csv"),
                                                               ("conformal", "conformal_metrics.csv"))]
        if all(f.exists() for f in metric_files):
            acc, cal = build_period_tables(root)
        else:
            acc, cal = _read_period(ASSETS)
            notes.append("per-period metrics read from the packaged extract")
    else:
        summary = json.loads((ASSETS / PAPER_JSON.name).read_text(encoding="utf-8"))
        tables = {n: pd.read_csv(ASSETS / "tables" / f"{n}.csv") for n in TABLE_NAMES}
        acc, cal = _read_period(ASSETS)
        source = "packaged copy of the paper results (gmpe_bench/app/assets)"
    return PaperData(summary, tables, acc, cal, source, notes)


def period_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Spectral-acceleration rows only, with a numeric period column for log-period plots."""
    out = df[df["im"].astype(str).str.startswith("SA_")].copy()
    out["period"] = pd.to_numeric(out["period"], errors="coerce")
    return out


def load_split_summary(folder: Path | None = None) -> pd.DataFrame | None:
    """The frozen leakage-audit table written when the splits were made (``split_summary.csv``)."""
    folder = folder or data_dir()
    path = folder / "split_summary.csv"
    return pd.read_csv(path) if path.exists() else None
