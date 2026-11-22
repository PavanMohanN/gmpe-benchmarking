"""Loads every Phase 1-5 output used by the paper; nothing is hard-coded.

Expected layout under ``root`` (the repository folder)::

    data/processed/{full,sp23}.parquet, attrition.csv, split_summary.csv, splits_<profile>.json
    results/baselines/baseline_metrics.csv, results/baselines/cache/<MODEL>.parquet
    results/ml/{ml_metrics.csv, oof_ml_predictions.parquet, cal_ml_predictions.parquet}
    results/conformal/{conformal_metrics.csv, conditional_coverage.csv, conformal_intervals_summary.parquet}
    results/baselines/oof_predictions.parquet     (bootstrap; site-corrected predictions)
    results/conformal/event_counts.parquet        (bootstrap of coverage; `gmpe-bench conformal
                                                   --event-counts-only` adds it to an existing run)

Optional, for the leakage-inflation figure (written by separate runs so the main
results are not overwritten)::

    results/ml_random/ml_metrics.csv, results/baselines_random/baseline_metrics.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from gmpe_bench.data import GMDataset

# Literature reference values (not results of this study): Sedaghati & Pezeshk (2023).
SP23_REFERENCE = {"records": 14_518, "events": 285, "stations": 2_347}


class MissingInput(FileNotFoundError):
    """A required Phase 1-5 output is absent; the artifact that needs it is skipped."""


def site_name(model: str) -> str:
    """``ASK14_cal`` -> ``ASK14_site`` (site-corrected mode of a Phase 3 model)."""
    return f"{model[:-4] if model.endswith('_cal') else model}_site"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MissingInput(str(path))
    return pd.read_csv(path)


@dataclass
class ReportConfig:
    root: Path = Path(".")
    profile: str = "full"
    compare_profile: str = "sp23"
    alpha: float = 0.10
    accuracy_scheme: str = "event_grouped"
    example_model: str = "Hybrid_ASK14_MLP"
    example_schemes: tuple[str, ...] = ("event_grouped", "spatial_block")
    focus_models: tuple[str, ...] = ("ASK14_cal", "BSSA14_cal", "MixedEffects", "XGBoost",
                                     "Hybrid_ASK14_MLP", "Hybrid_BSSA14_XGB")
    # NGA-West2 models also reported in site-corrected mode (Phase 3 `site_corrected`: median plus the
    # train_fit station term, 0 for unseen stations); they appear as "<NAME>_site", e.g. "ASK14 (site)".
    site_models: tuple[str, ...] = ("ASK14_cal", "BSSA14_cal")
    # Event-clustered bootstrap
    bootstrap_b: int = 1000
    bootstrap_seed: int = 20261122
    bootstrap_level: float = 0.95
    pair_models: tuple[str, ...] = ("Hybrid_BSSA14_XGB", "Hybrid_ASK14_MLP", "ASK14_cal", "ASK14_site")
    station_models: tuple[str, ...] = ("ASK14_site", "BSSA14_site", "Hybrid_ASK14_MLP", "Hybrid_BSSA14_XGB")
    out_dir: Path | None = None

    @property
    def data_dir(self) -> Path:
        return Path(self.root) / "data" / "processed"

    @property
    def results(self) -> Path:
        return Path(self.root) / "results"

    @property
    def paper_dir(self) -> Path:
        return Path(self.out_dir) if self.out_dir is not None else self.results / "paper"


@dataclass
class ReportInputs:
    cfg: ReportConfig
    used: set[Path] = field(default_factory=set)
    log: Callable[[str], None] = print

    def _csv(self, path: Path) -> pd.DataFrame:
        df = _read_csv(path)
        self.used.add(path)
        return df

    # -- Phase 1-2 -------------------------------------------------------------
    def dataset(self, profile: str | None = None) -> GMDataset:
        prof = profile or self.cfg.profile
        path = self.cfg.data_dir / f"{prof}.parquet"
        if not path.exists():
            raise MissingInput(str(path))
        self.used.add(path)
        return GMDataset.from_parquet(path, prof)

    @cached_property
    def attrition(self) -> pd.DataFrame:
        return self._csv(self.cfg.data_dir / "attrition.csv")

    @cached_property
    def split_summary(self) -> pd.DataFrame:
        return self._csv(self.cfg.data_dir / "split_summary.csv")

    # -- Phase 3-5 -------------------------------------------------------------
    @cached_property
    def baseline_metrics(self) -> pd.DataFrame:
        return self._csv(self.cfg.results / "baselines" / "baseline_metrics.csv")

    @cached_property
    def ml_metrics(self) -> pd.DataFrame:
        return self._csv(self.cfg.results / "ml" / "ml_metrics.csv")

    @cached_property
    def conformal_metrics(self) -> pd.DataFrame:
        return self._csv(self.cfg.results / "conformal" / "conformal_metrics.csv")

    @cached_property
    def conditional(self) -> pd.DataFrame:
        return self._csv(self.cfg.results / "conformal" / "conditional_coverage.csv")

    def accuracy(self, profile: str | None = None, include_random: bool = False,
                 include_site: bool = False) -> pd.DataFrame:
        """Test-set accuracy rows from Phase 3 (ergodic) and Phase 4 (test part), one schema.

        ``include_site`` adds the Phase 3 ``site_corrected`` rows of ``cfg.site_models``, renamed
        ``<NAME>_site`` (same folds, records and metric code as the ergodic rows).
        """
        prof = profile or self.cfg.profile
        cols = ["scheme", "fold", "model", "im", "period", "n", "rmse", "bias", "nll", "cov90", "width90",
                "tau_test", "phi_s2s_test", "phi_0_test"]
        parts = []
        try:
            b = self.baseline_metrics
            parts.append(b[(b["profile"] == prof) & (b["mode"] == "ergodic")][cols].assign(family="classical"))
            if include_site:
                site = b[(b["profile"] == prof) & (b["mode"] == "site_corrected")
                         & b["model"].isin(self.cfg.site_models)][cols]
                parts.append(site.assign(model=site["model"].map(site_name), family="classical"))
        except MissingInput:
            pass
        try:
            m = self.ml_metrics
            parts.append(m[(m["profile"] == prof) & (m["part"] == "test")][cols].assign(family="ml"))
        except MissingInput:
            pass
        if include_random:
            parts.append(self.random_accuracy(prof))
        if not parts:
            raise MissingInput("results/baselines/baseline_metrics.csv and results/ml/ml_metrics.csv")
        return pd.concat(parts, ignore_index=True)

    # -- record-level predictions (bootstrap) ----------------------------------------
    def iter_predictions(self, profile: str | None = None) -> Iterator[tuple[str, pd.DataFrame]]:
        """Test predictions one model at a time: (model, frame with scheme, fold, rsn, eqid, ssn and
        y_/mu_/sigma_ per IM). Covers the ergodic Phase 3 models, the site-corrected
        ``cfg.site_models`` (yielded as ``<NAME>_site`` from the ``mu_site_``/``sigma_site_`` columns)
        and the Phase 4 ML models (``part == "test"``). Streaming keeps memory to one model."""
        prof = profile or self.cfg.profile
        keys = ["scheme", "fold", "rsn", "eqid", "ssn"]
        found = False
        for path in (self.cfg.results / "baselines" / "oof_predictions.parquet",
                     self.cfg.results / "ml" / "oof_ml_predictions.parquet"):
            if not path.exists():
                continue
            found = True
            self.used.add(path)
            dset = pads.dataset(path)
            names = dset.schema.names
            flt = pads.field("profile") == prof
            if "part" in names:
                flt = flt & (pads.field("part") == "test")
            models = sorted(set(dset.to_table(columns=["model"], filter=flt).column("model").to_pylist()))
            for model in models:
                f = flt & (pads.field("model") == model)
                cols = keys + [c for c in names if c.startswith(("y_", "mu_", "sigma_"))]
                df = dset.to_table(columns=cols, filter=f).to_pandas()
                df["scheme"], df["fold"] = df["scheme"].astype(str), df["fold"].astype(str)
                plain = [c for c in df.columns if not c.startswith(("mu_site_", "sigma_site_"))]
                yield model, df[plain]
                if model in self.cfg.site_models and any(c.startswith("mu_site_") for c in df.columns):
                    site = df[keys + [c for c in df.columns if c.startswith(("y_", "mu_site_", "sigma_site_"))]]
                    site = site.rename(columns=lambda c: c.replace("mu_site_", "mu_").replace("sigma_site_", "sigma_"))
                    yield site_name(model), site
        if not found:
            raise MissingInput("results/baselines/oof_predictions.parquet and results/ml/oof_ml_predictions.parquet")

    def event_counts(self, profile: str | None = None) -> pd.DataFrame:
        """Per-earthquake covered/usable counts written by Phase 5 (``event_counts.parquet``)."""
        prof = profile or self.cfg.profile
        path = self.cfg.results / "conformal" / "event_counts.parquet"
        if not path.exists():
            raise MissingInput(f"{path} (run: gmpe-bench conformal --event-counts-only --profiles {prof})")
        self.used.add(path)
        e = pads.dataset(path).to_table(filter=pads.field("profile") == prof).to_pandas()
        for c in ("scheme", "fold", "model", "method"):
            e[c] = e[c].astype(str)
        return e

    def train_fit_stations(self, profile: str | None = None) -> pd.DataFrame:
        """(scheme, fold, ssn) for every station with a train_fit record: the 'seen' stations of a fold."""
        from gmpe_bench.splits import load_splits

        prof = profile or self.cfg.profile
        ds = self.dataset(prof)
        path = self.cfg.data_dir / f"splits_{prof}.json"
        if not path.exists():
            raise MissingInput(str(path))
        self.used.add(path)
        ssn = ds.df["ssn"].to_numpy()
        rows = []
        for scheme, folds in load_splits(path, ds).items():
            for f in folds:
                st = np.unique(ssn[f.train_fit])
                rows.append(pd.DataFrame({"scheme": scheme, "fold": str(f.fold), "ssn": st}))
        return pd.concat(rows, ignore_index=True)

    @cached_property
    def bootstrap(self):  # noqa: ANN201 - BootstrapResult (lazy import avoids a cycle)
        """Event-clustered bootstrap intervals, computed once per report run."""
        from gmpe_bench.reporting.bootstrap import run_bootstrap

        return run_bootstrap(self, log=self.log)

    def random_accuracy(self, profile: str | None = None) -> pd.DataFrame:
        """Rows for the leaky ``random`` scheme from the main or the ``*_random`` result folders."""
        prof = profile or self.cfg.profile
        cols = ["scheme", "fold", "model", "im", "period", "n", "rmse", "bias", "nll", "cov90", "width90",
                "tau_test", "phi_s2s_test", "phi_0_test"]
        frames = []
        for sub, kind in (("baselines", "classical"), ("baselines_random", "classical"),
                          ("ml", "ml"), ("ml_random", "ml")):
            name = "baseline_metrics.csv" if kind == "classical" else "ml_metrics.csv"
            path = self.cfg.results / sub / name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[(df["profile"] == prof) & (df["scheme"] == "random")]
            df = df[df["mode"] == "ergodic"] if kind == "classical" else df[df["part"] == "test"]
            if len(df):
                self.used.add(path)
                frames.append(df[cols].assign(family=kind))
        if not frames:
            raise MissingInput("no 'random' scheme results: run baselines/train with --schemes random "
                               "into results/baselines_random and results/ml_random")
        return pd.concat(frames, ignore_index=True).drop_duplicates(["scheme", "fold", "model", "im"])
