"""High-level plug-in API: benchmark any ground-motion model on the curated NGA-West2 data.

    import gmpe_bench as gb

    class MyGMM(gb.BaseGMM):
        def fit(self, X_train, y_train, metadata_train): ...
        def predict(self, X_test, metadata_test): return mu, sigma   # (n, 27) each

    report = gb.evaluate(MyGMM(), profile="full")
    print(report.summary())

Scope
-----
:func:`evaluate`, :func:`evaluate_predictions` and :func:`iter_folds` run only on a curated NGA-West2
profile built locally by ``gmpe-bench build`` (``data/processed/full.parquet`` or ``sp23.parquet``) with
the frozen splits shipped in the repository. They never accept synthetic or user-supplied datasets,
so every number they report is an NGA-West2 benchmark number. Installation checks without the PEER
flatfiles are the job of ``gmpe-bench selftest`` (a separate, clearly labelled smoke test).

What :func:`evaluate` guarantees
--------------------------------
* The profile must match its frozen ``splits_<profile>.json`` (RSN fingerprint), and every fold passes
  the leakage audit (:mod:`gmpe_bench.splits.leakage`) before a model is fitted; a failing fold raises
  :class:`LeakageError`.
* The model is refitted from an unfitted copy in every fold, on ``train_fit`` rows only. It never
  receives calibration or test targets; calibration targets are used by the interval methods alone.
* Accuracy and coverage carry event-clustered bootstrap intervals (whole earthquakes resampled).
* Intervals come from the paper's calibrators (event-clustered conformal prediction and HSCP).
* Results are compared with the stored paper benchmarks (``ASK14 (site)`` and ``Hybrid BSSA14+XGB``,
  ``results/paper/paper_summary_numbers.json``) when the profile is the one they were computed on.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gmpe_bench.conformal.evaluate import interval_metrics, joint_coverage
from gmpe_bench.conformal.methods import (
    HSCP,
    CalSet,
    ClusteredEventConformal,
    ConformalCalibrator,
    NaiveSplitConformal,
    TestInputs,
    UncalibratedGaussian,
    WeightedShiftConformal,
)
from gmpe_bench.conformal.shift import density_ratio_weights, shift_features
from gmpe_bench.data import GMDataset
from gmpe_bench.schema import IM_NAMES
from gmpe_bench.splits.freeze import load_splits, rsn_fingerprint
from gmpe_bench.splits.group_splits import SplitFold, assign_region
from gmpe_bench.splits.leakage import LeakageReport, audit_fold, leakage_table

K = len(IM_NAMES)
LEAKAGE_FREE_SCHEMES: tuple[str, ...] = ("event_grouped", "station_grouped", "spatial_block", "temporal",
                                         "magnitude_holdout", "cl1_to_cl2")
ALL_SCHEMES: tuple[str, ...] = LEAKAGE_FREE_SCHEMES + ("random",)
SCHEME_LABELS = {"event_grouped": "Event", "station_grouped": "Station", "spatial_block": "Region",
                 "temporal": "Temporal", "magnitude_holdout": "Mw>=7", "cl1_to_cl2": "CL1->CL2",
                 "random": "Random (leaky)"}
REFERENCE_MODELS = {"ASK14_site": "ASK14 (site)", "Hybrid_BSSA14_XGB": "Hybrid BSSA14+XGB"}
CALIBRATORS: dict[str, Callable[[], ConformalCalibrator]] = {
    "UncalibratedGaussian": UncalibratedGaussian,
    "NaiveSplitConformal": NaiveSplitConformal,
    "ClusteredEventConformal": ClusteredEventConformal,
    "WeightedShiftConformal": WeightedShiftConformal,
    "HSCP": HSCP,
}
METHOD_LABELS = {"UncalibratedGaussian": "Gaussian", "NaiveSplitConformal": "Split CP",
                 "ClusteredEventConformal": "Event-clustered CP", "WeightedShiftConformal": "Shift-weighted CP",
                 "HSCP": "HSCP"}
DEFAULT_METHODS: tuple[str, ...] = ("ClusteredEventConformal", "HSCP")
META_COLUMNS = ("rsn", "eqid", "ssn", "eq_name", "year", "cl_class", "mw", "rjb", "rrup", "vs30",
                "hypo_lat", "hypo_lon", "sta_lat", "sta_lon")
SIGMA_FLOOR = 1e-3
DEFAULT_DATA_DIR = Path("data") / "processed"
BOOT_SEED = 20261122
REFERENCE_BUNDLE = Path(__file__).with_name("reference") / "reference_benchmarks.json"
PAPER_JSON = Path("results") / "paper" / "paper_summary_numbers.json"


# =============================================================================== errors
class DataNotFoundError(FileNotFoundError):
    """The local NGA-West2 profile (or its frozen splits) is not available."""


class LeakageError(RuntimeError):
    """A fold failed the leakage audit; no model was fitted on it."""


def _missing_data_message(path: Path, what: str = "profile") -> str:
    return (
        f"gmpe-bench could not find the curated NGA-West2 {what} at '{path}'.\n\n"
        "evaluate() runs only on the real NGA-West2 data, which PEER does not allow us to redistribute.\n"
        "Build it once (about a minute):\n"
        "  1. Download the NGA-West2 flatfiles from the PEER ground-motion database (https://peer.berkeley.edu):\n"
        "     the horizontal RotD50 flatfile and the vertical flatfile (January 2015 update), saved as\n"
        "     nga_w_h.xlsx and nga_w_v.xlsx.\n"
        "  2. From the repository root run:\n"
        "         gmpe-bench build --h nga_w_h.xlsx --v nga_w_v.xlsx\n"
        "     This writes data/processed/full.parquet and sp23.parquet. The frozen splits\n"
        "     (data/processed/splits_full.json, splits_sp23.json) ship with the repository and must not be\n"
        "     regenerated.\n"
        "  3. Call evaluate() from the repository root, or pass data_dir=... (or set GMPE_BENCH_DATA).\n\n"
        "To check that the package itself is installed correctly, run `gmpe-bench selftest`."
    )


# =============================================================================== model interface
class BaseGMM(ABC):
    """Interface for a user ground-motion model (median and standard deviation for 27 IMs).

    ``fit(X_train, y_train, metadata_train)``
        ``X_train``  DataFrame of Tier-B inputs (``GMDataset.X("B")``; may contain NaN).
        ``y_train``  array (n, 27) of ln targets in ``gmpe_bench.IM_NAMES`` order; NaN where masked.
        ``metadata_train``  DataFrame with ``rsn``, ``eqid``, ``ssn``, ``region`` and other metadata, so
        models can learn station or event terms. Returns ``self``.

    ``predict(X_test, metadata_test)``
        Returns ``(mu, sigma)`` with shape (n, 27) each, or ``mu`` alone. ``sigma`` may also be a scalar
        or a length-27 vector. If it is missing, :attr:`default_sigma` is used when set; otherwise the
        per-IM standard deviation of the model's own train-fit residuals.

    The model passed to :func:`evaluate` is a template: it is deep-copied and refitted in every fold.
    """

    #: Fallback standard deviation (ln units) when ``predict`` returns ``mu`` only. ``None`` means
    #: "use the per-IM standard deviation of the train-fit residuals".
    default_sigma: float | None = None

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, metadata_train: pd.DataFrame) -> BaseGMM:
        ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame, metadata_test: pd.DataFrame
                ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        ...

    def predict_with_sigma(self, X: pd.DataFrame, metadata: pd.DataFrame,
                           fallback: np.ndarray | float | None = None) -> tuple[np.ndarray, np.ndarray, str]:
        """``(mu, sigma, source)`` with sigma always filled; ``source`` says where sigma came from."""
        return _resolve_output(self.predict(X, metadata), len(X), fallback if fallback is not None
                               else self.default_sigma)


def _as_2d(a: Any, n: int, what: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full((n, K), float(arr))
    elif arr.ndim == 1 and arr.size == K:
        arr = np.broadcast_to(arr, (n, K)).copy()
    if arr.shape != (n, K):
        raise ValueError(f"{what} must have shape ({n}, {K}) (records x IMs in gmpe_bench.IM_NAMES order); "
                         f"got {arr.shape}")
    return arr


def _resolve_output(out: Any, n: int, fallback: np.ndarray | float | None) -> tuple[np.ndarray, np.ndarray, str]:
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError("predict must return mu or (mu, sigma)")
        mu_raw, sd_raw = out
    else:
        mu_raw, sd_raw = out, None
    mu = _as_2d(mu_raw, n, "mu")
    if sd_raw is not None:
        sd, source = _as_2d(sd_raw, n, "sigma"), "model"
        if np.any(np.isfinite(sd) & (sd <= 0)):
            raise ValueError("sigma must be positive")
    elif fallback is not None:
        sd, source = _as_2d(fallback, n, "default sigma"), "fallback"
    else:
        sd, source = np.full((n, K), np.nan), "missing"
    return mu, np.clip(sd, SIGMA_FLOOR, None), source


def _train_residual_sigma(y: np.ndarray, mu: np.ndarray, default: float = 0.6) -> np.ndarray:
    """Per-IM standard deviation of train-fit residuals (the fallback sigma)."""
    r = np.where(np.isfinite(y) & np.isfinite(mu), y - mu, np.nan)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(r).sum(axis=0)
        sd = np.sqrt(np.nansum(r**2, axis=0) / np.maximum(cnt - 1, 1))
    return np.where(cnt > 2, np.clip(sd, 0.05, None), default)


# =============================================================================== built-in models
class LinearGMM(BaseGMM):
    """Ergodic, linear-in-parameters GMM fitted per IM by least squares (a published-model analog).

    ln y = b0 + b1 m + b2 m^2 + b3 max(M - 6.75, 0) + (b4 + b5 m) ln sqrt(R_JB^2 + 36) + b6 R_JB
           + b7 ln(VS30 / 760) + b8 F_NM + b9 F_RV,   m = M - 6.

    It has no event or station terms, so it cannot memorize either; sigma is the per-IM residual
    standard deviation. Missing inputs are filled with train-fit medians.
    """

    def __init__(self, ridge: float = 1e-6) -> None:
        self.ridge = ridge

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        x = X.fillna(self.medians_)
        m = x["mw"].to_numpy() - 6.0
        rjb = np.exp(x["ln_rjb"].to_numpy())
        ln_r = 0.5 * np.log(rjb**2 + 36.0)
        return np.column_stack([np.ones(len(x)), m, m**2, np.maximum(m - 0.75, 0.0), ln_r, m * ln_r, rjb,
                                x["ln_vs30"].to_numpy() - np.log(760.0), x["mech_nm"].to_numpy(),
                                x["mech_rv"].to_numpy()])

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, metadata_train: pd.DataFrame) -> LinearGMM:
        self.medians_ = X_train.median(numeric_only=True)
        A = self._design(X_train)
        p = A.shape[1]
        self.coef_ = np.zeros((p, K))
        self.sigma_ = np.full(K, 0.6)
        for k in range(K):
            ok = np.isfinite(y_train[:, k])
            if ok.sum() <= p:
                continue
            Ak, yk = A[ok], y_train[ok, k]
            self.coef_[:, k] = np.linalg.solve(Ak.T @ Ak + self.ridge * np.eye(p), Ak.T @ yk)
            self.sigma_[k] = float(np.std(yk - Ak @ self.coef_[:, k], ddof=p))
        return self

    def predict(self, X_test: pd.DataFrame, metadata_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = self._design(X_test) @ self.coef_
        return mu, np.broadcast_to(self.sigma_, mu.shape).copy()


# =============================================================================== data access
def _data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("GMPE_BENCH_DATA")
    return Path(env) if env else DEFAULT_DATA_DIR


PROFILES: tuple[str, ...] = ("full", "sp23")


@dataclass
class _Bench:
    ds: GMDataset
    splits: dict[str, list[SplitFold]]
    source: str          # "local" (curated NGA-West2) or "fixture" (internal smoke test only)
    data_dir: Path | None


def _is_fixture(ds: GMDataset) -> bool:
    return "data_source" in ds.df and bool(ds.df["data_source"].astype("string").notna().any())


def load_benchmark(profile: str = "full", data_dir: str | Path | None = None) -> _Bench:
    """Curated NGA-West2 profile and its frozen, fingerprint-checked splits."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    root = _data_dir(data_dir)
    path = root / f"{profile}.parquet"
    if not path.exists():
        raise DataNotFoundError(_missing_data_message(path))
    split_path = root / f"splits_{profile}.json"
    if not split_path.exists():
        raise DataNotFoundError(_missing_data_message(split_path, "frozen splits"))
    ds = GMDataset.from_parquet(path, profile)
    if _is_fixture(ds):
        raise ValueError(f"{path} is not a curated NGA-West2 profile (it carries a 'data_source' marker of "
                         "generated data). evaluate() runs only on profiles built by `gmpe-bench build` from the "
                         "PEER flatfiles.")
    return _Bench(ds, load_splits(split_path, ds), "local", root)


def _metadata(ds: GMDataset) -> pd.DataFrame:
    meta = ds.df[[c for c in META_COLUMNS if c in ds.df]].reset_index(drop=True).copy()
    meta["region"] = assign_region(ds.df).to_numpy()
    return meta


@dataclass
class FoldData:
    """One fold for external training. Targets are given for ``train`` rows only."""

    scheme: str
    fold: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    meta_train: pd.DataFrame
    X_cal: pd.DataFrame
    meta_cal: pd.DataFrame
    X_test: pd.DataFrame
    meta_test: pd.DataFrame

    @property
    def rsn_cal(self) -> np.ndarray:
        return self.meta_cal["rsn"].to_numpy()

    @property
    def rsn_test(self) -> np.ndarray:
        return self.meta_test["rsn"].to_numpy()

    def predictions_frame(self, mu_cal: np.ndarray, sigma_cal: np.ndarray | None, mu_test: np.ndarray,
                          sigma_test: np.ndarray | None) -> pd.DataFrame:
        """Long-format frame for :func:`evaluate_predictions` (cal and test rows of this fold)."""
        parts = []
        for rsn, mu, sd in ((self.rsn_cal, mu_cal, sigma_cal), (self.rsn_test, mu_test, sigma_test)):
            mu = _as_2d(mu, len(rsn), "mu")
            f = pd.DataFrame({"rsn": np.repeat(rsn, K), "im": np.tile(IM_NAMES, len(rsn)), "mu": mu.ravel()})
            if sd is not None:
                f["sigma"] = _as_2d(sd, len(rsn), "sigma").ravel()
            parts.append(f)
        return pd.concat(parts, ignore_index=True).assign(scheme=self.scheme, fold=self.fold)


def iter_folds(profile: str = "full", schemes: tuple[str, ...] | list[str] | None = None,
               data_dir: str | Path | None = None, audit: bool = True) -> Iterator[FoldData]:
    """Yield every audited fold for training outside gmpe-bench (PyTorch, JAX ...).

    Only train-fit targets are handed out. Carve any early-stopping set from the training rows;
    calibration and test targets stay inside the benchmark and are used by :func:`evaluate_predictions`.
    """
    schemes = _check_schemes(schemes)
    bench = load_benchmark(profile, data_dir)
    X, Y, meta = bench.ds.X("B"), bench.ds.Y("H").to_numpy(), _metadata(bench.ds)
    for scheme in schemes:
        for fold in bench.splits.get(scheme, []):
            if audit:
                _audit_or_raise(bench.ds, fold)
            tr, ca, te = fold.train_fit, fold.cal, fold.test
            yield FoldData(scheme, fold.fold, X.iloc[tr].reset_index(drop=True), Y[tr].copy(),
                           meta.iloc[tr].reset_index(drop=True), X.iloc[ca].reset_index(drop=True),
                           meta.iloc[ca].reset_index(drop=True), X.iloc[te].reset_index(drop=True),
                           meta.iloc[te].reset_index(drop=True))


def _check_schemes(schemes: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if schemes is None:
        return LEAKAGE_FREE_SCHEMES
    if isinstance(schemes, str):
        schemes = (schemes,)
    bad = [s for s in schemes if s not in ALL_SCHEMES]
    if bad:
        raise ValueError(f"unknown scheme(s) {bad}; choose from {list(ALL_SCHEMES)}")
    return tuple(schemes)


def _audit_or_raise(ds: GMDataset, fold: SplitFold) -> LeakageReport:
    rep = audit_fold(ds, fold)
    if not rep.passed:
        raise LeakageError(f"{fold.scheme}/{fold.fold} failed the leakage audit: {'; '.join(rep.issues)}")
    return rep


# =============================================================================== reference numbers
def build_reference_bundle(paper_json: str | Path, splits_dir: str | Path | None = None) -> dict[str, Any]:
    """Extract the stored paper benchmarks used for comparison (accuracy and calibration)."""
    d = json.loads(Path(paper_json).read_text(encoding="utf-8"))
    acc = {s: {m: {"rmse": v[m]["rmse"], "lo": v[m].get("rmse_ci", {}).get("lo"),
                   "hi": v[m].get("rmse_ci", {}).get("hi"), "cov90": v[m].get("cov90")}
               for m in REFERENCE_MODELS if m in v}
           for s, v in d.get("accuracy", {}).items()}
    conf = {lvl: {s: {m: {"coverage": v[m]["coverage"], "width": v[m]["width"],
                          "lo": v[m].get("coverage_ci", {}).get("lo"), "hi": v[m].get("coverage_ci", {}).get("hi"),
                          "joint_band_coverage": v[m].get("joint_band_coverage")}
                      for m in CALIBRATORS if m in v}
                  for s, v in by_s.items()}
            for lvl, by_s in d.get("conformal", {}).items()}
    fp: dict[str, str] = {}
    if splits_dir is not None:
        for f in sorted(Path(splits_dir).glob("splits_*.json")):
            payload = json.loads(f.read_text(encoding="utf-8"))
            if "rsn_sha256" in payload and "profile" in payload:   # skips splits_manifest.json
                fp[payload["profile"]] = payload["rsn_sha256"]
    return {"source": "results/paper/paper_summary_numbers.json", "profile": d.get("meta", {}).get("profile", "full"),
            "note": "Conformal references are means over the paper's eight models.",
            "rsn_sha256": fp, "accuracy": acc, "conformal": conf}


def load_reference(path: str | Path | None = None) -> dict[str, Any]:
    """Stored paper benchmarks: ``path``, else ``./results/paper/...`` or the repository copy, else the
    bundled extract shipped with the package (which also holds the dataset fingerprints)."""
    bundled = json.loads(REFERENCE_BUNDLE.read_text(encoding="utf-8"))
    candidates = [Path(path)] if path is not None else [Path.cwd() / PAPER_JSON,
                                                        Path(__file__).resolve().parents[2] / PAPER_JSON]
    for c in candidates:
        if c.exists() and c.suffix == ".json":
            payload = json.loads(c.read_text(encoding="utf-8"))
            ref = build_reference_bundle(c) if "accuracy" in payload and "meta" in payload else payload
            ref["rsn_sha256"] = ref.get("rsn_sha256") or bundled["rsn_sha256"]
            ref["source"] = str(c)
            return ref
    if path is not None:
        raise FileNotFoundError(f"reference file {path} not found")
    bundled["source"] = f"bundled ({REFERENCE_BUNDLE.name})"
    return bundled


# =============================================================================== fold scoring
@dataclass
class _FoldOut:
    row: dict[str, Any]
    acc: pd.DataFrame                    # per (eqid): sse_<k>, n_<k>, cov_<k> (Gaussian 90 %)
    cal_rows: list[dict[str, Any]] = field(default_factory=list)
    cov: dict[tuple[str, float], pd.DataFrame] = field(default_factory=dict)   # per (eqid): c_<k>, n_<k>


def _per_event(eqid: np.ndarray, **mats: np.ndarray) -> pd.DataFrame:
    cols = {f"{p}_{k}": m[:, k] for p, m in mats.items() for k in range(K)}
    return pd.DataFrame(cols).assign(eqid=eqid).groupby("eqid", sort=True).sum()


def _score_fold(ds: GMDataset, fold: SplitFold, Y: np.ndarray, mu_c: np.ndarray | None, sd_c: np.ndarray | None,
                mu_t: np.ndarray, sd_t: np.ndarray, calibrate: bool, alphas: tuple[float, ...],
                methods: tuple[str, ...]) -> _FoldOut:
    df = ds.df
    test_df = df.iloc[fold.test]
    y_t = Y[fold.test]
    ok = np.isfinite(y_t) & np.isfinite(mu_t)
    err2 = np.where(ok, (y_t - mu_t) ** 2, 0.0)
    z = 1.6448536269514722
    gauss_ok = ok & np.isfinite(sd_t)
    gcov = gauss_ok & (np.abs(y_t - mu_t) <= z * np.where(gauss_ok, sd_t, 0.0))
    acc = _per_event(test_df["eqid"].to_numpy(), sse=err2, n=ok.astype(float), gc=gcov.astype(float),
                     gn=gauss_ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse_k = np.sqrt(err2.sum(0) / np.where(ok.sum(0) > 0, ok.sum(0), np.nan))
        g_k = gcov.sum(0) / np.where(gauss_ok.sum(0) > 0, gauss_ok.sum(0), np.nan)
    row = {"scheme": fold.scheme, "fold": fold.fold, "n_train_fit": int(fold.train_fit.size),
           "n_cal": int(fold.cal.size), "n_test": int(fold.test.size),
           "n_test_events": int(test_df["eqid"].nunique()), "rmse": float(np.nanmean(rmse_k)),
           "gauss_cov90": float(np.nanmean(g_k)) if np.isfinite(g_k).any() else np.nan,
           "pred_coverage": float(ok.sum() / max(np.isfinite(y_t).sum(), 1))}
    out = _FoldOut(row, acc)
    if not calibrate or mu_c is None or sd_c is None or fold.cal.size == 0:
        return out
    cal_df = df.iloc[fold.cal]
    y_c = Y[fold.cal]
    keep = np.isfinite(mu_c).all(axis=1) & np.isfinite(sd_c).all(axis=1)
    if keep.sum() < 20 or not (np.isfinite(mu_t).all() and np.isfinite(sd_t).all()):
        out.row["calibration_note"] = "skipped: calibration or test predictions incomplete"
        return out
    Xc, Xt = shift_features(cal_df[keep]), shift_features(test_df)
    wc, wt = density_ratio_weights(Xc, Xt)
    train_st = np.unique(df["ssn"].to_numpy()[fold.train_fit])
    cal = CalSet(y_c[keep], mu_c[keep], sd_c[keep], cal_df["eqid"].to_numpy()[keep], cal_df["ssn"].to_numpy()[keep],
                 Xc, train_st, wc)
    test = TestInputs(mu_t, sd_t, test_df["eqid"].to_numpy(), test_df["ssn"].to_numpy(), Xt, wt)
    for mname in methods:
        method = CALIBRATORS[mname]()
        for alpha, iv in method.calibrate_many(cal, test, alphas).items():
            m = interval_metrics(y_t, iv)
            fin = np.isfinite(y_t)
            covered = (y_t >= iv.lower) & (y_t <= iv.upper) & fin
            rec = {"scheme": fold.scheme, "fold": fold.fold, "method": mname, "nominal": round(1 - alpha, 4),
                   "coverage": float(np.nanmean(m["coverage"])), "width": float(np.nanmean(m["width"])),
                   "joint_marginal": joint_coverage(y_t, iv.lower, iv.upper),
                   "joint_band": (joint_coverage(y_t, iv.joint_lower, iv.joint_upper)
                                  if iv.joint_lower is not None and iv.joint_upper is not None else np.nan),
                   "n_inf": int(iv.n_inf), "ess": float(iv.ess) if iv.ess is not None else np.nan}
            out.cal_rows.append(rec)
            out.cov[(mname, round(1 - alpha, 4))] = _per_event(test_df["eqid"].to_numpy(),
                                                               c=covered.astype(float), n=fin.astype(float))
    return out


# =============================================================================== bootstrap
def _stack(frames: list[pd.DataFrame], prefixes: tuple[str, str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Event x (fold cell, IM) matrices of two per-event tables, aligned on the union of events."""
    events = pd.Index(sorted(set().union(*[f.index for f in frames])))
    a = np.hstack([f.reindex(events, fill_value=0.0)[[f"{prefixes[0]}_{k}" for k in range(K)]].to_numpy()
                   for f in frames])
    b = np.hstack([f.reindex(events, fill_value=0.0)[[f"{prefixes[1]}_{k}" for k in range(K)]].to_numpy()
                   for f in frames])
    return a, b, len(frames)


def cluster_weights(n_clusters: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, n_clusters) multiplicities of a with-replacement draw of whole earthquakes.

    Same algorithm as :func:`gmpe_bench.reporting.bootstrap.cluster_weights`; repeated here so the API does
    not import the plotting stack.
    """
    draws = rng.integers(0, n_clusters, size=(b, n_clusters))
    w = np.zeros((b, n_clusters), dtype=np.float64)
    np.add.at(w, (np.repeat(np.arange(b), n_clusters), draws.ravel()), 1.0)
    return w


def _rmse_from(W: np.ndarray, sse: np.ndarray, n: np.ndarray) -> np.ndarray:
    valid = n.sum(axis=0) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        num, den = W @ sse[:, valid], W @ n[:, valid]
        return np.nanmean(np.sqrt(num / np.where(den > 0, den, np.nan)), axis=1)


def _cov_from(W: np.ndarray, c: np.ndarray, n: np.ndarray, cells: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = ((W @ c) / np.where(W @ n > 0, W @ n, np.nan)).reshape(W.shape[0], cells, K)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.nanmean(r, axis=2), axis=1)


def _ci(reps: np.ndarray) -> tuple[float, float]:
    reps = reps[np.isfinite(reps)]
    if reps.size < 2:
        return np.nan, np.nan
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def _rng(seed: int, *tags: object) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32("|".join(map(str, tags)).encode())])


# =============================================================================== report
@dataclass
class BenchmarkReport:
    """Result of :func:`evaluate` / :func:`evaluate_predictions`."""

    model: str
    profile: str
    dataset: str
    accuracy: pd.DataFrame
    calibration: pd.DataFrame
    folds: pd.DataFrame
    audit: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- views
    def to_dataframe(self) -> pd.DataFrame:
        """Tidy table: one row per (scheme, metric[, method, nominal]) with CI and paper references."""
        rows: list[dict[str, Any]] = []
        for r in self.accuracy.to_dict("records"):
            base = {"model": self.model, "profile": self.profile, "dataset": self.dataset, "scheme": r["scheme"]}
            rows.append({**base, "metric": "rmse", "method": None, "nominal": None, "value": r["rmse"],
                         "ci_lo": r["rmse_lo"], "ci_hi": r["rmse_hi"],
                         **{f"ref_{m}": r.get(f"ref_{m}") for m in REFERENCE_MODELS}})
            rows.append({**base, "metric": "gaussian_coverage_90", "method": "UncalibratedGaussian",
                         "nominal": 0.9, "value": r["gauss_cov90"], "ci_lo": np.nan, "ci_hi": np.nan})
        for r in self.calibration.to_dict("records"):
            base = {"model": self.model, "profile": self.profile, "dataset": self.dataset, "scheme": r["scheme"],
                    "method": r["method"], "nominal": r["nominal"]}
            rows.append({**base, "metric": "coverage", "value": r["coverage"], "ci_lo": r["coverage_lo"],
                         "ci_hi": r["coverage_hi"], "ref_paper_mean": r.get("ref_coverage")})
            rows.append({**base, "metric": "width", "value": r["width"], "ci_lo": np.nan, "ci_hi": np.nan,
                         "ref_paper_mean": r.get("ref_width")})
            if np.isfinite(r.get("joint_band", np.nan)):
                rows.append({**base, "metric": "joint_band_coverage", "value": r["joint_band"],
                             "ci_lo": np.nan, "ci_hi": np.nan})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Plain-text summary table (also printed by ``print(report)``)."""
        lines: list[str] = []
        m = self.meta
        audit_ok = bool(self.audit["passed"].all()) if len(self.audit) else False
        lines.append(f"gmpe-bench {m.get('version', '')} | model {self.model} | profile {self.profile} | "
                     f"data {self.dataset} | {len(self.folds)} folds | leakage audit "
                     f"{'PASSED' if audit_ok else 'FAILED'}")
        b = m.get("n_boot", 0)
        lines.append("Accuracy: RMSE in ln units, mean over folds and 27 IMs"
                     + (f"; event-clustered 95% CI (B = {b})" if b else ""))
        hdr = f"  {'regime':<16}{'RMSE':>7} {'95% CI':<17}{'ASK14 (site)':>13}{'Hyb BSSA14+XGB':>16}{'Gauss cov90':>13}"
        lines.append(hdr)
        for r in self.accuracy.to_dict("records"):
            ci = f"[{r['rmse_lo']:.3f}, {r['rmse_hi']:.3f}]" if np.isfinite(r["rmse_lo"]) else ""
            refs = [f"{r.get('ref_' + k):.3f}" if r.get("ref_" + k) is not None and np.isfinite(r.get("ref_" + k))
                    else "-" for k in REFERENCE_MODELS]
            lines.append(f"  {SCHEME_LABELS.get(r['scheme'], r['scheme']):<16}{r['rmse']:>7.3f} {ci:<17}"
                         f"{refs[0]:>13}{refs[1]:>16}{r['gauss_cov90']:>13.3f}")
        if m.get("comparable", False):
            lines.append(f"  References: paper results on the same folds ({m.get('reference_source')}); "
                         "compare intervals, not only point values.")
        else:
            lines.append(f"  Paper references not shown: {m.get('comparable_reason', 'dataset differs')}.")
        if "leakage_inflation" in m:
            lines.append(f"  Leakage inflation (Event vs Random RMSE): {m['leakage_inflation']:+.1%}")
        if len(self.calibration):
            for nominal, g in self.calibration.groupby("nominal", sort=False):
                lines.append(f"Calibration at {nominal:.0%}: coverage [95% CI] / mean width (ln units)")
                methods = list(dict.fromkeys(g["method"]))
                lines.append("  " + f"{'regime':<16}" + "".join(f"{METHOD_LABELS.get(x, x):>28}" for x in methods)
                             + (f"{'HSCP joint band':>17}" if "HSCP" in methods else ""))
                for scheme, gs in g.groupby("scheme", sort=False):
                    cells = []
                    for x in methods:
                        rr = gs[gs["method"] == x]
                        if rr.empty:
                            cells.append(f"{'-':>28}")
                            continue
                        r = rr.iloc[0]
                        ci = f"[{r['coverage_lo']:.3f},{r['coverage_hi']:.3f}]" if np.isfinite(r["coverage_lo"]) else ""
                        cells.append(f"{r['coverage']:>8.3f} {ci:<15}{r['width']:>4.2f}")
                    jb = gs[gs["method"] == "HSCP"]["joint_band"]
                    jtxt = f"{jb.iloc[0]:>17.3f}" if len(jb) and np.isfinite(jb.iloc[0]) else ""
                    lines.append(f"  {SCHEME_LABELS.get(scheme, scheme):<16}" + "".join(cells) + jtxt)
        for note in m.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        return "<pre>" + self.summary() + "</pre>"

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Write ``report.json`` plus accuracy, calibration, fold and audit CSVs and ``summary.txt``."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.accuracy.to_csv(out / "accuracy.csv", index=False)
        self.calibration.to_csv(out / "calibration.csv", index=False)
        self.folds.to_csv(out / "folds.csv", index=False)
        self.audit.to_csv(out / "audit.csv", index=False)
        self.to_dataframe().to_csv(out / "report_long.csv", index=False)
        (out / "summary.txt").write_text(self.summary() + "\n", encoding="utf-8")

        def clean(o: Any) -> Any:
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, float)):
                return None if not np.isfinite(o) else float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return o

        payload = {"model": self.model, "profile": self.profile, "dataset": self.dataset, "meta": clean(self.meta),
                   "accuracy": clean(self.accuracy.to_dict("records")),
                   "calibration": clean(self.calibration.to_dict("records")),
                   "folds": clean(self.folds.to_dict("records"))}
        (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkReport:
        p = Path(path)
        d = json.loads((p / "report.json").read_text(encoding="utf-8"))
        return cls(d["model"], d["profile"], d["dataset"], pd.read_csv(p / "accuracy.csv"),
                   pd.read_csv(p / "calibration.csv"), pd.read_csv(p / "folds.csv"), pd.read_csv(p / "audit.csv"),
                   d["meta"])


def _assemble(name: str, bench: _Bench, profile: str, outs: list[_FoldOut], audits: list[LeakageReport],
              n_boot: int, seed: int, reference: dict[str, Any] | None, meta: dict[str, Any]) -> BenchmarkReport:
    folds = pd.DataFrame([o.row for o in outs])
    acc_rows, cal_rows = [], []
    ref = reference or {}
    fp = rsn_fingerprint(bench.ds)
    ref_fp = ref.get("rsn_sha256", {}).get(profile)
    comparable = bool(ref) and bench.source == "local" and ref_fp == fp and profile == ref.get("profile", "full")
    reason = ("installation smoke-test fixture, not NGA-West2" if bench.source == "fixture" else
              f"no stored benchmarks for profile '{profile}'" if profile != ref.get("profile", "full") else
              "dataset fingerprint differs from the paper data" if ref_fp != fp else "")
    for scheme in dict.fromkeys(folds["scheme"]) if len(folds) else []:
        so = [o for o in outs if o.row["scheme"] == scheme]
        sse, n, cells = _stack([o.acc for o in so], ("sse", "n"))
        gc, gn, _ = _stack([o.acc for o in so], ("gc", "gn"))
        est = float(_rmse_from(np.ones((1, sse.shape[0])), sse, n)[0])
        gest = float(_cov_from(np.ones((1, gc.shape[0])), gc, gn, cells)[0])
        lo = hi = np.nan
        if n_boot > 0 and sse.shape[0] > 1:
            W = cluster_weights(sse.shape[0], n_boot, _rng(seed, scheme, "rmse"))
            lo, hi = _ci(_rmse_from(W, sse, n))
        row = {"scheme": scheme, "rmse": est, "rmse_lo": lo, "rmse_hi": hi, "gauss_cov90": gest,
               "n_folds": len(so), "n_test_records": int(sum(o.row["n_test"] for o in so)),
               "n_test_events": int(sse.shape[0])}
        for key in REFERENCE_MODELS:
            r = ref.get("accuracy", {}).get(scheme, {}).get(key) if comparable else None
            row[f"ref_{key}"] = r["rmse"] if r else np.nan
            row[f"ref_{key}_lo"] = r["lo"] if r and r.get("lo") is not None else np.nan
            row[f"ref_{key}_hi"] = r["hi"] if r and r.get("hi") is not None else np.nan
        acc_rows.append(row)

        keys = list(dict.fromkeys(k for o in so for k in o.cov))
        for mname, nominal in keys:
            frames = [o.cov[(mname, nominal)] for o in so if (mname, nominal) in o.cov]
            c, nn, cells_c = _stack(frames, ("c", "n"))
            est_c = float(_cov_from(np.ones((1, c.shape[0])), c, nn, cells_c)[0])
            clo = chi = np.nan
            if n_boot > 0 and c.shape[0] > 1:
                W = cluster_weights(c.shape[0], n_boot, _rng(seed, scheme, mname, nominal))
                clo, chi = _ci(_cov_from(W, c, nn, cells_c))
            fr = pd.DataFrame([r for o in so for r in o.cal_rows if r["method"] == mname and r["nominal"] == nominal])
            lvl = f"{nominal:.2f}"
            rr = ref.get("conformal", {}).get(lvl, {}).get(scheme, {}).get(mname) if comparable else None
            cal_rows.append({"scheme": scheme, "method": mname, "nominal": nominal, "coverage": est_c,
                             "coverage_lo": clo, "coverage_hi": chi, "width": float(fr["width"].mean()),
                             "joint_marginal": float(fr["joint_marginal"].mean()),
                             "joint_band": float(fr["joint_band"].mean()), "n_inf": int(fr["n_inf"].sum()),
                             "ess": float(fr["ess"].mean()) if fr["ess"].notna().any() else np.nan,
                             "n_folds": len(fr), "ref_coverage": rr["coverage"] if rr else np.nan,
                             "ref_width": rr["width"] if rr else np.nan})
    accuracy = pd.DataFrame(acc_rows)
    meta = {**meta, "comparable": comparable, "comparable_reason": reason or None, "rsn_sha256": fp,
            "n_boot": n_boot, "bootstrap_seed": seed, "reference_source": ref.get("source")}
    if {"random", "event_grouped"} <= set(accuracy.get("scheme", [])):
        r = accuracy.set_index("scheme")["rmse"]
        meta["leakage_inflation"] = float(r["event_grouped"] / r["random"] - 1)
    return BenchmarkReport(name, profile, "NGA-West2 (curated, local)" if bench.source == "local"
                           else "SMOKE-TEST FIXTURE (synthetic; not NGA-West2)",
                           accuracy, pd.DataFrame(cal_rows), folds, leakage_table(audits), meta)


def _base_meta(n_boot: int, calibrate: bool, alphas: tuple[float, ...], methods: tuple[str, ...]) -> dict[str, Any]:
    from gmpe_bench import __version__

    return {"version": __version__, "python": platform.python_version(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "calibrate": calibrate, "alphas": list(alphas), "methods": list(methods) if calibrate else [],
            "notes": []}


def _check_methods(methods: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    methods = tuple(methods)
    bad = [x for x in methods if x not in CALIBRATORS]
    if bad:
        raise ValueError(f"unknown calibration method(s) {bad}; choose from {list(CALIBRATORS)}")
    return methods


# =============================================================================== evaluate
def evaluate(model: BaseGMM, profile: str = "full", schemes: tuple[str, ...] | list[str] | None = None,
             calibrate: bool = True, n_boot: int = 200, *, data_dir: str | Path | None = None,
             alphas: tuple[float, ...] = (0.10,), methods: tuple[str, ...] | list[str] = DEFAULT_METHODS,
             seed: int = BOOT_SEED, reference: str | Path | None = None,
             log: Callable[[str], None] | None = None) -> BenchmarkReport:
    """Benchmark ``model`` on the curated NGA-West2 profile and its frozen, audited splits.

    Parameters
    ----------
    model : BaseGMM (or any object with the same ``fit`` / ``predict`` signature); an unfitted template.
    profile : ``"full"`` or ``"sp23"``; read from ``data/processed`` (or ``data_dir`` / ``GMPE_BENCH_DATA``).
    schemes : regimes to run; default the six leakage-free regimes. Add ``"random"`` for the leaky reference.
    calibrate : calibrate intervals with ``methods`` (default event-clustered CP and HSCP).
    n_boot : event-clustered bootstrap replicates for 95 % intervals (0 disables).

    Raises
    ------
    DataNotFoundError : the curated profile or its frozen splits are missing (message explains the setup).
    LeakageError : a fold fails the leakage audit.
    """
    if not (callable(getattr(model, "fit", None)) and callable(getattr(model, "predict", None))):
        raise TypeError("model must implement fit(X_train, y_train, metadata_train) and "
                        "predict(X_test, metadata_test); subclass gmpe_bench.BaseGMM")
    schemes, methods = _check_schemes(schemes), _check_methods(methods)
    bench = load_benchmark(profile, data_dir)
    return _run_model(bench, model, profile, schemes, calibrate, n_boot, alphas, methods, seed,
                      load_reference(reference), log)


def _run_model(bench: _Bench, model: BaseGMM, profile: str, schemes: tuple[str, ...], calibrate: bool,
               n_boot: int, alphas: tuple[float, ...], methods: tuple[str, ...], seed: int,
               reference: dict[str, Any] | None, log: Callable[[str], None] | None) -> BenchmarkReport:
    """Engine behind :func:`evaluate` (also used by the installation smoke test on its fixture)."""
    say = log or (lambda s: None)
    ds = bench.ds
    X, Y, meta = ds.X("B"), ds.Y("H").to_numpy(), _metadata(ds)
    name = getattr(model, "name", type(model).__name__)
    info = _base_meta(n_boot, calibrate, alphas, methods)
    outs: list[_FoldOut] = []
    audits: list[LeakageReport] = []
    sigma_sources: set[str] = set()
    for scheme in schemes:
        folds = bench.splits.get(scheme, [])
        if not folds:
            info["notes"].append(f"scheme {scheme} has no folds for this dataset")
            continue
        for fold in folds:
            audits.append(_audit_or_raise(ds, fold))
            t0 = time.perf_counter()
            tr, ca, te = fold.train_fit, fold.cal, fold.test
            m = copy.deepcopy(model)
            ret = m.fit(X.iloc[tr].reset_index(drop=True), Y[tr].copy(), meta.iloc[tr].reset_index(drop=True))
            m = ret if ret is not None and callable(getattr(ret, "predict", None)) else m
            fallback = getattr(model, "default_sigma", None)

            def pred(idx: np.ndarray, m: Any = m, fallback: Any = fallback) -> tuple[np.ndarray, np.ndarray, str]:
                return _resolve_output(m.predict(X.iloc[idx].reset_index(drop=True),
                                                 meta.iloc[idx].reset_index(drop=True)), len(idx), fallback)

            mu_t, sd_t, src = pred(te)
            mu_c, sd_c, _ = pred(ca) if ca.size else (None, None, src)
            if src == "missing":
                mu_tr, _, _ = pred(tr)
                fb = _train_residual_sigma(Y[tr], mu_tr)
                sd_t = np.broadcast_to(fb, mu_t.shape).copy()
                sd_c = np.broadcast_to(fb, mu_c.shape).copy() if mu_c is not None else None
                src = "train_residual"
            sigma_sources.add(src)
            out = _score_fold(ds, fold, Y, mu_c, sd_c, mu_t, sd_t, calibrate, alphas, methods)
            out.row["seconds"] = round(time.perf_counter() - t0, 3)
            out.row["sigma_source"] = src
            outs.append(out)
            say(f"{scheme}/{fold.fold}: RMSE {out.row['rmse']:.3f} ({out.row['seconds']:.1f} s)")
    if not outs:
        raise ValueError(f"no folds found for schemes {list(schemes)}")
    info["sigma_source"] = sorted(sigma_sources)
    info["provenance"] = "fitted by gmpe-bench on train-fit rows of each fold"
    if "random" in schemes:
        info["notes"].append("'random' is the leaky reference split: test records share earthquakes and "
                             "stations with training.")
    return _assemble(name, bench, profile, outs, audits, n_boot, seed, reference, info)


# =============================================================================== evaluate_predictions
def evaluate_predictions(df: pd.DataFrame, profile: str = "full", calibrate: bool = True, *,
                         data_dir: str | Path | None = None, schemes: tuple[str, ...] | list[str] | None = None,
                         n_boot: int = 200, alphas: tuple[float, ...] = (0.10,),
                         methods: tuple[str, ...] | list[str] = DEFAULT_METHODS, name: str = "external",
                         default_sigma: float | None = None, seed: int = BOOT_SEED,
                         reference: str | Path | None = None) -> BenchmarkReport:
    """Score externally produced predictions on the audited folds of the curated NGA-West2 profile.

    ``df`` is long format with columns ``rsn``, ``im``, ``mu`` and optionally ``sigma``:

    * **Per-fold mode** (recommended): also give ``scheme`` and ``fold`` (see :func:`iter_folds`). Rows of
      the fold's calibration records calibrate the intervals; rows of its test records are scored.
    * **Out-of-fold mode**: without ``scheme``/``fold`` the frame holds one prediction per record from a
      cross-validated run of a single scheme (``schemes=["event_grouped"]`` by default): each record's
      prediction must come from the model of the fold in which it is a test record. Intervals then use
      cross-fitted calibration scores (as in CV+), which the report notes.

    Training provenance cannot be verified for external predictions; the report records it as declared.
    """
    need = {"rsn", "im", "mu"}
    if not need <= set(df.columns):
        raise ValueError(f"df needs columns {sorted(need)} (optional: sigma, scheme, fold); got {list(df.columns)}")
    per_fold = {"scheme", "fold"} <= set(df.columns)
    if per_fold:
        schemes = _check_schemes(schemes or tuple(dict.fromkeys(df["scheme"].astype(str))))
    else:
        schemes = _check_schemes(schemes or ("event_grouped",))
        if len(schemes) != 1:
            raise ValueError("without 'scheme' and 'fold' columns, pass exactly one scheme (out-of-fold mode)")
    methods = _check_methods(methods)
    bad_im = sorted(set(df["im"].astype(str)) - set(IM_NAMES))
    if bad_im:
        raise ValueError(f"unknown IM name(s) {bad_im[:5]}; use gmpe_bench.IM_NAMES: {IM_NAMES[:4]} ...")
    key = ["scheme", "fold", "rsn", "im"] if per_fold else ["rsn", "im"]
    if df.duplicated(key).any():
        raise ValueError(f"duplicate predictions for the same {key}")
    if "sigma" in df and (pd.to_numeric(df["sigma"], errors="coerce") <= 0).any():
        raise ValueError("sigma must be positive")

    bench = load_benchmark(profile, data_dir)
    return _run_predictions(bench, df, profile, schemes, per_fold, calibrate, n_boot, alphas, methods, name,
                            default_sigma, seed, load_reference(reference))


def _run_predictions(bench: _Bench, df: pd.DataFrame, profile: str, schemes: tuple[str, ...], per_fold: bool,
                     calibrate: bool, n_boot: int, alphas: tuple[float, ...], methods: tuple[str, ...], name: str,
                     default_sigma: float | None, seed: int, reference: dict[str, Any] | None) -> BenchmarkReport:
    """Engine behind :func:`evaluate_predictions`."""
    ds = bench.ds
    Y = ds.Y("H").to_numpy()
    rsn_all = ds.df["rsn"].to_numpy(dtype=np.int64)
    unknown = np.setdiff1d(pd.to_numeric(df["rsn"]).to_numpy(dtype=np.int64), rsn_all)
    if unknown.size:
        raise ValueError(f"{unknown.size} RSN(s) are not in the {profile} dataset (e.g. {unknown[:5].tolist()})")
    pos = pd.Series(np.arange(len(ds)), index=rsn_all)
    im_pos = {n: i for i, n in enumerate(IM_NAMES)}

    def wide(frame: pd.DataFrame, col: str) -> np.ndarray:
        out = np.full((len(ds), K), np.nan)
        if col in frame:
            out[pos.loc[frame["rsn"].to_numpy(dtype=np.int64)].to_numpy(), frame["im"].map(im_pos).to_numpy()] = \
                pd.to_numeric(frame[col], errors="coerce").to_numpy()
        return out

    info = _base_meta(n_boot, calibrate, alphas, methods)
    info["provenance"] = "external predictions (training provenance declared, not verified)"
    info["mode"] = "per-fold" if per_fold else "out-of-fold (cross-fitted calibration scores)"
    outs: list[_FoldOut] = []
    audits: list[LeakageReport] = []
    groups = {k: g for k, g in df.groupby(["scheme", "fold"])} if per_fold else {}
    mu_oof = wide(df, "mu") if not per_fold else None
    sd_oof = wide(df, "sigma") if not per_fold else None
    for scheme in schemes:
        for fold in bench.splits.get(scheme, []):
            audits.append(_audit_or_raise(ds, fold))
            if per_fold:
                g = groups.get((scheme, fold.fold))
                if g is None:
                    info["notes"].append(f"no predictions for {scheme}/{fold.fold}; fold skipped")
                    continue
                MU, SD = wide(g, "mu"), wide(g, "sigma")
                n_train = int(np.isfinite(MU[fold.train_fit]).any(axis=1).sum())
                if n_train:
                    info["notes"].append(f"{scheme}/{fold.fold}: {n_train} train-fit records had predictions "
                                         "(ignored)")
            else:
                MU, SD = mu_oof, sd_oof
            mu_t, sd_t = MU[fold.test], SD[fold.test]
            mu_c, sd_c = MU[fold.cal], SD[fold.cal]
            if not np.isfinite(sd_t).any():
                fb = default_sigma if default_sigma is not None else 0.6
                sd_t = np.where(np.isfinite(mu_t), fb, np.nan)
                sd_c = np.where(np.isfinite(mu_c), fb, np.nan)
                info["sigma_source"] = f"default ({fb})"
            sd_t, sd_c = np.clip(sd_t, SIGMA_FLOOR, None), np.clip(sd_c, SIGMA_FLOOR, None)
            out = _score_fold(ds, fold, Y, mu_c, sd_c, mu_t, sd_t, calibrate, alphas, methods)
            if out.row["pred_coverage"] < 1.0:
                info["notes"].append(f"{scheme}/{fold.fold}: predictions cover {out.row['pred_coverage']:.1%} "
                                     "of usable test targets; metrics use predicted pairs only")
            if calibrate and not out.cal_rows:
                info["notes"].append(f"{scheme}/{fold.fold}: no calibration (calibration-record predictions "
                                     "missing or incomplete)")
            outs.append(out)
    if not outs:
        raise ValueError("no fold had predictions; check the scheme/fold labels (see gmpe_bench.iter_folds)")
    return _assemble(name, bench, profile, outs, audits, n_boot, seed, reference, info)


__all__ = [
    "ALL_SCHEMES",
    "LEAKAGE_FREE_SCHEMES",
    "PROFILES",
    "BaseGMM",
    "BenchmarkReport",
    "DataNotFoundError",
    "FoldData",
    "LeakageError",
    "LinearGMM",
    "build_reference_bundle",
    "evaluate",
    "evaluate_predictions",
    "iter_folds",
    "load_benchmark",
    "load_reference",
]
