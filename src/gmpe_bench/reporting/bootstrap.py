"""Event-clustered bootstrap confidence intervals for the paper.

Records of one earthquake share its source term, so record--IM pairs are not independent. Every
replicate therefore resamples whole earthquakes (``eqid`` clusters) with replacement, within each
split scheme, and recomputes the *same* statistic the tables report:

RMSE (Table 3)
    mean over (fold, IM) cells of sqrt(sum w_e SSE_e / sum w_e n_e), where w_e is the number of
    times earthquake e was drawn. With w_e = 1 this is exactly the Table 3 value.
RMSE difference
    the RMSE of model A minus that of model B on their common (fold, IM) cells, with the *same*
    resampled earthquakes for both models (paired bootstrap).
Coverage (Table 4)
    mean over (fold, model) of the mean over IMs of sum w_e covered_e / sum w_e n_e, from the
    per-earthquake counts in ``results/conformal/event_counts.parquet``.
Seen / unseen stations
    RMSE as above on the test records whose station has (seen) or has no (unseen) train_fit record.

An earthquake that appears in several folds (Station regime) is one cluster with one weight in
every fold. Intervals are percentile intervals at ``cfg.bootstrap_level`` from ``cfg.bootstrap_b``
replicates with the fixed seed ``cfg.bootstrap_seed``; the seed is offset by a CRC32 of the scheme
and statistic so that every interval is reproducible on its own.
"""

from __future__ import annotations

import itertools
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from gmpe_bench.reporting.inputs import MissingInput, ReportInputs
from gmpe_bench.reporting.style import MODEL_ORDER, SCHEME_ORDER, ordered
from gmpe_bench.schema import IM_NAMES

MAIN_METHODS = ("UncalibratedGaussian", "NaiveSplitConformal", "ClusteredEventConformal",
                "WeightedShiftConformal", "HSCP")
K = len(IM_NAMES)


@dataclass
class BootstrapResult:
    rmse: pd.DataFrame = field(default_factory=pd.DataFrame)        # scheme, model, estimate, lo, hi, ...
    rmse_diff: pd.DataFrame = field(default_factory=pd.DataFrame)   # scheme, model_a, model_b, ...
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)    # scheme, method, nominal, ...
    station: pd.DataFrame = field(default_factory=pd.DataFrame)     # scheme, model, status, ...
    meta: dict[str, object] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ resampling core
def cluster_weights(n_clusters: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, n_clusters) multiplicities of a with-replacement draw of n_clusters clusters."""
    draws = rng.integers(0, n_clusters, size=(b, n_clusters))
    w = np.zeros((b, n_clusters), dtype=np.float64)
    rows = np.repeat(np.arange(b), n_clusters)
    np.add.at(w, (rows, draws.ravel()), 1.0)
    return w


def _rng(cfg_seed: int, *tags: object) -> np.random.Generator:
    return np.random.default_rng([cfg_seed, zlib.crc32("|".join(map(str, tags)).encode())])


def _scatter(frame: pd.DataFrame, events: pd.Index, cells: list[tuple], cell_cols: list[str],
             prefixes: tuple[str, ...]) -> list[np.ndarray]:
    """Event x (cell, IM) matrices of the wide columns ``<prefix>_<im>``; duplicate rows are summed."""
    n_e, n_c = len(events), len(cells)
    out = [np.zeros((n_e, n_c * K)) for _ in prefixes]
    if frame.empty:
        return out
    ei = events.get_indexer(frame["eqid"].to_numpy())
    lookup = {c: i for i, c in enumerate(cells)}
    keys = list(zip(*(frame[c].to_numpy() for c in cell_cols)))
    ci = np.array([lookup[k] for k in keys])
    cols = (ci[:, None] * K + np.arange(K)[None, :]).ravel()
    rows = np.repeat(ei, K)
    for mat, p in zip(out, prefixes):
        vals = frame[[f"{p}_{im}" for im in IM_NAMES]].to_numpy(dtype=np.float64).ravel()
        np.add.at(mat, (rows, cols), vals)
    return out


def _rmse_stat(W: np.ndarray, sse: np.ndarray, n: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean over valid (cell, IM) columns of sqrt(weighted SSE / weighted n), one value per row of W."""
    num, den = W @ sse[:, valid], W @ n[:, valid]
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.sqrt(num / np.where(den > 0, den, np.nan))
        return np.nanmean(r, axis=1) if r.shape[1] else np.full(W.shape[0], np.nan)


def _coverage_stat(W: np.ndarray, cov: np.ndarray, n: np.ndarray, n_cells: int) -> np.ndarray:
    """Mean over cells of the mean over IMs of weighted coverage (Table 4 'ALL', averaged)."""
    num, den = W @ cov, W @ n
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (num / np.where(den > 0, den, np.nan)).reshape(W.shape[0], n_cells, K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.nanmean(c, axis=2), axis=1)


def _interval(est: float, reps: np.ndarray, level: float) -> dict[str, float]:
    reps = reps[np.isfinite(reps)]
    tail = (1 - level) / 2 * 100
    lo, hi = (np.percentile(reps, [tail, 100 - tail]) if len(reps) else (np.nan, np.nan))
    return {"estimate": float(est), "lo": float(lo), "hi": float(hi), "se": float(reps.std(ddof=1))
            if len(reps) > 1 else np.nan, "n_valid_reps": int(len(reps))}


# ------------------------------------------------------------------ residual statistics
def residual_stats(inp: ReportInputs, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Per (scheme, fold, model, eqid, seen): SSE and n per IM and the record count.

    Uses the same mask as the Phase 3/4 RMSE (finite y, mu, sigma and sigma > 0); ``seen`` is True
    when the record's station has a train_fit record in that fold (NaN if splits are unavailable).
    """
    try:
        seen_tab = inp.train_fit_stations().assign(_seen=True)
    except MissingInput as exc:
        log(f"bootstrap: station status unavailable ({exc})")
        seen_tab = None
    frames = []
    for model, df in inp.iter_predictions():
        y = df[[f"y_{im}" for im in IM_NAMES]].to_numpy(dtype=np.float64)
        mu = df[[f"mu_{im}" for im in IM_NAMES]].to_numpy(dtype=np.float64)
        sd = df[[f"sigma_{im}" for im in IM_NAMES]].to_numpy(dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
        e2 = np.where(ok, (y - mu) ** 2, 0.0)
        w = pd.DataFrame(np.column_stack([e2, ok.astype(np.float64)]),
                         columns=[f"sse_{im}" for im in IM_NAMES] + [f"n_{im}" for im in IM_NAMES])
        w["scheme"], w["fold"] = df["scheme"].to_numpy(), df["fold"].to_numpy()
        w["eqid"], w["ssn"] = df["eqid"].to_numpy(), df["ssn"].to_numpy()
        w["n_rec"] = 1.0
        if seen_tab is not None:
            w = w.merge(seen_tab, on=["scheme", "fold", "ssn"], how="left")
            w["seen"] = w.pop("_seen").fillna(False).astype(bool)
        else:
            w["seen"] = np.nan
        g = w.drop(columns="ssn").groupby(["scheme", "fold", "eqid", "seen"], dropna=False, sort=False).sum()
        frames.append(g.reset_index().assign(model=model))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ statistics
def _rmse_tables(stats: pd.DataFrame, inp: ReportInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = inp.cfg
    rows, diffs = [], []
    for scheme in ordered(stats["scheme"].unique(), SCHEME_ORDER):
        s = stats[stats["scheme"] == scheme]
        events = pd.Index(np.unique(s["eqid"].to_numpy()))
        folds = sorted(s["fold"].unique())
        cells = [(f,) for f in folds]
        W = cluster_weights(len(events), cfg.bootstrap_b, _rng(cfg.bootstrap_seed, "rmse", scheme))
        ones = np.ones((1, len(events)))
        mats = {}
        for model in ordered(s["model"].unique(), MODEL_ORDER):
            sse, n = _scatter(s[s["model"] == model], events, cells, ["fold"], ("sse", "n"))
            valid = n.sum(axis=0) > 0
            mats[model] = (sse, n, valid)
            est = _rmse_stat(ones, sse, n, valid)[0]
            rows.append({"scheme": scheme, "model": model, **_interval(est, _rmse_stat(W, sse, n, valid),
                                                                       cfg.bootstrap_level),
                         "n_events": int((n.sum(axis=1) > 0).sum()), "n_folds": int(valid.reshape(-1, K)
                                                                                     .any(axis=1).sum())})
        pair_models = [m for m in cfg.pair_models if m in mats]
        for a, b in itertools.combinations(pair_models, 2):
            (sa, na, va), (sb, nb, vb) = mats[a], mats[b]
            common = va & vb
            if not common.any():
                continue
            est = _rmse_stat(ones, sa, na, common)[0] - _rmse_stat(ones, sb, nb, common)[0]
            reps = _rmse_stat(W, sa, na, common) - _rmse_stat(W, sb, nb, common)
            fin = reps[np.isfinite(reps)]
            p = float(min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean()))) if len(fin) else np.nan
            diffs.append({"scheme": scheme, "model_a": a, "model_b": b,
                          "rmse_a": _rmse_stat(ones, sa, na, common)[0], "rmse_b": _rmse_stat(ones, sb, nb, common)[0],
                          **_interval(est, reps, cfg.bootstrap_level), "p_two_sided": p,
                          "n_events": int(((na[:, common].sum(1) > 0) | (nb[:, common].sum(1) > 0)).sum())})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def _station_table(stats: pd.DataFrame, inp: ReportInputs) -> pd.DataFrame:
    cfg = inp.cfg
    if stats["seen"].isna().all():
        raise MissingInput("station status needs data/processed/<profile>.parquet and splits_<profile>.json")
    rows = []
    models = [m for m in cfg.station_models if m in set(stats["model"])]
    for scheme in ordered(stats["scheme"].unique(), SCHEME_ORDER):
        for status, flag in (("seen", True), ("unseen", False)):
            s = stats[(stats["scheme"] == scheme) & (stats["seen"] == flag)]
            if s.empty:
                continue
            events = pd.Index(np.unique(s["eqid"].to_numpy()))
            cells = [(f,) for f in sorted(s["fold"].unique())]
            W = cluster_weights(len(events), cfg.bootstrap_b, _rng(cfg.bootstrap_seed, "station", scheme, status))
            for model in models:
                sub = s[s["model"] == model]
                if sub.empty:
                    continue
                sse, n = _scatter(sub, events, cells, ["fold"], ("sse", "n"))
                valid = n.sum(axis=0) > 0
                est = _rmse_stat(np.ones((1, len(events))), sse, n, valid)[0]
                rows.append({"scheme": scheme, "model": model, "status": status,
                             **_interval(est, _rmse_stat(W, sse, n, valid), cfg.bootstrap_level),
                             "n_records": int(sub["n_rec"].sum()), "n_events": int(sub["eqid"].nunique())})
    return pd.DataFrame(rows)


def _coverage_table(inp: ReportInputs) -> pd.DataFrame:
    cfg = inp.cfg
    e = inp.event_counts()
    e = e[e["method"].isin(MAIN_METHODS)]
    table4 = inp.conformal_metrics
    table4 = table4[(table4["profile"] == cfg.profile) & (table4["im"] == "ALL")]
    ref = table4.groupby(["scheme", "method", "alpha"])["coverage"].mean()
    rows = []
    for scheme in ordered(e["scheme"].unique(), SCHEME_ORDER):
        s = e[e["scheme"] == scheme]
        events = pd.Index(np.unique(s["eqid"].to_numpy()))
        W = cluster_weights(len(events), cfg.bootstrap_b, _rng(cfg.bootstrap_seed, "coverage", scheme))
        for alpha in sorted(s["alpha"].unique(), reverse=True):
            for method in [m for m in MAIN_METHODS if m in set(s["method"])]:
                sub = s[(s["alpha"] == alpha) & (s["method"] == method)]
                cells = sorted(set(zip(sub["fold"], sub["model"])))
                cov, n = _scatter(sub, events, cells, ["fold", "model"], ("cov", "n"))
                est = _coverage_stat(np.ones((1, len(events))), cov, n, len(cells))[0]
                table_value = ref.get((scheme, method, alpha), np.nan)
                rows.append({"scheme": scheme, "method": method, "alpha": float(alpha), "nominal": 1 - float(alpha),
                             **_interval(est, _coverage_stat(W, cov, n, len(cells)), cfg.bootstrap_level),
                             "table4_value": float(table_value), "abs_diff_vs_table4": abs(est - table_value),
                             "n_events": int((n.sum(axis=1) > 0).sum()), "n_cells": len(cells)})
    return pd.DataFrame(rows)


def run_bootstrap(inp: ReportInputs, log: Callable[[str], None] = print) -> BootstrapResult:
    """All bootstrap intervals; parts whose inputs are missing are listed in ``skipped``."""
    cfg = inp.cfg
    res = BootstrapResult(meta={"b": cfg.bootstrap_b, "seed": cfg.bootstrap_seed, "level": cfg.bootstrap_level,
                                "cluster": "eqid (earthquake), resampled with replacement within each scheme",
                                "interval": "percentile", "profile": cfg.profile})
    try:
        stats = residual_stats(inp, log)
        res.rmse, res.rmse_diff = _rmse_tables(stats, inp)
        acc = inp.accuracy(include_site=True).groupby(["scheme", "model"])["rmse"].mean()
        diff = (res.rmse.set_index(["scheme", "model"])["estimate"] - acc).abs()
        res.meta["rmse_max_abs_diff_vs_table3"] = float(diff.max()) if diff.notna().any() else None
        res.meta["events_per_scheme"] = {k: int(v) for k, v in stats.groupby("scheme")["eqid"].nunique().items()}
        try:
            res.station = _station_table(stats, inp)
        except MissingInput as exc:
            res.skipped["station"] = str(exc)
    except MissingInput as exc:
        res.skipped["rmse"] = str(exc)
    try:
        res.coverage = _coverage_table(inp)
        res.meta["coverage_max_abs_diff_vs_table4"] = float(res.coverage["abs_diff_vs_table4"].max())
    except MissingInput as exc:
        res.skipped["coverage"] = str(exc)
    for k, v in res.skipped.items():
        log(f"bootstrap {k}: SKIPPED ({v})")
    for k in ("rmse_max_abs_diff_vs_table3", "coverage_max_abs_diff_vs_table4"):
        v = res.meta.get(k)
        if v is not None and v > 1e-4:
            log(f"bootstrap WARNING: {k} = {v:.2e}; the counts do not reproduce the table "
                "(re-run the stage that wrote them)")
    return res
