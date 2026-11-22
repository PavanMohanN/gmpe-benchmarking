"""Evaluate uncertainty calibration methods on frozen splits.

Inputs (no retraining):
* ML models   - ``results/ml/{oof,cal}_ml_predictions.parquet`` (Phase 4)
* Classical   - ``results/baselines/cache/<MODEL>.parquet`` (Phase 3 pygmm cache); the
                bias term is re-estimated per fold and IM on ``train_fit`` residuals, so
                ``<MODEL>_cal`` matches Phase 3 exactly and has ``cal`` predictions.

Outputs in ``out_dir``:
* ``conformal_metrics.csv``        per IM, plus ``ALL`` (mean over IMs), ``JOINT_MARGINAL``
                                   (all usable IMs inside the marginal bands) and
                                   ``JOINT_BAND`` (HSCP simultaneous band)
* ``conditional_coverage.csv``     coverage by Mw, R_rup, VS30 class and seen/unseen station
* ``conformal_intervals_summary.parquet``  per IM: q quantiles, half-widths, site shift, ESS
* ``event_counts.parquet``         (``event_counts=True``) covered and usable counts per test
                                   earthquake and IM for every fold, model, method and level; the
                                   event-clustered bootstrap of the paper resamples these rows.
                                   Integer counts, so coverage is reproduced exactly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from gmpe_bench.baselines.base import target
from gmpe_bench.conformal.hierarchy import crossed_terms
from gmpe_bench.conformal.methods import CalSet, ConformalCalibrator, Intervals, TestInputs, default_methods
from gmpe_bench.conformal.shift import density_ratio_weights, shift_features
from gmpe_bench.data import GMDataset
from gmpe_bench.schema import IM_NAMES, period_of
from gmpe_bench.splits import SplitFold, load_splits

CLASSICAL = ("ASK14", "BSSA14", "CB14", "CY14")
ALPHAS = (0.10, 0.05)
MW_BINS = (-np.inf, 5.0, 6.5, np.inf)
MW_LABELS = ("Mw<5", "5<=Mw<6.5", "Mw>=6.5")
R_BINS = (-np.inf, 30.0, 100.0, np.inf)
R_LABELS = ("Rrup<30", "30<=Rrup<100", "Rrup>=100")
VS_BINS = (-np.inf, 180.0, 360.0, 760.0, np.inf)
VS_LABELS = ("E (<180)", "D (180-360)", "C (360-760)", "A/B (>=760)")
DEFAULT_SCHEMES = ("event_grouped", "station_grouped", "spatial_block", "temporal",
                   "magnitude_holdout", "cl1_to_cl2")


# ------------------------------------------------------------------ predictions
def _y_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([target(df, im) for im in IM_NAMES])


def classical_predictions(ds: GMDataset, fold: SplitFold, model: str, cache_dir: Path,
                          table: pd.DataFrame | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bias-calibrated (train_fit intercept per IM) mu and published sigma for cal and test."""
    t = table if table is not None else pd.read_parquet(Path(cache_dir) / f"{model}.parquet")
    mu_cols, sd_cols = [f"mu_{n}" for n in IM_NAMES], [f"sigma_{n}" for n in IM_NAMES]
    rsn = ds.df["rsn"].to_numpy(dtype=np.int64)

    def get(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sub = t.reindex(rsn[idx])
        return sub[mu_cols].to_numpy(dtype=float), sub[sd_cols].to_numpy(dtype=float)

    mu_tr, _ = get(fold.train_fit)
    tr = ds.df.iloc[fold.train_fit]
    ct = crossed_terms(_y_matrix(tr) - mu_tr, tr["eqid"].to_numpy(), tr["ssn"].to_numpy())
    bias = np.where(np.isfinite(ct.intercept), ct.intercept, 0.0)
    out = {}
    for part, idx in (("cal", fold.cal), ("test", fold.test)):
        mu, sd = get(idx)
        out[part] = (mu + bias, sd)
    return out


def _load_ml(ml_dir: Path, profile: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for part, fname in (("test", "oof_ml_predictions.parquet"), ("cal", "cal_ml_predictions.parquet")):
        path = Path(ml_dir) / fname
        if path.exists():
            out[part] = pads.dataset(path).to_table(filter=pads.field("profile") == profile).to_pandas()
    return out


def _ml_arrays(frame: pd.DataFrame, scheme: str, fold: str, model: str, rsn: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray] | None:
    sub = frame[(frame["scheme"] == scheme) & (frame["fold"] == fold) & (frame["model"] == model)]
    if sub.empty:
        return None
    sub = sub.set_index("rsn").reindex(rsn)
    mu = sub[[f"mu_{n}" for n in IM_NAMES]].to_numpy(dtype=float)
    sd = sub[[f"sigma_{n}" for n in IM_NAMES]].to_numpy(dtype=float)
    return mu, sd


# ------------------------------------------------------------------ scoring
def interval_metrics(y: np.ndarray, iv: Intervals) -> dict[str, np.ndarray]:
    """Per-IM coverage, width and Winkler score on finite targets."""
    fin = np.isfinite(y)
    L, U = iv.lower, iv.upper
    cov = ((y >= L) & (y <= U)) & fin
    width = np.where(fin, U - L, np.nan)
    a = iv.alpha
    wink = np.where(fin, (U - L) + (2 / a) * np.clip(L - y, 0, None) + (2 / a) * np.clip(y - U, 0, None), np.nan)
    n = fin.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"n": n, "coverage": cov.sum(axis=0) / np.where(n > 0, n, np.nan),
                "width": np.nanmean(width, axis=0), "winkler": np.nanmean(wink, axis=0),
                "q_median": np.nanmedian(iv.q, axis=0)}


def width_roughness(iv: Intervals) -> float:
    """Mean squared second difference of ln half-width across adjacent SA periods."""
    sa = [i for i, n in enumerate(IM_NAMES) if period_of(n) is not None]
    hw = np.log(np.clip((iv.upper - iv.lower)[:, sa] / 2, 1e-9, None))
    d2 = hw[:, 2:] - 2 * hw[:, 1:-1] + hw[:, :-2]
    return float(np.nanmean(d2**2))


def joint_coverage(y: np.ndarray, L: np.ndarray, U: np.ndarray) -> float:
    fin = np.isfinite(y)
    ok = (~fin) | ((y >= L) & (y <= U))
    has = fin.any(axis=1)
    return float(ok[has].all(axis=1).mean()) if has.any() else float("nan")


def strata(df: pd.DataFrame, seen: np.ndarray) -> dict[str, np.ndarray]:
    vs = pd.to_numeric(df["vs30"], errors="coerce").to_numpy()
    return {
        "mw": np.asarray(pd.cut(df["mw"].astype(float), MW_BINS, labels=MW_LABELS, right=False).astype("string")),
        "rrup": np.asarray(pd.cut(df["rrup"].astype(float), R_BINS, labels=R_LABELS, right=False).astype("string")),
        "vs30_class": np.asarray(pd.cut(pd.Series(vs), VS_BINS, labels=VS_LABELS, right=False).astype("string")),
        "station": np.where(seen, "seen", "unseen"),
    }


# ------------------------------------------------------------------ main loop
EVENT_COUNTS_FILE = "event_counts.parquet"


def event_coverage_counts(y: np.ndarray, iv: Intervals, eqid: np.ndarray) -> pd.DataFrame:
    """Covered and usable (finite-target) counts per earthquake and IM: columns eqid, cov_<im>, n_<im>."""
    fin = np.isfinite(y)
    cov = (y >= iv.lower) & (y <= iv.upper) & fin
    codes, uniq = pd.factorize(np.asarray(eqid), sort=True)
    c = np.zeros((len(uniq), len(IM_NAMES)), dtype=np.int64)
    n = np.zeros_like(c)
    np.add.at(c, codes, cov.astype(np.int64))
    np.add.at(n, codes, fin.astype(np.int64))
    if c.size and max(c.max(), n.max()) > np.iinfo(np.uint16).max:
        raise ValueError("per-event count exceeds uint16")
    out = {"eqid": np.asarray(uniq, dtype=np.int64)}
    out.update({f"cov_{im}": c[:, j].astype(np.uint16) for j, im in enumerate(IM_NAMES)})
    out.update({f"n_{im}": n[:, j].astype(np.uint16) for j, im in enumerate(IM_NAMES)})
    return pd.DataFrame(out)


def evaluate_conformal(data_dir: Path, out_dir: Path, ml_dir: Path | None = None,
                       baseline_cache: Path | None = None, profiles: tuple[str, ...] = ("full", "sp23"),
                       schemes: tuple[str, ...] = DEFAULT_SCHEMES, models: tuple[str, ...] | None = None,
                       methods: list[ConformalCalibrator] | None = None, alphas: tuple[float, ...] = ALPHAS,
                       log: Callable[[str], None] = print, event_counts: bool = True,
                       metrics: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Calibrate every method for every model, split and fold; write the outputs.

    ``event_counts`` also writes ``event_counts.parquet``. ``metrics=False`` writes *only* that file
    (the three Phase 5 outputs already on disk are left untouched), which is how the counts are added
    to an existing run without recomputing anything else: no model is refitted in either case.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (metrics or event_counts):
        raise ValueError("nothing to write: metrics=False and event_counts=False")
    methods = methods or default_methods()
    metric_rows: list[dict[str, object]] = []
    cond_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    count_frames: list[pd.DataFrame] = []

    for prof in profiles:
        ds = GMDataset.from_parquet(data_dir / f"{prof}.parquet", prof)
        splits = load_splits(data_dir / f"splits_{prof}.json", ds)
        rsn_all = ds.df["rsn"].to_numpy(dtype=np.int64)
        ml = _load_ml(ml_dir, prof) if ml_dir is not None else {}
        ml_models = sorted(set(ml["test"]["model"])) if "test" in ml else []
        cls_models = [m for m in CLASSICAL if baseline_cache is not None and (Path(baseline_cache) / f"{m}.parquet").exists()]
        tables = {m: pd.read_parquet(Path(baseline_cache) / f"{m}.parquet") for m in cls_models}  # type: ignore[arg-type]
        names = [f"{m}_cal" for m in cls_models] + ml_models
        if models:
            names = [n for n in names if n in models]
        log(f"[{prof}] models: {', '.join(names) or 'none found'}")

        for scheme in schemes:
            for fold in splits.get(scheme, []):
                t0 = time.perf_counter()
                cal_df, test_df = ds.df.iloc[fold.cal], ds.df.iloc[fold.test]
                if len(cal_df) == 0 or len(test_df) == 0:
                    continue
                Xc, Xt = shift_features(cal_df), shift_features(test_df)
                wc, wt = density_ratio_weights(Xc, Xt)          # covariates only
                y_cal, y_test = _y_matrix(cal_df), _y_matrix(test_df)
                train_st = np.unique(ds.df["ssn"].to_numpy()[fold.train_fit])
                seen = np.isin(test_df["ssn"].to_numpy(), np.union1d(train_st, cal_df["ssn"].to_numpy()))
                st = strata(test_df, seen)
                for name in names:
                    if name.endswith("_cal") and name[:-4] in tables:
                        p = classical_predictions(ds, fold, name[:-4], Path(baseline_cache), tables[name[:-4]])  # type: ignore[arg-type]
                        (mu_c, sd_c), (mu_t, sd_t) = p["cal"], p["test"]
                    else:
                        a = _ml_arrays(ml["cal"], scheme, fold.fold, name, rsn_all[fold.cal]) if "cal" in ml else None
                        b = _ml_arrays(ml["test"], scheme, fold.fold, name, rsn_all[fold.test])
                        if a is None or b is None:
                            continue
                        (mu_c, sd_c), (mu_t, sd_t) = a, b
                    cal = CalSet(y_cal, mu_c, sd_c, cal_df["eqid"].to_numpy(), cal_df["ssn"].to_numpy(), Xc,
                                 train_st, wc)
                    test = TestInputs(mu_t, sd_t, test_df["eqid"].to_numpy(), test_df["ssn"].to_numpy(), Xt, wt)
                    for method in methods:
                        for alpha, iv in method.calibrate_many(cal, test, alphas).items():
                            key = {"profile": prof, "scheme": scheme, "fold": fold.fold, "model": name,
                                   "method": method.name, "alpha": alpha, "nominal": 1 - alpha}
                            if metrics:
                                _record(key, y_test, iv, mu_t, st, metric_rows, cond_rows, summary_rows)
                            if event_counts:
                                count_frames.append(event_coverage_counts(y_test, iv, test_df["eqid"].to_numpy())
                                                    .assign(**key))
                log(f"[{prof}] {scheme}/{fold.fold}: {len(names)} models x {len(methods)} methods "
                    f"in {time.perf_counter() - t0:.1f} s (cal {len(cal_df):,}, test {len(test_df):,})")

    if event_counts:
        counts = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
        keys = ["profile", "scheme", "fold", "model", "method", "alpha", "nominal"]
        if len(counts):
            counts = counts[keys + [c for c in counts.columns if c not in keys]]
            for k in ("profile", "scheme", "fold", "model", "method"):
                counts[k] = counts[k].astype("category")
        counts.to_parquet(out_dir / EVENT_COUNTS_FILE, index=False)
        log(f"event counts: {len(counts):,} rows -> {out_dir / EVENT_COUNTS_FILE}")
    summary_path = out_dir / "conformal_intervals_summary.parquet"
    metric_df = pd.DataFrame(metric_rows)
    cond = _aggregate_conditional(pd.DataFrame(cond_rows))
    if metrics:
        metric_df.to_csv(out_dir / "conformal_metrics.csv", index=False)
        cond.to_csv(out_dir / "conditional_coverage.csv", index=False)
        pd.DataFrame(summary_rows).to_parquet(summary_path, index=False)
    return metric_df, cond, summary_path


def _record(key: dict[str, object], y: np.ndarray, iv: Intervals, mu: np.ndarray, st: dict[str, np.ndarray],
            metric_rows: list, cond_rows: list, summary_rows: list) -> None:
    m = interval_metrics(y, iv)
    for j, im in enumerate(IM_NAMES):
        metric_rows.append({**key, "im": im, "period": period_of(im), "n": int(m["n"][j]),
                            "coverage": m["coverage"][j], "width": m["width"][j], "winkler": m["winkler"][j],
                            "q_median": m["q_median"][j]})
    rough = width_roughness(iv)
    metric_rows.append({**key, "im": "ALL", "period": np.nan, "n": int(m["n"].sum()),
                        "coverage": float(np.nanmean(m["coverage"])), "width": float(np.nanmean(m["width"])),
                        "winkler": float(np.nanmean(m["winkler"])), "q_median": float(np.nanmedian(iv.q)),
                        "width_roughness": rough, "n_inf": iv.n_inf, "ess": iv.ess})
    metric_rows.append({**key, "im": "JOINT_MARGINAL", "period": np.nan, "n": int(np.isfinite(y).any(1).sum()),
                        "coverage": joint_coverage(y, iv.lower, iv.upper),
                        "width": float(np.nanmean(m["width"]))})
    if iv.joint_lower is not None and iv.joint_upper is not None:
        jw = np.where(np.isfinite(y), iv.joint_upper - iv.joint_lower, np.nan)
        metric_rows.append({**key, "im": "JOINT_BAND", "period": np.nan, "n": int(np.isfinite(y).any(1).sum()),
                            "coverage": joint_coverage(y, iv.joint_lower, iv.joint_upper),
                            "width": float(np.nanmean(jw))})

    fin = np.isfinite(y)
    covered = ((y >= iv.lower) & (y <= iv.upper) & fin).sum(axis=1)
    n_rec = fin.sum(axis=1)
    wsum = np.nansum(np.where(fin, iv.upper - iv.lower, np.nan), axis=1)
    for stratum, labels in st.items():
        frame = pd.DataFrame({"bin": labels, "covered": covered, "n": n_rec, "width_sum": wsum})
        g = frame.groupby("bin", dropna=True)[["covered", "n", "width_sum"]].sum()
        for b, row in g.iterrows():
            cond_rows.append({**key, "stratum": stratum, "bin": b, "covered": int(row["covered"]),
                              "n": int(row["n"]), "width_sum": float(row["width_sum"])})

    hw = (iv.upper - iv.lower) / 2
    shift = np.abs(iv.center - mu)
    for j, im in enumerate(IM_NAMES):
        summary_rows.append({**key, "im": im, "period": period_of(im),
                             "q_p10": float(np.nanpercentile(iv.q[:, j], 10)), "q_median": float(np.nanmedian(iv.q[:, j])),
                             "q_p90": float(np.nanpercentile(iv.q[:, j], 90)),
                             "halfwidth_median": float(np.nanmedian(hw[:, j])),
                             "center_shift_mean": float(np.nanmean(shift[:, j])),
                             "n_inf": iv.n_inf, "ess": iv.ess})


def _aggregate_conditional(cond: pd.DataFrame) -> pd.DataFrame:
    if cond.empty:
        return cond
    keys = ["profile", "scheme", "model", "method", "alpha", "nominal", "stratum", "bin"]
    g = cond.groupby(keys, sort=False)[["covered", "n", "width_sum"]].sum().reset_index()
    g["coverage"] = g["covered"] / g["n"].where(g["n"] > 0)
    g["width"] = g["width_sum"] / g["n"].where(g["n"] > 0)
    return g.drop(columns="width_sum")
