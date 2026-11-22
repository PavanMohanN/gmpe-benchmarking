"""Out-of-fold evaluation of the classical baselines on frozen splits.

For every profile x scheme x fold x model x IM:

1. fit on ``train_fit`` (bias term for NGA-West2 models, full fit for the mixed model);
2. predict ``test``; decompose train_fit residuals to get station terms;
3. score two prediction modes on ``test``:
   * ``ergodic``         model median and model sigma
   * ``site_corrected``  median + dS2S from train_fit (0 for unseen stations),
                         sigma sqrt(tau^2 + phi_0^2) for seen stations, total sigma otherwise.

``cal`` records are never touched here: they are reserved for conformal calibration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gmpe_bench.baselines.base import BaseGMM, target
from gmpe_bench.baselines.mixed_effects import MixedEffectsGMM
from gmpe_bench.baselines.ngawest2 import MODEL_CLASSES, NGAWest2Baseline
from gmpe_bench.baselines.residuals import apply_site_terms, decompose_residuals
from gmpe_bench.data import GMDataset
from gmpe_bench.schema import IM_NAMES, period_of
from gmpe_bench.splits import SplitFold, load_splits

Z90 = 1.6448536269514722
DEFAULT_SCHEMES = ("event_grouped", "station_grouped", "spatial_block", "temporal",
                   "magnitude_holdout", "cl1_to_cl2")
DEFAULT_MODELS = ("ASK14", "BSSA14", "CB14", "CY14", "MixedEffects")


def make_model(name: str, calibrate_bias: bool = True, cache_dir: Path | None = None) -> BaseGMM:
    if name in MODEL_CLASSES:
        return NGAWest2Baseline(name, calibrate_bias=calibrate_bias, cache_dir=cache_dir)
    if name == "MixedEffects":
        return MixedEffectsGMM()
    raise KeyError(f"Unknown baseline {name!r}")


def score(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    """Point and Gaussian probabilistic scores on finite entries."""
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    if not ok.any():
        return {k: np.nan for k in ("n", "rmse", "mae", "bias", "nll", "cov90", "width90", "sigma_mean")}
    r, s = y[ok] - mu[ok], sigma[ok]
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "mae": float(np.mean(np.abs(r))),
        "bias": float(np.mean(r)),
        "nll": float(np.mean(0.5 * np.log(2 * np.pi * s**2) + r**2 / (2 * s**2))),
        "cov90": float(np.mean(np.abs(r) <= Z90 * s)),
        "width90": float(np.mean(2 * Z90 * s)),
        "sigma_mean": float(np.mean(s)),
    }


@dataclass
class OOFWriter:
    """Streams wide out-of-fold rows to one Parquet file (float32 values)."""

    path: Path
    _writer: pq.ParquetWriter | None = field(default=None, init=False)

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self._writer.write_table(table.cast(self._writer.schema))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def evaluate_fold(ds: GMDataset, fold: SplitFold, model: BaseGMM, ims: list[str]
                  ) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Metrics rows and the wide OOF frame for one fold and one model."""
    df = ds.df
    tr, te = df.iloc[fold.train_fit], df.iloc[fold.test]
    eq_te, st_te = te["eqid"].to_numpy(), te["ssn"].to_numpy()
    oof = pd.DataFrame({"profile": ds.profile, "scheme": fold.scheme, "fold": fold.fold,
                        "model": model.name, "rsn": te["rsn"].to_numpy(dtype=np.int64),
                        "eqid": eq_te.astype(np.int64), "ssn": st_te.astype(np.int64)})
    rows: list[dict[str, object]] = []
    for im in ims:
        y_tr, y_te = target(tr, im), target(te, im)
        keep = np.isfinite(y_tr)
        if keep.sum() < 20 or not np.isfinite(y_te).any():
            continue
        model.fit(tr[keep], im)
        mu_tr, _ = model.predict(tr[keep])
        dec = decompose_residuals(y_tr[keep] - mu_tr, tr["eqid"].to_numpy()[keep], tr["ssn"].to_numpy()[keep],
                                  fit_intercept=False)
        mu, sd = model.predict(te)
        mu_site, sd_site, seen = apply_site_terms(mu, st_te, dec)
        test_dec = decompose_residuals(y_te - mu, eq_te, st_te)
        base = {"profile": ds.profile, "scheme": fold.scheme, "fold": fold.fold, "model": model.name,
                "im": im, "period": period_of(im), "n_events": int(np.unique(eq_te[np.isfinite(y_te)]).size),
                "frac_station_seen": float(seen[np.isfinite(y_te)].mean()),
                "bias_cal": float(getattr(model, "bias", 0.0)),
                "tau_train": dec.tau, "phi_s2s_train": dec.phi_s2s, "phi_0_train": dec.phi_0,
                "tau_test": test_dec.tau, "phi_s2s_test": test_dec.phi_s2s, "phi_0_test": test_dec.phi_0}
        rows.append({**base, "mode": "ergodic", **score(y_te, mu, sd)})
        rows.append({**base, "mode": "site_corrected", **score(y_te, mu_site, sd_site)})
        oof[f"y_{im}"] = y_te.astype(np.float32)
        oof[f"mu_{im}"] = mu.astype(np.float32)
        oof[f"sigma_{im}"] = sd.astype(np.float32)
        oof[f"mu_site_{im}"] = mu_site.astype(np.float32)
        oof[f"sigma_site_{im}"] = sd_site.astype(np.float32)
    for im in ims:  # constant schema for the Parquet writer
        for p in ("y", "mu", "sigma", "mu_site", "sigma_site"):
            if f"{p}_{im}" not in oof:
                oof[f"{p}_{im}"] = np.float32(np.nan)
    cols = list(oof.columns[:7]) + [f"{p}_{im}" for im in ims for p in ("y", "mu", "sigma", "mu_site", "sigma_site")]
    return rows, oof[cols]


def evaluate_baselines(data_dir: Path, out_dir: Path, profiles: tuple[str, ...] = ("full", "sp23"),
                       schemes: tuple[str, ...] = DEFAULT_SCHEMES, models: tuple[str, ...] = DEFAULT_MODELS,
                       ims: list[str] | None = None, calibrate_bias: bool = True,
                       log: Callable[[str], None] = print) -> tuple[pd.DataFrame, Path]:
    """Run every baseline on frozen splits; write ``baseline_metrics.csv`` and ``oof_predictions.parquet``."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ims = list(ims or IM_NAMES)
    writer = OOFWriter(out_dir / "oof_predictions.parquet")
    rows: list[dict[str, object]] = []
    try:
        for prof in profiles:
            ds = GMDataset.from_parquet(data_dir / f"{prof}.parquet", prof)
            splits = load_splits(data_dir / f"splits_{prof}.json", ds)
            fitted: dict[str, BaseGMM] = {}
            for name in models:
                t0 = time.perf_counter()
                model = make_model(name, calibrate_bias, out_dir / "cache")
                if isinstance(model, NGAWest2Baseline):
                    model.precompute(ds.df)
                    log(f"[{prof}] {name}: evaluated {len(ds):,} records in {time.perf_counter() - t0:.1f} s")
                fitted[name] = model
            for scheme in schemes:
                if scheme not in splits:
                    log(f"[{prof}] scheme {scheme} not in splits file; skipped")
                    continue
                for fold in splits[scheme]:
                    for name, model in fitted.items():
                        t0 = time.perf_counter()
                        r, oof = evaluate_fold(ds, fold, model, ims)
                        rows += r
                        writer.write(oof)
                        log(f"[{prof}] {scheme}/{fold.fold} {name}: {len(r) // 2} IMs, "
                            f"{time.perf_counter() - t0:.1f} s")
    finally:
        writer.close()
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "baseline_metrics.csv", index=False)
    return metrics, writer.path
