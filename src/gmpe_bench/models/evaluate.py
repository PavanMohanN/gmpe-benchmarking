"""Train ML/DL spectral GMMs on frozen splits; save test and calibration predictions.

For every profile x scheme x fold x model:

* fit on ``train_fit`` (internal early stopping uses held-out events of ``train_fit``);
* predict ``test`` -> ``oof_ml_predictions.parquet``
* predict ``cal``  -> ``cal_ml_predictions.parquet`` (for Phase 5 conformal calibration)
* score both parts per IM -> ``ml_metrics.csv``

Prediction files are wide: ``y_<IM>``, ``mu_<IM>``, ``sigma_<IM>`` for all 27 IMs, float32.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gmpe_bench.baselines.evaluate import OOFWriter, score
from gmpe_bench.baselines.residuals import decompose_residuals
from gmpe_bench.data import GMDataset
from gmpe_bench.models.base import BaseSpectralModel, target_matrix
from gmpe_bench.models.gbdt import XGBConfig, XGBoostSpectralGMM
from gmpe_bench.models.hybrid import ResidualHybridGMM
from gmpe_bench.models.spectral_mlp import MLPConfig, SpectralMLP
from gmpe_bench.schema import IM_NAMES, period_of
from gmpe_bench.splits import SplitFold, load_splits

DEFAULT_SCHEMES = ("event_grouped", "station_grouped", "spatial_block", "temporal",
                   "magnitude_holdout", "cl1_to_cl2")
DEFAULT_MODELS = ("XGBoost", "SpectralMLP", "Hybrid_ASK14_MLP", "Hybrid_BSSA14_XGB")
FAST_MLP = MLPConfig(epochs=40, patience=8, hidden=128, depth=2)
FAST_XGB = XGBConfig(n_estimators=300, early_stopping_rounds=30, scale_estimators=150, max_depth=6)


def make_ml_model(name: str, cache_dir: Path | None = None, station_encoding: bool = True,
                  fast: bool = False, seed: int = 0) -> BaseSpectralModel:
    """``XGBoost``, ``SpectralMLP`` or ``Hybrid_<GMM>_<MLP|XGB>`` (e.g. ``Hybrid_ASK14_MLP``)."""
    mlp = FAST_MLP if fast else MLPConfig()
    xgbc = FAST_XGB if fast else XGBConfig()
    if name == "XGBoost":
        return XGBoostSpectralGMM(xgbc, station_encoding, seed)
    if name == "SpectralMLP":
        return SpectralMLP(mlp, station_encoding, seed)
    if name.startswith("Hybrid_"):
        _, gmm, corr = name.split("_")
        return ResidualHybridGMM(gmm, corr.lower(), cache_dir, station_encoding,
                                 mlp_config=mlp, xgb_config=xgbc, seed=seed)
    raise KeyError(f"Unknown ML model {name!r}")


def _frame(ds: GMDataset, fold: SplitFold, model_name: str, part: str, idx: np.ndarray,
           y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> pd.DataFrame:
    df = ds.df.iloc[idx]
    cols: dict[str, np.ndarray] = {
        "profile": np.repeat(ds.profile, idx.size), "scheme": np.repeat(fold.scheme, idx.size),
        "fold": np.repeat(fold.fold, idx.size), "model": np.repeat(model_name, idx.size),
        "part": np.repeat(part, idx.size), "rsn": df["rsn"].to_numpy(dtype=np.int64),
        "eqid": df["eqid"].to_numpy(dtype=np.int64), "ssn": df["ssn"].to_numpy(dtype=np.int64),
    }
    for j, im in enumerate(IM_NAMES):
        cols[f"y_{im}"] = y[:, j].astype(np.float32)
        cols[f"mu_{im}"] = mu[:, j].astype(np.float32)
        cols[f"sigma_{im}"] = sigma[:, j].astype(np.float32)
    return pd.DataFrame(cols)


def _metrics(ds: GMDataset, fold: SplitFold, model: BaseSpectralModel, part: str, idx: np.ndarray,
             y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, extra: dict[str, object]) -> list[dict[str, object]]:
    df = ds.df.iloc[idx]
    eq, st = df["eqid"].to_numpy(), df["ssn"].to_numpy()
    rows = []
    for j, im in enumerate(IM_NAMES):
        s = score(y[:, j], mu[:, j], sigma[:, j])
        dec = decompose_residuals(y[:, j] - mu[:, j], eq, st) if part == "test" else None
        rows.append({"profile": ds.profile, "scheme": fold.scheme, "fold": fold.fold, "model": model.name,
                     "part": part, "im": im, "period": period_of(im), **s,
                     "tau_test": dec.tau if dec else np.nan,
                     "phi_s2s_test": dec.phi_s2s if dec else np.nan,
                     "phi_0_test": dec.phi_0 if dec else np.nan, **extra})
    return rows


def evaluate_ml_fold(ds: GMDataset, fold: SplitFold, model: BaseSpectralModel
                     ) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    """Fit on train_fit; predict and score test and cal."""
    t0 = time.perf_counter()
    model.fit(ds.df.iloc[fold.train_fit])
    extra: dict[str, object] = {"fit_seconds": round(time.perf_counter() - t0, 2),
                                "n_train_fit": int(fold.train_fit.size),
                                "device": str(getattr(model, "device", getattr(getattr(model, "corrector", None),
                                                                                "device", "cpu")))}
    rows: list[dict[str, object]] = []
    frames: dict[str, pd.DataFrame] = {}
    for part, idx in (("test", fold.test), ("cal", fold.cal)):
        if idx.size == 0:
            frames[part] = pd.DataFrame()
            continue
        sub = ds.df.iloc[idx]
        mu, sigma = model.predict(sub)
        y = target_matrix(sub)
        rows += _metrics(ds, fold, model, part, idx, y, mu, sigma, extra)
        frames[part] = _frame(ds, fold, model.name, part, idx, y, mu, sigma)
    return rows, frames["test"], frames["cal"]


def evaluate_ml_models(data_dir: Path, out_dir: Path, profiles: tuple[str, ...] = ("full", "sp23"),
                       schemes: tuple[str, ...] = DEFAULT_SCHEMES, models: tuple[str, ...] = DEFAULT_MODELS,
                       physics_cache: Path | None = None, station_encoding: bool = True, fast: bool = False,
                       seed: int = 0, log: Callable[[str], None] = print) -> tuple[pd.DataFrame, Path, Path]:
    """Run every ML model on frozen splits. Returns (metrics, oof path, cal path)."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_w = OOFWriter(out_dir / "oof_ml_predictions.parquet")
    cal_w = OOFWriter(out_dir / "cal_ml_predictions.parquet")
    rows: list[dict[str, object]] = []
    try:
        for prof in profiles:
            ds = GMDataset.from_parquet(data_dir / f"{prof}.parquet", prof)
            splits = load_splits(data_dir / f"splits_{prof}.json", ds)
            for scheme in schemes:
                if scheme not in splits:
                    log(f"[{prof}] scheme {scheme} not in splits file; skipped")
                    continue
                for fold in splits[scheme]:
                    for name in models:
                        model = make_ml_model(name, physics_cache, station_encoding, fast, seed)
                        r, test_df, cal_df = evaluate_ml_fold(ds, fold, model)
                        rows += r
                        if len(test_df):
                            test_w.write(test_df)
                        if len(cal_df):
                            cal_w.write(cal_df)
                        t = [x for x in r if x["part"] == "test"]
                        rmse = np.nanmean([x["rmse"] for x in t]) if t else float("nan")
                        cov = np.nanmean([x["cov90"] for x in t]) if t else float("nan")
                        log(f"[{prof}] {scheme}/{fold.fold} {model.name}: fit {r[0]['fit_seconds'] if r else 0:.1f} s "
                            f"on {r[0]['device'] if r else '?'}, test RMSE {rmse:.3f}, cov90 {cov:.3f}")
    finally:
        test_w.close()
        cal_w.close()
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "ml_metrics.csv", index=False)
    return metrics, test_w.path, cal_w.path
