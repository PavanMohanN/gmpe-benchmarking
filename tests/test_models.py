"""Phase 4: leakage-free features, masked loss, smoothness, ML models end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gmpe_bench import io
from gmpe_bench.data import GMDataset
from gmpe_bench.models import (
    FeaturePipeline,
    MLPConfig,
    ResidualHybridGMM,
    SpectralMLP,
    XGBConfig,
    XGBoostSpectralGMM,
    evaluate_ml_models,
    target_matrix,
)
from gmpe_bench.models.base import N_IMS, grouped_holdout, sa_columns
from gmpe_bench.models.gbdt import resolve_device, to_long
from gmpe_bench.models.spectral_mlp import masked_gaussian_nll, spectral_smoothness
from gmpe_bench.schema import IM_NAMES, im_column
from gmpe_bench.splits import assert_fit_transform_clean, make_splits, save_splits

FAST_MLP = MLPConfig(epochs=80, patience=15, hidden=32, depth=2, batch_size=128)
FAST_XGB = XGBConfig(n_estimators=60, early_stopping_rounds=10, scale_estimators=40, max_depth=4)


def _sim(n_events: int = 40, per_event: int = 12, n_stations: int = 80, seed: int = 0) -> pd.DataFrame:
    """Simple synthetic records with crossed effects; no pygmm needed."""
    rng = np.random.default_rng(seed)
    eq = np.repeat(np.arange(1, n_events + 1), per_event)
    n = eq.size
    st = rng.integers(1, n_stations + 1, n)
    st[::30] = 1  # one well-recorded station with a strong site term
    mw = rng.uniform(4.5, 7.5, n_events + 1)[eq]
    rjb = rng.uniform(1, 150, n)
    vs30_st = rng.uniform(200, 1000, n_stations + 1)
    vs30_st[1] = 500.0  # mid-range, so the VS30 trend cannot absorb station 1's term
    vs30 = vs30_st[st]
    df = pd.DataFrame({
        "rsn": np.arange(1, n + 1), "eqid": eq, "ssn": st, "mw": mw, "rjb": rjb,
        "rrup": np.sqrt(rjb**2 + 16), "rx": rng.uniform(-50, 50, n), "ry0": np.nan, "vs30": vs30,
        "vs30_measured": 1.0, "z1": np.nan, "z2p5": np.nan, "ztor": 2.0, "dip": 90.0, "rake": 0.0,
        "width": 10.0, "hypo_depth": 8.0, "mechanism": 0, "cl_class": "CL1", "crjb": np.nan,
        "hypo_lat": 35.0, "hypo_lon": -118.0, "year": 2000.0,
    })
    dB = rng.normal(0, 0.4, n_events + 1)[eq]
    dS = rng.normal(0, 0.3, n_stations + 1)[st] + 1.0 * (st == 1)
    for k, im in enumerate(IM_NAMES):
        base = -2.0 + 1.2 * (mw - 6) - 1.1 * np.log(np.sqrt(rjb**2 + 36)) - 0.5 * np.log(vs30 / 760) - 0.02 * k
        df[im_column("H", im)] = np.exp(base + dB + dS + rng.normal(0, 0.45, n))
    df.loc[df.index[: n // 3], im_column("H", "SA_10s")] = np.nan   # T > T_max for some records
    return df


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _sim()


def _split(df: pd.DataFrame, n_train_events: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["eqid"] <= n_train_events], df[df["eqid"] > n_train_events]


# ------------------------------------------------------------------ features
def test_pipeline_is_fit_on_train_only(df: pd.DataFrame) -> None:
    train_idx = np.flatnonzero(df["eqid"] <= 30)
    assert_fit_transform_clean(FeaturePipeline().fit(df.iloc[train_idx]), df, train_idx)
    with pytest.raises(AssertionError, match="leakage"):
        assert_fit_transform_clean(FeaturePipeline().fit(df), df, train_idx)


def test_pipeline_shapes_and_no_nan(df: pd.DataFrame) -> None:
    tr, te = _split(df)
    p = FeaturePipeline().fit(tr)
    X = p.transform(te)
    assert X.shape == (len(te), p.n_features) and np.isfinite(X).all()
    assert "dS2S_PGA" in p.feature_names_ and "region_unseen" in p.feature_names_


def test_unseen_region_does_not_crash(df: pd.DataFrame) -> None:
    tr = df[df["eqid"] <= 30]
    te = df[df["eqid"] > 30].assign(hypo_lat=37.0, hypo_lon=140.0)  # Japan, absent from train
    p = FeaturePipeline().fit(tr)
    X = p.transform(te)
    names = p.feature_names_
    assert X[:, names.index("region_unseen")].min() == 1.0
    assert X[:, names.index("region_California_WUS")].max() == 0.0


def test_station_terms_zero_for_unseen(df: pd.DataFrame) -> None:
    tr = df[df["ssn"] != 2]
    te = df[df["ssn"] == 2]
    p = FeaturePipeline().fit(tr)
    X = p.transform(te)
    cols = [p.feature_names_.index(f"dS2S_{im}") for im in p.anchor_ims]
    assert np.all(X[:, cols] == 0.0)
    assert np.all(X[:, p.feature_names_.index("station_seen")] == 0.0)


def test_station_terms_learned_and_out_of_fold(df: pd.DataFrame) -> None:
    tr, _ = _split(df)
    p = FeaturePipeline()
    X_oof = p.fit_transform(tr)
    X_in = p.transform(tr)
    j = p.feature_names_.index("dS2S_PGA")
    s1 = (tr["ssn"] == 1).to_numpy()
    assert X_in[s1, j].mean() > 0.3                   # strong station effect recovered
    assert not np.allclose(X_oof[:, j], X_in[:, j])   # training rows use OOF terms


def test_no_station_encoding(df: pd.DataFrame) -> None:
    p = FeaturePipeline(station_encoding=False).fit(df)
    assert not any(n.startswith("dS2S") for n in p.feature_names_)


# ------------------------------------------------------------------ losses
def test_masked_nll_ignores_nan_targets() -> None:
    torch.manual_seed(0)
    mu = torch.randn(4, 5, requires_grad=True)
    ls = torch.zeros(4, 5, requires_grad=True)
    y = torch.randn(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[:, -1] = False
    y_nan = y.clone()
    y_nan[:, -1] = float("nan")
    loss = masked_gaussian_nll(mu, ls, torch.nan_to_num(y_nan), mask)
    ref = (0.5 * (y[:, :-1] - mu[:, :-1]) ** 2).mean()
    assert torch.isclose(loss, ref)
    loss.backward()
    assert torch.isfinite(mu.grad).all() and torch.all(mu.grad[:, -1] == 0)


def test_smoothness_penalty() -> None:
    idx = torch.arange(6)
    straight = torch.arange(6, dtype=torch.float32).repeat(3, 1)
    jagged = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]).repeat(3, 1)
    assert float(spectral_smoothness(straight, idx)) == pytest.approx(0.0)
    assert float(spectral_smoothness(jagged, idx)) > 1.0


def test_sa_columns_exclude_pga_pgv() -> None:
    cols = sa_columns()
    assert 0 not in cols and 1 not in cols and cols.size == N_IMS - 2


def test_grouped_holdout_whole_events() -> None:
    g = np.repeat(np.arange(20), 5)
    val = grouped_holdout(g, 0.2, 0)
    assert set(g[val]).isdisjoint(set(g[~val])) and len(set(g[val])) == 4


def test_to_long_drops_masked() -> None:
    X = np.ones((3, 2), dtype=np.float32)
    Y = np.zeros((3, N_IMS))
    Y[0, -1] = np.nan
    Xl, yl, rec, imi = to_long(X, Y)
    assert Xl.shape == (3 * N_IMS - 1, 5) and yl.shape[0] == 3 * N_IMS - 1


# ------------------------------------------------------------------ models
def test_mlp_fit_predict(df: pd.DataFrame) -> None:
    tr, te = _split(df)
    m = SpectralMLP(FAST_MLP, seed=1).fit(tr)
    mu, sigma = m.predict(te)
    assert mu.shape == sigma.shape == (len(te), N_IMS)
    assert np.isfinite(mu).all() and (sigma > 0).all()
    assert set(m.val_eqids_) <= set(tr["eqid"])            # early stopping uses train_fit events only
    assert m.history[-1][2] <= m.history[0][2]             # validation loss improved
    r = target_matrix(te) - mu
    assert np.sqrt(np.nanmean(r**2)) < 1.2


def test_mlp_deterministic(df: pd.DataFrame) -> None:
    tr, te = _split(df)
    a = SpectralMLP(FAST_MLP, seed=3).fit(tr).predict(te)[0]
    b = SpectralMLP(FAST_MLP, seed=3).fit(tr).predict(te)[0]
    assert np.allclose(a, b, atol=1e-5)


def test_xgb_fit_predict(df: pd.DataFrame) -> None:
    tr, te = _split(df)
    m = XGBoostSpectralGMM(FAST_XGB, seed=0).fit(tr)
    mu, sigma = m.predict(te)
    assert mu.shape == (len(te), N_IMS) and np.isfinite(mu).all() and (sigma > 0).all()
    assert m.best_iteration is not None and m.device in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"


def test_predict_requires_fit(df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        SpectralMLP(FAST_MLP).predict(df)


def test_hybrid_adds_physics_median(df: pd.DataFrame, tmp_path: Path) -> None:
    tr, te = _split(df)
    h = ResidualHybridGMM("BSSA14", "mlp", tmp_path / "cache", mlp_config=FAST_MLP).fit(tr)
    mu, sigma = h.predict(te)
    delta, _ = h.correction(te)
    assert np.allclose(mu, h.physics_median(te) + delta)
    assert np.abs(delta).max() <= 2.0 + 1e-9
    assert (tmp_path / "cache" / "BSSA14.parquet").exists()
    assert h.name == "Hybrid_BSSA14_MLP"


# ------------------------------------------------------------------ end to end
def test_evaluate_ml_end_to_end(df: pd.DataFrame, tmp_path: Path) -> None:
    data = tmp_path / "data"
    io.write_parquet(df, data / "toy.parquet")
    ds = GMDataset.from_parquet(data / "toy.parquet", "toy")
    splits = make_splits(ds, ["event_grouped"], seed=42)
    save_splits(ds, splits, data / "splits_toy.json", seed=42, cal_frac=0.25)

    metrics, oof_path, cal_path = evaluate_ml_models(
        data, tmp_path / "out", profiles=("toy",), schemes=("event_grouped",),
        models=("XGBoost", "SpectralMLP"), fast=True, log=lambda _: None)
    assert (tmp_path / "out" / "ml_metrics.csv").exists()
    assert set(metrics["part"]) == {"test", "cal"} and set(metrics["im"]) == set(IM_NAMES)

    oof, cal = pd.read_parquet(oof_path), pd.read_parquet(cal_path)
    n_test = sum(f.test.size for f in splits["event_grouped"])
    n_cal = sum(f.cal.size for f in splits["event_grouped"])
    assert len(oof) == 2 * n_test and len(cal) == 2 * n_cal
    for f in splits["event_grouped"]:
        cal_rsn = set(cal.loc[cal["fold"] == f.fold, "rsn"])
        test_rsn = set(oof.loc[oof["fold"] == f.fold, "rsn"])
        assert cal_rsn.isdisjoint(test_rsn)
        assert set(ds.df["eqid"].to_numpy()[f.cal]).isdisjoint(ds.df["eqid"].to_numpy()[f.train_fit])
    assert {"y_PGA", "mu_PGA", "sigma_PGA", "mu_SA_10s"} <= set(oof.columns)
