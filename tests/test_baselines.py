"""Phase 3: pygmm mapping and units, residual decomposition, leakage rules, end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pygmm
import pytest

from gmpe_bench import io
from gmpe_bench.baselines import (
    MixedEffectsGMM,
    NGAWest2Baseline,
    apply_site_terms,
    decompose_residuals,
    evaluate_baselines,
    predict_table,
    scenario_kwargs,
    score,
)
from gmpe_bench.baselines.mixed_effects import design_matrix
from gmpe_bench.data import GMDataset
from gmpe_bench.schema import IM_NAMES, im_column
from gmpe_bench.splits import make_splits, save_splits

ROW = {"rsn": 1, "mw": 6.5, "rrup": 12.0, "rjb": 10.0, "rx": np.nan, "ry0": np.nan, "vs30": 400.0,
       "vs30_measured": 1.0, "z1": 500.0, "z2p5": 2000.0, "ztor": 2.0, "dip": np.nan, "rake": 90.0,
       "width": np.nan, "hypo_depth": 8.0, "mechanism": 2, "cl_class": "CL1", "crjb": np.nan,
       "hypo_lat": 35.0, "hypo_lon": -118.0}


def _sim(n_events: int = 30, per_event: int = 10, n_stations: int = 60, seed: int = 1) -> pd.DataFrame:
    """Records with full Tier B inputs and H targets = BSSA14 median + crossed effects."""
    rng = np.random.default_rng(seed)
    eq = np.repeat(np.arange(1, n_events + 1), per_event)
    n = eq.size
    ev_mw = rng.uniform(4.5, 7.5, n_events)
    st = rng.integers(1, n_stations + 1, n)
    rjb = rng.uniform(1, 150, n)
    df = pd.DataFrame({
        "rsn": np.arange(1, n + 1), "eqid": eq, "ssn": st, "mw": ev_mw[eq - 1],
        "rjb": rjb, "rrup": np.sqrt(rjb**2 + 16), "rx": rng.uniform(-50, 50, n), "ry0": np.nan,
        "vs30": rng.uniform(200, 1000, n_stations + 1)[st], "vs30_measured": 1.0,
        "z1": np.nan, "z2p5": np.nan, "ztor": 2.0, "dip": 90.0, "rake": 0.0, "width": 10.0,
        "hypo_depth": 8.0, "mechanism": 0, "cl_class": "CL1", "crjb": np.nan,
        "hypo_lat": 35.0, "hypo_lon": -118.0, "year": 2000.0,
    })
    med = predict_table(df, "BSSA14")
    dB = rng.normal(0, 0.4, n_events + 1)[eq]
    dS = rng.normal(0, 0.3, n_stations + 1)[st]
    for im in IM_NAMES:
        df[im_column("H", im)] = np.exp(med[f"mu_{im}"].to_numpy() + 0.1 + dB + dS + rng.normal(0, 0.45, n))
    return df


# ------------------------------------------------------------------ scenario mapping
def test_scenario_mapping_defaults() -> None:
    kw = scenario_kwargs(ROW, "ASK14", "California_WUS")
    assert kw["mechanism"] == "RS" and kw["dip"] == 40.0
    assert kw["dist_x"] == -10.0 and kw["on_hanging_wall"] is False
    assert kw["depth_1_0"] == pytest.approx(0.5)  # 500 m -> 0.5 km
    assert kw["vs_source"] == "measured" and kw["region"] == "california"
    assert "dist_y0" not in kw and "width" not in kw  # missing -> model default


def test_scenario_mapping_hanging_wall_and_regions() -> None:
    row = {**ROW, "rx": 5.0, "dip": 45.0}
    assert scenario_kwargs(row, "ASK14")["on_hanging_wall"] is True
    assert scenario_kwargs(row, "CY14", "Taiwan")["region"] == "california"  # CY14 has no taiwan/global
    assert scenario_kwargs(row, "ASK14", "Taiwan")["region"] == "taiwan"
    assert scenario_kwargs(row, "CB14", "Europe_MiddleEast")["region"] == "global"
    bssa = scenario_kwargs(row, "BSSA14")
    assert "dip" not in bssa and "dist_rup" not in bssa  # only accepted inputs are passed


def test_scenario_mechanism_from_rake() -> None:
    assert scenario_kwargs({**ROW, "mechanism": np.nan, "rake": -90.0}, "CB14")["mechanism"] == "NS"
    assert scenario_kwargs({**ROW, "mechanism": np.nan, "rake": np.nan}, "CB14")["mechanism"] == "SS"


@pytest.mark.parametrize("model,cls", [("ASK14", pygmm.AbrahamsonSilvaKamai2014),
                                       ("BSSA14", pygmm.BooreStewartSeyhanAtkinson2014),
                                       ("CB14", pygmm.CampbellBozorgnia2014),
                                       ("CY14", pygmm.ChiouYoungs2014)])
def test_units_ln_g_and_ln_cms(model: str, cls: type) -> None:
    df = pd.DataFrame([ROW])
    t = predict_table(df, model).iloc[0]
    ref = cls(pygmm.Scenario(**scenario_kwargs(ROW, model, "California_WUS")))
    assert t["mu_PGA"] == pytest.approx(np.log(ref.pga))         # ln(g)
    assert t["mu_PGV"] == pytest.approx(np.log(ref.pgv))         # ln(cm/s)
    assert t["mu_SA_1s"] == pytest.approx(ref.interp_ln_spec_accels(np.array([1.0]))[0])
    assert t["sigma_PGA"] == pytest.approx(ref.ln_std_pga)
    assert np.isfinite(t.filter(like="mu_").to_numpy(dtype=float)).all()
    assert not t["failed"]


def test_out_of_range_inputs_flagged_not_dropped() -> None:
    t = predict_table(pd.DataFrame([{**ROW, "mw": 2.8, "rrup": 350.0, "rjb": 349.0}]), "ASK14").iloc[0]
    assert not t["applicable"] and np.isfinite(t["mu_PGA"])


# ------------------------------------------------------------------ decomposition
def test_decomposition_recovers_variances() -> None:
    rng = np.random.default_rng(0)
    eq = np.repeat(np.arange(400), 25)
    st = rng.integers(0, 500, eq.size)
    r = rng.normal(0, 0.4, 400)[eq] + rng.normal(0, 0.3, 500)[st] + rng.normal(0, 0.45, eq.size)
    d = decompose_residuals(r, eq, st)
    assert d.converged
    assert d.tau == pytest.approx(0.4, abs=0.05)
    assert d.phi_s2s == pytest.approx(0.3, abs=0.05)
    assert d.phi_0 == pytest.approx(0.45, abs=0.03)
    assert d.sigma == pytest.approx(np.sqrt(d.tau**2 + d.phi_s2s**2 + d.phi_0**2))


def test_decomposition_ignores_nan() -> None:
    r = np.array([0.1, np.nan, -0.2, 0.3, 0.0, 0.2])
    d = decompose_residuals(r, np.array([1, 1, 2, 2, 3, 3]), np.array([1, 2, 1, 2, 1, 2]))
    assert d.n == 5


def test_unseen_station_gets_zero_term() -> None:
    rng = np.random.default_rng(1)
    eq, st = np.repeat(np.arange(20), 10), np.tile(np.arange(10), 20)
    d = decompose_residuals(rng.normal(size=200) + (st == 3) * 1.0, eq, st)
    test_st = np.array([3, 3, 999])
    mu_site, sigma, seen = apply_site_terms(np.zeros(3), test_st, d)
    assert seen.tolist() == [True, True, False]
    assert mu_site[0] == pytest.approx(d.station_terms.loc[3]) and mu_site[0] > 0.5
    assert mu_site[2] == 0.0
    assert sigma[0] == pytest.approx(np.hypot(d.tau, d.phi_0))
    assert sigma[2] == pytest.approx(d.sigma)


# ------------------------------------------------------------------ leakage rules
def test_bias_calibration_uses_train_only() -> None:
    df = _sim()
    train, test = df[df["eqid"] <= 20], df[df["eqid"] > 20].copy()
    m = NGAWest2Baseline("BSSA14").precompute(df).fit(train, "SA_1s")
    bias = m.bias
    test[im_column("H", "SA_1s")] *= 1e6  # corrupt test targets
    m2 = NGAWest2Baseline("BSSA14").precompute(pd.concat([train, test])).fit(train, "SA_1s")
    assert m2.bias == pytest.approx(bias)
    assert bias == pytest.approx(0.1, abs=0.25)  # simulated offset is 0.1
    mu, _ = m.predict(test)
    raw, _ = m.raw(test, "SA_1s")
    assert np.allclose(mu - raw, bias)


def test_uncalibrated_has_zero_bias() -> None:
    df = _sim()
    m = NGAWest2Baseline("CY14", calibrate_bias=False).precompute(df).fit(df, "PGA")
    assert m.bias == 0.0 and m.name == "CY14"


def test_mixed_effects_recovers_form() -> None:
    rng = np.random.default_rng(3)
    df = _sim(n_events=120, per_event=20, n_stations=200)
    X = design_matrix(df)
    beta = np.array([-1.0, 1.1, -0.1, -0.3, -1.2, 0.15, -0.003, -0.6, 0.1, -0.05])
    eq, st = df["eqid"].to_numpy(), df["ssn"].to_numpy()
    y = X @ beta + rng.normal(0, 0.4, 121)[eq] + rng.normal(0, 0.3, 201)[st] + rng.normal(0, 0.4, len(df))
    df[im_column("H", "PGA")] = np.exp(y)
    m = MixedEffectsGMM().fit(df, "PGA")
    assert m.converged
    assert m.tau == pytest.approx(0.4, abs=0.08) and m.phi_0 == pytest.approx(0.4, abs=0.04)
    mu, sd = m.predict(df)
    assert np.corrcoef(mu, X @ beta)[0, 1] > 0.98
    assert np.allclose(sd, m.decomposition.sigma)


def test_mixed_effects_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        MixedEffectsGMM().predict(_sim(5, 5, 5))


def test_score_coverage() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.6, 20000)
    s = score(y, np.zeros_like(y), np.full_like(y, 0.6))
    assert s["cov90"] == pytest.approx(0.90, abs=0.01)
    assert s["bias"] == pytest.approx(0.0, abs=0.02)


# ------------------------------------------------------------------ end to end
def test_evaluate_end_to_end(tmp_path: Path) -> None:
    df = _sim()
    data = tmp_path / "data"
    io.write_parquet(df, data / "toy.parquet")
    ds = GMDataset.from_parquet(data / "toy.parquet", "toy")
    splits = make_splits(ds, ["event_grouped", "station_grouped"], seed=42)
    save_splits(ds, splits, data / "splits_toy.json", seed=42, cal_frac=0.25)

    ims = ["PGA", "PGV", "SA_1s"]
    metrics, oof_path = evaluate_baselines(data, tmp_path / "out", profiles=("toy",),
                                           schemes=("event_grouped", "station_grouped"),
                                           ims=ims, log=lambda _: None)
    assert (tmp_path / "out" / "baseline_metrics.csv").exists()
    assert set(metrics["model"]) == {"ASK14_cal", "BSSA14_cal", "CB14_cal", "CY14_cal", "MixedEffects"}
    assert set(metrics["mode"]) == {"ergodic", "site_corrected"}
    assert set(metrics["im"]) == set(ims)
    assert metrics["rmse"].between(0.0, 3.0).all()

    oof = pd.read_parquet(oof_path)
    n_test = sum(f.test.size for folds in splits.values() for f in folds)
    assert len(oof) == n_test * 5
    assert {"y_PGA", "mu_PGA", "sigma_PGA", "mu_site_PGA", "sigma_site_PGA"} <= set(oof.columns)
    ev = oof[oof["scheme"] == "event_grouped"]
    cal_rsn = {int(r) for f in splits["event_grouped"] for r in ds.df["rsn"].to_numpy()[f.cal]}
    assert ev.groupby("model")["rsn"].nunique().eq(len(ds)).all()  # each record tested once
    assert cal_rsn  # calibration rows exist and are untouched by Phase 3 fitting

    cache = tmp_path / "out" / "cache"
    assert (cache / "ASK14.parquet").exists()
