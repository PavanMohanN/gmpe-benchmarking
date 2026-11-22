"""Phase 5: conformal guarantees, shift recovery, zero test-label use, HSCP behaviour."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmpe_bench.conformal import (
    HSCP,
    CalSet,
    ClusteredEventConformal,
    NaiveSplitConformal,
    TestInputs,
    UncalibratedGaussian,
    WeightedShiftConformal,
    conformal_level,
    crossed_terms,
    default_methods,
    density_ratio_weights,
    effective_n,
    normalized_scores,
    split_conformal_quantile,
    weighted_conformal_quantile,
)
from gmpe_bench.conformal.evaluate import interval_metrics, joint_coverage, width_roughness
from gmpe_bench.schema import IM_NAMES

K = len(IM_NAMES)
N_STATIONS = 500


def _corr_chol() -> np.ndarray:
    idx = np.arange(K)
    C = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / 4.0) ** 2)
    return np.linalg.cholesky(C + 1e-6 * np.eye(K))


CHOL = _corr_chol()
DS_TRUE = np.random.default_rng(99).normal(0, 0.35, N_STATIONS)


def _records(n_ev: int, per: int, mw_lo: float, mw_hi: float, rng: np.random.Generator,
             eq_offset: int = 0, stations: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Residual-world simulation: y = dB + dS2S + heteroscedastic, period-correlated eps; mu = 0."""
    eq = np.repeat(np.arange(n_ev), per)
    n = eq.size
    mw = rng.uniform(mw_lo, mw_hi, n_ev)[eq]
    ssn = rng.choice(stations, n) if stations is not None else rng.integers(0, N_STATIONS, n)
    eps = (rng.normal(size=(n, K)) @ CHOL.T) * (0.35 + 0.12 * (mw - 4))[:, None]
    y = rng.normal(0, 0.35, n_ev)[eq][:, None] + DS_TRUE[ssn][:, None] + eps
    X = np.column_stack([mw, np.zeros(n)])
    return {"y": y, "mu": np.zeros((n, K)), "sigma": np.full((n, K), 0.55), "eqid": eq + eq_offset,
            "ssn": ssn, "X": X}


def _cal_test(seed: int, cal_mw: tuple[float, float] = (4, 7.5), test_mw: tuple[float, float] = (4, 7.5),
              train_stations: np.ndarray | None = None) -> tuple[CalSet, TestInputs, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = _records(150, 20, *cal_mw, rng)
    t = _records(60, 20, *test_mw, rng, eq_offset=10_000)
    cal = CalSet(c["y"], c["mu"], c["sigma"], c["eqid"], c["ssn"], c["X"],
                 np.arange(300) if train_stations is None else train_stations)
    test = TestInputs(t["mu"], t["sigma"], t["eqid"], t["ssn"], t["X"])
    return cal, test, t["y"]


def _cov(iv, y: np.ndarray) -> float:
    fin = np.isfinite(y)
    return float(((y >= iv.lower) & (y <= iv.upper))[fin].mean())


# ------------------------------------------------------------------ scores
def test_conformal_level_and_quantile() -> None:
    assert conformal_level(9, 0.1) == pytest.approx(1.0)
    assert conformal_level(99, 0.1) == pytest.approx(90 / 99)
    s = np.arange(1, 100, dtype=float)          # n = 99 -> 90th smallest
    assert split_conformal_quantile(s, 0.1) == 90.0
    assert np.isinf(split_conformal_quantile(np.arange(5.0), 0.1))  # n too small


def test_weighted_quantile_reduces_to_split() -> None:
    s = np.random.default_rng(0).normal(size=257)
    q_w = weighted_conformal_quantile(s, np.ones_like(s), 1.0, 0.1)[0]
    assert q_w == split_conformal_quantile(s, 0.1)


def test_effective_n_with_masked_long_periods() -> None:
    y = np.random.default_rng(1).normal(size=(50, 3))
    y[:20, 2] = np.nan                           # SA beyond T_max for 20 records
    s = normalized_scores(y, np.zeros_like(y), np.ones_like(y))
    assert effective_n(s).tolist() == [50, 50, 30]


# ------------------------------------------------------------------ guarantees
@pytest.mark.parametrize("method", [NaiveSplitConformal(), ClusteredEventConformal(), HSCP()])
def test_marginal_coverage_exchangeable(method) -> None:
    covs = [_cov(method.calibrate(*_cal_test(seed)[:2], 0.1), _cal_test(seed)[2]) for seed in range(6)]
    assert np.mean(covs) >= 0.9 - 0.025          # 1 - alpha - O(n^-1/2), allowing event clustering


def test_uncalibrated_gaussian_undercovers() -> None:
    cal, test, y = _cal_test(0)
    assert _cov(UncalibratedGaussian().calibrate(cal, test, 0.1), y) < 0.85   # sigma too small


def test_covariate_shift_recovery() -> None:
    naive, weighted, hscp = [], [], []
    for seed in range(5):
        cal, test, y = _cal_test(seed, cal_mw=(4, 6.5), test_mw=(6, 7.5))
        naive.append(_cov(NaiveSplitConformal().calibrate(cal, test, 0.1), y))
        weighted.append(_cov(WeightedShiftConformal().calibrate(cal, test, 0.1), y))
        hscp.append(_cov(HSCP().calibrate(cal, test, 0.1), y))
    assert np.mean(naive) < 0.87                  # shift breaks the naive band
    assert np.mean(weighted) > np.mean(naive) + 0.02
    assert np.mean(hscp) > np.mean(naive) + 0.02 and np.mean(hscp) >= 0.86


def test_density_ratio_uses_covariates_only() -> None:
    rng = np.random.default_rng(0)
    wc, wt = density_ratio_weights(rng.normal(0, 1, (500, 2)), rng.normal(1, 1, (300, 2)))
    assert wc.shape == (500,) and wt.shape == (300,)
    assert 0.05 <= wc.min() and wc.max() <= 20.0
    params = inspect.signature(density_ratio_weights).parameters
    assert not any(p.startswith("y") for p in params)


# ------------------------------------------------------------------ no test labels
def test_test_inputs_have_no_targets() -> None:
    fields = TestInputs.__dataclass_fields__
    assert "y" not in fields
    with pytest.raises(TypeError):
        TestInputs(y=np.zeros((2, K)), mu=np.zeros((2, K)), sigma=np.ones((2, K)),  # type: ignore[call-arg]
                   eqid=np.zeros(2), ssn=np.zeros(2), X_shift=np.zeros((2, 2)))


@pytest.mark.parametrize("method", default_methods(ablations=True), ids=lambda m: m.name)
def test_bounds_independent_of_test_labels(method) -> None:
    cal, test, y = _cal_test(3, cal_mw=(4, 6.5), test_mw=(5.5, 7.5))
    a = method.calibrate(cal, test, 0.1)
    rng = np.random.default_rng(7)
    y_perm = rng.permutation(y)                  # randomise test labels...
    y_noise = y + rng.normal(0, 5, y.shape)
    b = method.calibrate(cal, test, 0.1)         # ...calibration cannot see them
    assert np.array_equal(a.lower, b.lower) and np.array_equal(a.upper, b.upper)
    m1, m2 = interval_metrics(y_perm, a), interval_metrics(y_noise, a)
    assert np.isfinite(m1["width"]).all() and np.allclose(m1["width"], m2["width"])
    assert "test" in inspect.signature(method.calibrate).parameters


# ------------------------------------------------------------------ HSCP behaviour
def test_hscp_seen_vs_unseen_stations() -> None:
    rng = np.random.default_rng(11)
    pool = np.arange(200)                         # cal and seen-test stations share this pool
    c = _records(150, 20, 4, 7.5, rng, stations=pool)
    t_seen = _records(30, 20, 4, 7.5, rng, eq_offset=10_000, stations=pool)
    t_new = _records(30, 20, 4, 7.5, rng, eq_offset=20_000, stations=np.arange(300, 500))
    cal = CalSet(c["y"], c["mu"], c["sigma"], c["eqid"], c["ssn"], c["X"], np.array([], dtype=np.int64))
    cat = lambda k: np.concatenate([t_seen[k], t_new[k]])  # noqa: E731
    test = TestInputs(cat("mu"), cat("sigma"), cat("eqid"), cat("ssn"), cat("X"))
    iv = HSCP().calibrate(cal, test, 0.1)
    seen = iv.seen
    assert seen is not None and seen[: len(t_seen["eqid"])].all() and not seen[len(t_seen["eqid"]):].any()
    width = (iv.upper - iv.lower).mean(axis=1)
    assert width[seen].mean() < width[~seen].mean()             # tighter where dS2S is known
    assert np.abs(iv.center[seen]).mean() > 0.05                 # seen stations are re-centred
    assert np.allclose(iv.center[~seen], 0.0)                    # unseen: no site term
    y = cat("y")
    assert _cov(iv, y) >= 0.87


def test_hscp_spectral_smoothness() -> None:
    cal, test, y = _cal_test(5)
    rough = width_roughness(HSCP(smooth=True).calibrate(cal, test, 0.1))
    raw = width_roughness(HSCP(smooth=False).calibrate(cal, test, 0.1))
    assert rough < raw


def test_hscp_joint_band() -> None:
    cal, test, y = _cal_test(2)
    iv = HSCP().calibrate(cal, test, 0.1)
    assert iv.joint_lower is not None
    assert np.all(iv.joint_upper - iv.joint_lower >= (iv.upper - iv.lower) - 1e-9)   # simultaneous >= marginal
    assert joint_coverage(y, iv.joint_lower, iv.joint_upper) >= 0.85
    assert joint_coverage(y, iv.lower, iv.upper) < joint_coverage(y, iv.joint_lower, iv.joint_upper)


def test_calibrate_many_matches_single() -> None:
    cal, test, _ = _cal_test(4)
    many = HSCP().calibrate_many(cal, test, (0.1, 0.05))
    one = HSCP().calibrate(cal, test, 0.05)
    assert np.allclose(many[0.05].upper, one.upper)
    assert (many[0.05].upper - many[0.05].lower).mean() > (many[0.1].upper - many[0.1].lower).mean()


def test_crossed_terms_vectorised_matches_scalar() -> None:
    from gmpe_bench.baselines import decompose_residuals

    rng = np.random.default_rng(0)
    eq, st = np.repeat(np.arange(100), 10), rng.integers(0, 80, 1000)
    R = np.column_stack([rng.normal(0, .4, 100)[eq] + rng.normal(0, .3, 80)[st] + rng.normal(0, .4, 1000)
                         for _ in range(3)])
    R[:100, 2] = np.nan
    ct = crossed_terms(R, eq, st)
    d = decompose_residuals(R[:, 2], eq, st)
    assert ct.tau[2] == pytest.approx(d.tau, rel=1e-3) and ct.phi_0[2] == pytest.approx(d.phi_0, rel=1e-3)


# ------------------------------------------------------------------ end to end
def test_evaluate_conformal_end_to_end(tmp_path: Path) -> None:
    from gmpe_bench import io
    from gmpe_bench.conformal import evaluate_conformal
    from gmpe_bench.data import GMDataset
    from gmpe_bench.models import evaluate_ml_models
    from gmpe_bench.splits import make_splits, save_splits

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_models import _sim  # noqa: E402

    df = _sim(n_events=40, per_event=15, n_stations=60)
    data = tmp_path / "data"
    io.write_parquet(df, data / "toy.parquet")
    ds = GMDataset.from_parquet(data / "toy.parquet", "toy")
    save_splits(ds, make_splits(ds, ["event_grouped", "spatial_block"], seed=1),
                data / "splits_toy.json", seed=1, cal_frac=0.25)
    evaluate_ml_models(data, tmp_path / "ml", profiles=("toy",), schemes=("event_grouped",),
                       models=("SpectralMLP",), fast=True, log=lambda _: None)
    metrics, cond, summary = evaluate_conformal(data, tmp_path / "conf", ml_dir=tmp_path / "ml",
                                                profiles=("toy",), schemes=("event_grouped",),
                                                log=lambda _: None)
    assert set(metrics["method"]) == {"UncalibratedGaussian", "NaiveSplitConformal", "ClusteredEventConformal",
                                      "WeightedShiftConformal", "HSCP"}
    assert {"ALL", "JOINT_MARGINAL", "JOINT_BAND", "PGA"} <= set(metrics["im"])
    assert set(metrics["alpha"]) == {0.10, 0.05}
    assert set(cond["stratum"]) == {"mw", "rrup", "vs30_class", "station"}
    assert summary.exists() and len(pd.read_parquet(summary)) > 0
    assert (tmp_path / "conf" / "conformal_metrics.csv").exists()


# ------------------------------------------------------------------ support rule and groups
def test_unsupported_region_does_not_saturate_weights() -> None:
    """spatial_block: a held-out region has no cal support -> region group dropped, no 0.05/20 weights."""
    from gmpe_bench.conformal.shift import supported_columns

    rng = np.random.default_rng(0)
    cont_c, cont_t = rng.normal(size=(400, 4)), rng.normal(size=(200, 4))
    cl2 = np.zeros((400, 1)), np.zeros((200, 1))
    reg_c = np.column_stack([np.ones(400), np.zeros(400)])      # cal: region A only
    reg_t = np.column_stack([np.zeros(200), np.ones(200)])      # test: region B only
    Xc = np.hstack([cont_c, cl2[0], reg_c])
    Xt = np.hstack([cont_t, cl2[1], reg_t])
    keep = supported_columns(Xc, Xt)
    assert keep[:5].all() and not keep[5:].any()                # whole region group dropped
    wc, wt = density_ratio_weights(Xc, Xt)
    assert 0.3 < wt.min() and wt.max() < 3.0


def test_unsupported_aftershock_flag_dropped() -> None:
    rng = np.random.default_rng(1)
    Xc = np.hstack([rng.normal(size=(300, 4)), np.zeros((300, 1))])   # cal: mainshocks only
    Xt = np.hstack([rng.normal(size=(100, 4)), np.ones((100, 1))])    # test: aftershocks only
    wc, wt = density_ratio_weights(Xc, Xt)
    assert wt.max() < 3.0


def test_hscp_three_station_groups() -> None:
    rng = np.random.default_rng(5)
    c = _records(150, 20, 4, 7.5, rng, stations=np.arange(0, 200))
    t = _records(60, 20, 4, 7.5, rng, eq_offset=10_000, stations=np.arange(0, 400))
    train_only = np.arange(200, 300)                              # in train_fit, absent from cal
    cal = CalSet(c["y"], c["mu"], c["sigma"], c["eqid"], c["ssn"], c["X"], train_only)
    test = TestInputs(t["mu"], t["sigma"], t["eqid"], t["ssn"], t["X"])
    h = HSCP()
    site = h._site_fit(cal, test)
    g = site.group_test
    assert set(g[t["ssn"] < 200]) == {0}                          # cal term available
    assert set(g[(t["ssn"] >= 200) & (t["ssn"] < 300)]) == {1}    # train_fit only
    assert set(g[t["ssn"] >= 300]) == {2}                         # new
    iv = h.calibrate(cal, test, 0.1)
    assert np.allclose(iv.center[g != 0], 0.0)                    # only cal-term stations are re-centred
    assert _cov(iv, t["y"]) >= 0.86
