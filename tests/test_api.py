"""High-level API: BaseGMM, evaluate(), evaluate_predictions(), iter_folds() and `gmpe-bench selftest`.

Without the PEER files, the API engines are exercised on the private smoke-test fixture by patching
``api.load_benchmark``; the public loaders are tested for their refusal of missing or synthetic data.
Tests marked ``real`` run only when the curated profile exists in ``data/processed``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import gmpe_bench as gb
from gmpe_bench import _smoke, api
from gmpe_bench.cli import app
from gmpe_bench.schema import IM_NAMES

K = len(IM_NAMES)
ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "processed"
HAVE_REAL = (REAL_DIR / "full.parquet").exists() and (REAL_DIR / "splits_full.json").exists()
real = pytest.mark.skipif(not HAVE_REAL, reason="curated NGA-West2 profile not built (gmpe-bench build)")


# ------------------------------------------------------------------ fixtures and helper models
@pytest.fixture(scope="module")
def bench():
    return _smoke.fixture_bench(("event_grouped", "random"))


@pytest.fixture
def patched(monkeypatch, bench):
    """Route the public API to the fixture (the public loader itself refuses synthetic data)."""
    monkeypatch.setattr(api, "load_benchmark", lambda profile="full", data_dir=None: bench)
    return bench


class MuOnly(gb.BaseGMM):
    """Returns the median only: sigma must come from the fallback."""

    def fit(self, X_train, y_train, metadata_train):
        self.mean_ = np.nanmean(y_train, axis=0)
        return self

    def predict(self, X_test, metadata_test):
        return np.tile(self.mean_, (len(X_test), 1))


class FixedSigma(MuOnly):
    default_sigma = 0.7


class Spy(gb.LinearGMM):
    """Records what the benchmark hands to the model."""

    seen: ClassVar[list] = []

    def fit(self, X_train, y_train, metadata_train):
        Spy.seen.append({"rsn": metadata_train["rsn"].to_numpy(), "n_y": y_train.shape[0], "n_x": len(X_train)})
        return super().fit(X_train, y_train, metadata_train)


# ------------------------------------------------------------------ BaseGMM
def test_basegmm_is_abstract():
    with pytest.raises(TypeError):
        gb.BaseGMM()


def test_package_exports_are_lazy_and_complete():
    assert len(gb.IM_NAMES) == K == 27
    for name in ("BaseGMM", "BenchmarkReport", "evaluate", "evaluate_predictions", "iter_folds", "LinearGMM"):
        assert hasattr(gb, name)
    assert not hasattr(gb, "make_synthetic_dataset")       # synthetic data is not part of the public API


@pytest.mark.parametrize("sigma", [0.5, np.full(K, 0.5), np.full((4, K), 0.5)])
def test_resolve_output_broadcasts_sigma(sigma):
    mu, sd, src = api._resolve_output((np.zeros((4, K)), sigma), 4, None)
    assert mu.shape == sd.shape == (4, K) and src == "model" and np.allclose(sd, 0.5)


def test_resolve_output_mu_only_uses_fallback_or_flags_missing():
    _, sd, src = api._resolve_output(np.zeros((3, K)), 3, 0.8)
    assert src == "fallback" and np.allclose(sd, 0.8)
    _, _, src = api._resolve_output(np.zeros((3, K)), 3, None)
    assert src == "missing"


@pytest.mark.parametrize("bad", [np.zeros((3, 5)), (np.zeros((3, K)), -np.ones((3, K))), (1, 2, 3)])
def test_resolve_output_rejects_bad_predictions(bad):
    with pytest.raises(ValueError):
        api._resolve_output(bad, 3, None)


def test_train_residual_sigma():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.5, (400, K))
    y[:, -1] = np.nan
    sd = api._train_residual_sigma(y, np.zeros_like(y))
    assert np.allclose(sd[:-1], 0.5, atol=0.06) and sd[-1] == 0.6


# ------------------------------------------------------------------ evaluate(): public guards
def test_evaluate_missing_data_message(tmp_path):
    with pytest.raises(gb.DataNotFoundError) as exc:
        gb.evaluate(gb.LinearGMM(), "full", data_dir=tmp_path)
    msg = str(exc.value)
    assert "PEER" in msg and "gmpe-bench build" in msg and "selftest" in msg
    assert "synthetic\")" not in msg          # no synthetic route is offered for benchmarking


def test_evaluate_missing_splits_message(tmp_path):
    _smoke.make_fixture_frame(25).to_parquet(tmp_path / "full.parquet")
    with pytest.raises(gb.DataNotFoundError, match="frozen splits"):
        gb.evaluate(gb.LinearGMM(), data_dir=tmp_path)


def test_evaluate_refuses_synthetic_profile(tmp_path):
    _smoke.make_fixture_frame(25).to_parquet(tmp_path / "full.parquet")
    (tmp_path / "splits_full.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a curated NGA-West2 profile"):
        gb.evaluate(gb.LinearGMM(), data_dir=tmp_path)


def test_evaluate_has_no_dataset_argument():
    with pytest.raises(TypeError):
        gb.evaluate(gb.LinearGMM(), dataset="synthetic")  # type: ignore[call-arg]


@pytest.mark.parametrize("kwargs, err", [({"profile": "tiny"}, ValueError), ({"schemes": ["nope"]}, ValueError),
                                         ({"methods": ["nope"]}, ValueError)])
def test_evaluate_rejects_bad_arguments(kwargs, err, tmp_path):
    with pytest.raises(err):
        gb.evaluate(gb.LinearGMM(), data_dir=tmp_path, **kwargs)


def test_evaluate_rejects_non_models():
    with pytest.raises(TypeError):
        gb.evaluate(object())


# ------------------------------------------------------------------ evaluate(): engine on the fixture
def test_evaluate_report(patched, tmp_path):
    rep = gb.evaluate(gb.LinearGMM(), "full", ["event_grouped", "random"], n_boot=100)
    assert isinstance(rep, gb.BenchmarkReport)
    assert set(rep.accuracy["scheme"]) == {"event_grouped", "random"}
    assert rep.audit["passed"].all() and len(rep.folds) == 10
    row = rep.accuracy.set_index("scheme").loc["event_grouped"]
    assert row["rmse_lo"] <= row["rmse"] <= row["rmse_hi"]
    assert not rep.meta["comparable"] and np.isnan(row["ref_ASK14_site"])   # fixture is never compared
    assert "leakage_inflation" in rep.meta
    cal = rep.calibration
    assert set(cal["method"]) == {"ClusteredEventConformal", "HSCP"}
    assert cal["coverage"].between(0.5, 1.0).all() and (cal["width"] > 0).all()
    assert np.isfinite(cal.loc[cal["method"] == "HSCP", "joint_band"]).all()

    text = rep.summary()
    assert "SMOKE-TEST FIXTURE" in text and "leakage audit PASSED" in text and "Paper references not shown" in text
    tidy = rep.to_dataframe()
    assert {"scheme", "metric", "method", "nominal", "value", "ci_lo", "ci_hi"} <= set(tidy.columns)
    assert {"rmse", "coverage", "width", "joint_band_coverage"} <= set(tidy["metric"])

    out = rep.save(tmp_path / "rep")
    for f in ("report.json", "accuracy.csv", "calibration.csv", "folds.csv", "audit.csv", "summary.txt"):
        assert (out / f).exists()
    payload = json.loads((out / "report.json").read_text())
    assert payload["model"] == "LinearGMM" and payload["meta"]["n_boot"] == 100
    back = gb.BenchmarkReport.load(out)
    assert np.allclose(back.accuracy["rmse"], rep.accuracy["rmse"])


def test_evaluate_is_deterministic(patched):
    a = gb.evaluate(gb.LinearGMM(), schemes=["event_grouped"], n_boot=50)
    b = gb.evaluate(gb.LinearGMM(), schemes=["event_grouped"], n_boot=50)
    pd.testing.assert_frame_equal(a.accuracy, b.accuracy)
    pd.testing.assert_frame_equal(a.calibration, b.calibration)


def test_model_sees_train_fit_rows_only(patched, bench):
    Spy.seen = []
    gb.evaluate(Spy(), schemes=["event_grouped"], calibrate=False, n_boot=0)
    rsn = bench.ds.df["rsn"].to_numpy()
    folds = bench.splits["event_grouped"]
    assert len(Spy.seen) == len(folds)
    for s, f in zip(Spy.seen, folds):
        assert s["n_y"] == s["n_x"] == f.train_fit.size
        assert set(s["rsn"]) == set(rsn[f.train_fit])
        assert not set(s["rsn"]) & set(rsn[f.test]) and not set(s["rsn"]) & set(rsn[f.cal])


def test_template_model_is_not_mutated(patched):
    m = gb.LinearGMM()
    gb.evaluate(m, schemes=["event_grouped"], calibrate=False, n_boot=0)
    assert not hasattr(m, "coef_")


def test_sigma_fallbacks(patched):
    rep = gb.evaluate(MuOnly(), schemes=["event_grouped"], n_boot=0)
    assert rep.meta["sigma_source"] == ["train_residual"] and len(rep.calibration) == 2
    rep = gb.evaluate(FixedSigma(), schemes=["event_grouped"], n_boot=0)
    assert rep.meta["sigma_source"] == ["fallback"]


def test_leakage_error_stops_before_fitting(monkeypatch, bench):
    f = bench.splits["event_grouped"][0]
    bad = api.SplitFold(f.scheme, f.fold, f.train_fit, f.cal, np.concatenate([f.test, f.train_fit[:3]]),
                        f.excluded, f.group_key)
    leaky = api._Bench(bench.ds, {"event_grouped": [bad]}, "fixture", None)
    monkeypatch.setattr(api, "load_benchmark", lambda profile="full", data_dir=None: leaky)
    Spy.seen = []
    with pytest.raises(gb.LeakageError, match="leakage audit"):
        gb.evaluate(Spy(), schemes=["event_grouped"], n_boot=0)
    assert Spy.seen == []


# ------------------------------------------------------------------ iter_folds and evaluate_predictions
def _external_predictions(scheme: str = "event_grouped") -> pd.DataFrame:
    frames = []
    for f in gb.iter_folds("full", [scheme]):
        assert not hasattr(f, "y_test") and not hasattr(f, "y_cal")     # targets stay inside the benchmark
        m = gb.LinearGMM().fit(f.X_train, f.y_train, f.meta_train)
        (mc, sc), (mt, st) = m.predict(f.X_cal, f.meta_cal), m.predict(f.X_test, f.meta_test)
        frames.append(f.predictions_frame(mc, sc, mt, st))
    return pd.concat(frames, ignore_index=True)


def test_evaluate_predictions_matches_evaluate(patched):
    df = _external_predictions()
    assert {"rsn", "im", "mu", "sigma", "scheme", "fold"} == set(df.columns)
    ext = gb.evaluate_predictions(df, "full", n_boot=100)
    own = gb.evaluate(gb.LinearGMM(), schemes=["event_grouped"], n_boot=100)
    cols = ["rmse", "rmse_lo", "rmse_hi", "gauss_cov90"]
    assert np.allclose(ext.accuracy[cols].to_numpy(dtype=float), own.accuracy[cols].to_numpy(dtype=float))
    assert np.allclose(ext.calibration["coverage"], own.calibration["coverage"])
    assert ext.meta["mode"] == "per-fold" and "not verified" in ext.meta["provenance"]


def test_evaluate_predictions_out_of_fold_mode(patched):
    df = _external_predictions()
    tests = {f.fold: set(f.rsn_test) for f in gb.iter_folds("full", ["event_grouped"])}
    oof = df[[r in tests[fo] for r, fo in zip(df["rsn"], df["fold"])]].drop(columns=["scheme", "fold"])
    rep = gb.evaluate_predictions(oof, "full", n_boot=0)
    own = gb.evaluate(gb.LinearGMM(), schemes=["event_grouped"], calibrate=False, n_boot=0)
    assert np.isclose(rep.accuracy["rmse"].iloc[0], own.accuracy["rmse"].iloc[0])
    assert rep.meta["mode"].startswith("out-of-fold") and len(rep.calibration) == 2


def test_evaluate_predictions_without_sigma(patched):
    df = _external_predictions().drop(columns="sigma")
    rep = gb.evaluate_predictions(df, "full", n_boot=0, default_sigma=0.65)
    assert rep.meta["sigma_source"] == "default (0.65)" and rep.calibration["coverage"].between(0.5, 1).all()


def test_evaluate_predictions_validation(patched):
    df = _external_predictions()
    with pytest.raises(ValueError, match="needs columns"):
        gb.evaluate_predictions(df.drop(columns="mu"))
    with pytest.raises(ValueError, match="unknown IM"):
        gb.evaluate_predictions(df.assign(im="SA_99s"))
    with pytest.raises(ValueError, match="duplicate"):
        gb.evaluate_predictions(pd.concat([df, df.iloc[:5]]))
    with pytest.raises(ValueError, match="not in the"):
        gb.evaluate_predictions(df.assign(rsn=df["rsn"] + 10**7))
    with pytest.raises(ValueError, match="positive"):
        gb.evaluate_predictions(df.assign(sigma=-1.0))
    with pytest.raises(ValueError, match="exactly one scheme"):
        gb.evaluate_predictions(df.drop(columns=["scheme", "fold"]), schemes=["event_grouped", "random"])


def test_evaluate_predictions_requires_real_profile(tmp_path):
    df = pd.DataFrame({"rsn": [1], "im": ["PGA"], "mu": [0.0], "sigma": [0.6], "scheme": ["event_grouped"],
                       "fold": ["fold0"]})
    with pytest.raises(gb.DataNotFoundError):
        gb.evaluate_predictions(df, data_dir=tmp_path)


# ------------------------------------------------------------------ references
def test_reference_bundle_in_sync_with_repository():
    paper = ROOT / "results" / "paper" / "paper_summary_numbers.json"
    if not paper.exists():
        pytest.skip("results/paper not present")
    fresh = api.build_reference_bundle(paper, ROOT / "data" / "processed")
    bundled = json.loads(api.REFERENCE_BUNDLE.read_text(encoding="utf-8"))
    assert fresh["accuracy"] == bundled["accuracy"] and fresh["conformal"] == bundled["conformal"]
    assert fresh["rsn_sha256"] == bundled["rsn_sha256"]
    assert bundled["accuracy"]["event_grouped"]["ASK14_site"]["rmse"] == pytest.approx(0.7224, abs=5e-4)
    assert bundled["accuracy"]["event_grouped"]["Hybrid_BSSA14_XGB"]["rmse"] == pytest.approx(0.7220, abs=5e-4)


def test_load_reference_falls_back_to_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = api.load_reference()
    assert "full" in ref["rsn_sha256"] and "event_grouped" in ref["accuracy"]


# ------------------------------------------------------------------ CLI smoke test
def test_selftest_runs_labelled_smoke_test():
    res = CliRunner().invoke(app, ["selftest"])
    assert res.exit_code == 0, res.stdout
    assert "INSTALLATION SMOKE TEST" in res.stdout and "NOT NGA-West2" in res.stdout
    for line in ("PASS  leakage audit", "PASS  conformal", "selftest passed"):
        assert line in res.stdout


def test_selftest_no_smoke_skips_fixture():
    res = CliRunner().invoke(app, ["selftest", "--no-smoke"])
    assert res.exit_code == 0 and "SMOKE TEST" not in res.stdout and "selftest passed" in res.stdout


def test_fixture_is_marked_and_rejected_by_public_loader():
    ds = _smoke.make_fixture(25)
    assert api._is_fixture(ds)
    assert (ds.df["data_source"].str.contains("not NGA-West2")).all()


# ------------------------------------------------------------------ real NGA-West2 (local only)
@real
def test_real_evaluate_event_split():
    rep = gb.evaluate(gb.LinearGMM(), "full", ["event_grouped"], n_boot=50, data_dir=REAL_DIR)
    assert rep.meta["comparable"] and rep.dataset.startswith("NGA-West2")
    row = rep.accuracy.iloc[0]
    assert row["n_test_records"] == 20926 and row["n_test_events"] == 575
    assert row["ref_ASK14_site"] == pytest.approx(0.7224, abs=5e-4)
    assert 0.6 < row["rmse"] < 1.2 and rep.audit["passed"].all()


@real
def test_real_evaluate_predictions_matches_evaluate():
    frames = []
    for f in gb.iter_folds("full", ["event_grouped"], data_dir=REAL_DIR):
        m = gb.LinearGMM().fit(f.X_train, f.y_train, f.meta_train)
        frames.append(f.predictions_frame(*m.predict(f.X_cal, f.meta_cal), *m.predict(f.X_test, f.meta_test)))
    ext = gb.evaluate_predictions(pd.concat(frames), "full", data_dir=REAL_DIR, n_boot=0)
    own = gb.evaluate(gb.LinearGMM(), "full", ["event_grouped"], data_dir=REAL_DIR, n_boot=0)
    assert np.isclose(ext.accuracy["rmse"].iloc[0], own.accuracy["rmse"].iloc[0])
