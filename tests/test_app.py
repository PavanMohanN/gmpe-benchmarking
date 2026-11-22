"""Headless tests of the optional app: data loaders, packaged assets, Tab 3 validation and audit, CLI.

No browser and no PEER data are needed. The Streamlit page itself is exercised with Streamlit's
``AppTest`` only when Streamlit is installed (``pip install -e ".[app]"``).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gmpe_bench.app import data as D
from gmpe_bench.app import validation as V
from gmpe_bench.cli import app as cli_app
from gmpe_bench.schema import IM_NAMES

ROOT = Path(__file__).resolve().parents[1]
HAVE_RESULTS = (ROOT / D.PAPER_JSON).exists()
HAVE_METRICS = all((ROOT / "results" / d / f).exists() for d, f in (
    ("baselines", "baseline_metrics.csv"), ("ml", "ml_metrics.csv"), ("conformal", "conformal_metrics.csv")))
HAVE_SPLITS = (ROOT / "data" / "processed" / "splits_full.json").exists()
HAVE_REAL = (ROOT / "data" / "processed" / "full.parquet").exists() and HAVE_SPLITS
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="results/paper not present")
needs_splits = pytest.mark.skipif(not HAVE_SPLITS, reason="frozen splits not present")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ packaged assets
def test_packaged_assets_exist():
    assert (D.ASSETS / "paper_summary_numbers.json").exists()
    for name in D.TABLE_NAMES:
        assert (D.ASSETS / "tables" / f"{name}.csv").exists(), name
    for f in D.PERIOD_FILES:
        assert (D.ASSETS / f).exists()


@needs_results
def test_packaged_copies_identical_to_paper_outputs():
    assert _sha(D.ASSETS / "paper_summary_numbers.json") == _sha(ROOT / D.PAPER_JSON)
    for name in D.TABLE_NAMES:
        assert _sha(D.ASSETS / "tables" / f"{name}.csv") == _sha(ROOT / D.PAPER_TABLES / f"{name}.csv"), name


@pytest.mark.skipif(not HAVE_METRICS, reason="per-IM metric files not present")
def test_period_extract_matches_repository_metrics():
    acc, cal = D.build_period_tables(ROOT)
    pa, pc = pd.read_csv(D.ASSETS / D.PERIOD_FILES[0]), pd.read_csv(D.ASSETS / D.PERIOD_FILES[1])
    pd.testing.assert_frame_equal(acc.reset_index(drop=True), pa, check_dtype=False, atol=1e-6)
    pd.testing.assert_frame_equal(cal.reset_index(drop=True), pc, check_dtype=False, atol=1e-6)


@needs_results
def test_period_extract_reproduces_paper_rmse():
    pa = pd.read_csv(D.ASSETS / D.PERIOD_FILES[0])
    summary = json.loads((ROOT / D.PAPER_JSON).read_text())
    for scheme in D.SCHEMES:
        for model in ("ASK14_cal", "ASK14_site", "Hybrid_BSSA14_XGB", "XGBoost"):
            got = pa[(pa["scheme"] == scheme) & (pa["model"] == model)]["rmse"].mean()
            assert got == pytest.approx(summary["accuracy"][scheme][model]["rmse"], abs=1e-4), (scheme, model)


# ------------------------------------------------------------------ loaders
@pytest.fixture(scope="module")
def packaged():
    return D.load_paper_data(prefer_packaged=True)


def test_load_packaged_without_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(D, "repo_roots", lambda: [tmp_path])
    d = D.load_paper_data()
    assert d.source.startswith("packaged") and set(D.TABLE_NAMES) <= set(d.tables)


@needs_results
def test_load_from_repository():
    d = D.load_paper_data(ROOT)
    assert d.source.startswith("repository")


def test_leakage_view(packaged):
    leak = packaged.leakage().set_index("model")
    assert leak.loc["XGBoost", "inflation_pct"] == pytest.approx(37.09, abs=0.01)
    assert leak.loc["Hybrid_BSSA14_XGB", "inflation_pct"] == pytest.approx(35.96, abs=0.01)
    published = leak[leak["family"] == "Published NGA-West2"]["inflation_pct"]
    assert len(published) == 4 and published.between(-0.6, -0.3).all()


def test_accuracy_view(packaged):
    acc = packaged.accuracy()
    assert set(acc["scheme"]) == set(D.SCHEMES)
    evt = acc[acc["scheme"] == "event_grouped"].set_index("model")
    assert evt.loc["ASK14_site", "rmse"] == pytest.approx(0.7224, abs=1e-4)
    assert evt.loc["Hybrid_BSSA14_XGB", "rmse"] == pytest.approx(0.7220, abs=1e-4)
    assert (acc["lo"] <= acc["rmse"]).all() and (acc["rmse"] <= acc["hi"]).all()


def test_differences_view(packaged):
    diff = packaged.differences()
    row = diff[(diff["scheme"] == "event_grouped") & (diff["model_a"] == "Hybrid_BSSA14_XGB")
               & (diff["model_b"] == "ASK14_site")].iloc[0]
    assert row["lo"] < 0 < row["hi"] and not row["resolved"]


def test_calibration_and_mace_views(packaged):
    cal = packaged.calibration("0.90")
    hscp = cal[(cal["method"] == "HSCP") & (cal["scheme"] == "event_grouped")].iloc[0]
    assert hscp["coverage"] == pytest.approx(0.901, abs=1e-3)
    assert hscp["joint_band"] == pytest.approx(0.892, abs=1e-3) and hscp["joint_marginal"] < 0.7
    mace = packaged.mace("0.90").set_index("method")["mace"]
    assert mace.idxmin() == "ClusteredEventConformal" and mace["ClusteredEventConformal"] == pytest.approx(0.044, abs=1e-3)
    assert set(packaged.calibration("0.95")["scheme"]) >= set(D.SCHEMES)


def test_period_views(packaged):
    pa = packaged.period_accuracy
    assert set(IM_NAMES) == set(pa["im"])
    assert pa.loc[pa["model"].str.endswith("_site"), "tau"].isna().all()     # not defined for (site) models
    sa = D.period_axis(pa)
    assert sa["period"].notna().all() and not sa["im"].isin(["PGA", "PGV"]).any()
    pc = packaged.period_calibration
    assert {0.9, 0.95} == set(pc["nominal"].round(2)) and pc["coverage"].between(0, 1).all()


def test_labels():
    assert D.label("ASK14_site") == "ASK14 (site)" and D.label("HSCP") == "HSCP"
    assert D.label("magnitude_holdout") == "Mw ≥ 7" and D.label("unknown") == "unknown"


def test_local_profile_detection(tmp_path, monkeypatch):
    assert not D.local_profile_available("full", tmp_path)
    (tmp_path / "full.parquet").write_bytes(b"x")
    (tmp_path / "splits_full.json").write_text("{}")
    assert D.local_profile_available("full", tmp_path)
    monkeypatch.setenv("GMPE_BENCH_DATA", str(tmp_path))
    assert D.data_dir() == tmp_path


# ------------------------------------------------------------------ Tab 3: schema audit
@pytest.mark.parametrize("raw, want", [("PGA", "PGA"), ("pga", "PGA"), ("SA_1.0s", "SA_1s"), ("SA(0.2)", "SA_0.2s"),
                                       ("0.75", "SA_0.75s"), ("sa_0.01s", "SA_0.01s"), ("SA_9s", None), ("PGD", None)])
def test_normalize_im(raw, want):
    assert V.normalize_im(raw) == want


def _frame(rsn=(1, 2), sigma=0.6, **extra) -> pd.DataFrame:
    rows = [{"rsn": r, "im": im, "mu": -2.0, "sigma": sigma, **extra} for r in rsn for im in IM_NAMES]
    return pd.DataFrame(rows)


def test_schema_only_without_splits():
    res = V.audit_predictions(_frame(), None)
    assert res.ok and not res.split_audit
    assert res.table().set_index("name").loc["split audit", "status"] == "SKIP"


@pytest.mark.parametrize("mutate, check", [
    (lambda d: d.drop(columns="mu"), "columns"),
    (lambda d: d.assign(rsn="abc"), "rsn values"),
    (lambda d: d.assign(im="SA_99s"), "IM names"),
    (lambda d: d.assign(mu=np.nan), "mu"),
    (lambda d: d.assign(sigma=-1.0), "sigma"),
    (lambda d: pd.concat([d, d.iloc[:3]]), "duplicates"),
])
def test_schema_failures(mutate, check):
    res = V.audit_predictions(mutate(_frame()), None)
    assert not res.ok
    assert any(c.name == check and c.status == "FAIL" for c in res.checks)


def test_schema_warnings():
    res = V.audit_predictions(_frame().drop(columns="sigma").assign(extra=1), None)
    assert res.ok
    names = {c.name for c in res.checks if c.status == "WARN"}
    assert {"sigma", "columns"} <= names
    lin = V.audit_predictions(_frame().assign(mu=0.3), None)
    assert any(c.name == "units" and c.status == "WARN" for c in lin.checks)


def test_read_csv_normalizes_columns():
    df = V.read_predictions_csv(b" RSN ,IM,Mu,Sigma\n1,PGA,-2,0.6\n")
    assert list(df.columns) == ["rsn", "im", "mu", "sigma"]


# ------------------------------------------------------------------ Tab 3: split audit (synthetic payload)
def _payload() -> dict:
    """Two-fold scheme over 30 records with a correct fingerprint."""
    rsn = np.arange(1, 31)
    folds = [{"fold": "fold0", "group_key": "eqid", "train_fit": list(range(11, 26)), "cal": list(range(26, 31)),
              "test": list(range(1, 11)), "excluded": []},
             {"fold": "fold1", "group_key": "eqid", "train_fit": list(range(1, 16)), "cal": list(range(16, 21)),
              "test": list(range(21, 31)), "excluded": []}]
    return {"profile": "full", "n_records": 30, "rsn_sha256": V.rsn_fingerprint(rsn),
            "schemes": {"event_grouped": folds}}


def test_split_payload_verification():
    p = _payload()
    assert all(c.status == "PASS" for c in V.verify_splits_payload(p))
    bad = json.loads(json.dumps(p))
    bad["rsn_sha256"] = "0" * 64
    assert V.verify_splits_payload(bad)[0].status == "FAIL"
    bad = json.loads(json.dumps(p))
    bad["schemes"]["event_grouped"][0]["test"].append(11)          # overlaps train_fit
    assert V.verify_splits_payload(bad)[1].status == "FAIL"


def test_per_fold_audit_complete_upload():
    p = _payload()
    rows = []
    for f in p["schemes"]["event_grouped"]:
        rows.append(_frame(f["cal"] + f["test"], scheme="event_grouped", fold=f["fold"]))
    res = V.audit_predictions(pd.concat(rows), p)
    assert res.ok and res.mode == "per-fold" and res.split_audit
    assert (res.folds["test_coverage"] == 1).all() and (res.folds["cal_coverage"] == 1).all()
    assert res.folds["ims_per_test_record"].eq(27).all() and not res.warnings


def test_per_fold_audit_flags_train_rows_and_partial_coverage():
    p = _payload()
    df = _frame([1, 2, 3, 11, 12], scheme="event_grouped", fold="fold0")     # 11, 12 are train-fit rows
    res = V.audit_predictions(df, p)
    assert res.ok
    st = res.table().set_index("name")["status"]
    assert st["train-fit rows"] == "WARN" and st["test coverage"] == "WARN" and st["calibration rows"] == "WARN"
    assert int(res.folds["train_fit_rows"].iloc[0]) == 2


def test_per_fold_audit_rejects_unknown_labels_and_rsn():
    p = _payload()
    res = V.audit_predictions(_frame([1], scheme="event_grouped", fold="fold9"), p)
    assert any(c.name == "scheme/fold labels" and c.status == "FAIL" for c in res.checks)
    res = V.audit_predictions(_frame([1, 999], scheme="event_grouped", fold="fold0"), p)
    assert any(c.name == "RSNs in profile" and c.status == "FAIL" for c in res.checks)


def test_out_of_fold_audit():
    p = _payload()
    res = V.audit_predictions(_frame(range(1, 31)), p, "event_grouped")
    assert res.ok and res.mode == "out-of-fold" and (res.folds["test_coverage"] == 1).all()
    res = V.audit_predictions(_frame([1, 2]), p, "nope")
    assert not res.ok


# ------------------------------------------------------------------ Tab 3: real frozen splits (shipped in repo)
@needs_splits
def test_real_splits_fingerprint_and_template():
    p = V.load_splits_payload("full", ROOT / "data" / "processed")
    assert all(c.status == "PASS" for c in V.verify_splits_payload(p))
    tpl = V.sample_template(p)
    assert len(tpl) == 10 * len(IM_NAMES)
    res = V.audit_predictions(V.read_predictions_csv(io.BytesIO(tpl.to_csv(index=False).encode())), p)
    assert res.ok and res.split_audit


@needs_splits
def test_fold_skeleton():
    p = V.load_splits_payload("full", ROOT / "data" / "processed")
    fold = p["schemes"]["event_grouped"][0]
    sk = V.fold_skeleton(p, "event_grouped", fold["fold"])
    assert len(sk) == (len(fold["cal"]) + len(fold["test"])) * len(IM_NAMES)
    assert set(sk["part"]) == {"cal", "test"} and sk["mu"].isna().all()


def test_template_without_splits_and_snippet():
    tpl = V.sample_template(None)
    assert V.audit_predictions(tpl, None).ok
    code = V.local_snippet("my.csv", "full", "event_grouped")
    assert "gb.evaluate_predictions" in code and "my.csv" in code and "gmpe-bench build" in code
    compile(code, "<snippet>", "exec")


def test_local_evaluation_requires_curated_profile(tmp_path):
    import gmpe_bench as gb

    res = V.audit_predictions(_frame(), None)
    with pytest.raises(gb.DataNotFoundError):
        V.run_local_evaluation(res, tmp_path, n_boot=0)


@pytest.mark.skipif(not HAVE_REAL, reason="curated NGA-West2 profile not built")
def test_local_evaluation_matches_api():
    import gmpe_bench as gb

    frames = []
    for f in gb.iter_folds("full", ["event_grouped"], data_dir=ROOT / "data" / "processed"):
        m = gb.LinearGMM().fit(f.X_train, f.y_train, f.meta_train)
        frames.append(f.predictions_frame(*m.predict(f.X_cal, f.meta_cal), *m.predict(f.X_test, f.meta_test)))
    df = V.read_predictions_csv(io.BytesIO(pd.concat(frames).to_csv(index=False).encode()))
    p = V.load_splits_payload("full", ROOT / "data" / "processed")
    res = V.audit_predictions(df, p)
    assert res.ok and not res.warnings
    rep = V.run_local_evaluation(res, ROOT / "data" / "processed", n_boot=0)
    own = gb.evaluate(gb.LinearGMM(), "full", ["event_grouped"], data_dir=ROOT / "data" / "processed", n_boot=0)
    assert rep.accuracy["rmse"].iloc[0] == pytest.approx(own.accuracy["rmse"].iloc[0], abs=1e-6)


# ------------------------------------------------------------------ CLI
def test_app_command_without_streamlit(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "streamlit" else real(n, *a))
    res = CliRunner().invoke(cli_app, ["app"])
    assert res.exit_code == 1 and 'pip install "gmpe-bench[app]"' in res.output


def test_app_command_builds_streamlit_call(monkeypatch, tmp_path):
    import importlib.util
    import subprocess

    calls = {}
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: object() if n == "streamlit" else real(n, *a))
    monkeypatch.setattr(subprocess, "call", lambda cmd, env=None: calls.update(cmd=cmd, env=env) or 0)
    res = CliRunner().invoke(cli_app, ["app", "--port", "9999", "--headless", "--data", str(tmp_path)])
    assert res.exit_code == 0, res.output
    cmd = calls["cmd"]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("streamlit_app.py") and Path(cmd[4]).exists()
    assert "9999" in cmd and "--server.headless" in cmd
    assert calls["env"]["GMPE_BENCH_DATA"] == str(tmp_path.resolve())


# ------------------------------------------------------------------ Streamlit page (optional)
def test_streamlit_page_renders_headless():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src" / "gmpe_bench" / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Benchmark explorer", "Period & spectral bands", "Evaluate your predictions"]
    at.multiselect(key="t1_mod").set_value(["ASK14_cal", "XGBoost"]).run()
    at.radio(key="t1_lvl").set_value("0.95").run()
    at.selectbox(key="t2_reg").set_value("spatial_block").run()
    at.selectbox(key="t2_met").set_value("tau").run()
    assert not at.exception, [e.value for e in at.exception]


def test_root_entrypoint_renders_headless():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 3
