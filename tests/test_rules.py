"""Phase 1: rules R2-R11 on small synthetic flatfiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gmpe_bench import schema
from gmpe_bench.data import build, features, rules
from gmpe_bench.data.dataset import GMDataset

N_EVENTS, PER_EVENT = 3, 6
N = N_EVENTS * PER_EVENT


def _raw(component: str) -> pd.DataFrame:
    """Three events x six records, real header spellings, all values usable."""
    rng = np.random.default_rng(0)
    d: dict[str, object] = {
        "Record Sequence Number": np.arange(1, N + 1),
        "EQID": np.repeat([1, 2, 3], PER_EVENT),
        "Earthquake Name": np.repeat(["A_ California", "B_ Japan", "C_ Italy"], PER_EVENT),
        "Station Sequence Number": np.arange(100, 100 + N),
        "Earthquake Magnitude": np.repeat([5.0, 6.0, 7.0], PER_EVENT),
        "Rake Angle (deg)": np.repeat([0.0, 90.0, 180.0], PER_EVENT),
        "Mechanism Based on Rake Angle": np.repeat([0, 2, 0], PER_EVENT),
        "Hypocenter Depth (km)": 10.0,
        "Depth to Top Of Fault Rupture Model": 2.0,
        "Joyner-Boore Dist. (km)": rng.uniform(1, 100, N),
        "ClstD (km)": rng.uniform(2, 120, N),
        "Vs30 (m/s) selected for analysis": 400.0,
        "Measured/Inferred Class": 0,
        "TYPE\n(CRjb = 5)": 0,
        "CRjb": -999,
        "Late P-trigger": 0,
        "Late S-trigger": 0,
        "Spectra Quality Flag": 0,
        "PGA (g)": 0.2,
    }
    if component == "H":
        d.update({
            "HP-H1 (Hz)": 0.08, "HP-H2 (Hz)": 0.08,
            "Lowest Usable Freq - Ave. Component (Hz)": 0.1,  # T_max = 10 s
            "RotD percentile": 50, "pgv": 10.0,
        })
        sa = 0.1
    else:
        d.update({"HP-V (Hz)": 0.08, "Lowest Usable Freq - V (Hz)": 0.1, "PGV (cm/sec)": 5.0})
        sa = 0.05
    for p in schema.PERIODS:
        d[f"T{p:.3f}S"] = sa
    return pd.DataFrame(d)


def _canon(h: pd.DataFrame | None = None, v: pd.DataFrame | None = None) -> pd.DataFrame:
    h = schema.normalise_columns(_raw("H") if h is None else h, "H")
    v = schema.normalise_columns(_raw("V") if v is None else v, "V")
    h, _ = rules.replace_sentinels(h)
    v, _ = rules.replace_sentinels(v)
    df, _ = rules.join_hv(h, v)
    df, _ = rules.lock_keys(df)
    return df


# --------------------------------------------------------------------- R2
def test_r2_sentinels_and_blanks() -> None:
    df = pd.DataFrame({"a": [1.0, -999.0, 3.0], "b": ["x", "", "-999"]})
    out, step = rules.replace_sentinels(df)
    assert out["a"].isna().tolist() == [False, True, False]
    assert out["b"].isna().tolist() == [False, True, True]
    assert step.removed == 0 and "3 cells" in step.note


# --------------------------------------------------------------------- R3/R4
def test_r3_join_brings_v_columns() -> None:
    df = _canon()
    assert len(df) == N
    assert {"luf_v", "V_PGA", "H_PGA"} <= set(df.columns)


def test_r4_drops_non_rotd50() -> None:
    h = _raw("H")
    h.loc[0, "RotD percentile"] = -999
    df = _canon(h=h)
    assert len(df) == N - 1
    assert str(df["eqid"].dtype) == "Int64"


# --------------------------------------------------------------------- R5
def test_r5_masks_beyond_tmax_and_flags_anomaly() -> None:
    h = _raw("H")
    h.loc[0, "Lowest Usable Freq - Ave. Component (Hz)"] = 0.5  # T_max = 2 s
    h.loc[1, "Lowest Usable Freq - Ave. Component (Hz)"] = 0.08  # = corner -> anomaly
    df, step = rules.mask_unusable_periods(_canon(h=h))
    row = df[df["rsn"] == 1].iloc[0]
    for p in schema.PERIODS:
        col = schema.im_column("H", schema.sa_name(p))
        assert pd.isna(row[col]) == (p > 2.0), p
    assert row["tmax_h"] == pytest.approx(2.0)
    assert bool(df.loc[df["rsn"] == 2, "luf_factor_anomaly_h"].iloc[0])
    assert int(df["luf_factor_anomaly_h"].sum()) == 1
    assert step.removed == 0


def test_r5_missing_luf_keeps_values() -> None:
    h = _raw("H")
    h.loc[0, "Lowest Usable Freq - Ave. Component (Hz)"] = -999
    df, _ = rules.mask_unusable_periods(_canon(h=h))
    assert df.loc[0, schema.im_columns("H")].notna().all()


# --------------------------------------------------------------------- R6
def test_r6_drops_late_p_and_masks_v() -> None:
    h, v = _raw("H"), _raw("V")
    h.loc[0, "Late P-trigger"] = 1
    h.loc[1, "Spectra Quality Flag"] = 1
    h.loc[2, "Late S-trigger"] = 1
    v.loc[3, "Late P-trigger"] = 1
    df, step = rules.apply_quality_flags(_canon(h=h, v=v))
    assert step.removed == 2
    assert bool(df.loc[df["rsn"] == 3, "late_s"].iloc[0])
    v_row = df.loc[df["rsn"] == 4, schema.im_columns("V")]
    assert v_row.isna().all(axis=None)


def test_r6_late_p_string_codes() -> None:
    """2015 flatfiles: 'Y' (any case) is a late trigger; 'D', '?', 'PT' codes are not."""
    h, v = _raw("H"), _raw("V")
    h["Late P-trigger"] = h["Late P-trigger"].astype(object)
    h["Late S-trigger"] = h["Late S-trigger"].astype(object)
    h.loc[0, "Late P-trigger"] = "Y"
    h.loc[1, "Late P-trigger"] = " y "
    h.loc[2, "Late P-trigger"] = "D"
    h.loc[3, "Late P-trigger"] = "?"
    h.loc[4, "Late P-trigger"] = "D PT"
    h.loc[5, "Late S-trigger"] = "Y"
    h.loc[6, "Late S-trigger"] = "D"
    v["Late P-trigger"] = v["Late P-trigger"].astype(object)
    v.loc[7, "Late P-trigger"] = "Y"
    df, step = rules.apply_quality_flags(_canon(h=h, v=v))
    assert step.removed == 2
    assert set(df["rsn"]) == set(range(3, N + 1))
    flagged = set(df.loc[df["late_s"], "rsn"])
    assert flagged == {6}
    assert df.loc[df["rsn"] == 8, schema.im_columns("V")].isna().all(axis=None)
    assert df.loc[df["rsn"] == 9, schema.im_columns("V")].notna().all(axis=None)


def test_is_yes_handles_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    assert not rules.is_yes(df, "late_p_h").any()


# --------------------------------------------------------------------- R7
def test_r7_drops_missing_keys() -> None:
    h = _raw("H")
    h.loc[0, "Earthquake Magnitude"] = -999
    h.loc[1, "Vs30 (m/s) selected for analysis"] = np.nan
    df, step = rules.drop_missing_keys(_canon(h=h))
    assert step.removed == 2


# --------------------------------------------------------------------- R8
@pytest.mark.parametrize("c1_code,c2_code", [("C1", "C2 - 0137"), ("C1f - 1186", "C2 - 0277"), ("CL1", "CL2")])
def test_r8_class_codes(c1_code: str, c2_code: str) -> None:
    h = _raw("H")
    h["TYPE\n(CRjb = 5)"] = c1_code
    h.loc[0, "TYPE\n(CRjb = 5)"] = c2_code
    df, _ = rules.assign_cl1_cl2(_canon(h=h))
    assert df.loc[0, "cl_class"] == "CL2"
    assert df.loc[1, "cl_class"] == "CL1"


def test_r8_text_codes() -> None:
    h = _raw("H")
    h["TYPE\n(CRjb = 5)"] = "CL1"
    h.loc[0, "TYPE\n(CRjb = 5)"] = "CL2"
    df, step = rules.assign_cl1_cl2(_canon(h=h))
    assert df.loc[0, "cl_class"] == "CL2"
    assert (df.loc[1:, "cl_class"] == "CL1").all()
    assert "TYPE (CRjb = 5) column" in step.note


def test_cl_from_crjb_missing_is_cl1() -> None:
    crjb = pd.Series([np.nan, 2.0, 0.0, 5.0, 40.0])
    assert rules.cl_from_crjb(crjb).tolist() == ["CL1", "CL2", "CL2", "CL1", "CL1"]


def test_r8_crjb_fallback_mainshock_nan_is_cl1() -> None:
    """No TYPE column: CRJB = -999 (mainshock) -> CL1; CRJB < 5 km -> CL2."""
    h = _raw("H").drop(columns=["TYPE\n(CRjb = 5)"])
    h["CRjb"] = -999.0          # mainshocks
    h.loc[0, "CRjb"] = 2.0      # aftershock within 5 km
    h.loc[1, "CRjb"] = 12.0     # aftershock beyond 5 km
    df, step = rules.assign_cl1_cl2(_canon(h=h))
    assert df.loc[0, "cl_class"] == "CL2"
    assert (df.loc[1:, "cl_class"] == "CL1").all()
    assert "CRJB < 5 km" in step.note and "CRJB missing" in step.note


def test_r8_unrecognised_type_falls_back_to_crjb() -> None:
    h = _raw("H")
    h["TYPE\n(CRjb = 5)"] = "X"
    h["CRjb"] = -999.0
    h.loc[0, "CRjb"] = 1.0
    df, step = rules.assign_cl1_cl2(_canon(h=h))
    assert df.loc[0, "cl_class"] == "CL2"
    assert (df.loc[1:, "cl_class"] == "CL1").all()
    assert "not recognised" in step.note


def test_r9d_keeps_unclassified() -> None:
    h = _raw("H")
    hv, v = schema.normalise_columns(h, "H"), schema.normalise_columns(_raw("V"), "V")
    base, _ = build.run_rules(hv, v)
    base.loc[0, "cl_class"] = pd.NA
    out, steps = rules.select_profile_sp23(base)
    r9d = next(s for s in steps if s.rule == "R9d")
    assert r9d.removed == 0


# --------------------------------------------------------------------- R10
def test_r10_vh_only_where_both_usable() -> None:
    h = _raw("H")
    h.loc[0, "Lowest Usable Freq - Ave. Component (Hz)"] = 0.5  # H masked above 2 s
    df, _ = rules.mask_unusable_periods(_canon(h=h))
    df, _ = rules.add_vh_ratios(df)
    p_long = max(schema.PERIODS)
    col = schema.im_column("VH", schema.sa_name(p_long))
    assert pd.isna(df.loc[0, col])
    assert df.loc[1, col] == pytest.approx(np.log(0.05 / 0.1))


# --------------------------------------------------------------------- R11
def test_r11_ztor_cy14_values() -> None:
    mw = pd.Series([7.0, 6.0])
    rev = pd.Series([False, True])
    out = features.ztor_cy14(mw, rev)
    assert out[0] == pytest.approx((2.673 - 1.136 * (7.0 - 4.970)) ** 2)
    assert out[1] == pytest.approx((2.704 - 1.226 * (6.0 - 5.849)) ** 2)


def test_r11_fills_and_flags() -> None:
    h = _raw("H")
    h.loc[0, "Depth to Top Of Fault Rupture Model"] = -999
    df, _ = rules.fill_site_source(_canon(h=h))
    assert bool(df.loc[0, "ztor_imputed"]) and not bool(df.loc[1, "ztor_imputed"])
    assert df.loc[0, "ztor"] == pytest.approx(features.ztor_cy14(pd.Series([5.0]), pd.Series([False]))[0])
    assert df["z1_imputed"].all()  # no basin values in the fixture
    ca = np.exp(-7.15 / 4 * np.log((400.0**4 + 571.0**4) / (1360.0**4 + 571.0**4)))
    assert df.loc[0, "z1"] == pytest.approx(ca)
    jp = df.loc[df["eqid"] == 2, "region_japan"]
    assert jp.all()
    assert df["vs30_measured"].all()


# --------------------------------------------------------------------- R9
def test_r9_sp23_filters() -> None:
    h = _raw("H")
    h.loc[0, "Earthquake Magnitude"] = 2.5      # Mw < 3
    h.loc[1, "ClstD (km)"] = 350.0             # R_rup > 300 km
    h.loc[2, "TYPE\n(CRjb = 5)"] = 1           # CL2
    h.loc[12:, "Depth to Top Of Fault Rupture Model"] = -999  # event 3 loses ZTOR
    hv, v = schema.normalise_columns(h, "H"), schema.normalise_columns(_raw("V"), "V")
    base, _ = build.run_rules(hv, v)
    out, steps = rules.select_profile_sp23(base)
    assert set(out["eqid"]) == {2}               # event 1 falls to 3 records; event 3 has no ZTOR
    assert [s.rule for s in steps][-1] == "R9g"
    assert any("absent" in s.note for s in steps)


# --------------------------------------------------------------------- end to end
def test_build_end_to_end(tmp_path) -> None:
    h_path, v_path = tmp_path / "h.csv", tmp_path / "v.csv"
    _raw("H").to_csv(h_path, index=False)
    _raw("V").to_csv(v_path, index=False)
    res = build.build_dataset(h_path, v_path, tmp_path / "out", profile="both")
    assert res.outputs["full"].exists() and res.outputs["sp23"].exists()
    assert res.manifest_path.exists()
    att = res.attrition
    for _, g in att.groupby("profile"):
        assert (g["n_after"].diff().dropna() <= 0).all()
    ds = GMDataset.from_parquet(res.outputs["full"])
    assert len(ds) == N
    assert ds.Y("H").shape == (N, len(schema.IM_NAMES))
    assert ds.X("A").shape[1] == 7 and ds.X("B").shape[1] == 17
    assert ds.Y("VH").iloc[0, 0] == pytest.approx(0.0)  # PGA 0.2 / 0.2
    assert ds.subset([0, 1]).df["rsn"].tolist() == [1, 2]
    assert ds.describe()["n_events"] == N_EVENTS
