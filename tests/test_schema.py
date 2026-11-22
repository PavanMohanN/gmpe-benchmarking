"""Phase 1: column mapping and schema validation."""

from __future__ import annotations

import pandas as pd
import pytest

from gmpe_bench import schema


def _raw(component: str, n: int = 3) -> pd.DataFrame:
    """Minimal raw flatfile with real NGA-West2 header spellings."""
    d: dict[str, object] = {
        "Record Sequence Number": list(range(1, n + 1)),
        "EQID": [10] * n,
        "Station Sequence Number": list(range(100, 100 + n)),
        "Earthquake Magnitude": [6.0] * n,
        "Joyner-Boore Dist. (km)": [10.0] * n,
        "ClstD (km)": [12.0] * n,
        "Vs30 (m/s) selected for analysis": [400.0] * n,
        "TYPE\n(CRjb = 5)": [0] * n,
        "PGA (g)": [0.1] * n,
    }
    if component == "H":
        d["Lowest Usable Freq - Ave. Component (Hz)"] = [0.2] * n
        d["pgv"] = [5.0] * n
    else:
        d["Lowest Usable Freq - V (Hz)"] = [0.3] * n
        d["PGV (cm/sec)"] = [3.0] * n
    for p in schema.PERIODS:
        d[f"T{p:.3f}S"] = [0.05] * n
    d["T20.000S"] = [0.001] * n  # outside the configured set: must be ignored
    return pd.DataFrame(d)


def test_sa_name_round_trip() -> None:
    assert schema.sa_name(0.01) == "SA_0.01s"
    assert schema.sa_name(10.0) == "SA_10s"
    assert schema.sa_name(0.075) == "SA_0.075s"
    assert schema.period_of("SA_0.075s") == pytest.approx(0.075)
    assert schema.period_of("PGA") is None


def test_periods_follow_im_names() -> None:
    assert schema.IM_NAMES[:2] == ["PGA", "PGV"]
    assert len(schema.PERIODS) == len(schema.IM_NAMES) - 2


@pytest.mark.parametrize("component", ["H", "V"])
def test_all_ims_mapped(component: str) -> None:
    out = schema.normalise_columns(_raw(component), component)
    for col in schema.im_columns(component):
        assert col in out.columns
    assert f"{component}_SA_20s" not in out.columns


def test_whitespace_in_headers_is_collapsed() -> None:
    out = schema.normalise_columns(_raw("H"), "H")
    assert "cl_type_5" in out.columns


def test_component_suffixes_and_pgv_spellings() -> None:
    h = schema.normalise_columns(_raw("H"), "H")
    v = schema.normalise_columns(_raw("V"), "V")
    assert "luf_h" in h.columns and "luf_v" in v.columns
    assert h["H_PGV"].iloc[0] == 5.0
    assert v["V_PGV"].iloc[0] == 3.0


def test_unmapped_columns_dropped() -> None:
    raw = _raw("H")
    raw["Some Unused Column"] = 1
    out = schema.normalise_columns(raw, "H")
    assert "Some Unused Column" not in out.columns


def test_validate_passes() -> None:
    schema.validate_schema(schema.normalise_columns(_raw("H"), "H"), "H")
    schema.validate_schema(schema.normalise_columns(_raw("V"), "V"), "V")


def test_validate_reports_missing() -> None:
    raw = _raw("H").drop(columns=["Joyner-Boore Dist. (km)"])
    with pytest.raises(schema.SchemaError, match="rjb"):
        schema.validate_schema(schema.normalise_columns(raw, "H"), "H")


def test_validate_rejects_duplicate_rsn() -> None:
    raw = _raw("H")
    raw.loc[1, "Record Sequence Number"] = 1
    with pytest.raises(schema.SchemaError, match="duplicate"):
        schema.validate_schema(schema.normalise_columns(raw, "H"), "H")


def test_bad_component() -> None:
    with pytest.raises(ValueError):
        schema.normalise_columns(_raw("H"), "X")
