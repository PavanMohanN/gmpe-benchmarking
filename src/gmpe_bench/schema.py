"""Canonical schema for the NGA-West2 horizontal (H) and vertical (V) flatfiles.

Raw flatfile headers are matched after collapsing whitespace, so
``"TYPE\\n(CRjb = 5)"`` and ``"TYPE (CRjb = 5)"`` are the same key.

Canonical layout after :func:`normalise_columns`:

* shared metadata  -> snake_case names (``rsn``, ``eqid``, ``mw`` ...)
* component fields -> snake_case + ``_h`` / ``_v`` (``luf_h``, ``late_p_v`` ...)
* intensity values -> ``<C>_<IM>`` (``H_PGA``, ``V_SA_0.2s`` ...), SA in g, PGV in cm/s
"""

from __future__ import annotations

import re

import pandas as pd

# --------------------------------------------------------------------------- IMs
_DEFAULT_PERIODS: tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3,
    0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0,
)


def sa_name(period: float) -> str:
    """Canonical SA name, e.g. 0.01 -> ``SA_0.01s``, 10.0 -> ``SA_10s``."""
    return f"SA_{period:g}s"


def period_of(im_name: str) -> float | None:
    """Period in seconds for an SA name, ``None`` for PGA/PGV."""
    m = re.fullmatch(r"SA_([0-9]*\.?[0-9]+)s", im_name)
    return float(m.group(1)) if m else None


def _load_im_names() -> list[str]:
    try:
        from gmpe_bench import config

        names = getattr(config, "IM_NAMES", None)
        if names:
            return list(names)
    except ImportError:  # pragma: no cover - package always importable in practice
        pass
    return ["PGA", "PGV"] + [sa_name(p) for p in _DEFAULT_PERIODS]


IM_NAMES: list[str] = _load_im_names()
PERIODS: list[float] = [p for p in (period_of(n) for n in IM_NAMES) if p is not None]
COMPONENTS: tuple[str, ...] = ("H", "V")

# ---------------------------------------------------------------- column maps
META_MAP: dict[str, str] = {
    "Record Sequence Number": "rsn",
    "EQID": "eqid",
    "Earthquake Name": "eq_name",
    "YEAR": "year",
    "MODY": "mody",
    "HRMN": "hrmn",
    "Station Name": "station_name",
    "Station Sequence Number": "ssn",
    "Station ID No.": "station_id",
    "Earthquake Magnitude": "mw",
    "Magnitude Type": "mag_type",
    "Strike (deg)": "strike",
    "Dip (deg)": "dip",
    "Rake Angle (deg)": "rake",
    "Mechanism Based on Rake Angle": "mechanism",
    "Hypocenter Latitude (deg)": "hypo_lat",
    "Hypocenter Longitude (deg)": "hypo_lon",
    "Hypocenter Depth (km)": "hypo_depth",
    "Depth to Top Of Fault Rupture Model": "ztor",
    "Fault Rupture Width (km)": "width",
    "EpiD (km)": "repi",
    "HypD (km)": "rhypo",
    "Joyner-Boore Dist. (km)": "rjb",
    "ClstD (km)": "rrup",
    "Rx": "rx",
    "Ry 2": "ry0",
    "FW/HW Indicator": "fw_hw",
    "Vs30 (m/s) selected for analysis": "vs30",
    "Measured/Inferred Class": "vs30_class",
    "Station Latitude": "sta_lat",
    "Station Longitude": "sta_lon",
    "Northern CA/Southern CA - H11 Z1 (m)": "z1_h11",
    "Northern CA/Southern CA - S4 Z1 (m)": "z1_s4",
    "Northern CA/Southern CA - H11 Z2.5 (m)": "z2p5_h11",
    "Northern CA/Southern CA - S4 Z2.5 (m)": "z2p5_s4",
    "TYPE (CRjb = 5)": "cl_type_5",
    "CRjb": "crjb",
}

# Canonical names here get a ``_h`` / ``_v`` suffix.
COMPONENT_MAP: dict[str, dict[str, str]] = {
    "H": {
        "Lowest Usable Freq - Ave. Component (Hz)": "luf",
        "HP-H1 (Hz)": "hp1",
        "HP-H2 (Hz)": "hp2",
        "Late P-trigger": "late_p",
        "Late S-trigger": "late_s",
        "Spectra Quality Flag": "spectra_qflag",
        "RotD percentile": "rotd_pct",
    },
    "V": {
        "Lowest Usable Freq - V (Hz)": "luf",
        "HP-V (Hz)": "hp1",
        "Late P-trigger": "late_p",
        "Late S-trigger": "late_s",
        "Spectra Quality Flag": "spectra_qflag",
    },
}

PGA_RAW: tuple[str, ...] = ("PGA (g)",)
PGV_RAW: tuple[str, ...] = ("pgv", "PGV (cm/sec)")

REQUIRED_META: tuple[str, ...] = ("rsn", "eqid", "ssn", "mw", "rjb", "rrup", "vs30")


class SchemaError(ValueError):
    """Raised when a flatfile lacks required columns or has duplicate records."""


# ------------------------------------------------------------------- helpers
def _key(name: object) -> str:
    return " ".join(str(name).split())


def _raw_period(name: str) -> float | None:
    m = re.fullmatch(r"T(\d+\.\d+)S", _key(name))
    return float(m.group(1)) if m else None


def im_column(component: str, im: str) -> str:
    """Canonical intensity column, e.g. ``("H", "SA_0.2s") -> "H_SA_0.2s"``."""
    return f"{component}_{im}"


def im_columns(component: str) -> list[str]:
    return [im_column(component, n) for n in IM_NAMES]


def normalise_columns(df: pd.DataFrame, component: str) -> pd.DataFrame:
    """Return a new frame with only canonical columns for one flatfile."""
    component = component.upper()
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")

    by_key = {_key(c): c for c in df.columns}
    out: dict[str, pd.Series] = {}

    for raw, canon in META_MAP.items():
        if raw in by_key:
            out[canon] = df[by_key[raw]]

    suffix = f"_{component.lower()}"
    for raw, canon in COMPONENT_MAP[component].items():
        if raw in by_key:
            out[canon + suffix] = df[by_key[raw]]

    for raw in PGA_RAW:
        if raw in by_key:
            out[im_column(component, "PGA")] = df[by_key[raw]]
    for raw in PGV_RAW:
        if raw in by_key:
            out[im_column(component, "PGV")] = df[by_key[raw]]

    wanted = {round(p, 6): sa_name(p) for p in PERIODS}
    for col in df.columns:
        p = _raw_period(col)
        if p is not None and round(p, 6) in wanted:
            out[im_column(component, wanted[round(p, 6)])] = df[col]

    return pd.DataFrame(out, index=df.index)


def validate_schema(df: pd.DataFrame, component: str) -> None:
    """Check a normalised frame has every required column and unique RSNs."""
    component = component.upper()
    required = list(REQUIRED_META) + [f"luf_{component.lower()}"] + im_columns(component)
    if component == "V":
        required = ["rsn", "luf_v"] + im_columns("V")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{component} flatfile missing canonical columns: {missing}")
    if df["rsn"].duplicated().any():
        n = int(df["rsn"].duplicated().sum())
        raise SchemaError(f"{component} flatfile has {n} duplicate RSN values")
