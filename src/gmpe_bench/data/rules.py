"""Data-cleaning rules R2-R11. Each rule is a pure function ``df -> (df, AttritionStep)``.

Rule order used by :mod:`gmpe_bench.data.build`:
R2 sentinels -> R3 join -> R4 keys/RotD50 -> R5 usable period -> R6 quality ->
R7 missing keys -> R8 CL1/CL2 -> R10 V/H -> R11 fills -> R9 profile.
R11 runs before R9 because the ``sp23`` profile needs to know which ZTOR values are filled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gmpe_bench.data import features, targets
from gmpe_bench.schema import IM_NAMES, im_column, period_of

SENTINEL_MAX = -998.0  # NGA-West2 uses -999 for "unknown"
LUF_ANOMALY_TOL = 0.01  # LUF / corner within 1 % of 1.00
CL_CUTOFF_KM = 5.0


@dataclass(frozen=True)
class AttritionStep:
    rule: str
    description: str
    n_before: int
    n_after: int
    n_events: int
    n_stations: int
    note: str = ""

    @property
    def removed(self) -> int:
        return self.n_before - self.n_after

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "removed": self.removed}


def _count(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique()) if col in df else 0


def make_step(rule: str, description: str, before: int, after: pd.DataFrame, note: str = "") -> AttritionStep:
    return AttritionStep(rule, description, before, len(after),
                         _count(after, "eqid"), _count(after, "ssn"), note)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _value_counts(s: pd.Series, top: int = 8) -> str:
    vc = s.value_counts(dropna=False).head(top)
    return "; ".join(f"{k}: {v}" for k, v in vc.items())


# ----------------------------------------------------------------------- R2
def replace_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R2: -999 (any value <= -998) and blank strings become missing."""
    out = df.copy()
    n_replaced = 0
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            bad = s <= SENTINEL_MAX
        else:
            txt = s.astype("string").str.strip()
            num = pd.to_numeric(txt, errors="coerce")
            bad = (txt == "") | (num <= SENTINEL_MAX)
            bad = bad.fillna(False)
        k = int(bad.sum())
        if k:
            out[col] = s.mask(bad)
            n_replaced += k
    return out, make_step("R2", "Sentinels (-999, blanks) -> missing", len(df), out,
                          f"{n_replaced} cells replaced")


# ----------------------------------------------------------------------- R3
def join_hv(h: pd.DataFrame, v: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R3: one row per RSN; metadata from H, component fields and IMs from V."""
    v_cols = ["rsn"] + [c for c in v.columns if c.endswith("_v") or c.startswith("V_")]
    out = h.merge(v[v_cols], on="rsn", how="left", validate="one_to_one")
    n_v = int(out[[c for c in v_cols if c.startswith("V_")]].notna().any(axis=1).sum())
    return out, make_step("R3", "Join H and V on RSN", len(h), out, f"{n_v} records have V values")


# ----------------------------------------------------------------------- R4
def lock_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R4: EQID and SSN as integer keys; keep only RotD50 horizontal records."""
    out = df.copy()
    for col in ("rsn", "eqid", "ssn"):
        out[col] = _num(out, col).round().astype("Int64")
    note = ""
    if "rotd_pct_h" in out:
        note = f"RotD percentile: {_value_counts(out['rotd_pct_h'])}"
        out = out[_num(out, "rotd_pct_h") == 50]
    out = out.sort_values("rsn").reset_index(drop=True)
    return out, make_step("R4", "Lock EQID/SSN keys; keep RotD50 only", len(df), out, note)


# ----------------------------------------------------------------------- R5
def compute_usable_period(df: pd.DataFrame, component: str) -> pd.Series:
    """T_max = 1 / LUF [s]. The flatfile LUF already includes the 1.25 factor."""
    luf = _num(df, f"luf_{component.lower()}")
    return 1.0 / luf.where(luf > 0)


def flag_luf_anomaly(df: pd.DataFrame, component: str) -> pd.Series:
    """True where LUF / high-pass corner is ~1.00 instead of ~1.25."""
    c = component.lower()
    corners = [_num(df, f"{k}_{c}") for k in ("hp1", "hp2") if f"{k}_{c}" in df]
    if not corners:
        return pd.Series(False, index=df.index)
    corner = pd.concat(corners, axis=1).max(axis=1)
    ratio = _num(df, f"luf_{c}") / corner
    return ((ratio - 1.0).abs() < LUF_ANOMALY_TOL).fillna(False)


def mask_unusable_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R5: set SA(T) to missing where T > T_max; flag LUF anomalies (no records dropped)."""
    out = df.copy()
    notes = []
    for comp in ("H", "V"):
        c = comp.lower()
        tmax = compute_usable_period(out, comp)
        out[f"tmax_{c}"] = tmax
        out[f"luf_factor_anomaly_{c}"] = flag_luf_anomaly(out, comp)
        masked = 0
        for name in IM_NAMES:
            p = period_of(name)
            col = im_column(comp, name)
            if p is None or col not in out:
                continue
            beyond = tmax.notna() & (p > tmax) & out[col].notna()
            masked += int(beyond.sum())
            out[col] = out[col].mask(beyond)
        notes.append(
            f"{comp}: {masked} SA values masked; {int(tmax.isna().sum())} records without LUF (kept, unmasked); "
            f"{int(out[f'luf_factor_anomaly_{c}'].sum())} LUF anomalies flagged"
        )
    return out, make_step("R5", "Mask SA beyond usable period T_max = 1/LUF", len(df), out, " | ".join(notes))


# ----------------------------------------------------------------------- R6
def is_yes(df: pd.DataFrame, col: str) -> pd.Series:
    """True where a trigger flag is set: string ``'Y'`` (any case) or numeric 1.

    The 2015 flatfiles store confirmed late triggers as ``'Y'``; other codes
    (``'D'``, ``'?'``, ``'PT'`` ...) are not treated as failures.
    """
    if col not in df:
        return pd.Series(False, index=df.index)
    text = df[col].astype("string").str.strip().str.upper()
    return ((text == "Y") | (_num(df, col) == 1)).fillna(False).astype(bool)


def _failed_quality(df: pd.DataFrame, c: str) -> pd.Series:
    """Late P-trigger set, or a non-zero Spectra Quality Flag."""
    q = _num(df, f"spectra_qflag_{c}")
    return (is_yes(df, f"late_p_{c}") | (q.notna() & (q != 0))).fillna(False).astype(bool)


def apply_quality_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R6: drop records failing H quality; mask V values failing V quality; keep Late S as a flag."""
    out = df.copy()
    note = (f"H late P: {_value_counts(out.get('late_p_h', pd.Series(dtype=float)))} | "
            f"H spectra flag: {_value_counts(out.get('spectra_qflag_h', pd.Series(dtype=float)))}")
    out["late_s"] = is_yes(out, "late_s_h")

    h_bad = _failed_quality(out, "h")
    v_bad = _failed_quality(out, "v")
    for name in IM_NAMES:
        col = im_column("V", name)
        if col in out:
            out[col] = out[col].mask(v_bad)
    note += (f" | H dropped: late P 'Y' {int(is_yes(out, 'late_p_h').sum())}, "
             f"any failure {int(h_bad.sum())} | V values masked for {int(v_bad.sum())} records "
             f"| late S 'Y' flagged {int(out['late_s'].sum())}")
    out = out[~h_bad].reset_index(drop=True)
    return out, make_step("R6", "Drop failed H quality; mask failed V", len(df), out, note)


# ----------------------------------------------------------------------- R7
def drop_missing_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R7: need EQID, SSN, Mw, R_JB, R_rup, VS30 and at least one usable H target."""
    keys = ["eqid", "ssn", "mw", "rjb", "rrup", "vs30"]
    ok = pd.concat([_num(df, k).notna() for k in keys], axis=1).all(axis=1)
    h_cols = [im_column("H", n) for n in IM_NAMES if im_column("H", n) in df]
    ok &= df[h_cols].notna().any(axis=1)
    out = df[ok].reset_index(drop=True)
    return out, make_step("R7", "Drop missing keys or all-missing H targets", len(df), out)


# ----------------------------------------------------------------------- R8
_CL_TEXT = {"CL1": "CL1", "CL2": "CL2", "1": "CL1", "2": "CL2", "CLASS1": "CL1", "CLASS2": "CL2"}


def _cl_from_type(raw: pd.Series) -> pd.Series | None:
    """Parse NGA-West2 TYPE (CRjb = 5) codes: C1/C1f.../CL1/0/1 -> CL1, C2.../CL2/2 -> CL2."""
    txt = raw.astype("string").str.strip().str.upper()
    is_c1 = txt.str.startswith("C1") | txt.isin(["CL1", "1", "0"])
    is_c2 = txt.str.startswith("C2") | txt.isin(["CL2", "2"])
    if not (is_c1 | is_c2).any():
        return None
    out = pd.Series(pd.NA, index=raw.index, dtype="string")
    out[is_c1] = "CL1"
    out[is_c2] = "CL2"
    return out


def cl_from_crjb(crjb: pd.Series, cutoff: float = CL_CUTOFF_KM) -> pd.Series:
    """CL2 only where CRJB is known and below the cutoff; missing CRJB (mainshock) is CL1."""
    is_cl2 = (crjb.notna() & (crjb < cutoff)).to_numpy(dtype=bool)
    return pd.Series(np.where(is_cl2, "CL2", "CL1"), index=crjb.index, dtype="string")


def _crjb_summary(crjb: pd.Series) -> str:
    return (f"CRJB missing {int(crjb.isna().sum())}, <0 {int((crjb < 0).sum())}, "
            f"=0 {int((crjb == 0).sum())}, (0,5) {int(((crjb > 0) & (crjb < 5)).sum())}, "
            f">=5 {int((crjb >= 5).sum())}")


def assign_cl1_cl2(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R8: Wooddell & Abrahamson (2014) class at CRJB = 5 km, stored as ``cl_class``.

    Uses the flatfile's TYPE (CRjb = 5) column when its codes are recognisable;
    otherwise CRJB: missing (mainshock) -> CL1, CRJB < 5 km -> CL2.
    Any record still unclassified falls back to the CRJB rule.
    """
    out = df.copy()
    crjb = _num(out, "crjb")
    from_crjb = cl_from_crjb(crjb)
    raw = out["cl_type_5"] if "cl_type_5" in out else pd.Series(pd.NA, index=out.index)
    from_type = _cl_from_type(raw)

    if from_type is not None:
        out["cl_class"] = from_type.fillna(from_crjb)
        source = "TYPE (CRjb = 5) column"
    else:
        out["cl_class"] = from_crjb
        source = "CRJB < 5 km (TYPE codes not recognised)"
    note = (f"source: {source} | TYPE raw: {_value_counts(raw.astype('string'))} | "
            f"{_crjb_summary(crjb)} | {_value_counts(out['cl_class'])}")
    return out, make_step("R8", "CL1/CL2 class at CRJB = 5 km", len(df), out, note)


# ----------------------------------------------------------------------- R9
def select_profile_full(df: pd.DataFrame) -> tuple[pd.DataFrame, list[AttritionStep]]:
    """R9 ``full``: every usable record."""
    return df.copy(), [make_step("R9", "Profile full: no further filters", len(df), df)]


def min_records_per_event(df: pd.DataFrame, n_min: int = 5) -> pd.DataFrame:
    counts = df.groupby("eqid")["rsn"].transform("size")
    return df[counts >= n_min].reset_index(drop=True)


def select_profile_sp23(df: pd.DataFrame) -> tuple[pd.DataFrame, list[AttritionStep]]:
    """R9 ``sp23``: Sedaghati & Pezeshk (2023) selection, applied in their order."""
    steps: list[AttritionStep] = []
    cur = df

    def apply(rule: str, desc: str, keep: pd.Series, note: str = "") -> None:
        nonlocal cur
        before = len(cur)
        cur = cur[keep.fillna(False)].reset_index(drop=True)
        steps.append(make_step(rule, desc, before, cur, note))

    filled = cur["ztor_imputed"].astype(bool) if "ztor_imputed" in cur else pd.Series(False, index=cur.index)
    known = cur["ztor"].notna() & ~filled & cur[im_column("H", "PGA")].notna()
    apply("R9a", "sp23: known ZTOR (not filled) and PGA", known)
    apply("R9b", "sp23: Mw >= 3", _num(cur, "mw") >= 3.0)
    apply("R9c", "sp23: R_rup <= 300 km", _num(cur, "rrup") <= 300.0)
    apply("R9d", "sp23: drop CL2 at CRJB = 5 km", (cur["cl_class"] != "CL2").fillna(True))
    steps.append(make_step(
        "R9e", "sp23: multiple-event flag", len(cur), cur,
        "Not applied: column absent from the 2015 flatfile",
    ))
    steps.append(make_step(
        "R9f", "sp23: residual outliers > 4", len(cur), cur,
        "Not applied: needs a fitted model; handled at modelling stage",
    ))
    before = len(cur)
    cur = min_records_per_event(cur, 5)
    steps.append(make_step("R9g", "sp23: events with >= 5 records", before, cur))
    return cur, steps


PROFILES = {"full": select_profile_full, "sp23": select_profile_sp23}


# ----------------------------------------------------------------------- R10
def add_vh_ratios(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R10: ln(V/H) only where both components are usable."""
    out = targets.compute_vh(df)
    n = int(out[[im_column("VH", n) for n in IM_NAMES]].notna().any(axis=1).sum())
    return out, make_step("R10", "ln(V/H) where H and V both usable", len(df), out, f"{n} records with V/H")


# ----------------------------------------------------------------------- R11
def fill_site_source(df: pd.DataFrame) -> tuple[pd.DataFrame, AttritionStep]:
    """R11: fill ZTOR, Z1.0, Z2.5 with default relations; add flags and ``vs30_measured``."""
    out = features.impute_ztor(df)
    out = features.impute_z1(out)
    cls = _num(out, "vs30_class")
    out["vs30_measured"] = (cls == 0).where(cls.notna())
    note = (f"ZTOR filled {int(out['ztor_imputed'].sum())}; Z1.0 filled {int(out['z1_imputed'].sum())}; "
            f"Z2.5 filled {int(out['z2p5_imputed'].sum())}; VS30 class codes: {_value_counts(out.get('vs30_class', pd.Series(dtype=float)))} "
            "(0 taken as measured)")
    return out, make_step("R11", "Fill ZTOR (CY14), Z1.0 (CY14), Z2.5 (CB14) with flags", len(df), out, note)
