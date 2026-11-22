"""Model inputs (Tier A, Tier B) and default fills for ZTOR, Z1.0 and Z2.5.

Fill relations (each filled value is flagged):

* ZTOR  - Chiou & Youngs (2014): reverse / reverse-oblique
          ``max(2.704 - 1.226 max(M - 5.849, 0), 0)^2``, otherwise
          ``max(2.673 - 1.136 max(M - 4.970, 0), 0)^2``  [km]
* Z1.0  - Chiou & Youngs (2014): California
          ``ln Z1 = -7.15/4 ln((Vs30^4 + 571^4) / (1360^4 + 571^4))``, Japan
          ``ln Z1 = -5.23/2 ln((Vs30^2 + 412^2) / (1360^2 + 412^2))``  [m]
* Z2.5  - Campbell & Bozorgnia (2014): California ``ln Z2.5 = 7.089 - 1.144 ln Vs30``,
          Japan ``ln Z2.5 = 5.359 - 1.102 ln Vs30``  [km, stored in m]

Japan is detected from the earthquake name. All other regions use the California form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R_FLOOR_KM = 0.1  # floor before taking logs of distances

TIER_A: list[str] = ["mw", "ln_rjb", "ln_vs30", "mech_ss", "mech_nm", "mech_rv", "hypo_depth"]
TIER_B: list[str] = TIER_A + [
    "ln_rrup", "rx", "ry0", "ztor", "dip", "rake", "width",
    "vs30_measured", "z1", "z2p5",
]

# NGA-West2 "Mechanism Based on Rake Angle":
# 0 strike-slip, 1 normal, 2 reverse, 3 reverse-oblique, 4 normal-oblique
_MECH_GROUP = {0: "ss", 1: "nm", 4: "nm", 2: "rv", 3: "rv"}


def is_japan(df: pd.DataFrame) -> pd.Series:
    if "eq_name" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["eq_name"].astype("string").str.contains("japan", case=False, na=False)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _is_reverse(df: pd.DataFrame) -> pd.Series:
    """Reverse or reverse-oblique: rake in [30, 150]; mechanism code 2/3 if rake is missing."""
    rake, mech = _num(df, "rake"), _num(df, "mechanism")
    by_rake = rake.between(30.0, 150.0)
    by_mech = mech.isin([2, 3])
    return by_rake.where(rake.notna(), by_mech).astype(bool)


def ztor_cy14(mw: pd.Series, reverse: pd.Series) -> pd.Series:
    rv = np.maximum(2.704 - 1.226 * np.maximum(mw - 5.849, 0.0), 0.0) ** 2
    ss = np.maximum(2.673 - 1.136 * np.maximum(mw - 4.970, 0.0), 0.0) ** 2
    return pd.Series(np.where(reverse, rv, ss), index=mw.index)


def z1_cy14(vs30: pd.Series, japan: pd.Series) -> pd.Series:
    ca = np.exp(-7.15 / 4.0 * np.log((vs30**4 + 571.0**4) / (1360.0**4 + 571.0**4)))
    jp = np.exp(-5.23 / 2.0 * np.log((vs30**2 + 412.0**2) / (1360.0**2 + 412.0**2)))
    return pd.Series(np.where(japan, jp, ca), index=vs30.index)


def z2p5_cb14(vs30: pd.Series, japan: pd.Series) -> pd.Series:
    ca = np.exp(7.089 - 1.144 * np.log(vs30))
    jp = np.exp(5.359 - 1.102 * np.log(vs30))
    return pd.Series(np.where(japan, jp, ca) * 1000.0, index=vs30.index)  # km -> m


def impute_ztor(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ZTOR (CY14); adds ``ztor_imputed``."""
    out = df.copy()
    ztor = _num(out, "ztor")
    miss = ztor.isna() & out["mw"].notna()
    fill = ztor_cy14(out["mw"].astype(float), _is_reverse(out))
    out["ztor"] = ztor.where(~miss, fill)
    out["ztor_imputed"] = miss
    return out


def impute_z1(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Z1.0 and Z2.5 [m]; adds ``z1``, ``z2p5``, flags and ``z1_source``.

    Basin-model values are taken first (SCEC CVM-S4, then CVM-H 11.9),
    then the default relations.
    """
    out = df.copy()
    japan = is_japan(out)
    vs30 = out["vs30"].astype(float)

    z1 = pd.Series(np.nan, index=out.index)
    source = pd.Series(pd.NA, index=out.index, dtype="string")
    for col, tag in (("z1_s4", "S4"), ("z1_h11", "H11")):
        vals = _num(out, col)
        take = z1.isna() & vals.notna()
        z1 = z1.where(~take, vals)
        source = source.where(~take, tag)
    miss = z1.isna() & vs30.notna()
    z1 = z1.where(~miss, z1_cy14(vs30, japan))
    source = source.where(~miss, "CY14_default")
    out["z1"], out["z1_imputed"], out["z1_source"] = z1, miss, source

    z25 = pd.Series(np.nan, index=out.index)
    for col in ("z2p5_s4", "z2p5_h11"):
        z25 = z25.fillna(_num(out, col))
    miss25 = z25.isna() & vs30.notna()
    out["z2p5"] = z25.where(~miss25, z2p5_cb14(vs30, japan))
    out["z2p5_imputed"] = miss25
    out["region_japan"] = japan
    return out


def mechanism_onehot(df: pd.DataFrame) -> pd.DataFrame:
    mech = _num(df, "mechanism").map(_MECH_GROUP)
    return pd.DataFrame(
        {f"mech_{g}": (mech == g).astype(float) for g in ("ss", "nm", "rv")},
        index=df.index,
    ).where(mech.notna())


def _ln_dist(s: pd.Series) -> pd.Series:
    return np.log(pd.to_numeric(s, errors="coerce").clip(lower=R_FLOOR_KM))


def tier_a_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mw, ln R_JB, ln VS30, mechanism one-hot, hypocentral depth."""
    x = pd.DataFrame(index=df.index)
    x["mw"] = df["mw"].astype(float)
    x["ln_rjb"] = _ln_dist(df["rjb"])
    x["ln_vs30"] = np.log(df["vs30"].astype(float))
    x = x.join(mechanism_onehot(df))
    x["hypo_depth"] = _num(df, "hypo_depth")
    return x[TIER_A]


def tier_b_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier A plus the inputs the NGA-West2 GMMs need."""
    x = tier_a_features(df)
    x["ln_rrup"] = _ln_dist(df["rrup"])
    for col in ("rx", "ry0", "ztor", "dip", "rake", "width", "z1", "z2p5"):
        x[col] = _num(df, col)
    x["vs30_measured"] = _num(df, "vs30_measured")
    return x[TIER_B]
