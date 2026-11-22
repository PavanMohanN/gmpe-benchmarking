"""Targets: ln SA (g), ln PGV (cm/s) and ln(V/H), with usable-value masks.

Stored columns hold linear H and V values (``H_SA_0.2s``) and the
log ratio ``VH_SA_0.2s``. Missing means masked or unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gmpe_bench.schema import IM_NAMES, im_column

TARGET_COMPONENTS: tuple[str, ...] = ("H", "V", "VH")


def _check(component: str) -> str:
    component = component.upper()
    if component not in TARGET_COMPONENTS:
        raise ValueError(f"component must be one of {TARGET_COMPONENTS}, got {component!r}")
    return component


def _positive(df: pd.DataFrame, component: str) -> pd.DataFrame:
    cols = [im_column(component, n) for n in IM_NAMES]
    vals = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    return vals.where(vals > 0)


def get_targets(df: pd.DataFrame, component: str = "H") -> pd.DataFrame:
    """Natural-log targets with IM names as columns (``PGA``, ``SA_0.2s`` ...)."""
    component = _check(component)
    if component == "VH":
        cols = [im_column("VH", n) for n in IM_NAMES]
        y = df[cols].astype(float)
    else:
        y = np.log(_positive(df, component))
    y.columns = list(IM_NAMES)
    return y


def target_mask(df: pd.DataFrame, component: str = "H") -> pd.DataFrame:
    """True where the target is usable."""
    return get_targets(df, component).notna()


def compute_vh(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``VH_<IM>`` = ln SA_V - ln SA_H where both components are usable."""
    out = df.copy()
    h = _positive(out, "H")
    v = _positive(out, "V")
    for n, hc, vc in zip(IM_NAMES, h.columns, v.columns):
        out[im_column("VH", n)] = np.log(v[vc]) - np.log(h[hc])
    return out
