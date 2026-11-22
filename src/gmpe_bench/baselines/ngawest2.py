"""NGA-West2 GMMs (ASK14, BSSA14, CB14, CY14) via ``pygmm`` for horizontal RotD50.

Input mapping (canonical column -> ``pygmm.Scenario``), with defaults for missing values:

=================  =================  =========================================================
canonical          Scenario           default when missing
=================  =================  =========================================================
mw                 mag                (required)
rrup, rjb          dist_rup, dist_jb  (required)
rx                 dist_x             -R_JB (footwall side)
ry0                dist_y0            omitted (model default)
vs30               v_s30              (required)
vs30_measured      vs_source          "inferred" unless flagged measured
z1  [m]            depth_1_0 [km]     omitted (model's own VS30 relation)
z2p5 [m]           depth_2_5 [km]     omitted (model's own relation)
ztor               depth_tor          omitted (model's own relation)
dip                dip                SS 90, NS 50, RS 40 degrees
width              width              omitted (model's own relation)
hypo_depth         depth_hyp          omitted
mechanism / rake   mechanism          code 0 SS, 1/4 NS, 2/3 RS; else rake; else SS
rx >= 0, dip < 90  on_hanging_wall    False
cl_class == CL2    is_aftershock      False
crjb               dist_crjb          model default (15 km)
hypocentre box     region             mapped to the model's options, else global/california
=================  =================  =========================================================

Every model is evaluated once per record for all IMs (about 1 ms per record for
all four models). Results are cached by RSN, so folds only re-fit the bias term.
Out-of-range inputs are kept and flagged in ``applicable``; pygmm's range warnings are silenced.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygmm

from gmpe_bench.baselines.base import BaseGMM, target
from gmpe_bench.baselines.residuals import ResidualDecomposition, decompose_residuals
from gmpe_bench.schema import IM_NAMES, period_of
from gmpe_bench.splits.group_splits import assign_region

MODEL_CLASSES: dict[str, type] = {
    "ASK14": pygmm.AbrahamsonSilvaKamai2014,
    "BSSA14": pygmm.BooreStewartSeyhanAtkinson2014,
    "CB14": pygmm.CampbellBozorgnia2014,
    "CY14": pygmm.ChiouYoungs2014,
}

DEFAULT_DIP = {"SS": 90.0, "NS": 50.0, "RS": 40.0}
REGION_TO_PYGMM = {"California_WUS": "california", "Japan": "japan", "Taiwan": "taiwan",
                   "China": "china", "Europe_MiddleEast": "global", "Other": "global"}
_MECH_CODE = {0: "SS", 1: "NS", 4: "NS", 2: "RS", 3: "RS"}

INPUT_COLUMNS = ("rsn", "mw", "rrup", "rjb", "rx", "ry0", "vs30", "vs30_measured", "z1", "z2p5",
                 "ztor", "dip", "rake", "width", "hypo_depth", "mechanism", "cl_class", "crjb",
                 "hypo_lat", "hypo_lon")


def _f(v: object) -> float | None:
    """Finite float or None."""
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def mechanism_of(code: object, rake: object) -> str:
    c = _f(code)
    if c is not None and int(c) in _MECH_CODE:
        return _MECH_CODE[int(c)]
    r = _f(rake)
    if r is not None:
        if 30.0 <= r <= 150.0:
            return "RS"
        if -150.0 <= r <= -30.0:
            return "NS"
    return "SS"


def region_for(model: str, region_label: str) -> str:
    """Map our region label to an option the model accepts."""
    options = _options(MODEL_CLASSES[model], "region")
    want = REGION_TO_PYGMM.get(region_label, "global")
    if want in options:
        return want
    return "global" if "global" in options else "california"


def _options(cls: type, name: str) -> list:
    for p in cls.PARAMS:
        if p.name == name:
            return list(getattr(p, "options", []) or [])
    return []


def scenario_kwargs(row: dict, model: str, region_label: str = "Other") -> dict:
    """Build pygmm Scenario keyword arguments for one record and one model."""
    cls = MODEL_CLASSES[model]
    accepted = {p.name for p in cls.PARAMS}
    mech = mechanism_of(row.get("mechanism"), row.get("rake"))
    rjb = _f(row.get("rjb"))
    dip = _f(row.get("dip")) or DEFAULT_DIP[mech]
    rx = _f(row.get("rx"))
    if rx is None and rjb is not None:
        rx = -rjb
    z1, z25 = _f(row.get("z1")), _f(row.get("z2p5"))
    measured = _f(row.get("vs30_measured"))
    kw = {
        "mag": _f(row.get("mw")),
        "dist_rup": _f(row.get("rrup")),
        "dist_jb": rjb,
        "dist_x": rx,
        "dist_y0": _f(row.get("ry0")),
        "v_s30": _f(row.get("vs30")),
        "vs_source": "measured" if measured == 1.0 else "inferred",
        "depth_1_0": z1 / 1000.0 if z1 is not None else None,
        "depth_2_5": z25 / 1000.0 if z25 is not None else None,
        "depth_tor": _f(row.get("ztor")),
        "depth_hyp": _f(row.get("hypo_depth")),
        "dip": dip,
        "width": _f(row.get("width")),
        "mechanism": mech,
        "on_hanging_wall": bool(rx is not None and rx >= 0.0 and dip < 90.0),
        "is_aftershock": str(row.get("cl_class")) == "CL2",
        "dist_crjb": _f(row.get("crjb")),
        "region": region_for(model, region_label),
        "dpp_centered": 0.0,
    }
    return {k: v for k, v in kw.items() if k in accepted and v is not None}


def in_limits(model: str, kw: dict) -> bool:
    """True when every numeric input is inside the model's recommended limits."""
    for p in MODEL_CLASSES[model].PARAMS:
        v = kw.get(p.name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = getattr(p, "min", None), getattr(p, "max", None)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                return False
    return True


def _ln_outputs(m: object, periods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ln medians and ln stds for IM_NAMES order: PGA [ln g], PGV [ln cm/s], SA [ln g]."""
    sa_mu = np.asarray(m.interp_ln_spec_accels(periods), dtype=float)  # type: ignore[attr-defined]
    sa_sd = np.asarray(m.interp_ln_stds(periods), dtype=float)  # type: ignore[attr-defined]
    lo, hi = m.PERIODS[m.INDICES_PSA].min(), m.PERIODS[m.INDICES_PSA].max()  # type: ignore[attr-defined]
    outside = (periods < lo) | (periods > hi)
    sa_mu[outside] = np.nan
    sa_sd[outside] = np.nan
    mu = np.concatenate([[np.log(m.pga), np.log(m.pgv)], sa_mu])  # type: ignore[attr-defined]
    sd = np.concatenate([[m.ln_std_pga, m.ln_std_pgv], sa_sd])  # type: ignore[attr-defined]
    return mu, sd


def predict_table(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Medians and sigmas for every record and IM; index is RSN.

    Columns: ``mu_<IM>``, ``sigma_<IM>``, ``applicable``, ``failed``.
    """
    if model not in MODEL_CLASSES:
        raise KeyError(f"Unknown model {model!r}; choose from {list(MODEL_CLASSES)}")
    if not IM_NAMES[:2] == ["PGA", "PGV"]:
        raise ValueError("IM_NAMES must start with PGA, PGV")
    cls = MODEL_CLASSES[model]
    periods = np.array([period_of(n) for n in IM_NAMES[2:]], dtype=float)
    regions = assign_region(df).to_numpy()
    rows = df[[c for c in INPUT_COLUMNS if c in df]].to_dict("records")

    mu = np.full((len(rows), len(IM_NAMES)), np.nan)
    sd = np.full_like(mu, np.nan)
    applicable = np.zeros(len(rows), dtype=bool)
    failed = np.zeros(len(rows), dtype=bool)
    level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, row in enumerate(rows):
                kw = scenario_kwargs(row, model, regions[i])
                applicable[i] = in_limits(model, kw)
                try:
                    mu[i], sd[i] = _ln_outputs(cls(pygmm.Scenario(**kw)), periods)
                except Exception:  # noqa: BLE001 - a bad record must not stop the benchmark
                    failed[i] = True
    finally:
        logging.disable(level)

    out = pd.DataFrame(mu, columns=[f"mu_{n}" for n in IM_NAMES])
    out[[f"sigma_{n}" for n in IM_NAMES]] = sd
    out["applicable"], out["failed"] = applicable, failed
    out.index = pd.Index(df["rsn"].to_numpy(dtype=np.int64), name="rsn")
    return out


def row_hashes(df: pd.DataFrame) -> np.ndarray:
    """One uint64 per record over the model inputs (and the pygmm version)."""
    cols = [c for c in INPUT_COLUMNS if c in df]
    frame = df[cols].astype("string").assign(_pygmm=pygmm.__version__)
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)


def cached_predict_table(df: pd.DataFrame, model: str, cache_dir: Path | None) -> pd.DataFrame:
    """:func:`predict_table` with an on-disk cache per model.

    A record is reused only if its RSN and its input hash both match, so a rebuilt
    dataset or another profile recomputes exactly the records that changed.
    """
    if cache_dir is None:
        return predict_table(df, model)
    path = Path(cache_dir) / f"{model}.parquet"
    hashes = pd.Series(row_hashes(df), index=pd.Index(df["rsn"].to_numpy(dtype=np.int64), name="rsn"))
    cache = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if len(cache):
        cache = cache[~cache.index.duplicated(keep="last")]
        hit = hashes.index.isin(cache.index)
        hit[hit] = cache.loc[hashes.index[hit], "_hash"].to_numpy(dtype=np.uint64) == hashes[hit].to_numpy()
    else:
        hit = np.zeros(len(df), dtype=bool)
    if (~hit).any():
        fresh = predict_table(df[~hit], model)
        fresh["_hash"] = hashes[~hit].to_numpy()
        cache = pd.concat([cache.drop(index=fresh.index, errors="ignore"), fresh]) if len(cache) else fresh
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(path)
    return cache.loc[hashes.index].drop(columns="_hash")


class NGAWest2Baseline(BaseGMM):
    """A published NGA-West2 model, optionally with a train_fit bias term.

    With ``calibrate_bias=True`` the crossed random-effects intercept of the
    train_fit residuals is added to every prediction (zero-mean calibration).
    """

    def __init__(self, model: str = "ASK14", calibrate_bias: bool = True,
                 cache_dir: Path | None = None) -> None:
        super().__init__()
        if model not in MODEL_CLASSES:
            raise KeyError(f"Unknown model {model!r}; choose from {list(MODEL_CLASSES)}")
        self.model = model
        self.name = model + ("_cal" if calibrate_bias else "")
        self.calibrate_bias = calibrate_bias
        self.cache_dir = cache_dir
        self.bias: float = 0.0
        self.train_decomposition: ResidualDecomposition | None = None
        self._table = pd.DataFrame()

    def precompute(self, df: pd.DataFrame) -> NGAWest2Baseline:
        """Evaluate the model once for every record of ``df`` (e.g. the whole profile)."""
        self._table = cached_predict_table(df, self.model, self.cache_dir)
        return self

    def _lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        rsn = df["rsn"].to_numpy(dtype=np.int64)
        missing = ~np.isin(rsn, self._table.index.to_numpy())
        if missing.any():
            extra = predict_table(df[missing], self.model)
            self._table = pd.concat([self._table, extra])
        return self._table.loc[rsn]

    def raw(self, df: pd.DataFrame, im: str) -> tuple[np.ndarray, np.ndarray]:
        t = self._lookup(df)
        return t[f"mu_{im}"].to_numpy(dtype=float), t[f"sigma_{im}"].to_numpy(dtype=float)

    def fit(self, df_train: pd.DataFrame, im: str) -> NGAWest2Baseline:
        self.im = im
        mu, _ = self.raw(df_train, im)
        resid = target(df_train, im) - mu
        dec = decompose_residuals(resid, df_train["eqid"].to_numpy(), df_train["ssn"].to_numpy())
        self.train_decomposition = dec
        self.bias = float(dec.intercept) if self.calibrate_bias and np.isfinite(dec.intercept) else 0.0
        self.tau, self.phi = None, None  # pygmm 0.8.0 exposes total sigma only
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        im = self._check_fitted()
        mu, sd = self.raw(df, im)
        return mu + self.bias, sd

    def applicable(self, df: pd.DataFrame) -> np.ndarray:
        return self._lookup(df)["applicable"].to_numpy(dtype=bool)
