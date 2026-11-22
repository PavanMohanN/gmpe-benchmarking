"""Benchmark split schemes on a :class:`GMDataset`.

Each scheme returns *outer* folds (train, test, excluded) as positional row
indices. Calibration hold-outs are carved from the train part afterwards
(:mod:`gmpe_bench.splits.calibration`), always respecting the scheme's group key.

Schemes
-------
* ``event_grouped``      5 folds by EQID, stratified on event (Mw, median R_JB) bins
* ``station_grouped``    5 folds by SSN, stratified on station VS30 bins
* ``random``             5 folds by record; deliberately leaky reference
* ``spatial_block``      leave-one-region-out by hypocentre region
* ``temporal``           train year < 2008, test year >= 2008
* ``magnitude_holdout``  test events with Mw >= 7 (frozen plan S3)
* ``cl1_to_cl2``         train CL1 events, test CL2 events (frozen plan S4)
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gmpe_bench.data import GMDataset

Outer = tuple[np.ndarray, np.ndarray, np.ndarray, str]  # train, test, excluded, label

# Hypocentre boxes: (name, lat_min, lat_max, lon_min, lon_max). First match wins.
REGION_BOXES: tuple[tuple[str, float, float, float, float], ...] = (
    ("California_WUS", 30.0, 43.0, -126.0, -113.0),
    ("Japan", 30.0, 46.0, 128.0, 146.0),
    ("Taiwan", 21.0, 26.5, 119.0, 123.0),
    ("China", 20.0, 45.0, 73.0, 119.0),
    ("Europe_MiddleEast", 34.0, 48.0, -10.0, 45.0),
)
REGION_OTHER = "Other"


def scheme_seed(name: str, seed: int) -> int:
    """Stable per-scheme seed (independent of Python's hash randomisation)."""
    return (seed + zlib.crc32(name.encode())) % (2**32)


def _idx(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask).astype(np.int64)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def balanced_group_folds(groups: np.ndarray, strata: np.ndarray, n_folds: int,
                         rng: np.random.Generator) -> dict[object, int]:
    """Assign each group to a fold, balancing record counts within every stratum.

    Groups are processed stratum by stratum, largest first (random tie-break);
    each goes to the fold with the fewest records in that stratum, then fewest overall.
    """
    g = pd.DataFrame({"group": groups, "stratum": strata})
    sizes = g.groupby("group").agg(n=("group", "size"), stratum=("stratum", "first"))
    sizes["tie"] = rng.permutation(len(sizes))
    total = np.zeros(n_folds, dtype=np.int64)
    assign: dict[object, int] = {}
    for _, block in sizes.groupby("stratum", sort=True):
        in_stratum = np.zeros(n_folds, dtype=np.int64)
        for grp, row in block.sort_values(["n", "tie"], ascending=[False, True]).iterrows():
            k = int(np.lexsort((np.arange(n_folds), total, in_stratum))[0])
            assign[grp] = k
            in_stratum[k] += row["n"]
            total[k] += row["n"]
    return assign


def _quantile_bins(values: pd.Series, q: int) -> pd.Series:
    if values.nunique() < 2:
        return pd.Series(0, index=values.index)
    return pd.qcut(values.rank(method="first"), q=min(q, values.nunique()), labels=False)


def assign_region(df: pd.DataFrame) -> pd.Series:
    """Region label from hypocentre latitude/longitude."""
    lat, lon = _num(df, "hypo_lat"), _num(df, "hypo_lon")
    region = pd.Series(REGION_OTHER, index=df.index, dtype="string")
    done = pd.Series(False, index=df.index)
    for name, la0, la1, lo0, lo1 in REGION_BOXES:
        hit = ~done & lat.between(la0, la1) & lon.between(lo0, lo1)
        region[hit] = name
        done |= hit
    return region


@dataclass
class SplitScheme:
    """Base class. Subclasses implement :meth:`outer`."""

    name: str = "base"
    group_key: str | None = None
    description: str = ""
    seed: int = 42

    def outer(self, ds: GMDataset) -> list[Outer]:  # pragma: no cover - abstract
        raise NotImplementedError

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(scheme_seed(self.name, self.seed))


@dataclass
class EventGroupedKFold(SplitScheme):
    name: str = "event_grouped"
    group_key: str | None = "eqid"
    description: str = "5-fold by EQID, stratified on event Mw x median R_JB bins"
    n_folds: int = 5
    mw_bins: int = 4
    r_bins: int = 3

    def outer(self, ds: GMDataset) -> list[Outer]:
        df = ds.df
        ev = df.groupby("eqid").agg(mw=("mw", "first"), r=("rjb", "median"))
        stratum = (_quantile_bins(ev["mw"], self.mw_bins) * self.r_bins
                   + _quantile_bins(np.log1p(ev["r"]), self.r_bins))
        strata = df["eqid"].map(stratum).to_numpy()
        assign = balanced_group_folds(df["eqid"].to_numpy(), strata, self.n_folds, self.rng())
        fold_of = df["eqid"].map(assign).to_numpy()
        none = np.array([], dtype=np.int64)
        return [(_idx(fold_of != k), _idx(fold_of == k), none, f"fold{k}") for k in range(self.n_folds)]


@dataclass
class StationGroupedKFold(SplitScheme):
    name: str = "station_grouped"
    group_key: str | None = "ssn"
    description: str = "5-fold by SSN, stratified on station VS30 bins"
    n_folds: int = 5
    vs30_bins: int = 5

    def outer(self, ds: GMDataset) -> list[Outer]:
        df = ds.df
        st = df.groupby("ssn")["vs30"].first()
        stratum = _quantile_bins(np.log(st), self.vs30_bins)
        strata = df["ssn"].map(stratum).to_numpy()
        assign = balanced_group_folds(df["ssn"].to_numpy(), strata, self.n_folds, self.rng())
        fold_of = df["ssn"].map(assign).to_numpy()
        none = np.array([], dtype=np.int64)
        return [(_idx(fold_of != k), _idx(fold_of == k), none, f"fold{k}") for k in range(self.n_folds)]


@dataclass
class RandomKFold(SplitScheme):
    name: str = "random"
    group_key: str | None = None
    description: str = "5-fold by record (leaky reference)"
    n_folds: int = 5

    def outer(self, ds: GMDataset) -> list[Outer]:
        n = len(ds)
        fold_of = np.empty(n, dtype=np.int64)
        fold_of[self.rng().permutation(n)] = np.arange(n) % self.n_folds
        none = np.array([], dtype=np.int64)
        return [(_idx(fold_of != k), _idx(fold_of == k), none, f"fold{k}") for k in range(self.n_folds)]


@dataclass
class SpatialBlockSplit(SplitScheme):
    name: str = "spatial_block"
    group_key: str | None = "eqid"
    description: str = "Leave one hypocentre region out"
    min_events: int = 10

    def outer(self, ds: GMDataset) -> list[Outer]:
        region = assign_region(ds.df).to_numpy()
        events = ds.df.groupby(region)["eqid"].nunique()
        held = [r for r in events.index if events[r] >= self.min_events and r != REGION_OTHER]
        none = np.array([], dtype=np.int64)
        return [(_idx(region != r), _idx(region == r), none, f"test_{r}") for r in sorted(held)]


@dataclass
class TemporalSplit(SplitScheme):
    name: str = "temporal"
    group_key: str | None = "eqid"
    description: str = "Train year < 2008, test year >= 2008"
    cutoff_year: int = 2008

    def outer(self, ds: GMDataset) -> list[Outer]:
        year = _num(ds.df, "year").to_numpy()
        known = ~np.isnan(year)
        return [(_idx(known & (year < self.cutoff_year)), _idx(known & (year >= self.cutoff_year)),
                 _idx(~known), f"cut{self.cutoff_year}")]


@dataclass
class MagnitudeHoldout(SplitScheme):
    name: str = "magnitude_holdout"
    group_key: str | None = "eqid"
    description: str = "Test events with Mw >= 7"
    mw_min: float = 7.0

    def outer(self, ds: GMDataset) -> list[Outer]:
        mw = ds.df.groupby("eqid")["mw"].transform("first").to_numpy()
        none = np.array([], dtype=np.int64)
        return [(_idx(mw < self.mw_min), _idx(mw >= self.mw_min), none, f"mw{self.mw_min:g}+")]


@dataclass
class CL1ToCL2(SplitScheme):
    name: str = "cl1_to_cl2"
    group_key: str | None = "eqid"
    description: str = "Train mainshock (CL1) events, test aftershock (CL2) events"

    def outer(self, ds: GMDataset) -> list[Outer]:
        df = ds.df
        is_cl2 = (df["cl_class"].astype("string") == "CL2").fillna(False)
        event_cl2 = is_cl2.groupby(df["eqid"]).transform("mean").to_numpy() > 0.5
        none = np.array([], dtype=np.int64)
        return [(_idx(~event_cl2), _idx(event_cl2), none, "cl1_to_cl2")]


SCHEMES: dict[str, type[SplitScheme]] = {
    "event_grouped": EventGroupedKFold,
    "station_grouped": StationGroupedKFold,
    "random": RandomKFold,
    "spatial_block": SpatialBlockSplit,
    "temporal": TemporalSplit,
    "magnitude_holdout": MagnitudeHoldout,
    "cl1_to_cl2": CL1ToCL2,
}


def get_scheme(name: str, seed: int = 42) -> SplitScheme:
    if name not in SCHEMES:
        raise KeyError(f"Unknown scheme {name!r}; choose from {list(SCHEMES)}")
    return SCHEMES[name](seed=seed)


@dataclass
class SplitFold:
    """One fold: disjoint positional indices for train_fit, cal, test and excluded rows."""

    scheme: str
    fold: str
    train_fit: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    excluded: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    group_key: str | None = None

    @property
    def train(self) -> np.ndarray:
        return np.sort(np.concatenate([self.train_fit, self.cal]))

    def parts(self) -> dict[str, np.ndarray]:
        return {"train_fit": self.train_fit, "cal": self.cal, "test": self.test, "excluded": self.excluded}


def make_splits(ds: GMDataset, schemes: list[str] | None = None, seed: int = 42,
                cal_frac: float = 0.25) -> dict[str, list[SplitFold]]:
    """All requested schemes, each fold with its calibration hold-out."""
    from gmpe_bench.splits.calibration import carve_calibration

    out: dict[str, list[SplitFold]] = {}
    for name in schemes or list(SCHEMES):
        scheme = get_scheme(name, seed)
        folds = []
        for i, (train, test, excluded, label) in enumerate(scheme.outer(ds)):
            fit, cal = carve_calibration(ds, train, scheme.group_key, cal_frac,
                                         seed=scheme_seed(f"{name}/{label}/cal", seed))
            folds.append(SplitFold(name, label, fit, cal, np.sort(test), np.sort(excluded), scheme.group_key))
        out[name] = folds
    return out
