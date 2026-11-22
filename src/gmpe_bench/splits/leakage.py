"""Leakage detector for benchmark splits.

Hard checks (a failure means the split is unusable):

* disjoint: ``train_fit``, ``cal``, ``test`` and ``excluded`` share no rows
* complete: together they cover every row of the dataset
* group isolation: no shared group key across ``train_fit``, ``cal`` and ``test``
  (EQID for event-keyed schemes, SSN for ``station_grouped``)

Reported, never failed: the share of test records whose event or station also
appears in training. For ``random`` this is the leakage rate in Table 2; for
``event_grouped`` the station overlap is expected (stations record many events).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from gmpe_bench.data import GMDataset
from gmpe_bench.splits.group_splits import SplitFold

PARTS = ("train_fit", "cal", "test")


@dataclass
class LeakageReport:
    profile: str
    scheme: str
    fold: str
    group_key: str | None
    n_train_fit: int
    n_cal: int
    n_test: int
    n_excluded: int
    events_train_fit: int
    events_cal: int
    events_test: int
    stations_train_fit: int
    stations_cal: int
    stations_test: int
    disjoint: bool
    complete: bool
    shared_events: int          # events in test that also occur in train_fit or cal
    shared_stations: int
    test_records_seen_event: float    # share of test records whose EQID is in training
    test_records_seen_station: float  # share of test records whose SSN is in training
    cal_fit_shared_groups: int        # group-key values shared by cal and train_fit
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = "; ".join(self.issues)
        row["passed"] = self.passed
        return row


def _shared(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.intersect1d(a, b).size)


def audit_fold(ds: GMDataset, fold: SplitFold) -> LeakageReport:
    """Run every check on one fold."""
    n = len(ds)
    eq, st = ds.df["eqid"].to_numpy(), ds.df["ssn"].to_numpy()
    parts = {k: np.asarray(v, dtype=np.int64) for k, v in fold.parts().items()}
    issues: list[str] = []

    allidx = np.concatenate(list(parts.values()))
    disjoint = allidx.size == np.unique(allidx).size
    complete = np.array_equal(np.unique(allidx), np.arange(n))
    if not disjoint:
        issues.append("row overlap between parts")
    if not complete:
        issues.append(f"rows covered {np.unique(allidx).size} of {n}")
    if parts["test"].size == 0:
        issues.append("empty test set")

    train = np.concatenate([parts["train_fit"], parts["cal"]])
    ev_test, ev_train = np.unique(eq[parts["test"]]), np.unique(eq[train])
    st_test, st_train = np.unique(st[parts["test"]]), np.unique(st[train])
    seen_ev = float(np.isin(eq[parts["test"]], ev_train).mean()) if parts["test"].size else 0.0
    seen_st = float(np.isin(st[parts["test"]], st_train).mean()) if parts["test"].size else 0.0

    cal_fit_shared = 0
    if fold.group_key is not None:
        keys = ds.df[fold.group_key].to_numpy()
        sets = {p: np.unique(keys[parts[p]]) for p in PARTS}
        for i, a in enumerate(PARTS):
            for b in PARTS[i + 1:]:
                k = _shared(sets[a], sets[b])
                if k:
                    issues.append(f"{fold.group_key} shared by {a} and {b}: {k}")
        cal_fit_shared = _shared(sets["cal"], sets["train_fit"])

    return LeakageReport(
        profile=ds.profile, scheme=fold.scheme, fold=fold.fold, group_key=fold.group_key,
        n_train_fit=int(parts["train_fit"].size), n_cal=int(parts["cal"].size),
        n_test=int(parts["test"].size), n_excluded=int(parts["excluded"].size),
        events_train_fit=int(np.unique(eq[parts["train_fit"]]).size),
        events_cal=int(np.unique(eq[parts["cal"]]).size), events_test=int(ev_test.size),
        stations_train_fit=int(np.unique(st[parts["train_fit"]]).size),
        stations_cal=int(np.unique(st[parts["cal"]]).size), stations_test=int(st_test.size),
        disjoint=disjoint, complete=complete,
        shared_events=_shared(ev_test, ev_train), shared_stations=_shared(st_test, st_train),
        test_records_seen_event=round(seen_ev, 6), test_records_seen_station=round(seen_st, 6),
        cal_fit_shared_groups=cal_fit_shared, issues=issues,
    )


def audit_splits(ds: GMDataset, splits: dict[str, list[SplitFold]] | Iterable[SplitFold]) -> list[LeakageReport]:
    """Audit every fold of every scheme."""
    folds = [f for v in splits.values() for f in v] if isinstance(splits, dict) else list(splits)
    return [audit_fold(ds, f) for f in folds]


def leakage_table(reports: Iterable[LeakageReport]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in reports])


def assert_fit_transform_clean(fitted: object, X: np.ndarray | pd.DataFrame,
                               train_fit_idx: np.ndarray, rtol: float = 1e-7) -> None:
    """Check a fitted preprocessor (scaler, imputer ...) was fitted on ``train_fit`` only.

    A fresh clone is fitted on ``X[train_fit_idx]``; every learned attribute
    (name ending in ``_``) must match the given object. Raises ``AssertionError`` otherwise.
    """
    from sklearn.base import clone

    rows = X.iloc[train_fit_idx] if isinstance(X, pd.DataFrame) else np.asarray(X)[train_fit_idx]
    ref = clone(fitted).fit(rows)
    learned = [a for a in vars(ref) if a.endswith("_") and not a.startswith("_")]
    if not learned:
        raise AssertionError("No learned attributes found; was the object fitted?")
    for attr in learned:
        a, b = getattr(ref, attr), getattr(fitted, attr, None)
        if b is None:
            raise AssertionError(f"Fitted object lacks attribute {attr!r}")
        if isinstance(a, (np.ndarray, float, int, np.floating, np.integer)):
            if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                               rtol=rtol, equal_nan=True):
                raise AssertionError(f"{attr} differs from a fit on train_fit only: leakage suspected")
