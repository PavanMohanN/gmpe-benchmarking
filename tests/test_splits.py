"""Phase 2: split schemes, calibration hold-outs, leakage audit and freezing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gmpe_bench.data import GMDataset
from gmpe_bench.splits import (
    SCHEMES,
    SplitFold,
    assert_fit_transform_clean,
    audit_splits,
    carve_calibration,
    load_splits,
    make_splits,
    save_splits,
)
from gmpe_bench.splits.group_splits import assign_region, balanced_group_folds
from gmpe_bench.splits.leakage import audit_fold

N_EVENTS, PER_EVENT, N_STATIONS = 60, 8, 90

# (lat, lon) per region, used to place hypocentres
REGION_POINTS = {"California_WUS": (35.0, -118.0), "Japan": (37.0, 140.0),
                 "Taiwan": (23.8, 120.8), "Europe_MiddleEast": (42.0, 13.0)}


@pytest.fixture(scope="module")
def ds() -> GMDataset:
    """60 events x 8 records on 90 crossed stations, 4 regions, years 1990-2019."""
    rng = np.random.default_rng(7)
    eqid = np.repeat(np.arange(1, N_EVENTS + 1), PER_EVENT)
    n = eqid.size
    regions = list(REGION_POINTS)
    ev_region = np.array([regions[i % 4] for i in range(N_EVENTS)])
    ev_mw = rng.uniform(4.0, 7.8, N_EVENTS)
    ev_year = rng.integers(1990, 2020, N_EVENTS).astype(float)
    ev_year[0] = np.nan  # one event with unknown year
    ev_cl2 = rng.random(N_EVENTS) < 0.3
    ssn = rng.integers(1, N_STATIONS + 1, n)
    st_vs30 = rng.uniform(150, 1500, N_STATIONS + 1)
    lat = np.array([REGION_POINTS[r][0] for r in ev_region])
    lon = np.array([REGION_POINTS[r][1] for r in ev_region])
    df = pd.DataFrame({
        "rsn": np.arange(1, n + 1), "eqid": eqid, "ssn": ssn,
        "mw": ev_mw[eqid - 1], "rjb": rng.uniform(0, 300, n), "rrup": rng.uniform(1, 300, n),
        "vs30": st_vs30[ssn], "year": ev_year[eqid - 1],
        "hypo_lat": lat[eqid - 1], "hypo_lon": lon[eqid - 1],
        "cl_class": np.where(ev_cl2[eqid - 1], "CL2", "CL1"),
    })
    return GMDataset(df, "toy")


@pytest.fixture(scope="module")
def splits(ds: GMDataset) -> dict[str, list[SplitFold]]:
    return make_splits(ds, seed=42, cal_frac=0.25)


def _keys(ds: GMDataset, col: str, idx: np.ndarray) -> set:
    return set(ds.df[col].to_numpy()[idx])


# ------------------------------------------------------------------ structure
def test_all_schemes_present(splits: dict[str, list[SplitFold]]) -> None:
    assert set(splits) == set(SCHEMES)
    for name in ("event_grouped", "station_grouped", "random"):
        assert len(splits[name]) == 5


def test_every_fold_passes_audit(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    for r in audit_splits(ds, splits):
        assert r.passed, (r.scheme, r.fold, r.issues)
        assert r.disjoint and r.complete


# ------------------------------------------------------------------ event grouped
def test_event_grouped_zero_eqid_overlap(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    seen_in_test: list[int] = []
    for f in splits["event_grouped"]:
        fit, cal, test = (_keys(ds, "eqid", f.train_fit), _keys(ds, "eqid", f.cal), _keys(ds, "eqid", f.test))
        assert not (fit & cal) and not (fit & test) and not (cal & test)
        seen_in_test += list(test)
    assert sorted(seen_in_test) == list(range(1, N_EVENTS + 1))  # each event tested once


def test_event_grouped_balanced(splits: dict[str, list[SplitFold]]) -> None:
    sizes = [f.test.size for f in splits["event_grouped"]]
    assert max(sizes) / min(sizes) < 1.5


def test_event_grouped_stations_overlap_is_reported(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    r = audit_fold(ds, splits["event_grouped"][0])
    assert r.test_records_seen_event == 0.0
    assert r.test_records_seen_station > 0.5  # crossed design: stations shared, by construction


# ------------------------------------------------------------------ station grouped
def test_station_grouped_zero_ssn_overlap(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    for f in splits["station_grouped"]:
        fit, cal, test = (_keys(ds, "ssn", f.train_fit), _keys(ds, "ssn", f.cal), _keys(ds, "ssn", f.test))
        assert not (fit & cal) and not (fit & test) and not (cal & test)


# ------------------------------------------------------------------ random
def test_random_overlap_detected(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    for r in audit_splits(ds, splits["random"]):
        assert r.passed  # no group key: overlap is reported, not failed
        assert r.shared_events > 0 and r.test_records_seen_event > 0.9


# ------------------------------------------------------------------ temporal
def test_temporal_cutoff(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    (f,) = splits["temporal"]
    year = ds.df["year"].to_numpy()
    assert np.all(year[f.train] < 2008)
    assert np.all(year[f.test] >= 2008)
    assert np.all(np.isnan(year[f.excluded])) and f.excluded.size == PER_EVENT


# ------------------------------------------------------------------ spatial
def test_spatial_leave_one_region_out(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    region = assign_region(ds.df).to_numpy()
    held = []
    for f in splits["spatial_block"]:
        name = f.fold.removeprefix("test_")
        assert set(region[f.test]) == {name}
        assert name not in set(region[f.train])
        held.append(name)
    assert sorted(held) == sorted(REGION_POINTS)


# ------------------------------------------------------------------ plan extras
def test_magnitude_holdout(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    (f,) = splits["magnitude_holdout"]
    mw = ds.df["mw"].to_numpy()
    assert np.all(mw[f.test] >= 7.0) and np.all(mw[f.train] < 7.0)


def test_cl1_to_cl2(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    (f,) = splits["cl1_to_cl2"]
    cl = ds.df["cl_class"].to_numpy()
    assert set(cl[f.test]) == {"CL2"} and set(cl[f.train]) == {"CL1"}


# ------------------------------------------------------------------ calibration
@pytest.mark.parametrize("scheme,key", [("event_grouped", "eqid"), ("station_grouped", "ssn"),
                                         ("temporal", "eqid"), ("spatial_block", "eqid")])
def test_calibration_groups_disjoint(ds: GMDataset, splits: dict[str, list[SplitFold]],
                                     scheme: str, key: str) -> None:
    for f in splits[scheme]:
        assert not (_keys(ds, key, f.cal) & _keys(ds, key, f.train_fit))
        share = f.cal.size / f.train.size
        assert 0.2 <= share <= 0.4, share


def test_calibration_random_is_rowwise(ds: GMDataset) -> None:
    train = np.arange(100)
    fit, cal = carve_calibration(ds, train, None, 0.25, seed=1)
    assert cal.size == 25 and fit.size == 75
    assert np.intersect1d(fit, cal).size == 0


def test_calibration_rejects_bad_fraction(ds: GMDataset) -> None:
    with pytest.raises(ValueError):
        carve_calibration(ds, np.arange(10), "eqid", 1.5)


# ------------------------------------------------------------------ audit catches leaks
def test_audit_flags_planted_event_leak(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    f = splits["event_grouped"][0]
    moved = f.test[:1]  # move one test record into train_fit: its event now straddles
    leaky = SplitFold(f.scheme, "leaky", np.sort(np.concatenate([f.train_fit, moved])),
                      f.cal, f.test[1:], f.excluded, f.group_key)
    r = audit_fold(ds, leaky)
    assert not r.passed and any("eqid shared by train_fit and test" in i for i in r.issues)


def test_audit_flags_missing_rows(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    f = splits["random"][0]
    broken = SplitFold(f.scheme, "broken", f.train_fit[1:], f.cal, f.test, f.excluded, None)
    r = audit_fold(ds, broken)
    assert not r.complete and not r.passed


def test_fit_transform_clean() -> None:
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[150:] += 5.0
    fit_idx = np.arange(150)
    assert_fit_transform_clean(StandardScaler().fit(X[fit_idx]), X, fit_idx)
    with pytest.raises(AssertionError, match="leakage"):
        assert_fit_transform_clean(StandardScaler().fit(X), X, fit_idx)


# ------------------------------------------------------------------ determinism + freezing
def test_deterministic(ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    again = make_splits(ds, seed=42, cal_frac=0.25)
    for name, folds in splits.items():
        for a, b in zip(folds, again[name]):
            for p in ("train_fit", "cal", "test", "excluded"):
                assert np.array_equal(getattr(a, p), getattr(b, p))


def test_json_round_trip(tmp_path, ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    path = save_splits(ds, splits, tmp_path / "splits_toy.json", seed=42, cal_frac=0.25)
    shuffled = GMDataset(ds.df.sample(frac=1.0, random_state=3).reset_index(drop=True), ds.profile)
    loaded = load_splits(path, shuffled)
    for name, folds in splits.items():
        for a, b in zip(folds, loaded[name]):
            for p in ("train_fit", "cal", "test", "excluded"):
                rsn_a = np.sort(ds.df["rsn"].to_numpy()[getattr(a, p)])
                rsn_b = np.sort(shuffled.df["rsn"].to_numpy()[getattr(b, p)])
                assert np.array_equal(rsn_a, rsn_b)
    assert all(r.passed for r in audit_splits(shuffled, loaded))


def test_load_rejects_other_dataset(tmp_path, ds: GMDataset, splits: dict[str, list[SplitFold]]) -> None:
    path = save_splits(ds, splits, tmp_path / "s.json", seed=42, cal_frac=0.25)
    with pytest.raises(ValueError, match="different dataset"):
        load_splits(path, ds.subset(np.arange(len(ds) - 1)))


def test_balanced_group_folds_assigns_every_group() -> None:
    groups = np.repeat(np.arange(20), 3)
    assign = balanced_group_folds(groups, groups % 2, 5, np.random.default_rng(0))
    assert set(assign) == set(range(20)) and set(assign.values()) == set(range(5))
