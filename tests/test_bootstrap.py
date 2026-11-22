"""Event-clustered bootstrap: resampling unit, exact point estimates, and per-event coverage counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gmpe_bench.conformal.evaluate import event_coverage_counts
from gmpe_bench.conformal.methods import Intervals
from gmpe_bench.reporting.bootstrap import _coverage_stat, _rmse_stat, _scatter, cluster_weights
from gmpe_bench.schema import IM_NAMES

K = len(IM_NAMES)


def test_cluster_weights_resample_whole_clusters() -> None:
    w = cluster_weights(37, 500, np.random.default_rng(0))
    assert w.shape == (500, 37) and (w.sum(axis=1) == 37).all() and (w >= 0).all()
    assert np.allclose(w, np.round(w))                     # integer multiplicities
    assert 0.3 < (w == 0).mean() < 0.43                     # ~ (1 - 1/n)^n of clusters left out
    assert np.array_equal(w, cluster_weights(37, 500, np.random.default_rng(0)))


def _frame(rng: np.random.Generator, n_ev: int = 12) -> pd.DataFrame:
    rows = []
    for fold in ("f0", "f1"):
        for e in range(n_ev):
            row = {"fold": fold, "eqid": 100 + e}
            n = rng.integers(1, 6, K).astype(float)
            n[rng.random(K) < 0.1] = 0
            row.update({f"n_{im}": n[j] for j, im in enumerate(IM_NAMES)})
            row.update({f"sse_{im}": n[j] * rng.uniform(0.2, 0.8) for j, im in enumerate(IM_NAMES)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_rmse_point_estimate_is_the_table_statistic() -> None:
    f = _frame(np.random.default_rng(1))
    events = pd.Index(sorted(f["eqid"].unique()))
    sse, n = _scatter(f, events, [("f0",), ("f1",)], ["fold"], ("sse", "n"))
    valid = n.sum(axis=0) > 0
    got = _rmse_stat(np.ones((1, len(events))), sse, n, valid)[0]
    direct = []
    for fold, g in f.groupby("fold"):
        for im in IM_NAMES:
            if g[f"n_{im}"].sum() > 0:
                direct.append(np.sqrt(g[f"sse_{im}"].sum() / g[f"n_{im}"].sum()))
    assert got == pytest.approx(np.mean(direct), rel=1e-12)


def test_weight_two_equals_duplicating_an_earthquake() -> None:
    f = _frame(np.random.default_rng(2))
    events = pd.Index(sorted(f["eqid"].unique()))
    cells = [("f0",), ("f1",)]
    sse, n = _scatter(f, events, cells, ["fold"], ("sse", "n"))
    valid = n.sum(axis=0) > 0
    w = np.ones((1, len(events)))
    w[0, 3] = 2.0
    dup = pd.concat([f, f[f["eqid"] == events[3]]], ignore_index=True)
    sse2, n2 = _scatter(dup, events, cells, ["fold"], ("sse", "n"))
    assert _rmse_stat(w, sse, n, valid)[0] == pytest.approx(_rmse_stat(np.ones((1, len(events))), sse2, n2, valid)[0])


def test_event_counts_reproduce_interval_coverage() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=(60, K))
    y[rng.random(y.shape) < 0.15] = np.nan
    lo, hi = np.full_like(y, -1.6), np.full_like(y, 1.6)
    iv = Intervals("test", 0.1, np.zeros_like(y), lo, hi, np.full_like(y, 1.6))
    eq = rng.integers(0, 9, 60)
    c = event_coverage_counts(y, iv, eq)
    assert c["eqid"].is_unique and len(c) == len(np.unique(eq))
    fin = np.isfinite(y)
    covered = ((y >= lo) & (y <= hi) & fin).sum(axis=0)
    assert np.array_equal(c[[f"cov_{im}" for im in IM_NAMES]].to_numpy().sum(axis=0), covered)
    assert np.array_equal(c[[f"n_{im}" for im in IM_NAMES]].to_numpy().sum(axis=0), fin.sum(axis=0))
    # Table 4 'ALL' for one (fold, model) cell: mean over IMs of per-IM coverage
    frame = c.assign(fold="f0", model="m")
    events = pd.Index(sorted(frame["eqid"]))
    cov, nn = _scatter(frame, events, [("f0", "m")], ["fold", "model"], ("cov", "n"))
    expected = np.nanmean(covered / np.where(fin.sum(0) > 0, fin.sum(0), np.nan))
    assert _coverage_stat(np.ones((1, len(events))), cov, nn, 1)[0] == pytest.approx(expected, abs=1e-15)
