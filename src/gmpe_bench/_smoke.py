"""Installation smoke test: a tiny in-memory synthetic fixture. NOT NGA-West2 data.

Used only by ``gmpe-bench selftest`` and the test suite to check, on a new machine and without the PEER
flatfiles, that the package, the split leakage audit and the conformal calibrators run end to end.
The public benchmark API (:func:`gmpe_bench.evaluate`) never accepts this fixture: every frame it makes
carries a ``data_source`` column, which :func:`gmpe_bench.api.load_benchmark` rejects.

The fixture has the canonical columns of a built profile and a smooth, physically shaped ground-motion
model in ln units,

    ln y(T) = f(M, R, VS30; T) + dR_region(T) + dCL2(T) + dB_e(T) + dS2S_s(T) + dWS_es(T),

with crossed, period-correlated random effects (tau = 0.40, phi_S2S = 0.38, phi_0 = 0.45) and a
usable-period mask ``T > 1/LUF`` as in rule R5. Its numbers mean nothing about real ground motion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gmpe_bench.data import GMDataset
from gmpe_bench.data.features import z1_cy14, z2p5_cb14
from gmpe_bench.schema import IM_NAMES, im_column, period_of
from gmpe_bench.splits.group_splits import REGION_BOXES

TAU, PHI_S2S, PHI_0 = 0.40, 0.38, 0.45
REGION_SD = 0.22
SMOKE_MARKER = "smoke-test fixture (synthetic, not NGA-West2)"

# Region name, sampling weight, (lat, lon) box used for hypocentres and stations.
_OTHER_BOX = ("Other", -40.0, -20.0, -75.0, -65.0)
_REGION_WEIGHTS = {"California_WUS": 0.46, "Japan": 0.10, "Taiwan": 0.12, "China": 0.10,
                   "Europe_MiddleEast": 0.14, "Other": 0.08}


def _im_period(name: str) -> float:
    """Period used for the smooth coefficient functions (PGA ~ 0.01 s, PGV ~ 0.6 s)."""
    p = period_of(name)
    if p is not None:
        return p
    return 0.01 if name == "PGA" else 0.6


def _correlation(periods: np.ndarray, length: float = 1.3) -> np.ndarray:
    lt = np.log(periods)
    return np.exp(-np.abs(lt[:, None] - lt[None, :]) / length)


def _median(mw: np.ndarray, rjb: np.ndarray, ztor: np.ndarray, vs30: np.ndarray,
            names: list[str]) -> np.ndarray:
    """ln median (SA in g, PGV in cm/s) for every record and IM: shape (n, K)."""
    out = np.empty((mw.size, len(names)))
    m6 = mw - 6.0
    for k, name in enumerate(names):
        t = _im_period(name)
        lt = np.log(t)
        if name == "PGV":
            base, c1, c6 = np.log(18.0), 1.25, -0.62
        else:
            # spectral shape at M6, R = 10 km, VS30 = 760 m/s: peak near 0.2 s, decay beyond 0.5 s
            base = np.log(0.21) + 0.85 * np.exp(-(np.log(t / 0.2)) ** 2 / (2 * 0.75**2)) \
                - 1.15 * max(np.log(t / 0.5), 0.0)
            c1 = 1.05 + 0.30 * np.clip((lt - np.log(0.1)) / np.log(100.0), 0.0, 1.0) * 2.0
            c6 = -0.42 - 0.30 * np.exp(-(np.log(t / 0.8)) ** 2 / 2.0)
        c2 = -0.12
        c3 = -1.22 + 0.12 * np.clip(lt + 2.0, 0.0, 4.0) / 4.0
        c4 = 0.16
        c5 = -0.0045 * np.exp(-t / 0.8)
        h = 6.0 + 2.0 * np.clip(m6, 0.0, None)
        r = np.sqrt(rjb**2 + h**2)
        ln_r = np.log(r / 10.0)
        out[:, k] = (base + c1 * m6 + c2 * np.minimum(m6, 1.5) ** 2 + (c3 + c4 * m6) * ln_r
                     + c5 * (r - 10.0) + c6 * np.log(vs30 / 760.0) + 0.03 * (ztor - 3.0) / 5.0)
    return out


def make_fixture_frame(n_events: int = 40, n_stations: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Synthetic canonical frame (one row per record); installation smoke test only."""
    if n_events < 20:
        raise ValueError("n_events must be at least 20")
    rng = np.random.default_rng(seed)
    names = list(IM_NAMES)
    K = len(names)
    periods = np.array([_im_period(n) for n in names])
    chol = np.linalg.cholesky(_correlation(periods) + 1e-10 * np.eye(K))

    def corr_normal(n: int, sd: float) -> np.ndarray:
        return sd * rng.standard_normal((n, K)) @ chol.T

    # ---------------------------------------------------------------- regions and stations
    regions = list(_REGION_WEIGHTS)
    boxes = {b[0]: b[1:] for b in REGION_BOXES}
    boxes["Other"] = _OTHER_BOX[1:]
    p_reg = np.array([_REGION_WEIGHTS[r] for r in regions])
    n_stations = n_stations or int(2.6 * n_events)
    st_region = rng.choice(regions, size=n_stations, p=p_reg)
    st_vs30 = np.clip(np.exp(rng.normal(np.log(420.0), 0.45, n_stations)), 150.0, 1500.0)
    st_term = corr_normal(n_stations, PHI_S2S)
    region_term = {r: corr_normal(1, REGION_SD)[0] for r in regions}
    region_term["California_WUS"] = np.zeros(K)  # reference region

    # ---------------------------------------------------------------- earthquakes
    n_big = max(3, round(0.07 * n_events))
    mw = np.concatenate([3.8 + 3.1 * rng.beta(1.15, 1.6, n_events - n_big), rng.uniform(7.0, 7.8, n_big)])
    rng.shuffle(mw)
    ev_region = rng.choice(regions, size=n_events, p=p_reg)
    for r in regions if n_events >= 100 else ():  # larger fixtures keep >= 10 events per region
        need = 10 - int((ev_region == r).sum())
        if need > 0 and r != "Other":
            idx = rng.choice(np.flatnonzero(ev_region == "California_WUS"), size=need, replace=False)
            ev_region[idx] = r
    year = rng.integers(1971, 2016, n_events)
    cl2 = rng.random(n_events) < 0.22
    cl2 &= mw < 7.0                       # aftershocks are smaller here
    mech = rng.choice([0, 1, 2, 3, 4], size=n_events, p=[0.45, 0.12, 0.30, 0.08, 0.05])
    rake = np.select([mech == 0, np.isin(mech, (1, 4)), np.isin(mech, (2, 3))],
                     [rng.uniform(-20, 20, n_events), rng.uniform(-110, -70, n_events),
                      rng.uniform(70, 110, n_events)])
    dip = np.where(mech == 0, 90.0, np.where(np.isin(mech, (1, 4)), 55.0, 40.0)) + rng.normal(0, 5, n_events)
    hypo_depth = np.clip(rng.gamma(4.0, 2.6, n_events), 2.0, 25.0)
    width = np.clip(10 ** (-1.01 + 0.32 * mw), 1.0, 40.0)
    ztor = np.clip(hypo_depth - 0.5 * width * np.sin(np.radians(dip)), 0.0, 15.0)
    ev_term = corr_normal(n_events, TAU)
    cl2_term = -0.18 * np.exp(-(np.log(periods / 0.1)) ** 2 / 4.0)  # lower short-period motion

    hypo_lat = np.empty(n_events)
    hypo_lon = np.empty(n_events)
    for i, r in enumerate(ev_region):
        la0, la1, lo0, lo1 = boxes[r]
        hypo_lat[i], hypo_lon[i] = rng.uniform(la0, la1), rng.uniform(lo0, lo1)

    # ---------------------------------------------------------------- records
    rows = []
    for e in range(n_events):
        pool = np.flatnonzero(st_region == ev_region[e])
        if pool.size == 0:
            pool = np.arange(n_stations)
        n_rec = int(np.clip(rng.poisson(6 + 9.0 * (mw[e] - 3.8)), 3, pool.size))
        for s in rng.choice(pool, size=n_rec, replace=False):
            rows.append((e, s))
    ev_idx = np.array([r[0] for r in rows])
    st_idx = np.array([r[1] for r in rows])
    n = ev_idx.size

    rmax = 40.0 + 70.0 * np.clip(mw[ev_idx] - 3.8, 0.0, None)
    rjb = np.exp(rng.uniform(np.log(0.5), np.log(rmax)))
    rjb[rng.random(n) < 0.04] = 0.0
    rrup = np.sqrt(rjb**2 + ztor[ev_idx] ** 2)
    rx = rjb * rng.choice([-1.0, 1.0], n)
    ry0 = np.abs(rng.normal(0.0, 0.3 * rjb + 1.0))
    vs30 = st_vs30[st_idx]
    japan = pd.Series(ev_region[ev_idx] == "Japan")
    z1 = z1_cy14(pd.Series(vs30), japan).to_numpy()
    z2p5 = z2p5_cb14(pd.Series(vs30), japan).to_numpy()

    ln_y = (_median(mw[ev_idx], rjb, ztor[ev_idx], vs30, names)
            + np.stack([region_term[r] for r in ev_region[ev_idx]])
            + np.where(cl2[ev_idx, None], cl2_term[None, :], 0.0)
            + ev_term[ev_idx] + st_term[st_idx] + corr_normal(n, PHI_0))

    # usable-period mask (rule R5): larger earthquakes have lower LUF
    luf = np.exp(rng.uniform(np.log(0.08), np.log(0.6), n) - 0.35 * np.clip(mw[ev_idx] - 5.5, 0, None))
    t_max = 1.0 / luf
    sa_period = np.array([period_of(nm) if period_of(nm) is not None else 0.0 for nm in names])
    masked = sa_period[None, :] > t_max[:, None]
    y_lin = np.where(masked, np.nan, np.exp(ln_y))
    ln_vh = -0.45 + 0.25 * np.tanh(np.log(np.where(sa_period > 0, sa_period, 0.01) / 0.3)) \
        + 0.25 * rng.standard_normal((n, K))
    v_lin = y_lin * np.exp(ln_vh)

    eq_names = np.array([f"Synthetic earthquake {e:03d}" + (" Japan" if ev_region[e] == "Japan" else "")
                         for e in range(n_events)])
    df = pd.DataFrame({
        "rsn": np.arange(1, n + 1, dtype=np.int64),
        "eqid": (ev_idx + 1).astype(np.int64),
        "ssn": (st_idx + 1001).astype(np.int64),
        "eq_name": eq_names[ev_idx],
        "year": year[ev_idx].astype(np.int64),
        "mw": np.round(mw[ev_idx], 2),
        "mechanism": mech[ev_idx].astype(np.int64),
        "rake": rake[ev_idx], "dip": dip[ev_idx], "width": width[ev_idx], "ztor": ztor[ev_idx],
        "hypo_lat": hypo_lat[ev_idx], "hypo_lon": hypo_lon[ev_idx], "hypo_depth": hypo_depth[ev_idx],
        "rjb": rjb, "rrup": rrup, "rx": rx, "ry0": ry0,
        "vs30": vs30, "vs30_measured": (rng.random(n) < 0.4).astype(float),
        "z1": z1, "z2p5": z2p5,
        "sta_lat": hypo_lat[ev_idx] + rng.normal(0, 0.5, n), "sta_lon": hypo_lon[ev_idx] + rng.normal(0, 0.5, n),
        "cl_class": np.where(cl2[ev_idx], "CL2", "CL1"),
        "luf_h": luf,
        "region_true": ev_region[ev_idx],
        "data_source": SMOKE_MARKER,
    })
    targets = {}
    for k, name in enumerate(names):
        targets[im_column("H", name)] = y_lin[:, k]
        targets[im_column("V", name)] = v_lin[:, k]
        targets[im_column("VH", name)] = np.where(np.isfinite(y_lin[:, k]), ln_vh[:, k], np.nan)
    return pd.concat([df, pd.DataFrame(targets)], axis=1)


def make_fixture(n_events: int = 40, seed: int = 0) -> GMDataset:
    """Tiny in-memory fixture as a :class:`GMDataset` (deterministic for a seed). Smoke test only."""
    return GMDataset(make_fixture_frame(n_events, seed=seed), "smoke")


SMOKE_BANNER = (
    "INSTALLATION SMOKE TEST: tiny synthetic in-memory fixture. These numbers are NOT NGA-West2\n"
    "benchmark output and must not be reported. For real results build the PEER profile\n"
    "(gmpe-bench build) and call gmpe_bench.evaluate()."
)


def fixture_bench(schemes: tuple[str, ...] = ("event_grouped", "random"), n_events: int = 40, seed: int = 0):
    """Fixture with fresh, seeded splits wrapped for the API engines (tests and selftest only)."""
    from gmpe_bench import api
    from gmpe_bench.splits import make_splits

    ds = make_fixture(n_events, seed)
    return api._Bench(ds, make_splits(ds, list(schemes), seed=42, cal_frac=0.25), "fixture", None)


def run_smoke_test(n_events: int = 40, seed: int = 0, log=print) -> dict[str, object]:
    """Fixture -> splits -> leakage audit -> LinearGMM -> event-clustered CP and HSCP. Returns a summary."""
    from gmpe_bench import api
    from gmpe_bench.splits import audit_splits

    bench = fixture_bench(("event_grouped", "random"), n_events, seed)
    ds, splits = bench.ds, bench.splits
    reports = audit_splits(ds, splits)
    audit_ok = all(r.passed for r in reports)
    if not audit_ok:
        raise RuntimeError("leakage audit failed on the smoke-test fixture")
    shared = max(r.shared_events for r in reports if r.scheme == "event_grouped")
    rep = api._run_model(bench, api.LinearGMM(), "smoke", ("event_grouped",), True, 0, (0.10,),
                         api.DEFAULT_METHODS, api.BOOT_SEED, None, None)
    cal = rep.calibration.set_index("method")
    out = {"records": len(ds), "events": int(ds.df["eqid"].nunique()), "folds_audited": len(reports),
           "audit_passed": audit_ok, "event_split_shared_events": int(shared),
           "rmse": float(rep.accuracy["rmse"].iloc[0]),
           "coverage": {m: float(cal.loc[m, "coverage"]) for m in cal.index},
           "width": {m: float(cal.loc[m, "width"]) for m in cal.index}}
    return out
