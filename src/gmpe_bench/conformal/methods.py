"""Uncertainty calibration methods, from the uncalibrated Gaussian baseline to HSCP.

All methods share one signature, which by construction never sees test targets:

    intervals = method.calibrate(cal: CalSet, test: TestInputs, alpha)

``CalSet`` carries labelled calibration records; ``TestInputs`` carries only
predictions and covariates for the test records (it has no ``y`` field and
refuses one).

Methods (ablation order)
------------------------
1. :class:`UncalibratedGaussian`    mu +- z_{1-alpha/2} sigma
2. :class:`NaiveSplitConformal`     marginal split conformal per IM (records exchangeable)
3. :class:`ClusteredEventConformal` each calibration *event* has total weight 1
                                    (pooled event CDFs, Dunn et al. 2023); test mass = one event
4. :class:`WeightedShiftConformal`  density-ratio weighted conformal (Tibshirani et al. 2019)
5. :class:`HierarchicalSpectralConformal` (HSCP), the proposed method:
   a. station terms dS2S cross-fitted on ``cal`` (event-grouped K-fold), so seen stations
      are re-centred and get scores free of their own site term; Mondrian quantiles
      separate seen and unseen stations;
   b. weights = density ratio x 1/n_event (shift-aware *and* cluster-aware);
   c. ln q(T) smoothed across ln T, plus an optional simultaneous band over the
      whole usable spectrum (max-over-spectrum score).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

from gmpe_bench.conformal.hierarchy import crossed_terms
from gmpe_bench.conformal.scores import abs_scores, normalized_scores, weighted_conformal_quantile
from gmpe_bench.conformal.shift import density_ratio_weights
from gmpe_bench.schema import IM_NAMES, period_of


# ------------------------------------------------------------------ containers
@dataclass
class CalSet:
    """Labelled calibration records (``cal`` part of a fold)."""

    y: np.ndarray                 # (n, K) ln observations, NaN where masked
    mu: np.ndarray                # (n, K)
    sigma: np.ndarray             # (n, K)
    eqid: np.ndarray              # (n,)
    ssn: np.ndarray               # (n,)
    X_shift: np.ndarray           # (n, p) physical covariates for density ratios
    train_stations: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    w_shift: np.ndarray | None = None   # optional precomputed density-ratio weights

    def __len__(self) -> int:
        return self.y.shape[0]


@dataclass
class TestInputs:
    """Predictions and covariates for test records. There is deliberately no ``y``."""

    __test__ = False  # not a pytest test class

    mu: np.ndarray
    sigma: np.ndarray
    eqid: np.ndarray
    ssn: np.ndarray
    X_shift: np.ndarray
    w_shift: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.mu.shape != self.sigma.shape:
            raise ValueError("mu and sigma must have the same shape")

    def __len__(self) -> int:
        return self.mu.shape[0]


@dataclass
class Intervals:
    method: str
    alpha: float
    center: np.ndarray            # (m, K)
    lower: np.ndarray
    upper: np.ndarray
    q: np.ndarray                 # (m, K) multiplier (score units)
    n_inf: int = 0                # thresholds that were +inf before capping
    joint_lower: np.ndarray | None = None
    joint_upper: np.ndarray | None = None
    seen: np.ndarray | None = None
    ess: float | None = None      # Kish effective sample size of shift weights


def shift_weights(cal: CalSet, test: TestInputs) -> tuple[np.ndarray, np.ndarray]:
    if cal.w_shift is not None and test.w_shift is not None:
        return cal.w_shift, test.w_shift
    return density_ratio_weights(cal.X_shift, test.X_shift)


def event_weights(eqid: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """(n, K) weight 1/n_e per record, n_e = finite records of that event for that IM."""
    codes, uniq = pd.factorize(np.asarray(eqid))
    A = sparse.csr_matrix((np.ones(codes.size), (codes, np.arange(codes.size))), shape=(uniq.size, codes.size))
    counts = (A @ finite.astype(float))[codes]
    with np.errstate(divide="ignore"):
        return np.where(finite & (counts > 0), 1.0 / counts, 0.0)


def smoothing_matrix(bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised Gaussian kernel over ln T for the SA IMs; returns (sa_idx, K)."""
    sa = np.array([i for i, n in enumerate(IM_NAMES) if period_of(n) is not None])
    lnT = np.log([period_of(IM_NAMES[i]) for i in sa])
    K = np.exp(-0.5 * ((lnT[:, None] - lnT[None, :]) / bandwidth) ** 2)
    return sa, K / K.sum(axis=1, keepdims=True)


def _cap_inf(q: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace +inf thresholds by the largest finite calibration score of that IM."""
    n_inf = int(np.isinf(q).sum())
    if n_inf:
        smax = np.nanmax(np.where(np.isfinite(scores), scores, np.nan), axis=0)
        smax = np.where(np.isfinite(smax), smax, 10.0)
        q = np.where(np.isinf(q), np.broadcast_to(smax, q.shape), q)
    return q, n_inf


# ------------------------------------------------------------------ base class
class ConformalCalibrator(ABC):
    name = "base"

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def scores(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return normalized_scores(y, mu, sigma) if self.normalize else abs_scores(y, mu)

    def _intervals(self, alpha: float, center: np.ndarray, q: np.ndarray, sigma: np.ndarray,
                   **extra: object) -> Intervals:
        half = q * sigma if self.normalize else q
        return Intervals(self.name, alpha, center, center - half, center + half, q, **extra)  # type: ignore[arg-type]

    @abstractmethod
    def calibrate(self, cal: CalSet, test: TestInputs, alpha: float) -> Intervals:
        """Intervals for every test record and IM at level 1 - alpha."""

    def calibrate_many(self, cal: CalSet, test: TestInputs, alphas: tuple[float, ...]) -> dict[float, Intervals]:
        return {a: self.calibrate(cal, test, a) for a in alphas}


# ------------------------------------------------------------------ 1-4
class UncalibratedGaussian(ConformalCalibrator):
    name = "UncalibratedGaussian"

    def __init__(self) -> None:
        super().__init__(normalize=True)

    def calibrate(self, cal: CalSet, test: TestInputs, alpha: float) -> Intervals:
        q = np.full(test.mu.shape, norm.ppf(1 - alpha / 2))
        return self._intervals(alpha, test.mu, q, test.sigma)


class NaiveSplitConformal(ConformalCalibrator):
    name = "NaiveSplitConformal"

    def calibrate(self, cal: CalSet, test: TestInputs, alpha: float) -> Intervals:
        s = self.scores(cal.y, cal.mu, cal.sigma)
        q = np.array([weighted_conformal_quantile(s[:, j], np.ones(len(cal)), 1.0, alpha)[0]
                      for j in range(s.shape[1])])
        q, n_inf = _cap_inf(np.broadcast_to(q, test.mu.shape).copy(), s)
        return self._intervals(alpha, test.mu, q, test.sigma, n_inf=n_inf)


class ClusteredEventConformal(ConformalCalibrator):
    name = "ClusteredEventConformal"

    def calibrate(self, cal: CalSet, test: TestInputs, alpha: float) -> Intervals:
        s = self.scores(cal.y, cal.mu, cal.sigma)
        w = event_weights(cal.eqid, np.isfinite(s))
        q = np.array([weighted_conformal_quantile(s[:, j], w[:, j], 1.0, alpha)[0] for j in range(s.shape[1])])
        q, n_inf = _cap_inf(np.broadcast_to(q, test.mu.shape).copy(), s)
        return self._intervals(alpha, test.mu, q, test.sigma, n_inf=n_inf)


class WeightedShiftConformal(ConformalCalibrator):
    name = "WeightedShiftConformal"

    def calibrate(self, cal: CalSet, test: TestInputs, alpha: float) -> Intervals:
        from gmpe_bench.conformal.shift import effective_sample_size

        s = self.scores(cal.y, cal.mu, cal.sigma)
        wc, wt = shift_weights(cal, test)
        q = np.column_stack([weighted_conformal_quantile(s[:, j], wc, wt, alpha) for j in range(s.shape[1])])
        q, n_inf = _cap_inf(q, s)
        return self._intervals(alpha, test.mu, q, test.sigma, n_inf=n_inf, ess=effective_sample_size(wc))


# ------------------------------------------------------------------ 5: HSCP
GROUP_CAL_TERM, GROUP_TRAIN_ONLY, GROUP_NEW = 0, 1, 2
GROUP_NAMES = {GROUP_CAL_TERM: "station term from cal", GROUP_TRAIN_ONLY: "station in train_fit only",
               GROUP_NEW: "new station"}


@dataclass
class _SiteFit:
    s_corr: np.ndarray       # (n, K) cal scores after out-of-fold station correction
    s_raw: np.ndarray        # (n, K) cal scores without correction
    found_cal: np.ndarray    # (n,) cal record has an out-of-fold station term
    train_cal: np.ndarray    # (n,) cal station appears in train_fit
    center: np.ndarray       # (m, K) test centre: mu + dS2S where the station has a cal term
    group_test: np.ndarray   # (m,) GROUP_* per test record
    seen_test: np.ndarray    # (m,) station in train_fit or cal


class HierarchicalSpectralConformal(ConformalCalibrator):
    """HSCP: hierarchical (event/station) + shift-weighted + spectrally smoothed conformal bands.

    Test records fall in one of three Mondrian groups, each calibrated on its own analog:

    * ``A`` station has a dS2S term estimated on ``cal`` -> centre mu + dS2S; calibrated on
      cal records whose term was *cross-fitted* from other cal events (corrected scores);
    * ``B`` station seen only in ``train_fit`` -> centre mu; raw scores of cal records at
      train_fit stations (the model's own station knowledge, no extra correction);
    * ``C`` new station -> centre mu; raw scores of cal records at stations absent from train_fit.

    Groups with fewer than ``min_group_events`` calibration events fall back to all raw scores.
    """

    name = "HSCP"

    def __init__(self, normalize: bool = True, site: bool = True, shift: bool = True, smooth: bool = True,
                 joint: bool = True, bandwidth: float = 0.35, n_crossfit: int = 5,
                 min_group_events: int = 10, test_mass: str = "record", seed: int = 0,
                 name: str | None = None) -> None:
        super().__init__(normalize)
        if test_mass not in ("record", "event"):
            raise ValueError("test_mass must be 'record' or 'event'")
        self.test_mass = test_mass
        self.site, self.shift, self.smooth, self.joint = site, shift, smooth, joint
        self.bandwidth, self.n_crossfit, self.min_group_events, self.seed = bandwidth, n_crossfit, min_group_events, seed
        if name:
            self.name = name

    # -- a. station terms, cross-fitted on cal --------------------------------
    def _site_fit(self, cal: CalSet, test: TestInputs) -> _SiteFit:
        raw = self.scores(cal.y, cal.mu, cal.sigma)
        train_cal = np.isin(cal.ssn, cal.train_stations)
        train_test = np.isin(test.ssn, cal.train_stations)
        if not self.site:
            group = np.where(train_test, GROUP_TRAIN_ONLY, GROUP_NEW)
            return _SiteFit(raw, raw, np.zeros(len(cal), bool), train_cal, test.mu.copy(), group, train_test)
        r = cal.y - cal.mu
        events = np.unique(cal.eqid)
        rng = np.random.default_rng(self.seed)
        fold_of = pd.Series(rng.permutation(events.size) % self.n_crossfit, index=events).reindex(cal.eqid).to_numpy()
        corr = np.zeros_like(r)
        found_cal = np.zeros(len(cal), dtype=bool)
        for k in range(self.n_crossfit):
            hold = fold_of == k
            if not hold.any() or hold.all():
                continue
            ct = crossed_terms(r[~hold], cal.eqid[~hold], cal.ssn[~hold], n_iter=30, tol=1e-4)
            corr[hold], found_cal[hold] = ct.station_lookup(cal.ssn[hold])
        ct_all = crossed_terms(r, cal.eqid, cal.ssn, n_iter=30, tol=1e-4)
        t_terms, t_found = ct_all.station_lookup(test.ssn)
        s_corr = self.scores(cal.y, cal.mu + corr, cal.sigma)
        group = np.where(t_found, GROUP_CAL_TERM, np.where(train_test, GROUP_TRAIN_ONLY, GROUP_NEW))
        return _SiteFit(s_corr, raw, found_cal, train_cal, test.mu + t_terms, group, t_found | train_test)

    def _analog(self, site: _SiteFit, g: int, eqid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(calibration rows, their scores) for Mondrian group ``g``."""
        if g == GROUP_CAL_TERM:
            rows, s = site.found_cal, site.s_corr
        elif g == GROUP_TRAIN_ONLY:
            rows, s = site.train_cal, site.s_raw
        else:
            rows, s = ~site.train_cal, site.s_raw
        if np.unique(eqid[rows]).size < self.min_group_events:
            rows, s = np.ones_like(rows), site.s_raw      # too few events: pool, uncorrected
        return rows, s[rows]

    # -- b. weighted Mondrian quantiles ---------------------------------------
    def _weights(self, s: np.ndarray, eqid: np.ndarray, w_shift: np.ndarray) -> np.ndarray:
        """Event x shift weights within one group (each event total 1 per IM).

        ``test_mass="record"`` rescales event weights to the group's record count, so the
        test point carries one record's mass (n_g + 1); ``"event"`` keeps one event's mass.
        """
        fin = np.isfinite(s)
        ev = event_weights(eqid, fin)
        if self.test_mass == "record":
            ev = ev * fin.sum(axis=0) / np.maximum(ev.sum(axis=0), 1e-12)
        return ev * w_shift[:, None]

    def _smooth(self, q: np.ndarray) -> np.ndarray:
        if not self.smooth:
            return q
        sa, Km = smoothing_matrix(self.bandwidth)
        out = q.copy()
        out[:, sa] = np.exp(np.log(np.clip(q[:, sa], 1e-6, None)) @ Km.T)
        return out

    def calibrate_many(self, cal: CalSet, test: TestInputs, alphas: tuple[float, ...]) -> dict[float, Intervals]:
        from gmpe_bench.conformal.shift import effective_sample_size

        site = self._site_fit(cal, test)
        wc_shift, wt_shift = shift_weights(cal, test) if self.shift else (np.ones(len(cal)), np.ones(len(test)))
        m, K = test.mu.shape
        out: dict[float, Intervals] = {}
        for alpha in alphas:
            q = np.full((m, K), np.inf)
            jl = np.full((m, K), np.nan)
            ju = np.full((m, K), np.nan)
            n_inf = 0
            for g in np.unique(site.group_test):
                rows_t = site.group_test == g
                rows_c, s_g = self._analog(site, g, cal.eqid)
                w = self._weights(s_g, cal.eqid[rows_c], wc_shift[rows_c])
                qg = np.column_stack([weighted_conformal_quantile(s_g[:, j], w[:, j], wt_shift[rows_t], alpha)
                                      for j in range(K)])
                qg, ni = _cap_inf(qg, s_g)
                n_inf += ni
                q[rows_t] = self._smooth(qg)
                if self.joint:
                    jl[rows_t], ju[rows_t] = self._joint(s_g, cal.eqid[rows_c], wc_shift[rows_c],
                                                         wt_shift[rows_t], site.center[rows_t],
                                                         test.sigma[rows_t], alpha)
            extra: dict[str, object] = {"n_inf": n_inf, "seen": site.seen_test,
                                        "ess": effective_sample_size(wc_shift)}
            if self.joint:
                extra.update({"joint_lower": jl, "joint_upper": ju})
            out[alpha] = self._intervals(alpha, site.center, q, test.sigma, **extra)
        return out

    def calibrate(self, cal: CalSet, test: TestInputs, alpha: float) -> Intervals:
        return self.calibrate_many(cal, test, (alpha,))[alpha]

    # -- c. simultaneous band over the usable spectrum -------------------------
    def _joint(self, s: np.ndarray, eqid: np.ndarray, w_shift: np.ndarray, w_test: np.ndarray,
               center: np.ndarray, sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Simultaneous band c(x) q0(T) sigma for one group.

        q0(T) is the group's smoothed marginal threshold (test mass 1); a calibration
        record's joint score is max_T s(T) / q0(T); c(x) is its weighted conformal quantile.
        """
        w = self._weights(s, eqid, w_shift)
        q0 = np.array([weighted_conformal_quantile(s[:, j], w[:, j], 1.0, alpha)[0] for j in range(s.shape[1])])
        q0 = self._smooth(_cap_inf(q0[None, :], s)[0])[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.isfinite(s), s / q0, -np.inf)
        S = ratio.max(axis=1)
        S = np.where(np.isfinite(S), S, np.nan)
        ev = event_weights(eqid, np.isfinite(S)[:, None])[:, 0]
        if self.test_mass == "record":
            ev = ev * np.isfinite(S).sum() / max(ev.sum(), 1e-12)
        c = weighted_conformal_quantile(S, ev * w_shift, w_test, alpha)
        c = np.where(np.isfinite(c), c, np.nanmax(S) if np.isfinite(S).any() else 10.0)
        half = c[:, None] * q0[None, :] * (sigma if self.normalize else 1.0)
        return center - half, center + half


# Names used in the paper / requested aliases
SplitConformal = NaiveSplitConformal
CovariateShiftConformal = WeightedShiftConformal
HSCP = HierarchicalSpectralConformal


def default_methods(ablations: bool = False) -> list[ConformalCalibrator]:
    methods: list[ConformalCalibrator] = [UncalibratedGaussian(), NaiveSplitConformal(),
                                          ClusteredEventConformal(), WeightedShiftConformal(), HSCP()]
    if ablations:
        methods += [HSCP(site=False, name="HSCP_noSite"), HSCP(shift=False, name="HSCP_noShift"),
                    HSCP(smooth=False, name="HSCP_noSmooth"), HSCP(normalize=False, name="HSCP_abs")]
    return methods
