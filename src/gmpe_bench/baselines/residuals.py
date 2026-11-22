"""Crossed random-effects decomposition of ln residuals (Al Atik et al. 2010 notation).

    R_es = c + dB_e + dS2S_s + dW0_es
    dB_e ~ N(0, tau^2),  dS2S_s ~ N(0, phi_S2S^2),  dW0_es ~ N(0, phi_0^2)

Estimated by an EM-style backfitting loop: each pass updates the intercept,
the event terms (BLUP shrinkage), the station terms, then the three variances.
Cost per pass is O(n) via ``np.bincount``, so 20k records converge in milliseconds.

Leakage rule: a decomposition is always estimated on ``train_fit`` residuals.
Applied to new records, an unseen event or station gets a term of 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VAR_FLOOR = 1e-8


@dataclass
class ResidualDecomposition:
    intercept: float
    tau: float
    phi_s2s: float
    phi_0: float
    event_terms: pd.Series      # dB_e, indexed by EQID
    station_terms: pd.Series    # dS2S_s, indexed by SSN
    n: int
    n_iter: int
    converged: bool

    @property
    def phi(self) -> float:
        """Within-event standard deviation: sqrt(phi_S2S^2 + phi_0^2)."""
        return float(np.hypot(self.phi_s2s, self.phi_0))

    @property
    def sigma(self) -> float:
        """Total standard deviation: sqrt(tau^2 + phi^2)."""
        return float(np.hypot(self.tau, self.phi))

    def seen_station(self, ssn: np.ndarray) -> np.ndarray:
        return np.isin(np.asarray(ssn), self.station_terms.index.to_numpy())

    def dS2S(self, ssn: np.ndarray) -> np.ndarray:
        """Station terms for ``ssn``; 0 for stations not seen in training."""
        return self.station_terms.reindex(np.asarray(ssn)).fillna(0.0).to_numpy(dtype=float)

    def dB(self, eqid: np.ndarray) -> np.ndarray:
        """Event terms for ``eqid``; 0 for events not seen in training."""
        return self.event_terms.reindex(np.asarray(eqid)).fillna(0.0).to_numpy(dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {"intercept": self.intercept, "tau": self.tau, "phi_s2s": self.phi_s2s,
                "phi_0": self.phi_0, "phi": self.phi, "sigma": self.sigma, "n": self.n}


class CrossedRE:
    """State of the backfitting loop; reused by :class:`MixedEffectsGMM`."""

    def __init__(self, eqid: np.ndarray, ssn: np.ndarray) -> None:
        self.e_codes, self.e_keys = pd.factorize(np.asarray(eqid), sort=True)
        self.s_codes, self.s_keys = pd.factorize(np.asarray(ssn), sort=True)
        self.n_e = np.bincount(self.e_codes).astype(float)
        self.n_s = np.bincount(self.s_codes).astype(float)
        self.n = self.e_codes.size
        self.dB = np.zeros(self.n_e.size)
        self.dS = np.zeros(self.n_s.size)
        self.c = 0.0
        self.tau2 = self.phis2 = self.phi02 = 0.1

    def init_variances(self, r: np.ndarray) -> None:
        v = max(float(np.var(r)), 3 * VAR_FLOOR)
        self.tau2 = self.phis2 = self.phi02 = v / 3.0

    def step(self, r: np.ndarray, fit_intercept: bool = True) -> None:
        e, s = self.e_codes, self.s_codes
        if fit_intercept:
            self.c = float(np.mean(r - self.dB[e] - self.dS[s]))
        rb = np.bincount(e, weights=r - self.c - self.dS[s], minlength=self.n_e.size)
        self.dB = rb / (self.n_e + self.phi02 / self.tau2)
        v_e = 1.0 / (self.n_e / self.phi02 + 1.0 / self.tau2)
        rs = np.bincount(s, weights=r - self.c - self.dB[e], minlength=self.n_s.size)
        self.dS = rs / (self.n_s + self.phi02 / self.phis2)
        v_s = 1.0 / (self.n_s / self.phi02 + 1.0 / self.phis2)
        eps = r - self.c - self.dB[e] - self.dS[s]
        self.tau2 = max(float(np.mean(self.dB**2 + v_e)), VAR_FLOOR)
        self.phis2 = max(float(np.mean(self.dS**2 + v_s)), VAR_FLOOR)
        self.phi02 = max(float((np.sum(eps**2) + np.sum(self.n_e * v_e) + np.sum(self.n_s * v_s)) / self.n),
                         VAR_FLOOR)

    def params(self) -> np.ndarray:
        return np.array([self.c, self.tau2, self.phis2, self.phi02])

    def result(self, n_iter: int, converged: bool) -> ResidualDecomposition:
        return ResidualDecomposition(
            intercept=self.c, tau=float(np.sqrt(self.tau2)), phi_s2s=float(np.sqrt(self.phis2)),
            phi_0=float(np.sqrt(self.phi02)),
            event_terms=pd.Series(self.dB, index=self.e_keys, name="dB"),
            station_terms=pd.Series(self.dS, index=self.s_keys, name="dS2S"),
            n=self.n, n_iter=n_iter, converged=converged,
        )


def _empty() -> ResidualDecomposition:
    nan = float("nan")
    return ResidualDecomposition(nan, nan, nan, nan, pd.Series(dtype=float), pd.Series(dtype=float), 0, 0, False)


def decompose_residuals(resid: np.ndarray, eqid: np.ndarray, ssn: np.ndarray, fit_intercept: bool = True,
                        tol: float = 1e-6, max_iter: int = 500) -> ResidualDecomposition:
    """Split residuals into intercept, dB_e, dS2S_s and dW0; missing residuals are ignored."""
    r = np.asarray(resid, dtype=float)
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return _empty()
    re = CrossedRE(np.asarray(eqid)[ok], np.asarray(ssn)[ok])
    r = r[ok]
    re.init_variances(r)
    prev = re.params()
    for i in range(1, max_iter + 1):
        re.step(r, fit_intercept)
        cur = re.params()
        if np.max(np.abs(cur - prev)) < tol:
            return re.result(i, True)
        prev = cur
    return re.result(max_iter, False)


def apply_site_terms(mu: np.ndarray, ssn: np.ndarray, dec: ResidualDecomposition
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Site-corrected prediction from a train_fit decomposition.

    Returns ``(mu + dS2S, sigma, seen)``: seen stations use sqrt(tau^2 + phi_0^2);
    unseen stations get dS2S = 0 and the full sqrt(tau^2 + phi_S2S^2 + phi_0^2).
    """
    seen = dec.seen_station(ssn)
    mu_site = np.asarray(mu, dtype=float) + dec.dS2S(ssn)
    sigma = np.where(seen, np.hypot(dec.tau, dec.phi_0), dec.sigma)
    return mu_site, sigma, seen
