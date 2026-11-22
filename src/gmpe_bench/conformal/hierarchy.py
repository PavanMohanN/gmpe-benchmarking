"""Crossed event/station effects for many IMs at once (vectorised, NaN-aware).

Same model and EM-style updates as :func:`gmpe_bench.baselines.decompose_residuals`,
but on an (n, k) residual matrix with sparse indicator products, so all 27 IMs
are decomposed in one pass. Used by HSCP to cross-fit station terms on ``cal``
and by the evaluation to recompute classical-model bias terms on ``train_fit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

VAR_FLOOR = 1e-8


@dataclass
class CrossedTerms:
    intercept: np.ndarray       # (k,)
    event_ids: np.ndarray       # (E,)
    event_terms: np.ndarray     # (E, k)  dB
    station_ids: np.ndarray     # (S,)
    station_terms: np.ndarray   # (S, k)  dS2S
    tau: np.ndarray             # (k,)
    phi_s2s: np.ndarray         # (k,)
    phi_0: np.ndarray           # (k,)

    def station_lookup(self, ssn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(terms (n, k), found (n,)); unseen stations get 0."""
        pos = pd.Series(np.arange(self.station_ids.size), index=self.station_ids)
        idx = pos.reindex(np.asarray(ssn)).to_numpy()
        found = ~np.isnan(idx)
        out = np.zeros((idx.size, self.station_terms.shape[1]))
        out[found] = self.station_terms[idx[found].astype(int)]
        return out, found


def _indicator(codes: np.ndarray, n_groups: int) -> sparse.csr_matrix:
    n = codes.size
    return sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n))


def crossed_terms(resid: np.ndarray, eqid: np.ndarray, ssn: np.ndarray, fit_intercept: bool = True,
                  n_iter: int = 100, tol: float = 1e-6) -> CrossedTerms:
    """Decompose an (n, k) residual matrix; NaN entries are ignored per column."""
    r = np.atleast_2d(np.asarray(resid, dtype=float))
    if r.shape[0] != np.asarray(eqid).size:
        r = r.T
    n, k = r.shape
    m = np.isfinite(r)
    r0 = np.where(m, r, 0.0)
    mf = m.astype(float)
    e_codes, e_ids = pd.factorize(np.asarray(eqid), sort=True)
    s_codes, s_ids = pd.factorize(np.asarray(ssn), sort=True)
    Ae, As = _indicator(e_codes, e_ids.size), _indicator(s_codes, s_ids.size)
    ne, ns = Ae @ mf, As @ mf                       # (E, k), (S, k) record counts
    cnt = np.maximum(mf.sum(axis=0), 1.0)

    var = np.nanvar(np.where(m, r, np.nan), axis=0) if m.any() else np.ones(k)
    var = np.where(np.isfinite(var) & (var > 0), var, 1.0)
    tau2, phis2, phi02 = var / 3, var / 3, var / 3
    c = np.zeros(k)
    dB = np.zeros((e_ids.size, k))
    dS = np.zeros((s_ids.size, k))
    prev = np.concatenate([c, tau2, phis2, phi02])
    for _ in range(n_iter):
        Bx, Sx = dB[e_codes] * mf, dS[s_codes] * mf
        if fit_intercept:
            c = (r0 - Bx - Sx).sum(axis=0) / cnt
        cm = c * mf
        dB = (Ae @ (r0 - cm - Sx)) / (ne + phi02 / tau2)
        ve = 1.0 / (ne / phi02 + 1.0 / tau2)
        Bx = dB[e_codes] * mf
        dS = (As @ (r0 - cm - Bx)) / (ns + phi02 / phis2)
        vs = 1.0 / (ns / phi02 + 1.0 / phis2)
        eps = (r0 - cm - Bx - dS[s_codes] * mf) * mf
        has_e, has_s = ne > 0, ns > 0
        tau2 = np.maximum(np.where(has_e, dB**2 + ve, 0).sum(0) / np.maximum(has_e.sum(0), 1), VAR_FLOOR)
        phis2 = np.maximum(np.where(has_s, dS**2 + vs, 0).sum(0) / np.maximum(has_s.sum(0), 1), VAR_FLOOR)
        phi02 = np.maximum(((eps**2).sum(0) + (ne * ve).sum(0) + (ns * vs).sum(0)) / cnt, VAR_FLOOR)
        cur = np.concatenate([c, tau2, phis2, phi02])
        if np.max(np.abs(cur - prev)) < tol:
            break
        prev = cur
    return CrossedTerms(c, np.asarray(e_ids), dB, np.asarray(s_ids), dS,
                        np.sqrt(tau2), np.sqrt(phis2), np.sqrt(phi02))
