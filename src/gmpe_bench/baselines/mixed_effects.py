"""Parametric mixed-effects GMM fitted on train_fit, with crossed event and station effects.

Functional form (linear in the coefficients, one fit per IM):

    mu = b0 + b1 (M-6) + b2 (M-6)^2 + b3 max(M - Mh, 0)
         + (b4 + b5 (M-6)) ln sqrt(R_JB^2 + h^2) + b6 R_JB
         + b7 ln(min(VS30, 1500) / 760) + b8 F_RV + b9 F_NM

with h = 6 km and Mh = 6.75. Fitting alternates generalised least squares for the
coefficients with one backfitting pass of the crossed random effects
(:class:`gmpe_bench.baselines.residuals.CrossedRE`) until both settle.
Unseen events and stations are predicted with zero random effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gmpe_bench.baselines.base import BaseGMM, target
from gmpe_bench.baselines.residuals import CrossedRE, ResidualDecomposition

FEATURES: tuple[str, ...] = ("const", "m", "m2", "m_hinge", "ln_r", "m_ln_r", "r", "ln_vs30", "f_rv", "f_nm")


def design_matrix(df: pd.DataFrame, h_km: float = 6.0, m_hinge: float = 6.75) -> np.ndarray:
    """Regressors in :data:`FEATURES` order; missing mechanism counts as strike-slip."""
    m = pd.to_numeric(df["mw"], errors="coerce").to_numpy(dtype=float) - 6.0
    rjb = pd.to_numeric(df["rjb"], errors="coerce").to_numpy(dtype=float)
    vs30 = pd.to_numeric(df["vs30"], errors="coerce").to_numpy(dtype=float)
    mech = pd.to_numeric(df.get("mechanism", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy()
    ln_r = np.log(np.sqrt(rjb**2 + h_km**2))
    return np.column_stack([
        np.ones_like(m), m, m**2, np.maximum(m + 6.0 - m_hinge, 0.0),
        ln_r, m * ln_r, rjb,
        np.log(np.minimum(vs30, 1500.0) / 760.0),
        np.isin(mech, [2, 3]).astype(float), np.isin(mech, [1, 4]).astype(float),
    ])


class MixedEffectsGMM(BaseGMM):
    """Statistical baseline: our functional form, our data, crossed random effects."""

    name = "MixedEffects"

    def __init__(self, h_km: float = 6.0, m_hinge: float = 6.75, tol: float = 1e-6,
                 max_iter: int = 300) -> None:
        super().__init__()
        self.h_km, self.m_hinge, self.tol, self.max_iter = h_km, m_hinge, tol, max_iter
        self.coef: np.ndarray | None = None
        self.decomposition: ResidualDecomposition | None = None
        self.phi_s2s: float | None = None
        self.phi_0: float | None = None
        self.converged = False

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return design_matrix(df, self.h_km, self.m_hinge)

    def fit(self, df_train: pd.DataFrame, im: str) -> MixedEffectsGMM:
        self.im = im
        X, y = self._X(df_train), target(df_train, im)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() <= X.shape[1] + 2:
            raise ValueError(f"{self.name}: too few usable records for {im} ({int(ok.sum())})")
        X, y = X[ok], y[ok]
        re = CrossedRE(df_train["eqid"].to_numpy()[ok], df_train["ssn"].to_numpy()[ok])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        re.init_variances(y - X @ coef)
        prev = np.concatenate([coef, re.params()])
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            re.step(y - X @ coef, fit_intercept=False)
            coef = np.linalg.lstsq(X, y - re.dB[re.e_codes] - re.dS[re.s_codes], rcond=None)[0]
            cur = np.concatenate([coef, re.params()])
            if np.max(np.abs(cur - prev)) < self.tol:
                self.converged = True
                break
            prev = cur
        self.coef = coef
        self.decomposition = re.result(n_iter, self.converged)
        d = self.decomposition
        self.tau, self.phi_s2s, self.phi_0, self.phi = d.tau, d.phi_s2s, d.phi_0, d.phi
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        assert self.coef is not None and self.decomposition is not None
        mu = self._X(df) @ self.coef
        return mu, np.full(len(df), self.decomposition.sigma)

    def coefficients(self) -> pd.Series:
        if self.coef is None:
            raise RuntimeError("not fitted")
        return pd.Series(self.coef, index=list(FEATURES), name=self.im)
