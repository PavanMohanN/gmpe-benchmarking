"""Leakage-free features for ML ground-motion models.

:class:`FeaturePipeline` learns everything (medians, scales, region levels,
station terms) from ``train_fit`` only. It follows the scikit-learn estimator
contract, so :func:`gmpe_bench.splits.assert_fit_transform_clean` can check it.

Station encoding (non-ergodic site term)
---------------------------------------
For each anchor IM a ridge model on the non-station features is fitted to
``train_fit``; its residuals are split into dB_e, dS2S_s and dW0 with
:func:`gmpe_bench.baselines.decompose_residuals` (BLUP shrinkage = regularisation).
``dS2S_s`` becomes a feature: 0 for stations unseen in ``train_fit``.
Training rows get *out-of-fold* station terms (event-grouped K-fold inside
``train_fit``) so a record never sees its own residual.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from gmpe_bench.baselines.base import target
from gmpe_bench.baselines.residuals import decompose_residuals
from gmpe_bench.splits.group_splits import assign_region

RRUP_C_KM = 5.0
BASE_NUMERIC: tuple[str, ...] = (
    "mw", "mw2", "rjb", "rrup", "ln_rrup", "mw_ln_rrup", "rx", "ry0", "vs30", "ln_vs30",
    "z1p0", "z2p5", "ztor", "dip", "rake", "width", "hw_flag", "sof_ss", "sof_nm", "sof_rv",
    "vs30_measured",
)
MISSING_FLAGS: tuple[str, ...] = ("rx", "ry0", "z1p0", "z2p5", "ztor", "dip", "width", "vs30_measured")
ANCHOR_IMS: tuple[str, ...] = ("PGA", "SA_0.2s", "SA_1s", "SA_3s")
_SOF = {0: "ss", 1: "nm", 4: "nm", 2: "rv", 3: "rv"}


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seismological predictors before imputation and scaling."""
    mw, rrup = _num(df, "mw"), _num(df, "rrup")
    dip, rx = _num(df, "dip"), _num(df, "rx")
    ln_rrup = np.log(np.clip(rrup, 0.0, None) + RRUP_C_KM)
    mech = pd.Series(_num(df, "mechanism")).map(_SOF)
    rake = _num(df, "rake")
    rake_sof = np.where((rake >= 30) & (rake <= 150), "rv", np.where((rake <= -30) & (rake >= -150), "nm", "ss"))
    sof = mech.where(mech.notna(), pd.Series(rake_sof)).to_numpy()
    out = {
        "mw": mw, "mw2": mw**2, "rjb": _num(df, "rjb"), "rrup": rrup, "ln_rrup": ln_rrup,
        "mw_ln_rrup": mw * ln_rrup, "rx": rx, "ry0": _num(df, "ry0"),
        "vs30": _num(df, "vs30"), "ln_vs30": np.log(_num(df, "vs30")),
        "z1p0": _num(df, "z1"), "z2p5": _num(df, "z2p5"), "ztor": _num(df, "ztor"),
        "dip": dip, "rake": rake, "width": _num(df, "width"),
        "hw_flag": ((rx >= 0) & (np.nan_to_num(dip, nan=90.0) < 90.0)).astype(float),
        "sof_ss": (sof == "ss").astype(float), "sof_nm": (sof == "nm").astype(float),
        "sof_rv": (sof == "rv").astype(float), "vs30_measured": _num(df, "vs30_measured"),
    }
    return pd.DataFrame(out, index=df.index)


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xa = np.column_stack([np.ones(len(X)), X])
    reg = alpha * np.eye(Xa.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y)


def _ridge_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Impute, scale, one-hot regions and (optionally) encode station terms.

    Learned attributes (all end in ``_``): ``medians_``, ``means_``, ``scales_``,
    ``regions_``, ``station_ids_``, ``station_terms_``, ``feature_names_``.
    """

    def __init__(self, station_encoding: bool = True, anchor_ims: tuple[str, ...] = ANCHOR_IMS,
                 n_oof_folds: int = 5, ridge_alpha: float = 1.0, seed: int = 0) -> None:
        self.station_encoding = station_encoding
        self.anchor_ims = anchor_ims
        self.n_oof_folds = n_oof_folds
        self.ridge_alpha = ridge_alpha
        self.seed = seed

    # ---------------------------------------------------------------- base block
    def _base(self, df: pd.DataFrame) -> np.ndarray:
        raw = raw_features(df)
        flags = raw[list(MISSING_FLAGS)].isna().to_numpy(dtype=float)
        vals = raw[list(BASE_NUMERIC)].to_numpy(dtype=float)
        vals = np.where(np.isnan(vals), self.medians_, vals)
        vals = (vals - self.means_) / self.scales_
        region = assign_region(df).to_numpy()
        onehot = np.column_stack([(region == r).astype(float) for r in self.regions_]) \
            if self.regions_ else np.zeros((len(df), 0))
        unseen = (~np.isin(region, list(self.regions_))).astype(float)[:, None]
        return np.column_stack([vals, flags, onehot, unseen])

    def _fit_station_terms(self, df: pd.DataFrame, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Station ids and dS2S per anchor IM from residuals of a ridge on the base block."""
        terms = []
        for im in self.anchor_ims:
            y = target(df, im)
            ok = np.isfinite(y)
            if ok.sum() < X.shape[1] + 5:
                terms.append(pd.Series(dtype=float))
                continue
            coef = _ridge(X[ok], y[ok], self.ridge_alpha)
            r = np.full(len(df), np.nan)
            r[ok] = y[ok] - _ridge_predict(coef, X[ok])
            dec = decompose_residuals(r, df["eqid"].to_numpy(), df["ssn"].to_numpy())
            terms.append(dec.station_terms)
        ids = np.unique(np.concatenate([t.index.to_numpy() for t in terms] + [np.array([], dtype=np.int64)]))
        mat = np.column_stack([t.reindex(ids).fillna(0.0).to_numpy() for t in terms]) \
            if terms else np.zeros((ids.size, 0))
        return ids.astype(np.int64), mat

    def _lookup(self, ssn: np.ndarray, ids: np.ndarray, mat: np.ndarray) -> np.ndarray:
        pos = pd.Series(np.arange(ids.size), index=ids)
        idx = pos.reindex(ssn).to_numpy()
        out = np.zeros((ssn.size, mat.shape[1]))
        hit = ~np.isnan(idx)
        out[hit] = mat[idx[hit].astype(int)]
        return out

    def _station_block(self, ssn: np.ndarray, ids: np.ndarray, mat: np.ndarray) -> np.ndarray:
        seen = np.isin(ssn, ids).astype(float)[:, None]
        return np.column_stack([self._lookup(ssn, ids, mat), seen])

    # ---------------------------------------------------------------- sklearn API
    def fit(self, df: pd.DataFrame, y: object = None) -> FeaturePipeline:
        raw = raw_features(df)[list(BASE_NUMERIC)].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-missing columns -> median 0
            med = np.nanmedian(raw, axis=0)
        self.medians_ = np.where(np.isnan(med), 0.0, med)
        filled = np.where(np.isnan(raw), self.medians_, raw)
        self.means_ = filled.mean(axis=0)
        sd = filled.std(axis=0)
        self.scales_ = np.where(sd > 1e-12, sd, 1.0)
        self.regions_ = tuple(sorted(set(assign_region(df).to_numpy())))  # tuple: not compared numerically
        base = self._base(df)
        names = list(BASE_NUMERIC) + [f"{c}_missing" for c in MISSING_FLAGS] \
            + [f"region_{r}" for r in self.regions_] + ["region_unseen"]
        if self.station_encoding:
            self.station_ids_, self.station_terms_ = self._fit_station_terms(df, base)
            names += [f"dS2S_{im}" for im in self.anchor_ims] + ["station_seen"]
        else:
            self.station_ids_ = np.array([], dtype=np.int64)
            self.station_terms_ = np.zeros((0, 0))
        self.feature_names_ = names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Features for records *not* in the fitting set (cal, test, new data)."""
        base = self._base(df)
        if not self.station_encoding:
            return base.astype(np.float32)
        st = self._station_block(df["ssn"].to_numpy(dtype=np.int64), self.station_ids_, self.station_terms_)
        return np.column_stack([base, st]).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, y: object = None, **_: object) -> np.ndarray:
        """Fit on ``df`` and return its features with out-of-fold station terms."""
        self.fit(df)
        base = self._base(df)
        if not self.station_encoding:
            return base.astype(np.float32)
        eq = df["eqid"].to_numpy()
        ssn = df["ssn"].to_numpy(dtype=np.int64)
        events = np.unique(eq)
        rng = np.random.default_rng(self.seed)
        fold_of_event = pd.Series(rng.permutation(events.size) % self.n_oof_folds, index=events)
        fold = fold_of_event.reindex(eq).to_numpy()
        st = np.zeros((len(df), len(self.anchor_ims) + 1))
        for k in range(self.n_oof_folds):
            inner, outer = fold != k, fold == k
            if not outer.any():
                continue
            ids, mat = self._fit_station_terms(df[inner], base[inner])
            st[outer] = self._station_block(ssn[outer], ids, mat)
        return np.column_stack([base, st]).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)
