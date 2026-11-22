"""Checks for uploaded prediction files (Tab 3 of the app). No Streamlit, no PEER data needed.

Two levels:

* **Schema audit**, always: columns, IM names, numeric ``mu`` / ``sigma``, positive ``sigma``, duplicates,
  and a units check (ln values expected).
* **Split audit**, when the frozen ``splits_<profile>.json`` is available (it ships with the repository):
  the file's RSN fingerprint, RSN-level disjointness of every fold, and for each uploaded row whether it
  belongs to that fold's calibration or test set (train-fit rows are flagged: they cannot be scored and
  would hint at in-sample predictions). Coverage of each fold's test and calibration records is reported.

With the curated ``full.parquet`` present, :func:`run_local_evaluation` calls
:func:`gmpe_bench.evaluate_predictions` for the full audited metrics and HSCP intervals; otherwise the
app shows :func:`local_snippet`, the code to run the same evaluation on a machine with the PEER data.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gmpe_bench.schema import IM_NAMES, period_of, sa_name

REQUIRED = ("rsn", "im", "mu")
OPTIONAL = ("sigma", "scheme", "fold")
PARTS = ("train_fit", "cal", "test", "excluded")
K = len(IM_NAMES)
MAX_ERRORS_SHOWN = 8


# ------------------------------------------------------------------ IM names
_ALIASES = {"PGA": "PGA", "PGV": "PGV"}


def normalize_im(name: object) -> str | None:
    """Canonical IM name for common spellings (``pga``, ``SA_1.0s``, ``SA(1.0)``, ``1.0``) or None."""
    s = str(name).strip()
    if s in IM_NAMES:
        return s
    up = s.upper().replace(" ", "")
    if up in _ALIASES:
        return _ALIASES[up]
    m = re.fullmatch(r"(?:P?SA)?[_(]?T?([0-9]*\.?[0-9]+)\s*S?\)?", up)
    if m:
        cand = sa_name(float(m.group(1)))
        return cand if cand in IM_NAMES else None
    return None


# ------------------------------------------------------------------ results
@dataclass
class Check:
    name: str
    status: str          # PASS, WARN, FAIL, SKIP
    detail: str


@dataclass
class AuditResult:
    df: pd.DataFrame                      # cleaned predictions (canonical IM names)
    mode: str                             # "per-fold" or "out-of-fold"
    checks: list[Check] = field(default_factory=list)
    folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    split_audit: bool = False
    profile: str = "full"
    schemes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)

    @property
    def errors(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if c.status == "FAIL"]

    @property
    def warnings(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if c.status == "WARN"]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))


# ------------------------------------------------------------------ frozen splits
def rsn_fingerprint(rsn: np.ndarray) -> str:
    """Same fingerprint as :func:`gmpe_bench.splits.freeze.rsn_fingerprint` (sorted int64 RSN set)."""
    return hashlib.sha256(np.sort(np.asarray(rsn, dtype=np.int64)).tobytes()).hexdigest()


def load_splits_payload(profile: str = "full", folder: Path | None = None) -> dict[str, Any] | None:
    from gmpe_bench.app.data import data_dir

    path = (folder or data_dir()) / f"splits_{profile}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def all_rsn(payload: dict[str, Any]) -> np.ndarray:
    """Every RSN of the profile (each scheme's first fold covers all records)."""
    folds = next(iter(payload["schemes"].values()))
    f = folds[0]
    return np.sort(np.concatenate([np.asarray(f[p], dtype=np.int64) for p in PARTS]))


def verify_splits_payload(payload: dict[str, Any]) -> list[Check]:
    """Fingerprint and RSN-level disjointness/completeness of every frozen fold."""
    out: list[Check] = []
    rsn = all_rsn(payload)
    ok_fp = rsn.size == payload.get("n_records") and rsn_fingerprint(rsn) == payload.get("rsn_sha256")
    out.append(Check("split file fingerprint", "PASS" if ok_fp else "FAIL",
                     f"{rsn.size:,} records; SHA-256 {'matches' if ok_fp else 'does NOT match'} the frozen value"))
    bad = []
    universe = set(rsn.tolist())
    for scheme, folds in payload["schemes"].items():
        for f in folds:
            parts = [np.asarray(f[p], dtype=np.int64) for p in PARTS]
            allp = np.concatenate(parts)
            if allp.size != np.unique(allp).size or set(allp.tolist()) != universe or not len(f["test"]):
                bad.append(f"{scheme}/{f['fold']}")
    n_folds = sum(len(v) for v in payload["schemes"].values())
    out.append(Check("fold disjointness (RSN level)", "PASS" if not bad else "FAIL",
                     f"{n_folds} folds disjoint and complete" if not bad else f"problems in {', '.join(bad[:5])}"))
    return out


# ------------------------------------------------------------------ reading
def read_predictions_csv(source: str | Path | bytes | io.IOBase) -> pd.DataFrame:
    """Read an uploaded CSV; column names are stripped and lower-cased."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# ------------------------------------------------------------------ audit
def audit_predictions(df: pd.DataFrame, payload: dict[str, Any] | None = None, scheme: str = "event_grouped",
                      profile: str = "full") -> AuditResult:
    """Schema audit always; split audit when ``payload`` (the frozen splits JSON) is given."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    per_fold = {"scheme", "fold"} <= set(df.columns)
    res = AuditResult(df, "per-fold" if per_fold else "out-of-fold", profile=profile)

    # -------------------------------------------------------------- schema
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        res.add("columns", "FAIL", f"missing {missing}; need rsn, im, mu (optional sigma, scheme, fold)")
        return res
    extra = sorted(set(df.columns) - set(REQUIRED) - set(OPTIONAL))
    res.add("columns", "PASS" if not extra else "WARN",
            "rsn, im, mu" + (", sigma" if "sigma" in df else "") + (", scheme, fold" if per_fold else "")
            + (f"; ignored: {extra}" if extra else ""))
    if len(df) == 0:
        res.add("rows", "FAIL", "the file has no rows")
        return res

    rsn = pd.to_numeric(df["rsn"], errors="coerce")
    bad_rsn = int(rsn.isna().sum() + (rsn.notna() & (rsn != rsn.round())).sum())
    res.add("rsn values", "FAIL" if bad_rsn else "PASS",
            f"{bad_rsn:,} non-integer RSN values" if bad_rsn else f"{rsn.nunique():,} distinct records")
    if bad_rsn:
        return res
    df["rsn"] = rsn.astype(np.int64)

    canon = df["im"].map(normalize_im)
    unknown = sorted(set(df.loc[canon.isna(), "im"].astype(str)))
    renamed = int((canon.notna() & (canon != df["im"].astype(str))).sum())
    if unknown:
        res.add("IM names", "FAIL", f"unknown {unknown[:MAX_ERRORS_SHOWN]}; use {IM_NAMES[0]}, {IM_NAMES[1]}, "
                f"{IM_NAMES[2]} ... {IM_NAMES[-1]}")
        return res
    df["im"] = canon
    res.add("IM names", "PASS", f"{df['im'].nunique()} of {K} IMs" + (f"; {renamed:,} names normalized" if renamed else ""))

    mu = pd.to_numeric(df["mu"], errors="coerce")
    n_bad = int((~np.isfinite(mu)).sum())
    res.add("mu", "FAIL" if n_bad else "PASS", f"{n_bad:,} missing or non-numeric values" if n_bad else "all finite")
    df["mu"] = mu
    if "sigma" in df:
        sd = pd.to_numeric(df["sigma"], errors="coerce")
        n_nan, n_neg = int(sd.isna().sum()), int((sd <= 0).sum())
        status = "FAIL" if n_neg else ("WARN" if n_nan else "PASS")
        res.add("sigma", status, f"{n_neg:,} non-positive" if n_neg else
                (f"{n_nan:,} missing (fallback sigma used)" if n_nan else "all positive"))
        df["sigma"] = sd
    else:
        res.add("sigma", "WARN", "no sigma column: a default sigma is used; calibrated intervals still apply")

    sa = df["im"].map(lambda n: period_of(n) is not None)
    if sa.any() and (df.loc[sa, "mu"] > 0).mean() > 0.95:
        res.add("units", "WARN", "most SA medians are positive: mu must be ln(SA in g); linear values suspected")
    else:
        res.add("units", "PASS", "values look like natural logs (SA in g, PGV in cm/s)")

    key = ["scheme", "fold", "rsn", "im"] if per_fold else ["rsn", "im"]
    dup = int(df.duplicated(key).sum())
    res.add("duplicates", "FAIL" if dup else "PASS", f"{dup:,} duplicate {key} rows" if dup else "none")
    res.df = df
    if payload is None:
        res.add("split audit", "SKIP", f"frozen splits_{profile}.json not found; schema checks only")
        return res

    # -------------------------------------------------------------- split audit
    res.split_audit = True
    res.checks += verify_splits_payload(payload)
    universe = all_rsn(payload)
    outside = np.setdiff1d(df["rsn"].unique(), universe)
    res.add("RSNs in profile", "FAIL" if outside.size else "PASS",
            f"{outside.size:,} RSNs are not in the {profile} profile (e.g. {outside[:5].tolist()})"
            if outside.size else f"all in the {profile} profile ({universe.size:,} records)")
    schemes = payload["schemes"]
    rows = []
    if per_fold:
        df["scheme"], df["fold"] = df["scheme"].astype(str), df["fold"].astype(str)
        bad = sorted({f"{s}/{f}" for s, f in df[["scheme", "fold"]].drop_duplicates().itertuples(index=False)
                      if s not in schemes or f not in {x["fold"] for x in schemes[s]}})
        if bad:
            valid = {s: [x["fold"] for x in v] for s, v in schemes.items()}
            res.add("scheme/fold labels", "FAIL", f"unknown {bad[:MAX_ERRORS_SHOWN]}; valid: "
                    + "; ".join(f"{s}: {', '.join(v)}" for s, v in valid.items() if s in set(df['scheme'])) or str(valid))
            return res
        res.add("scheme/fold labels", "PASS", ", ".join(sorted(df["scheme"].unique())))
        res.schemes = sorted(df["scheme"].unique())
        for (s, f), g in df.groupby(["scheme", "fold"]):
            fold = next(x for x in schemes[s] if x["fold"] == f)
            rows.append(_fold_row(s, f, fold, set(g["rsn"].unique()), g))
    else:
        if scheme not in schemes:
            res.add("scheme", "FAIL", f"{scheme!r} not in the frozen splits ({list(schemes)})")
            return res
        res.schemes = [scheme]
        got = set(df["rsn"].unique())
        test_all = set()
        for fold in schemes[scheme]:
            test_all |= set(fold["test"])
            g = df[df["rsn"].isin(fold["test"]) | df["rsn"].isin(fold["cal"])]
            rows.append(_fold_row(scheme, fold["fold"], fold, got, g, oof=True))
        never = len(got - test_all)
        res.add("out-of-fold rows", "WARN" if never else "PASS",
                f"{never:,} records are not a test record of any {scheme} fold and cannot be scored" if never
                else f"every record is a test record of one {scheme} fold")
    res.folds = pd.DataFrame(rows)
    f = res.folds
    leaked = int(f["train_fit_rows"].sum())
    res.add("train-fit rows", "WARN" if leaked else "PASS",
            f"{leaked:,} predictions for records the fold trains on: ignored (in-sample predictions?)"
            if leaked else "no predictions for training records")
    full = f["test_coverage"].min()
    res.add("test coverage", "PASS" if full >= 0.999 else ("WARN" if f["test_predicted"].sum() else "FAIL"),
            f"{f['test_coverage'].mean():.1%} of fold test records have predictions (min {full:.1%})")
    cal = f["cal_coverage"]
    res.add("calibration rows", "PASS" if cal.min() >= 0.999 else "WARN",
            "calibration records predicted in every fold: HSCP and event-clustered intervals possible"
            if cal.min() >= 0.999 else
            f"calibration coverage {cal.mean():.1%} (min {cal.min():.1%}); folds without it get no intervals")
    return res


def _fold_row(scheme: str, fold_name: str, fold: dict[str, Any], got: set[int], g: pd.DataFrame,
              oof: bool = False) -> dict[str, Any]:
    test, cal, fit = set(fold["test"]), set(fold["cal"]), set(fold["train_fit"])
    n_im = g.groupby("rsn")["im"].nunique() if len(g) else pd.Series(dtype=float)
    test_pred = [r for r in test if r in got]
    return {"scheme": scheme, "fold": fold_name, "n_test": len(test), "test_predicted": len(test_pred),
            "test_coverage": len(test_pred) / max(len(test), 1),
            "ims_per_test_record": float(n_im.reindex(test_pred).mean()) if test_pred else 0.0,
            "n_cal": len(cal), "cal_predicted": len(cal & got), "cal_coverage": len(cal & got) / max(len(cal), 1),
            "train_fit_rows": 0 if oof else len(fit & got)}


# ------------------------------------------------------------------ templates and snippets
def sample_template(payload: dict[str, Any] | None = None, scheme: str = "event_grouped", n_records: int = 5,
                    seed: int = 0) -> pd.DataFrame:
    """Small per-fold template: a few calibration and test records of the first fold, all 27 IMs.

    The values are placeholders (ln medians around 0.1 g and sigma 0.65), so the file passes the audit.
    """
    rng = np.random.default_rng(seed)
    if payload is not None and scheme in payload["schemes"]:
        fold = payload["schemes"][scheme][0]
        rsn = list(fold["cal"][:n_records]) + list(fold["test"][:n_records])
        fold_name = fold["fold"]
    else:
        rsn, fold_name = list(range(1, 2 * n_records + 1)), "fold0"
    rows = []
    for r in rsn:
        for im in IM_NAMES:
            mu = np.log(15.0) if im == "PGV" else np.log(0.1) - 0.5 * np.log1p(period_of(im) or 0.0)
            rows.append({"rsn": int(r), "im": im, "mu": round(mu + rng.normal(0, 0.05), 4), "sigma": 0.65,
                         "scheme": scheme, "fold": fold_name})
    return pd.DataFrame(rows)


def fold_skeleton(payload: dict[str, Any], scheme: str, fold_name: str) -> pd.DataFrame:
    """All calibration and test records of one fold x 27 IMs, with empty mu and sigma to fill in."""
    fold = next(f for f in payload["schemes"][scheme] if f["fold"] == fold_name)
    rsn = np.concatenate([np.asarray(fold["cal"], dtype=np.int64), np.asarray(fold["test"], dtype=np.int64)])
    part = np.repeat(["cal", "test"], [len(fold["cal"]), len(fold["test"])])
    return pd.DataFrame({"rsn": np.repeat(rsn, K), "im": np.tile(IM_NAMES, rsn.size), "mu": np.nan, "sigma": np.nan,
                         "scheme": scheme, "fold": fold_name, "part": np.repeat(part, K)})


def local_snippet(filename: str = "predictions.csv", profile: str = "full", scheme: str | None = None) -> str:
    schemes = f", schemes=[\"{scheme}\"]" if scheme else ""
    return f'''# 1. Once: download the NGA-West2 flatfiles from PEER (https://peer.berkeley.edu), then
#    gmpe-bench build --h nga_w_h.xlsx --v nga_w_v.xlsx      (from the repository root)
# 2. Score the predictions on the audited folds:
import pandas as pd
import gmpe_bench as gb

df = pd.read_csv("{filename}")
report = gb.evaluate_predictions(df, profile="{profile}", calibrate=True{schemes})
print(report.summary())
report.save("results/my_predictions")    # report.json, CSVs and summary.txt
'''


def run_local_evaluation(res: AuditResult, folder: Path | None = None, n_boot: int = 200):
    """Full audited NGA-West2 metrics via :func:`gmpe_bench.evaluate_predictions` (needs full.parquet)."""
    import gmpe_bench as gb

    df = res.df.drop(columns=[c for c in res.df.columns if c not in REQUIRED + OPTIONAL])
    if "sigma" in df and df["sigma"].isna().all():
        df = df.drop(columns="sigma")
    kwargs = {"schemes": res.schemes} if res.schemes else {}
    return gb.evaluate_predictions(df, res.profile, calibrate=True, data_dir=folder, n_boot=n_boot, **kwargs)
