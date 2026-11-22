"""``results/paper/paper_summary_numbers.json``: every metric, interval and count behind the paper.

One self-describing file, so that ``results/paper/`` alone is enough to write or check the manuscript.
Values are unrounded floats (NaN is written as null). Sections:

meta            configuration, bootstrap settings, consistency checks, skipped parts
data            records / events / stations per profile and rule (Table 1)
splits          per scheme: folds, mean train-fit / cal / test sizes, seen-event and seen-station shares
accuracy        per scheme and model: RMSE, Gaussian 90 % coverage and width, tau / phi_S2S / phi_0,
                averaged over folds and IMs (Table 3, Figure 3), with bootstrap CIs for RMSE
rmse_differences  paired bootstrap of RMSE differences (Table S3 A)
station_rmse    seen vs unseen stations (Table S4)
conformal       per nominal level, scheme and method: coverage, width, joint coverage, simultaneous-band
                coverage, roughness, ESS, infinite thresholds (Table 4, Figures 4-5), with bootstrap CIs
conditional     pooled coverage by stratum and bin (Table 5, Figure 6)
leakage         random vs event-grouped RMSE per model (Figure S1)
numbers         every manuscript macro: key -> {text, value}
claims          every directional statement with PASS / FAIL / NA
tables          every table exactly as printed (list of rows)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gmpe_bench.reporting.inputs import MissingInput, ReportInputs


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [] if df is None or df.empty else df.to_dict(orient="records")


def _section(fn, skipped: dict[str, str], name: str) -> Any:  # noqa: ANN001
    try:
        return fn()
    except (MissingInput, KeyError, ValueError) as exc:
        skipped[name] = f"{type(exc).__name__}: {exc}"[:240]
        return None


def build_summary(inp: ReportInputs, tables: dict[str, Any] | None = None, book: Any = None) -> dict[str, Any]:
    cfg = inp.cfg
    skipped: dict[str, str] = {}
    out: dict[str, Any] = {}

    def data() -> Any:
        a = inp.attrition
        return _records(a[["profile", "rule", "description", "n_before", "n_after", "removed", "n_events",
                           "n_stations"]])

    def splits() -> Any:
        s = inp.split_summary
        s = s[s["profile"] == cfg.profile]
        cols = [c for c in ("n_train_fit", "n_cal", "n_test", "n_excluded", "events_test", "stations_test",
                            "test_records_seen_event", "test_records_seen_station") if c in s.columns]
        g = s.groupby("scheme")[cols].mean()
        g["folds"] = s.groupby("scheme").size()
        g["audit_passed"] = s.groupby("scheme")["passed"].all() if "passed" in s else np.nan
        return {k: v for k, v in g.to_dict(orient="index").items()}

    def accuracy() -> Any:
        acc = inp.accuracy(include_site=True)
        g = acc.groupby(["scheme", "model"])[["rmse", "cov90", "width90", "tau_test", "phi_s2s_test",
                                              "phi_0_test", "n"]].mean()
        g["n_folds"] = acc.groupby(["scheme", "model"])["fold"].nunique()
        res: dict[str, dict[str, Any]] = {}
        ci = inp.bootstrap.rmse.set_index(["scheme", "model"]) if not inp.bootstrap.rmse.empty else None
        for (scheme, model), row in g.iterrows():
            d = row.to_dict()
            if ci is not None and (scheme, model) in ci.index:
                r = ci.loc[(scheme, model)]
                d["rmse_ci"] = {"lo": r["lo"], "hi": r["hi"], "se": r["se"], "n_events": r["n_events"]}
            res.setdefault(scheme, {})[model] = d
        return res

    def conformal() -> Any:
        cm = inp.conformal_metrics
        cm = cm[cm["profile"] == cfg.profile]
        res: dict[str, Any] = {}
        cov_ci = inp.bootstrap.coverage
        ci = cov_ci.set_index(["alpha", "scheme", "method"]) if not cov_ci.empty else None
        for alpha in sorted(cm["alpha"].unique(), reverse=True):
            a = cm[cm["alpha"].round(4) == round(float(alpha), 4)]
            allim = a[a["im"] == "ALL"]
            cols = [c for c in ("coverage", "width", "winkler", "width_roughness", "ess", "n_inf") if c in allim]
            g = allim.groupby(["scheme", "method"])[cols].mean()
            if "n_inf" in allim:
                g["n_inf"] = allim.groupby(["scheme", "method"])["n_inf"].sum()
            for kind in ("JOINT_MARGINAL", "JOINT_BAND"):
                j = a[a["im"] == kind].groupby(["scheme", "method"])[["coverage", "width"]].mean()
                g[f"{kind.lower()}_coverage"] = j["coverage"]
                g[f"{kind.lower()}_width"] = j["width"]
            g["n_models"] = allim.groupby(["scheme", "method"])["model"].nunique()
            level: dict[str, Any] = {}
            for (scheme, method), row in g.iterrows():
                d = row.to_dict()
                key = (float(alpha), scheme, method)
                if ci is not None and key in ci.index:
                    r = ci.loc[key]
                    d["coverage_ci"] = {"lo": r["lo"], "hi": r["hi"], "se": r["se"], "n_events": r["n_events"]}
                level.setdefault(scheme, {})[method] = d
            res[f"{1 - float(alpha):.2f}"] = level
        return res

    def conditional() -> Any:
        c = inp.conditional
        c = c[(c["profile"] == cfg.profile)]
        g = c.groupby(["alpha", "method", "stratum", "bin"])[["covered", "n"]].sum().reset_index()
        g["coverage"] = g["covered"] / g["n"]
        return _records(g)

    def leakage() -> Any:
        rnd = inp.random_accuracy().groupby("model")["rmse"].mean()
        ev = inp.accuracy()
        ev = ev[ev["scheme"] == cfg.accuracy_scheme].groupby("model")["rmse"].mean()
        return {m: {"rmse_random": rnd[m], "rmse_event": ev[m], "inflation_pct": (ev[m] / rnd[m] - 1) * 100}
                for m in rnd.index if m in ev.index}

    out["meta"] = {"profile": cfg.profile, "compare_profile": cfg.compare_profile, "alpha": cfg.alpha,
                   "accuracy_scheme": cfg.accuracy_scheme, "site_models": list(cfg.site_models),
                   "pair_models": list(cfg.pair_models), "station_models": list(cfg.station_models),
                   "bootstrap": inp.bootstrap.meta, "bootstrap_skipped": inp.bootstrap.skipped,
                   "units": "RMSE and widths in natural-log units; coverage as a fraction"}
    out["data"] = _section(data, skipped, "data")
    out["splits"] = _section(splits, skipped, "splits")
    out["accuracy"] = _section(accuracy, skipped, "accuracy")
    out["rmse_differences"] = _records(inp.bootstrap.rmse_diff)
    out["station_rmse"] = _records(inp.bootstrap.station)
    out["conformal"] = _section(conformal, skipped, "conformal")
    out["conditional"] = _section(conditional, skipped, "conditional")
    out["leakage"] = _section(leakage, skipped, "leakage")
    if book is not None:
        out["numbers"] = {k: {"text": v, "value": book.raw.get(k)} for k, v in sorted(book.text.items())}
        out["claims"] = [c.__dict__ for c in book.claims]
    if tables:
        out["tables"] = {name: {"caption": t.caption, "note": t.note, "rows": _records(t.frame.astype(str))}
                         for name, t in tables.items()}
    out["meta"]["skipped"] = skipped
    return _clean(out)


def write_summary(summary: dict[str, Any], out_dir: Path) -> Path:
    path = Path(out_dir) / "paper_summary_numbers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    return path
