"""Numbers quoted in the manuscript, computed from the same Phase 1-5 outputs as the tables.

Outputs under ``results/paper/``:

``numbers.tex``
    One ``\\GB{key}`` value per number used in the prose. Values shown in a table
    are copied *as strings* from that table's cells, so text and tables cannot
    disagree. Numbers that appear only in figures use the figure's own formula
    (for example the leakage-inflation labels of Figure S1). A key the data cannot
    supply is written as ``--`` and listed in ``numbers_missing.txt``.

``claims_check.csv``
    Every directional statement in the manuscript (``X is lower than Y``), with
    the values behind it and ``PASS`` / ``FAIL`` / ``NA``. Run after every new
    result set; a ``FAIL`` means the sentence tagged with that claim ID in
    ``paper/main.tex`` must be rewritten.

LaTeX usage::

    \\input{../results/paper/numbers.tex}
    ... a test RMSE of \\GB{rmse.best.evt} (\\GB{rmse.bestmodel.evt}) ...

An unknown key prints ``??key??`` and raises a ``gmpe-bench`` package warning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gmpe_bench.reporting.inputs import SP23_REFERENCE, MissingInput, ReportInputs
from gmpe_bench.reporting.style import (
    METHOD_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    SCHEME_LABELS_PLAIN,
)
from gmpe_bench.reporting.tables import (
    MAIN_METHODS,
    TABLE5_BINS,
    Table,
    latex_escape,
    table2_splits,
    table3_accuracy,
    table4_conformal_ablation,
    table5_conditional_coverage,
    tableS2_sp23_comparison,
)
from gmpe_bench.schema import IM_NAMES, period_of

# ------------------------------------------------------------------ key vocabulary
SCHEME_KEY = {"random": "rand", "event_grouped": "evt", "station_grouped": "stn", "spatial_block": "reg",
              "temporal": "temp", "magnitude_holdout": "mag", "cl1_to_cl2": "cl"}
METHOD_KEY = {"UncalibratedGaussian": "gauss", "NaiveSplitConformal": "split",
              "ClusteredEventConformal": "evtcp", "WeightedShiftConformal": "shiftcp", "HSCP": "hscp",
              "HSCP_noSite": "hscp-nosite", "HSCP_noShift": "hscp-noshift", "HSCP_noSmooth": "hscp-nosmooth",
              "HSCP_abs": "hscp-abs"}
MODEL_KEY = {"ASK14_cal": "ask14", "BSSA14_cal": "bssa14", "CB14_cal": "cb14", "CY14_cal": "cy14",
             "MixedEffects": "me", "XGBoost": "xgb", "SpectralMLP": "mlp", "Hybrid_ASK14_MLP": "hyb-ask",
             "Hybrid_BSSA14_XGB": "hyb-bssa", "ASK14_site": "ask14-site", "BSSA14_site": "bssa14-site",
             "CB14_site": "cb14-site", "CY14_site": "cy14-site"}
BIN_KEY = {"Mw<5": "mw-lt5", "5<=Mw<6.5": "mw-5to6.5", "Mw>=6.5": "mw-ge6.5", "Rrup<30": "r-lt30",
           "30<=Rrup<100": "r-30to100", "Rrup>=100": "r-ge100", "A/B (>=760)": "vs-ab", "C (360-760)": "vs-c",
           "D (180-360)": "vs-d", "E (<180)": "vs-e", "seen": "seen", "unseen": "unseen"}
STRATUM_KEY = {"mw": "mw", "rrup": "r", "vs30_class": "vs", "station": "station"}
PUBLISHED = ("ASK14_cal", "BSSA14_cal", "CB14_cal", "CY14_cal")
ML_MODELS = ("XGBoost", "SpectralMLP", "Hybrid_ASK14_MLP", "Hybrid_BSSA14_XGB")
HYBRID_PARENT = {"Hybrid_ASK14_MLP": "ASK14_cal", "Hybrid_BSSA14_XGB": "BSSA14_cal"}
SHIFT_SCHEMES = ("station_grouped", "spatial_block", "temporal", "magnitude_holdout", "cl1_to_cl2")
VC_IMS = {"PGA": "pga", "SA_0.2s": "t0.2", "SA_1s": "t1", "SA_3s": "t3"}
VC_COMPS = {"tau_test": "tau", "phi_s2s_test": "phis2s", "phi_0_test": "phi0"}
MISSING = "--"
_KEY_RE = re.compile(r"^[A-Za-z0-9.\-]+$")

_PREAMBLE = r"""\makeatletter
\providecommand{\GB}[1]{\@ifundefined{GB@#1}{\textbf{??#1??}\PackageWarning{gmpe-bench}{Number `#1' undefined}}%
{\@nameuse{GB@#1}}}
"""


@dataclass
class Claim:
    cid: str
    statement: str
    status: str
    detail: str


@dataclass
class NumberBook:
    """Formatted values (``text``), the floats behind them (``raw``) and directional claims."""

    text: dict[str, str] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def put(self, key: str, value: str, raw: float | None = None) -> None:
        if not _KEY_RE.match(key):
            raise ValueError(f"invalid number key {key!r}")
        self.text[key] = value
        if raw is not None and np.isfinite(raw):
            self.raw[key] = float(raw)
        if value == MISSING and key not in self.missing:
            self.missing.append(key)

    def num(self, key: str) -> float:
        """Raw value of ``key``; KeyError if the data could not supply it."""
        return self.raw[key]

    def claim(self, cid: str, statement: str, test: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, detail = test()
            status = "PASS" if ok else "FAIL"
        except (KeyError, ValueError, IndexError, ZeroDivisionError) as exc:
            status, detail = "NA", f"missing input: {exc}"
        self.claims.append(Claim(cid, statement, status, detail))


# ------------------------------------------------------------------ formatting
def f3(x: float) -> str:
    return MISSING if not np.isfinite(x) else f"{x:.3f}"


def f2(x: float) -> str:
    return MISSING if not np.isfinite(x) else f"{x:.2f}"


def f1(x: float) -> str:
    return MISSING if not np.isfinite(x) else f"{x:.1f}"


def pct(x: float, signed: bool = False) -> str:
    """Percentage with one decimal and an escaped percent sign."""
    if not np.isfinite(x):
        return MISSING
    if round(x, 1) == 0:
        x = 0.0                                   # never print -0.0 or +0.0
        signed = False
    return (f"{x:+.1f}" if signed else f"{x:.1f}").replace("-", "\\textminus{}") + r"\%"


def count(x: float) -> str:
    return MISSING if not np.isfinite(x) else f"{int(round(x)):,}"


def sci(x: float) -> str:
    """Three significant figures; scientific notation in math mode when small."""
    if not np.isfinite(x):
        return MISSING
    if x == 0 or 1e-2 <= abs(x) < 1e4:
        return f"{x:.3g}"
    e = int(math.floor(math.log10(abs(x))))
    return f"${x / 10**e:.2f}\\times10^{{{e}}}$"


def _cell_float(s: str) -> float:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(s).replace(",", ""))
    return float(m.group(1)) if m else float("nan")


def _inverse(labels: dict[str, str]) -> dict[str, str]:
    """Display label -> internal name; the first name listed for a label wins (``ASK14`` -> ``ASK14_cal``)."""
    inv: dict[str, str] = {}
    for k, v in labels.items():
        inv.setdefault(v, k)
    return inv


def _row_lookup(t: Table, col: str, labels: dict[str, str]) -> dict[str, pd.Series]:
    """Map internal names (via their display labels) to the rows of a table frame."""
    inv = _inverse(labels)
    return {inv.get(str(r[col]), str(r[col])): r for _, r in t.frame.iterrows()}


def _scheme_columns(t: Table) -> dict[str, str]:
    inv = {v: k for k, v in SCHEME_LABELS_PLAIN.items()}
    return {inv[c]: c for c in t.frame.columns if c in inv}


# ------------------------------------------------------------------ sections
def _data(inp: ReportInputs, b: NumberBook) -> None:
    a = inp.attrition
    for prof in (inp.cfg.profile, inp.cfg.compare_profile):
        rows = a[a["profile"] == prof]
        if rows.empty:
            continue
        last = rows.iloc[-1]
        b.put(f"n.{prof}.records", count(last["n_after"]), last["n_after"])
        b.put(f"n.{prof}.events", count(last["n_events"]), last["n_events"])
        b.put(f"n.{prof}.stations", count(last["n_stations"]), last["n_stations"])
        first = rows.iloc[0]
        b.put(f"n.{prof}.raw", count(first["n_after"]), first["n_after"])
        b.put(f"n.{prof}.removed", count(first["n_after"] - last["n_after"]), first["n_after"] - last["n_after"])
    b.put("n.full.label", latex_escape(inp.cfg.profile))
    b.put("n.compare.label", latex_escape(inp.cfg.compare_profile))
    for k, v in SP23_REFERENCE.items():
        b.put(f"ref.sp23.{k}", count(v), v)
    sa = [period_of(n) for n in IM_NAMES if period_of(n) is not None]
    b.put("n.ims", str(len(IM_NAMES)), len(IM_NAMES))
    b.put("n.sa", str(len(sa)), len(sa))
    b.put("t.min", f"{min(sa):g}", min(sa))
    b.put("t.max", f"{max(sa):g}", max(sa))


def _splits(inp: ReportInputs, b: NumberBook) -> None:
    t = table2_splits(inp)
    rows = _row_lookup(t, "Scheme", SCHEME_LABELS_PLAIN)
    for scheme, key in SCHEME_KEY.items():
        r = rows.get(scheme)
        for col, name in (("Folds", "folds"), ("Train-fit", "trainfit"), ("Cal", "cal"), ("Test", "test"),
                          ("Seen event (%)", "seen-event"), ("Seen station (%)", "seen-station")):
            if r is None:
                b.put(f"splits.{key}.{name}", MISSING)
                continue
            s = latex_escape(r[col])
            b.put(f"splits.{key}.{name}", s, _cell_float(r[col]))
    s = inp.split_summary
    s = s[s["profile"] == inp.cfg.profile]
    b.put("splits.n.folds", count(len(s)), len(s))
    b.put("splits.n.schemes", str(s["scheme"].nunique()), s["scheme"].nunique())
    n_bench = len(set(s["scheme"]) - {"random"})
    b.put("splits.n.benchmark", str(n_bench), n_bench)
    b.put("splits.audit", "all" if bool(s["passed"].all()) else "not all", float(s["passed"].all()))
    b.put("splits.audit.failed", count((~s["passed"].astype(bool)).sum()), (~s["passed"].astype(bool)).sum())


def _accuracy(inp: ReportInputs, b: NumberBook) -> None:
    t = table3_accuracy(inp)
    rows = _row_lookup(t, "Model", MODEL_LABELS)
    cols = _scheme_columns(t)
    for model, mkey in MODEL_KEY.items():
        for scheme, skey in SCHEME_KEY.items():
            if scheme == "random":
                continue
            cell = rows[model][cols[scheme]] if model in rows and scheme in cols else MISSING
            b.put(f"rmse.{mkey}.{skey}", str(cell), _cell_float(cell))
    for scheme, skey in SCHEME_KEY.items():
        if scheme == "random":
            continue
        vals = {m: b.raw.get(f"rmse.{MODEL_KEY[m]}.{skey}", np.nan) for m in MODEL_KEY}
        vals = {m: v for m, v in vals.items() if np.isfinite(v)}
        if vals:
            best = min(vals, key=lambda m: (vals[m], MODEL_ORDER.index(m)))
            b.put(f"rmse.best.{skey}", f3(vals[best]), vals[best])
            b.put(f"rmse.bestmodel.{skey}", latex_escape(MODEL_LABELS[best]))
            pub = {m: v for m, v in vals.items() if m in PUBLISHED}
            if pub:
                bp = min(pub, key=lambda m: (pub[m], MODEL_ORDER.index(m)))
                b.put(f"rmse.bestpub.{skey}", f3(pub[bp]), pub[bp])
                b.put(f"rmse.bestpubmodel.{skey}", latex_escape(MODEL_LABELS[bp]))
                b.put(f"rmse.gainbest.{skey}", pct((1 - vals[best] / pub[bp]) * 100), (1 - vals[best] / pub[bp]) * 100)
        else:
            for k in ("best", "bestmodel", "bestpub", "bestpubmodel", "gainbest"):
                b.put(f"rmse.{k}.{skey}", MISSING)
        for hyb, parent in HYBRID_PARENT.items():
            h, p = vals.get(hyb, np.nan), vals.get(parent, np.nan)
            g = (1 - h / p) * 100 if np.isfinite(h) and np.isfinite(p) else np.nan
            b.put(f"gain.{MODEL_KEY[hyb]}.{skey}", pct(g), g)

    # Regime difficulty for the best model of each regime (Table 3)
    best = {k: b.raw[f"rmse.best.{k}"] for s_, k in SCHEME_KEY.items()
            if s_ != "random" and f"rmse.best.{k}" in b.raw}
    if best:
        hard, easy = max(best, key=best.get), min(best, key=best.get)
        inv = {v: s_ for s_, v in SCHEME_KEY.items()}
        b.put("rmse.hardest.scheme", latex_escape(SCHEME_LABELS_PLAIN[inv[hard]]))
        b.put("rmse.hardest", f3(best[hard]), best[hard])
        b.put("rmse.easiest.scheme", latex_escape(SCHEME_LABELS_PLAIN[inv[easy]]))
        b.put("rmse.easiest", f3(best[easy]), best[easy])
    else:
        for k in ("hardest.scheme", "hardest", "easiest.scheme", "easiest"):
            b.put(f"rmse.{k}", MISSING)
    keys = [k for s_, k in SCHEME_KEY.items() if s_ != "random"]
    me = {k: b.raw.get(f"rmse.me.{k}", np.nan) for k in keys}
    pubbest = {k: b.raw.get(f"rmse.bestpub.{k}", np.nan) for k in keys}
    for k in keys:
        d = (1 - me[k] / pubbest[k]) * 100 if np.isfinite(me[k]) and np.isfinite(pubbest[k]) else np.nan
        b.put(f"gain.me.{k}", pct(d, signed=True), d)

    # Gaussian 90 % coverage of the models' own sigma (Figure 3b)
    acc = inp.accuracy(include_site=True)               # same rows as Figure 3
    cov = acc.groupby(["model", "scheme"])["cov90"].mean().unstack("scheme")
    for scheme, skey in SCHEME_KEY.items():
        if scheme == "random":
            continue
        col = cov[scheme].dropna() if scheme in cov.columns else pd.Series(dtype=float)
        if col.empty:
            for k in ("min", "max", "minmodel", "maxmodel", "below"):
                b.put(f"gausscov.{k}.{skey}", MISSING)
            continue
        b.put(f"gausscov.min.{skey}", f3(col.min()), col.min())
        b.put(f"gausscov.max.{skey}", f3(col.max()), col.max())
        b.put(f"gausscov.minmodel.{skey}", latex_escape(MODEL_LABELS.get(col.idxmin(), col.idxmin())))
        b.put(f"gausscov.maxmodel.{skey}", latex_escape(MODEL_LABELS.get(col.idxmax(), col.idxmax())))
        below = int((col < 1 - inp.cfg.alpha).sum())
        b.put(f"gausscov.below.{skey}", f"{below} of {len(col)}", below)
        for model in col.index:
            if model in MODEL_KEY:
                b.put(f"gausscov.{MODEL_KEY[model]}.{skey}", f3(col[model]), col[model])
    b.put("n.models.accuracy", str(acc["model"].nunique()), acc["model"].nunique())

    # Variance components of test residuals (Figure 2)
    sub = acc[acc["scheme"] == inp.cfg.accuracy_scheme]
    for comp, ckey in VC_COMPS.items():
        g = sub.groupby(["model", "im"])[comp].mean()
        for model, mkey in MODEL_KEY.items():
            for im, ikey in VC_IMS.items():
                v = g.get((model, im), np.nan)
                b.put(f"vc.{ckey}.{mkey}.{ikey}", f2(v), v)
        for im, ikey in VC_IMS.items():
            vals = g.xs(im, level="im") if im in g.index.get_level_values("im") else pd.Series(dtype=float)
            vals = vals.dropna()
            b.put(f"vc.{ckey}.min.{ikey}", f2(vals.min()) if len(vals) else MISSING, vals.min() if len(vals) else None)
            b.put(f"vc.{ckey}.max.{ikey}", f2(vals.max()) if len(vals) else MISSING, vals.max() if len(vals) else None)


def _calibration_drift(inp: ReportInputs, b: NumberBook) -> None:
    """Gaussian 90 % coverage of ML sigma on cal versus test (Phase 4 metrics)."""
    for k in ("worst.model", "worst.scheme", "worst.cal", "worst.test", "evt.maxdrop"):
        b.put(f"caltest.{k}", MISSING)
    b.put("ml.fit.seconds", MISSING)
    b.put("ml.device", MISSING)
    m = inp.ml_metrics
    m = m[m["profile"] == inp.cfg.profile]
    g = m.groupby(["model", "scheme", "part"])["cov90"].mean().unstack("part")
    if not {"cal", "test"} <= set(g.columns):
        raise MissingInput("ml_metrics.csv has no cal/test parts")
    g = g.dropna()
    g["drop"] = g["cal"] - g["test"]
    for (model, scheme), r in g.iterrows():
        if model in MODEL_KEY and scheme in SCHEME_KEY:
            key = f"{MODEL_KEY[model]}.{SCHEME_KEY[scheme]}"
            b.put(f"caltest.cal.{key}", f3(r["cal"]), r["cal"])
            b.put(f"caltest.test.{key}", f3(r["test"]), r["test"])
    worst = g["drop"].idxmax()
    b.put("caltest.worst.model", latex_escape(MODEL_LABELS.get(worst[0], worst[0])))
    b.put("caltest.worst.scheme", latex_escape(SCHEME_LABELS_PLAIN.get(worst[1], worst[1])))
    b.put("caltest.worst.cal", f3(g.loc[worst, "cal"]), g.loc[worst, "cal"])
    b.put("caltest.worst.test", f3(g.loc[worst, "test"]), g.loc[worst, "test"])
    ev = g.xs("event_grouped", level="scheme")["drop"] if "event_grouped" in g.index.get_level_values(1) else []
    b.put("caltest.evt.maxdrop", f3(max(ev)) if len(ev) else MISSING, max(ev) if len(ev) else None)
    secs = m.drop_duplicates(["scheme", "fold", "model"])["fit_seconds"].sum()
    b.put("ml.fit.seconds", count(secs), secs)
    dev = m["device"].dropna().astype(str)
    b.put("ml.device", latex_escape(dev.mode().iloc[0].upper()) if len(dev) else MISSING)


def _leakage(inp: ReportInputs, b: NumberBook) -> None:
    """Same formula as the Figure S1 labels: RMSE(event) / RMSE(random) - 1."""
    try:
        rnd = inp.random_accuracy().groupby("model")["rmse"].mean()
    except MissingInput as exc:                   # keys are still defined (as --) so the text compiles
        b.skipped["leakage"] = f"MissingInput: {exc}"[:240]
        rnd = pd.Series(dtype=float)
    ev = inp.accuracy()
    ev = ev[ev["scheme"] == inp.cfg.accuracy_scheme].groupby("model")["rmse"].mean()
    infl: dict[str, float] = {}
    for model, mkey in MODEL_KEY.items():
        if model in rnd.index and model in ev.index:
            infl[model] = (ev[model] / rnd[model] - 1) * 100
            b.put(f"infl.{mkey}", pct(infl[model], signed=True), infl[model])
            b.put(f"rmse.rand.{mkey}", f3(rnd[model]), rnd[model])
        else:
            b.put(f"infl.{mkey}", MISSING)
            b.put(f"rmse.rand.{mkey}", MISSING)
    for group, members in (("pub", PUBLISHED), ("ml", ML_MODELS)):
        vals = [infl[m] for m in members if m in infl]
        b.put(f"infl.{group}.min", pct(min(vals), signed=True) if vals else MISSING, min(vals) if vals else None)
        b.put(f"infl.{group}.max", pct(max(vals), signed=True) if vals else MISSING, max(vals) if vals else None)
    if infl:
        top = max(infl, key=infl.get)
        b.put("infl.top.model", latex_escape(MODEL_LABELS[top]))
        b.put("infl.top", pct(infl[top], signed=True), infl[top])
    else:
        b.put("infl.top.model", MISSING)
        b.put("infl.top", MISSING)


def _conformal(inp: ReportInputs, b: NumberBook) -> None:
    cm = inp.conformal_metrics
    b.put("n.cp.rows", count(len(cm)), len(cm))
    cm = cm[cm["profile"] == inp.cfg.profile]
    b.put("n.cp.models", str(cm["model"].nunique()), cm["model"].nunique())
    b.put("n.cp.methods", str(cm["method"].nunique()), cm["method"].nunique())
    nominal = 1 - inp.cfg.alpha

    # Table 4 cells, both levels
    t = table4_conformal_ablation(inp)
    panels = sorted(t.panels.items()) if t.panels else [(0, "")]
    bounds = [(start, panels[i + 1][0] if i + 1 < len(panels) else len(t.frame), title)
              for i, (start, title) in enumerate(panels)]
    cols = _scheme_columns(t)
    for start, stop, title in bounds:
        lvl = re.search(r"(\d+)%", title)
        prefix = "" if lvl is None or int(lvl.group(1)) == round(nominal * 100) else lvl.group(1)
        block = t.frame.iloc[start:stop]
        rows = _row_lookup(Table("x", block, ""), "Method", METHOD_LABELS)
        for method, mkey in METHOD_KEY.items():
            for scheme, skey in SCHEME_KEY.items():
                cell = rows[method][cols[scheme]] if method in rows and scheme in cols else MISSING
                m = re.match(r"\s*(\d\.\d+)\s*\((\d+\.\d+)\)", str(cell))
                if m:
                    b.put(f"cov{prefix}.{mkey}.{skey}", m.group(1), float(m.group(1)))
                    b.put(f"wid{prefix}.{mkey}.{skey}", m.group(2), float(m.group(2)))
                else:
                    b.put(f"cov{prefix}.{mkey}.{skey}", MISSING)
                    b.put(f"wid{prefix}.{mkey}.{skey}", MISSING)
        lvl_nom = nominal if not prefix else int(prefix) / 100
        for method in MAIN_METHODS:
            mkey = METHOD_KEY[method]
            vals = {s: b.raw[f"cov{prefix}.{mkey}.{SCHEME_KEY[s]}"] for s in SCHEME_KEY
                    if f"cov{prefix}.{mkey}.{SCHEME_KEY[s]}" in b.raw}
            if not vals:
                for k in ("mace", "min", "max", "minscheme"):
                    b.put(f"cov{prefix}.{k}.{mkey}", MISSING)
                continue
            mace = float(np.mean([abs(v - lvl_nom) for v in vals.values()]))
            lo = min(vals, key=vals.get)
            b.put(f"cov{prefix}.mace.{mkey}", f3(mace), mace)
            b.put(f"cov{prefix}.min.{mkey}", f3(vals[lo]), vals[lo])
            b.put(f"cov{prefix}.max.{mkey}", f3(max(vals.values())), max(vals.values()))
            b.put(f"cov{prefix}.minscheme.{mkey}", latex_escape(SCHEME_LABELS_PLAIN[lo]))

    # Width of HSCP relative to split conformal and to the Gaussian band (unrounded means)
    a = cm[(cm["alpha"].round(4) == round(inp.cfg.alpha, 4))]
    allim = a[a["im"] == "ALL"]
    wid = allim.groupby(["method", "scheme"])["width"].mean()
    for other in ("NaiveSplitConformal", "UncalibratedGaussian", "HSCP_noSite"):
        okey = METHOD_KEY[other]
        for scheme, skey in SCHEME_KEY.items():
            h, o = wid.get(("HSCP", scheme), np.nan), wid.get((other, scheme), np.nan)
            r = (h / o - 1) * 100 if np.isfinite(h) and np.isfinite(o) else np.nan
            b.put(f"widdiff.hscp-{okey}.{skey}", pct(r, signed=True), r)
            b.put(f"widabs.hscp-{okey}.{skey}", pct(abs(r)) if np.isfinite(r) else MISSING,
                  abs(r) if np.isfinite(r) else None)
            b.put(f"widcmp.hscp-{okey}.{skey}", MISSING if not np.isfinite(r) else ("wider" if r > 0 else "narrower"))

    # Simultaneous (whole-spectrum) coverage
    for kind, label_ in (("JOINT_BAND", "band"), ("JOINT_MARGINAL", "marg")):
        j = a[a["im"] == kind].groupby(["method", "scheme"])[["coverage", "width"]].mean()
        for method, mkey in METHOD_KEY.items():
            for scheme, skey in SCHEME_KEY.items():
                if (method, scheme) in j.index:
                    b.put(f"joint.{label_}.{mkey}.{skey}", f3(j.loc[(method, scheme), "coverage"]),
                          j.loc[(method, scheme), "coverage"])
                else:
                    b.put(f"joint.{label_}.{mkey}.{skey}", MISSING)
        for method in ("HSCP", "NaiveSplitConformal"):
            mkey = METHOD_KEY[method]
            vals = [b.raw[f"joint.{label_}.{mkey}.{SCHEME_KEY[s]}"] for s in SCHEME_KEY
                    if f"joint.{label_}.{mkey}.{SCHEME_KEY[s]}" in b.raw]
            b.put(f"joint.{label_}.{mkey}.min", f3(min(vals)) if vals else MISSING, min(vals) if vals else None)
            b.put(f"joint.{label_}.{mkey}.max", f3(max(vals)) if vals else MISSING, max(vals) if vals else None)
    jb = a[a["im"] == "JOINT_BAND"].groupby("scheme")["width"].mean()
    ratios = []
    for scheme, skey in SCHEME_KEY.items():
        h = wid.get(("HSCP", scheme), np.nan)
        r = jb.get(scheme, np.nan) / h if np.isfinite(h) and h > 0 else np.nan
        b.put(f"joint.widthratio.{skey}", f2(r), r)
        if np.isfinite(r):
            ratios.append(r)
    b.put("joint.widthratio.min", f2(min(ratios)) if ratios else MISSING, min(ratios) if ratios else None)
    b.put("joint.widthratio.max", f2(max(ratios)) if ratios else MISSING, max(ratios) if ratios else None)

    # Spectral roughness of the band width (mean squared 2nd difference of ln half-width)
    if "width_roughness" in allim.columns:
        rough = allim.groupby(["method", "scheme"])["width_roughness"].mean()
        reductions = []
        for method, mkey in METHOD_KEY.items():
            for scheme, skey in SCHEME_KEY.items():
                v = rough.get((method, scheme), np.nan)
                b.put(f"rough.{mkey}.{skey}", sci(v), v)
        for scheme, skey in SCHEME_KEY.items():
            h, n = rough.get(("HSCP", scheme), np.nan), rough.get(("HSCP_noSmooth", scheme), np.nan)
            red = (1 - h / n) * 100 if np.isfinite(h) and np.isfinite(n) and n > 0 else np.nan
            b.put(f"rough.reduction.{skey}", pct(red), red)
            if np.isfinite(red):
                reductions.append(red)
        b.put("rough.reduction.min", pct(min(reductions)) if reductions else MISSING,
              min(reductions) if reductions else None)
        b.put("rough.reduction.max", pct(max(reductions)) if reductions else MISSING,
              max(reductions) if reductions else None)

    # Effective sample size of the shift weights and infinite thresholds
    if "ess" in allim.columns:
        s = inp.split_summary
        s = s[s["profile"] == inp.cfg.profile].groupby("scheme")["n_cal"].mean()
        ess = allim[allim["method"] == "WeightedShiftConformal"].groupby("scheme")["ess"].mean()
        for scheme, skey in SCHEME_KEY.items():
            e, n = ess.get(scheme, np.nan), s.get(scheme, np.nan)
            b.put(f"ess.{skey}", count(e), e)
            frac = e / n * 100 if np.isfinite(e) and np.isfinite(n) and n > 0 else np.nan
            b.put(f"essfrac.{skey}", pct(frac), frac)
    if "n_inf" in allim.columns:
        ninf = allim.groupby("method")["n_inf"].sum()
        for method, mkey in METHOD_KEY.items():
            v = ninf.get(method, np.nan)
            b.put(f"ninf.{mkey}", count(v), v)


def _conditional(inp: ReportInputs, b: NumberBook) -> None:
    t = table5_conditional_coverage(inp)
    labels = {lab: key for st in TABLE5_BINS.values() for key, lab in st}
    frame = t.frame
    nominal = 1 - inp.cfg.alpha
    for bkey in BIN_KEY.values():                 # every key exists even if a bin is empty
        b.put(f"cond.n.{bkey}", MISSING)
        for method in MAIN_METHODS:
            b.put(f"cond.{METHOD_KEY[method]}.{bkey}", MISSING)
    for _, r in frame.iterrows():
        bin_ = labels.get(str(r["Bin"]))
        if bin_ is None:
            continue
        bkey = BIN_KEY[bin_]
        b.put(f"cond.n.{bkey}", str(r["Records x IMs"]), _cell_float(r["Records x IMs"]))
        for method in MAIN_METHODS:
            col = METHOD_LABELS[method]
            v = r[col] if col in frame.columns else MISSING
            b.put(f"cond.{METHOD_KEY[method]}.{bkey}", str(v), _cell_float(v))
    for method in MAIN_METHODS:
        mkey = METHOD_KEY[method]
        allv: dict[str, float] = {}
        for stratum, bins in TABLE5_BINS.items():
            vals = {bk: b.raw[f"cond.{mkey}.{BIN_KEY[bk]}"] for bk, _ in bins if f"cond.{mkey}.{BIN_KEY[bk]}" in b.raw}
            allv.update(vals)
            spread = max(vals.values()) - min(vals.values()) if vals else np.nan
            b.put(f"condspread.{mkey}.{STRATUM_KEY[stratum]}", f3(spread), spread)
        if allv:
            worst = max(allv, key=lambda k: abs(allv[k] - nominal))
            b.put(f"condworst.{mkey}", f3(allv[worst]), allv[worst])
            b.put(f"condworstbin.{mkey}", latex_escape(dict(sum(TABLE5_BINS.values(), ()))[worst]))
            low = min(allv, key=allv.get)
            b.put(f"condmin.{mkey}", f3(allv[low]), allv[low])
            b.put(f"condminbin.{mkey}", latex_escape(dict(sum(TABLE5_BINS.values(), ()))[low]))
        else:
            for k in ("condworst", "condworstbin", "condmin", "condminbin"):
                b.put(f"{k}.{mkey}", MISSING)


def _profiles(inp: ReportInputs, b: NumberBook) -> None:
    t = tableS2_sp23_comparison(inp)
    for _, r in t.frame.iterrows():
        m = re.match(r"RMSE (.+) \((\S+) / (\S+)\)", str(r["Item"]))
        if not m:
            continue
        model = _inverse(MODEL_LABELS).get(m.group(1))
        if model is None:
            continue
        a, c = str(r["Value"]).split(" / ")
        b.put(f"sp.rmse.main.{MODEL_KEY[model]}", a, _cell_float(a))
        b.put(f"sp.rmse.cmp.{MODEL_KEY[model]}", c, _cell_float(c))


def _bootstrap(inp: ReportInputs, b: NumberBook) -> None:
    """Event-clustered bootstrap intervals (Tables S3-S4); keys exist even when a part is missing."""
    bs = inp.bootstrap
    cfg = inp.cfg
    b.put("boot.b", count(cfg.bootstrap_b), cfg.bootstrap_b)
    b.put("boot.level", f"{cfg.bootstrap_level * 100:.0f}", cfg.bootstrap_level * 100)
    b.put("boot.seed", str(cfg.bootstrap_seed), cfg.bootstrap_seed)
    ev = bs.meta.get("events_per_scheme", {})
    for scheme, skey in SCHEME_KEY.items():
        b.put(f"boot.events.{skey}", count(ev.get(scheme, np.nan)), ev.get(scheme))
    # RMSE of every model
    r = bs.rmse.set_index(["scheme", "model"]) if not bs.rmse.empty else None
    for model, mkey in MODEL_KEY.items():
        for scheme, skey in SCHEME_KEY.items():
            row = r.loc[(scheme, model)] if r is not None and (scheme, model) in r.index else None
            for part in ("lo", "hi"):
                b.put(f"ci.rmse.{mkey}.{skey}.{part}", f3(row[part]) if row is not None else MISSING,
                      row[part] if row is not None else None)
    # paired differences
    pairs = [(a, c) for i, a in enumerate(cfg.pair_models) for c in cfg.pair_models[i + 1:]]
    d = bs.rmse_diff.set_index(["model_a", "model_b", "scheme"]) if not bs.rmse_diff.empty else None
    for a, c in pairs:
        ka, kc = MODEL_KEY.get(a, a.lower()), MODEL_KEY.get(c, c.lower())
        for scheme, skey in SCHEME_KEY.items():
            row = d.loc[(a, c, scheme)] if d is not None and (a, c, scheme) in d.index else None
            for part, name in (("estimate", "est"), ("lo", "lo"), ("hi", "hi")):
                v = row[part] if row is not None else np.nan
                b.put(f"ci.diff.{ka}.{kc}.{skey}.{name}", _signed3(v), v)
            v = row["p_two_sided"] if row is not None else np.nan
            b.put(f"ci.diff.{ka}.{kc}.{skey}.p", f3(v), v)
            sig = (row is not None and np.isfinite(row["lo"]) and (row["lo"] > 0 or row["hi"] < 0))
            b.put(f"ci.diff.{ka}.{kc}.{skey}.excludes0", MISSING if row is None else ("yes" if sig else "no"),
                  float(sig) if row is not None else None)
    # coverage
    cv = bs.coverage.set_index(["method", "alpha", "scheme"]) if not bs.coverage.empty else None
    for method in MAIN_METHODS:
        mkey = METHOD_KEY[method]
        for alpha, prefix in ((inp.cfg.alpha, ""), (0.05, "95")):
            for scheme, skey in SCHEME_KEY.items():
                row = None
                if cv is not None:
                    hit = [k for k in cv.index
                           if k[0] == method and round(k[1], 4) == round(alpha, 4) and k[2] == scheme]
                    row = cv.loc[hit[0]] if hit else None
                for part in ("lo", "hi"):
                    b.put(f"ci.cov{prefix}.{mkey}.{skey}.{part}", f3(row[part]) if row is not None else MISSING,
                          row[part] if row is not None else None)
    # seen / unseen stations
    st = bs.station.set_index(["model", "status", "scheme"]) if not bs.station.empty else None
    for model in cfg.station_models:
        mkey = MODEL_KEY.get(model, model.lower())
        for status in ("seen", "unseen"):
            for scheme, skey in SCHEME_KEY.items():
                key = (model, status, scheme)
                row = st.loc[key] if st is not None and key in st.index else None
                for part, name in (("estimate", "est"), ("lo", "lo"), ("hi", "hi")):
                    b.put(f"stn.rmse.{mkey}.{status}.{skey}.{name}", f3(row[part]) if row is not None else MISSING,
                          row[part] if row is not None else None)
                b.put(f"stn.n.{mkey}.{status}.{skey}", count(row["n_records"]) if row is not None else MISSING,
                      row["n_records"] if row is not None else None)
    for k, v in bs.skipped.items():
        b.skipped[f"bootstrap:{k}"] = v


def _signed3(x: float) -> str:
    if not np.isfinite(x):
        return MISSING
    s = f"{x:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s.replace("-", "\\textminus{}")


SECTIONS: dict[str, Callable[[ReportInputs, NumberBook], None]] = {
    "data": _data, "splits": _splits, "accuracy": _accuracy, "calibration_drift": _calibration_drift,
    "leakage": _leakage, "conformal": _conformal, "conditional": _conditional, "profiles": _profiles,
    "bootstrap": _bootstrap,
}


# ------------------------------------------------------------------ claims
def _claims(b: NumberBook, nominal: float) -> None:
    n = b.num
    S = [k for s, k in SCHEME_KEY.items() if s != "random"]

    def avail(prefix: str) -> list[str]:
        return [k for k in S if f"{prefix}.{k}" in b.raw]

    def c01() -> tuple[bool, str]:
        lo, hi = n("infl.ml.min"), n("infl.pub.max")
        return lo > hi, f"ML min {lo:+.1f}% vs NGA max {hi:+.1f}%"

    def c02() -> tuple[bool, str]:
        lo, hi = n("infl.pub.min"), n("infl.pub.max")
        return max(abs(lo), abs(hi)) < 2.0, f"NGA-West2 range {lo:+.1f}% to {hi:+.1f}%"

    def c03() -> tuple[bool, str]:
        best = b.text["rmse.bestmodel.evt"]
        return best.startswith("Hybrid"), f"best Event model: {best}"

    def c03b() -> tuple[bool, str]:
        hyb = [MODEL_KEY[h] for h in HYBRID_PARENT]
        out, ok, seen = [], True, False
        for k in S:
            if any(f"rmse.{h}.{k}" in b.raw for h in hyb):
                seen = True
                best = b.text.get(f"rmse.bestmodel.{k}", "")
                ok &= best.startswith("Hybrid")
                out.append(f"{k}: {best}")
        if not seen:
            raise KeyError("no hybrid results")
        return ok, "; ".join(out)

    def c04() -> tuple[bool, str]:
        out, ok = [], True
        for hyb, parent in HYBRID_PARENT.items():
            for k in S:
                hk, pk = f"rmse.{MODEL_KEY[hyb]}.{k}", f"rmse.{MODEL_KEY[parent]}.{k}"
                if hk in b.raw and pk in b.raw:
                    ok &= b.raw[hk] < b.raw[pk]
                    out.append(f"{MODEL_KEY[hyb]}/{k}: {b.raw[hk]:.3f} vs {b.raw[pk]:.3f}")
        if not out:
            raise KeyError("no hybrid/parent pair")
        return ok, "; ".join(out)

    def c05() -> tuple[bool, str]:
        vals = {k: n(f"gausscov.max.{k}") for k in ("reg", "cl")}
        return all(v < nominal for v in vals.values()), str({k: round(v, 3) for k, v in vals.items()})

    def c06() -> tuple[bool, str]:
        v = n("cov.split.evt")
        return abs(v - nominal) <= 0.02, f"split CP Event coverage {v:.3f}"

    def c07() -> tuple[bool, str]:
        m = {METHOD_KEY[x]: n(f"cov.mace.{METHOD_KEY[x]}") for x in MAIN_METHODS}
        return min(m, key=m.get) == "hscp", str({k: round(v, 3) for k, v in m.items()})

    def c07b() -> tuple[bool, str]:
        m = {METHOD_KEY[x]: n(f"cov95.mace.{METHOD_KEY[x]}") for x in MAIN_METHODS}
        return min(m, key=m.get) == "hscp", str({k: round(v, 3) for k, v in m.items()})

    def c08() -> tuple[bool, str]:
        h, s_ = n("cov.min.hscp"), n("cov.min.split")
        return h >= s_, f"HSCP worst {h:.3f} vs split {s_:.3f}"

    def c09() -> tuple[bool, str]:
        ks = [k for k in avail("joint.band.hscp") if f"joint.marg.hscp.{k}" in b.raw]
        if not ks:
            raise KeyError("joint coverage")
        return all(n(f"joint.band.hscp.{k}") > n(f"joint.marg.hscp.{k}") for k in ks), \
            "; ".join(f"{k}: {n(f'joint.band.hscp.{k}'):.3f} vs {n(f'joint.marg.hscp.{k}'):.3f}" for k in ks)

    def c10() -> tuple[bool, str]:
        ks = avail("joint.band.hscp")
        if not ks:
            raise KeyError("joint band")
        return all(n(f"joint.band.hscp.{k}") >= nominal - 0.05 for k in ks), \
            "; ".join(f"{k}: {n(f'joint.band.hscp.{k}'):.3f}" for k in ks)

    def c11() -> tuple[bool, str]:
        ks = [k for k in avail("rough.hscp") if f"rough.hscp-nosmooth.{k}" in b.raw]
        if not ks:
            raise KeyError("roughness")
        return all(n(f"rough.hscp.{k}") < n(f"rough.hscp-nosmooth.{k}") for k in ks), \
            f"reduction {b.text.get('rough.reduction.min')} to {b.text.get('rough.reduction.max')}"

    def c12() -> tuple[bool, str]:
        return n("condspread.hscp.vs") < n("condspread.split.vs"), \
            f"VS30 spread HSCP {n('condspread.hscp.vs'):.3f} vs split {n('condspread.split.vs'):.3f}"

    def c13() -> tuple[bool, str]:
        h, s = n("cond.hscp.unseen"), n("cond.split.unseen")
        return abs(h - nominal) < abs(s - nominal), f"unseen: HSCP {h:.3f} vs split {s:.3f}"

    def c14() -> tuple[bool, str]:
        ks = [k for k in b.raw if k.startswith("caltest.cal.") and k.endswith(".reg")]
        if not ks:
            raise KeyError("caltest region")
        pairs = [(k, n(k), n(k.replace(".cal.", ".test."))) for k in ks]
        return (all(c > t_ for _, c, t_ in pairs),
                "; ".join(f"{k.split('.')[2]}: {c:.3f}->{t_:.3f}" for k, c, t_ in pairs))

    def c15() -> tuple[bool, str]:
        shift = [k for k in ("stn", "reg", "temp", "mag", "cl") if f"cov.split.{k}" in b.raw]
        if not shift:
            raise KeyError("shift schemes")
        worst = min(shift, key=lambda k: n(f"cov.split.{k}"))
        v = n(f"cov.split.{worst}")
        return v < nominal - 0.02, f"split CP worst shift scheme {worst}: {v:.3f}"

    def c16() -> tuple[bool, str]:
        g, h = n("cov.mace.gauss"), n("cov.mace.hscp")
        return g > h, f"MACE Gaussian {g:.3f} vs HSCP {h:.3f}"

    def c17() -> tuple[bool, str]:
        out, ok = [], True
        for hyb, parent in HYBRID_PARENT.items():
            hk, pk = f"vc.phis2s.{MODEL_KEY[hyb]}.t1", f"vc.phis2s.{MODEL_KEY[parent]}.t1"
            if hk in b.raw and pk in b.raw:
                ok &= b.raw[hk] < b.raw[pk]
                out.append(f"{MODEL_KEY[hyb]}: {b.raw[hk]:.2f} vs {b.raw[pk]:.2f}")
        if not out:
            raise KeyError("no hybrid/parent phi_S2S pair")
        return ok, "; ".join(out)

    def c18() -> tuple[bool, str]:
        v = n("cov.hscp.evt")
        return abs(v - nominal) <= 0.02, f"HSCP Event coverage {v:.3f}"

    specs = (
        ("C01", "Random-split leakage inflates ML skill more than published NGA-West2 skill", c01),
        ("C02", "Published NGA-West2 models change by less than 2% between random and event splits", c02),
        ("C03", "The lowest Event-split RMSE belongs to a hybrid model", c03),
        ("C03b", "In every scheme with hybrid results, the lowest RMSE belongs to a hybrid", c03b),
        ("C04", "Each hybrid has lower RMSE than its physics parent in every reported scheme", c04),
        ("C05", "Gaussian 90% intervals undercover for every model under Region and CL1 to CL2", c05),
        ("C06", "Split conformal is within 0.02 of nominal under the Event split", c06),
        ("C07", "HSCP has the smallest mean absolute coverage error across schemes at 90%", c07),
        ("C07b", "HSCP has the smallest mean absolute coverage error across schemes at 95%", c07b),
        ("C08", "HSCP worst-scheme coverage is at least split conformal's worst-scheme coverage", c08),
        ("C09", "The HSCP simultaneous band covers whole spectra more often than its marginal bands", c09),
        ("C10", "HSCP simultaneous-band coverage is within 0.05 below nominal in every scheme", c10),
        ("C11", "Spectral smoothing lowers band-width roughness in every scheme", c11),
        ("C12", "HSCP coverage varies less across VS30 classes than split conformal", c12),
        ("C13", "HSCP unseen-station coverage is closer to nominal than split conformal", c13),
        ("C14", "ML Gaussian coverage on cal exceeds coverage on test under the Region split", c14),
        ("C15", "Split conformal undercovers by more than 0.02 in at least one shift scheme", c15),
        ("C16", "The uncalibrated Gaussian band has a larger mean coverage error than HSCP", c16),
        ("C17", "Each hybrid has a smaller site-to-site term phi_S2S than its parent at T = 1 s (Event split)", c17),
        ("C18", "HSCP is within 0.02 of nominal under the Event split", c18),
    )
    for cid, text, fn in specs:
        b.claim(cid, text, fn)


# ------------------------------------------------------------------ public API
def build_numbers(inp: ReportInputs, log: Callable[[str], None] = print) -> NumberBook:
    b = NumberBook()
    for name, fn in SECTIONS.items():
        try:
            fn(inp, b)
        except (MissingInput, KeyError, ValueError) as exc:
            b.skipped[name] = f"{type(exc).__name__}: {exc}"[:240]
            log(f"numbers {name}: SKIPPED ({b.skipped[name]})")
    _claims(b, 1 - inp.cfg.alpha)
    return b


def write_numbers(b: NumberBook, out_dir: Path, inp: ReportInputs | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = ["% numbers.tex -- generated by gmpe-bench (reporting/numbers.py). Do not edit by hand.",
            "% Every value is computed from the Phase 1-5 outputs; table values are copied from the table cells."]
    if inp is not None:
        head.append(f"% profile={inp.cfg.profile}  compare={inp.cfg.compare_profile}  alpha={inp.cfg.alpha}  "
                    f"accuracy_scheme={inp.cfg.accuracy_scheme}")
    body = [rf"\@namedef{{GB@{k}}}{{{v}}}" for k, v in sorted(b.text.items())]
    tex = out_dir / "numbers.tex"
    tex.write_text("\n".join(head) + "\n" + _PREAMBLE + "\n".join(body) + "\n\\makeatother\n", encoding="utf-8")
    claims = out_dir / "claims_check.csv"
    pd.DataFrame([c.__dict__ for c in b.claims], columns=["cid", "statement", "status", "detail"]) \
        .to_csv(claims, index=False)
    missing = out_dir / "numbers_missing.txt"
    lines = [f"{k}" for k in sorted(b.missing)] + [f"# section skipped: {k}: {v}" for k, v in b.skipped.items()]
    missing.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return [tex, claims, missing]


def claims_summary(b: NumberBook) -> str:
    counts = pd.Series([c.status for c in b.claims]).value_counts().to_dict()
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
