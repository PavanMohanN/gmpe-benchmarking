"""Paper tables (CSV + booktabs LaTeX) built from Phase 1-5 outputs.

LaTeX rules: ``booktabs`` horizontal rules only (\\toprule, \\midrule, \\bottomrule),
no vertical lines, no shading; every table sits in ``adjustbox`` with
``max width=6.25in``. Preamble needs ``\\usepackage{booktabs,adjustbox}``.

Main tables
    table1_attrition            record selection, rule by rule
    table2_splits               split sizes and leakage audit
    table3_accuracy             test RMSE by model and split scheme
    table4_conformal_ablation   coverage (width) by method and scheme, 90 % and 95 %
    table5_conditional_coverage coverage by Mw, R_rup, VS30 class and station status
Supplement
    tableS1_per_im_metrics      per-IM RMSE and calibrated-band performance
    tableS2_sp23_comparison     full vs sp23 profile, with the published sp23 counts
    tableS3_bootstrap_cis       event-clustered bootstrap CIs: RMSE differences and coverage
    tableS4_station_rmse        RMSE for seen vs unseen stations (equal site information)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gmpe_bench.reporting.inputs import SP23_REFERENCE, MissingInput, ReportInputs
from gmpe_bench.reporting.style import (
    METHOD_LABELS,
    METHOD_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    SCHEME_LABELS_PLAIN,
    SCHEME_ORDER,
    ordered,
)
from gmpe_bench.schema import IM_NAMES

MAX_WIDTH = "6.25in"
_LATEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


@dataclass
class Table:
    stem: str
    frame: pd.DataFrame
    caption: str
    note: str = ""
    align: str | None = None
    bold: set[tuple[int, int]] | None = None      # (row, col) cells to bold
    panels: dict[int, str] | None = None          # row index -> panel heading inserted above it
    stacked: bool = False                         # LaTeX only: "est [lo, hi]" cells as est over [lo, hi]


_RELATIONS = ((">=", r"$\geq$"), ("<=", r"$\leq$"), (">", r"$>$"), ("<", r"$<$"))


def latex_escape(value: object) -> str:
    """Escape LaTeX specials; render <, >, <=, >= as math so pdflatex's default encoding is safe."""
    text = str(value)
    if text.startswith("$") and text.endswith("$"):
        return text
    out, i = [], 0
    while i < len(text):
        for token, rep in _RELATIONS:
            if text.startswith(token, i):
                out.append(rep)
                i += len(token)
                break
        else:
            out.append(_LATEX_ESCAPES.get(text[i], text[i]))
            i += 1
    return "".join(out)


def fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, (float, np.floating)):
        return "--" if not np.isfinite(x) else f"{x:.{digits}f}"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return str(x)


_CI_CELL = re.compile(r"^([+-]?\d+\.\d+) \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]$")


def _stacked_ci(text: str) -> str | None:
    """'+0.012 [-0.004, +0.029]' -> two-line math cell (estimate above its interval), else None."""
    m = _CI_CELL.match(str(text))
    if not m:
        return None
    est, lo, hi = m.groups()
    return (rf"$\begin{{array}}{{@{{}}c@{{}}}}{est}\\[-1pt] \scriptstyle[{lo},\,{hi}]"
            rf"\end{{array}}$")


def to_booktabs(t: Table) -> str:
    """Render a table as booktabs LaTeX (no vertical rules, no shading)."""
    df = t.frame
    align = t.align or ("l" + "r" * (df.shape[1] - 1))
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{latex_escape(t.caption)}}}",
             rf"\label{{tab:{t.stem}}}", r"\small", rf"\begin{{adjustbox}}{{max width={MAX_WIDTH}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(latex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for i, row in enumerate(df.itertuples(index=False)):
        if t.panels and i in t.panels:
            if i:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{df.shape[1]}}}{{l}}{{\textit{{{latex_escape(t.panels[i])}}}}} \\")
        cells = []
        for j, v in enumerate(row):
            s = (_stacked_ci(v) if t.stacked else None) or latex_escape(v)
            cells.append(rf"\textbf{{{s}}}" if t.bold and (i, j) in t.bold else s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if t.note:
        lines += [r"\par\smallskip", rf"\parbox{{{MAX_WIDTH}}}{{\footnotesize {latex_escape(t.note)}}}"]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table(t: Table, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, tex = out_dir / f"{t.stem}.csv", out_dir / f"{t.stem}.tex"
    t.frame.to_csv(csv, index=False)
    tex.write_text(to_booktabs(t), encoding="utf-8")
    return [csv, tex]


# ------------------------------------------------------------------ main tables
def table1_attrition(inp: ReportInputs) -> Table:
    a = inp.attrition
    p, c = inp.cfg.profile, inp.cfg.compare_profile
    main = a[a["profile"] == p]
    extra = a[(a["profile"] == c) & a["rule"].astype(str).str.match(r"R9|R12")]
    rows = pd.concat([main.assign(prof=p), extra.assign(prof=c)], ignore_index=True)
    frame = pd.DataFrame({
        "Rule": rows["rule"], "Profile": rows["prof"], "Description": rows["description"],
        "Records": rows["n_after"].map(fmt), "Removed": rows["removed"].map(fmt),
        "Events": rows["n_events"].map(fmt), "Stations": rows["n_stations"].map(fmt),
    })
    first_c = int((rows["prof"] == c).to_numpy().argmax()) if (rows["prof"] == c).any() else None
    panels = {0: f"Profile: {p}"} | ({first_c: f"Profile: {c} (selection steps only)"} if first_c else {})
    return Table("table1_attrition", frame, f"Record selection from the NGA-West2 flatfiles, rule by rule "
                 f"({p} profile, then the {c}-specific steps).",
                 "Records, events and stations are counts after each rule.", align="llp{2.6in}rrrr", panels=panels)


def table2_splits(inp: ReportInputs) -> Table:
    s = inp.split_summary
    s = s[s["profile"] == inp.cfg.profile]
    g = s.groupby("scheme")
    frame = pd.DataFrame({
        "Folds": g.size(), "Train-fit": g["n_train_fit"].mean().round(0).astype(int),
        "Cal": g["n_cal"].mean().round(0).astype(int), "Test": g["n_test"].mean().round(0).astype(int),
        "Test events": g["events_test"].mean().round(1), "Test stations": g["stations_test"].mean().round(1),
        "Seen event (%)": (g["test_records_seen_event"].mean() * 100).round(1),
        "Seen station (%)": (g["test_records_seen_station"].mean() * 100).round(1),
        "Audit": g["passed"].all().map({True: "pass", False: "FAIL"}),
    })
    frame = frame.loc[ordered(frame.index, SCHEME_ORDER)]
    frame.insert(0, "Scheme", [SCHEME_LABELS_PLAIN.get(i, i) for i in frame.index])
    frame = frame.reset_index(drop=True)
    for col in ("Train-fit", "Cal", "Test"):
        frame[col] = frame[col].map(fmt)
    return Table("table2_splits", frame,
                 f"Benchmark split schemes ({inp.cfg.profile} profile): mean records per fold and the share "
                 "of test records whose event or station also occurs in training.",
                 "Seen event / seen station are reported, not failed, except where the scheme groups by that key.")


def table3_accuracy(inp: ReportInputs) -> Table:
    acc = inp.accuracy(include_site=True)
    piv = acc.groupby(["model", "scheme"])["rmse"].mean().unstack("scheme")
    piv = piv.loc[ordered(piv.index, MODEL_ORDER), ordered(piv.columns, SCHEME_ORDER)]
    frame = piv.apply(lambda col: col.map(fmt)).reset_index()
    frame.columns = ["Model"] + [SCHEME_LABELS_PLAIN.get(c, c) for c in piv.columns]
    frame["Model"] = frame["Model"].map(lambda m: MODEL_LABELS.get(m, m))
    bold = {(int(np.nanargmin(piv[c].to_numpy(dtype=float))), j + 1) for j, c in enumerate(piv.columns)
            if piv[c].notna().any()}
    return Table("table3_accuracy", frame,
                 f"Test RMSE (ln units) by model and split scheme ({inp.cfg.profile} profile), averaged over "
                 f"folds and {len(IM_NAMES)} intensity measures.",
                 "Classical models use the train_fit bias term (ergodic mode). (site) = the same model with "
                 "train_fit station terms added (zero for stations unseen in train_fit), i.e. with the same "
                 "site information as the machine-learning models. Lowest value per scheme in bold.",
                 bold=bold)


def _ablation_block(cm: pd.DataFrame, alpha: float) -> pd.DataFrame:
    a = cm[(cm["alpha"].round(4) == round(alpha, 4)) & (cm["im"] == "ALL")]
    cov = a.groupby(["method", "scheme"])["coverage"].mean().unstack("scheme")
    wid = a.groupby(["method", "scheme"])["width"].mean().unstack("scheme")
    methods, schemes = ordered(cov.index, METHOD_ORDER), ordered(cov.columns, SCHEME_ORDER)
    cells = {SCHEME_LABELS_PLAIN.get(s, s): [f"{cov.loc[m, s]:.3f} ({wid.loc[m, s]:.2f})"
                                             if np.isfinite(cov.loc[m, s]) else "--" for m in methods]
             for s in schemes}
    return pd.DataFrame({"Method": [METHOD_LABELS.get(m, m) for m in methods], **cells})


def table4_conformal_ablation(inp: ReportInputs) -> Table:
    cm = inp.conformal_metrics
    cm = cm[cm["profile"] == inp.cfg.profile]
    alphas = sorted(cm["alpha"].unique(), reverse=True)
    blocks = [_ablation_block(cm, a) for a in alphas]
    frame = pd.concat(blocks, ignore_index=True)
    panels, start = {}, 0
    for a, b in zip(alphas, blocks):
        panels[start] = f"Nominal coverage {1 - a:.0%}"
        start += len(b)
    n_models = cm["model"].nunique()
    return Table("table4_conformal_ablation", frame,
                 f"Empirical coverage and mean interval width (ln units, in parentheses) by uncertainty method "
                 f"and split scheme ({inp.cfg.profile} profile), averaged over {n_models} models, folds and "
                 f"{len(IM_NAMES)} intensity measures.", panels=panels)


TABLE5_BINS: dict[str, tuple[tuple[str, str], ...]] = {
    "mw": (("Mw<5", "Mw < 5"), ("5<=Mw<6.5", "5 <= Mw < 6.5"), ("Mw>=6.5", "Mw >= 6.5")),
    "rrup": (("Rrup<30", "Rrup < 30 km"), ("30<=Rrup<100", "30 <= Rrup < 100 km"), ("Rrup>=100", "Rrup >= 100 km")),
    "vs30_class": (("A/B (>=760)", "A/B (>= 760 m/s)"), ("C (360-760)", "C (360-760 m/s)"),
                   ("D (180-360)", "D (180-360 m/s)"), ("E (<180)", "E (< 180 m/s)")),   # stiff -> soft
    "station": (("seen", "Seen"), ("unseen", "Unseen")),
}
MAIN_METHODS = ("UncalibratedGaussian", "NaiveSplitConformal", "ClusteredEventConformal",
                "WeightedShiftConformal", "HSCP")


def table5_conditional_coverage(inp: ReportInputs) -> Table:
    c = inp.conditional
    c = c[(c["profile"] == inp.cfg.profile) & (c["alpha"].round(4) == round(inp.cfg.alpha, 4))
          & c["method"].isin(MAIN_METHODS)]
    g = c.groupby(["stratum", "bin", "method"])[["covered", "n"]].sum()
    cov = (g["covered"] / g["n"]).unstack("method")
    methods = ordered(cov.columns, METHOD_ORDER)
    names = {"mw": "Magnitude", "rrup": "Distance", "vs30_class": "VS30 class (NEHRP)", "station": "Station"}
    rows, panels = [], {}
    for stratum in [s for s in ("mw", "rrup", "vs30_class", "station") if s in cov.index.get_level_values(0)]:
        panels[len(rows)] = names[stratum]
        sub = cov.loc[stratum]
        n = g["n"].groupby(level=[0, 1]).max().loc[stratum]
        known = dict(TABLE5_BINS[stratum])
        order = [b for b, _ in TABLE5_BINS[stratum] if b in sub.index] + sorted(set(sub.index) - set(known))
        for b in order:
            rows.append({"Bin": known.get(b, b), "Records x IMs": fmt(int(n.loc[b])),
                         **{METHOD_LABELS.get(m, m): fmt(sub.loc[b, m]) for m in methods}})
    frame = pd.DataFrame(rows)
    return Table("table5_conditional_coverage", frame,
                 f"Conditional coverage at nominal {1 - inp.cfg.alpha:.0%} by magnitude, rupture distance, "
                 f"VS30 class and station status ({inp.cfg.profile} profile), pooled over schemes, models and folds.",
                 "Seen = station recorded in train_fit or cal.", panels=panels)


# ------------------------------------------------------------------ supplement
def tableS1_per_im_metrics(inp: ReportInputs) -> Table:
    acc = inp.accuracy()
    acc = acc[acc["scheme"] == inp.cfg.accuracy_scheme]
    models = [m for m in inp.cfg.focus_models if m in set(acc["model"])]
    rm = acc.groupby(["im", "model"])["rmse"].mean().unstack("model")
    cm = inp.conformal_metrics
    cm = cm[(cm["profile"] == inp.cfg.profile) & (cm["scheme"] == inp.cfg.accuracy_scheme)
            & (cm["alpha"].round(4) == round(inp.cfg.alpha, 4)) & cm["im"].isin(IM_NAMES)]
    cov = cm.groupby(["im", "method"])["coverage"].mean().unstack("method")
    wid = cm.groupby(["im", "method"])["width"].mean().unstack("method")
    short = {"Hybrid_ASK14_MLP": "Hyb-A", "Hybrid_BSSA14_XGB": "Hyb-B", "MixedEffects": "ME",
             "SpectralMLP": "MLP", "XGBoost": "XGB"}
    rows = []
    for im in IM_NAMES:
        r: dict[str, object] = {"IM": im.replace("SA_", "SA ").replace("s", " s") if im.startswith("SA_") else im}
        for m in models:
            r[short.get(m, MODEL_LABELS.get(m, m))] = fmt(rm.loc[im, m]) if im in rm.index else "--"
        for meth, head in (("NaiveSplitConformal", "Cov split"), ("HSCP", "Cov HSCP")):
            if meth in cov.columns:
                r[head] = fmt(cov.loc[im, meth]) if im in cov.index else "--"
        if "HSCP" in wid.columns:
            r["Width HSCP"] = fmt(wid.loc[im, "HSCP"]) if im in wid.index else "--"
        rows.append(r)
    used = [f"{short[m]} = {MODEL_LABELS.get(m, m)}" for m in models if m in short]
    return Table("tableS1_per_im_metrics", pd.DataFrame(rows),
                 f"Per-IM test RMSE (ln units; one column per model) and calibrated-band coverage and width at "
                 f"{1 - inp.cfg.alpha:.0%} ({SCHEME_LABELS_PLAIN.get(inp.cfg.accuracy_scheme)} split, "
                 f"{inp.cfg.profile} profile).",
                 ("Abbreviations: " + "; ".join(used) + ".") if used else "")


def tableS2_sp23_comparison(inp: ReportInputs) -> Table:
    p, c = inp.cfg.profile, inp.cfg.compare_profile
    rows = []
    for prof in (p, c):
        ds = inp.dataset(prof)
        rows.append({"Item": f"{prof}: records / events / stations",
                     "Value": f"{len(ds):,} / {ds.df['eqid'].nunique():,} / {ds.df['ssn'].nunique():,}"})
    ref = SP23_REFERENCE
    rows.append({"Item": "Sedaghati and Pezeshk (2023): records / events / stations",
                 "Value": f"{ref['records']:,} / {ref['events']:,} / {ref['stations']:,}"})
    acc_p = inp.accuracy(p)
    acc_c = inp.accuracy(c)
    sch = inp.cfg.accuracy_scheme
    for m in ordered(set(acc_p["model"]) & set(acc_c["model"]), MODEL_ORDER):
        a = acc_p[(acc_p["scheme"] == sch) & (acc_p["model"] == m)]["rmse"].mean()
        b = acc_c[(acc_c["scheme"] == sch) & (acc_c["model"] == m)]["rmse"].mean()
        rows.append({"Item": f"RMSE {MODEL_LABELS.get(m, m)} ({p} / {c})", "Value": f"{a:.3f} / {b:.3f}"})
    return Table("tableS2_sp23_comparison", pd.DataFrame(rows),
                 f"Profile comparison: {p} versus {c} (the Sedaghati and Pezeshk 2023 selection) and test RMSE "
                 f"under the {SCHEME_LABELS_PLAIN.get(sch)} split.",
                 "Published counts are reference values from the cited study; all other values are computed here.",
                 align="lr")


def _ci(est: float, lo: float, hi: float, signed: bool = False) -> str:
    if not np.isfinite(est):
        return "--"
    f = (lambda v: f"{v:+.3f}") if signed else (lambda v: f"{v:.3f}")
    return f"{f(est)} [{f(lo)}, {f(hi)}]"


def tableS3_bootstrap_cis(inp: ReportInputs) -> Table:
    """Pairwise RMSE differences (panel A) and coverage at each nominal level (panels B, C)."""
    bs = inp.bootstrap
    if bs.rmse_diff.empty and bs.coverage.empty:
        raise MissingInput("; ".join(bs.skipped.values()) or "no bootstrap results")
    schemes = ordered(list(bs.rmse_diff.get("scheme", [])) + list(bs.coverage.get("scheme", [])), SCHEME_ORDER)
    cols = [SCHEME_LABELS_PLAIN.get(x, x) for x in schemes]
    rows: list[dict[str, str]] = []
    panels: dict[int, str] = {}
    if not bs.rmse_diff.empty:
        panels[0] = "A. RMSE difference, first minus second model (ln units); negative favours the first"
        d = bs.rmse_diff.set_index(["model_a", "model_b", "scheme"])
        for a, b in dict.fromkeys(zip(bs.rmse_diff["model_a"], bs.rmse_diff["model_b"])):
            row = {"Quantity": f"{MODEL_LABELS.get(a, a)} - {MODEL_LABELS.get(b, b)}"}
            for sc, col in zip(schemes, cols):
                r = d.loc[(a, b, sc)] if (a, b, sc) in d.index else None
                row[col] = _ci(r["estimate"], r["lo"], r["hi"], signed=True) if r is not None else "--"
            rows.append(row)
    if not bs.coverage.empty:
        c = bs.coverage.set_index(["method", "alpha", "scheme"])
        for alpha in sorted(bs.coverage["alpha"].unique(), reverse=True):
            panels[len(rows)] = f"{'ABC'[len(panels)]}. Empirical coverage, nominal {1 - alpha:.0%}"
            for m in ordered(bs.coverage["method"].unique(), METHOD_ORDER):
                row = {"Quantity": METHOD_LABELS.get(m, m)}
                for sc, col in zip(schemes, cols):
                    r = c.loc[(m, alpha, sc)] if (m, alpha, sc) in c.index else None
                    row[col] = _ci(r["estimate"], r["lo"], r["hi"]) if r is not None else "--"
                rows.append(row)
    ev = bs.meta.get("events_per_scheme", {})
    events = ", ".join(f"{SCHEME_LABELS_PLAIN.get(k, k)} {ev[k]}" for k in ordered(ev, SCHEME_ORDER))
    return Table("tableS3_bootstrap_cis", pd.DataFrame(rows, columns=["Quantity"] + cols),
                 f"Event-clustered bootstrap {inp.cfg.bootstrap_level:.0%} confidence intervals "
                 f"({inp.cfg.profile} profile; {inp.cfg.bootstrap_b:,} replicates resampling whole earthquakes "
                 "within each split scheme): estimate [lower, upper].",
                 "Estimates are the Table 3 and Table 4 statistics recomputed from per-earthquake sums; each "
                 "replicate draws earthquakes with replacement and keeps all of their records and intensity "
                 "measures together, and both models of a difference use the same draw. "
                 + (f"Earthquakes per scheme: {events}." if events else ""),
                 align="l" + "c" * len(cols), panels=panels, stacked=True)


def tableS4_station_rmse(inp: ReportInputs) -> Table:
    """RMSE on test records at stations seen vs unseen in train_fit, for models with site information."""
    st = inp.bootstrap.station
    if st.empty:
        raise MissingInput(inp.bootstrap.skipped.get("station", "no station breakdown"))
    schemes = ordered(st["scheme"].unique(), SCHEME_ORDER)
    cols = [SCHEME_LABELS_PLAIN.get(x, x) for x in schemes]
    rows, panels = [], {}
    idx = st.set_index(["status", "model", "scheme"])
    for status, title in (("seen", "Stations with train_fit records (seen)"),
                          ("unseen", "Stations without train_fit records (unseen)")):
        sub = st[st["status"] == status]
        if sub.empty:
            continue
        panels[len(rows)] = title
        for m in ordered(sub["model"].unique(), MODEL_ORDER):
            row = {"Model": MODEL_LABELS.get(m, m)}
            for sc, col in zip(schemes, cols):
                r = idx.loc[(status, m, sc)] if (status, m, sc) in idx.index else None
                row[col] = _ci(r["estimate"], r["lo"], r["hi"]) if r is not None else "--"
            rows.append(row)
        cnt = sub.groupby("scheme")[["n_records", "n_events"]].max()
        rows.append({"Model": "Test records (earthquakes)",
                     **{col: (f"{int(cnt.loc[sc, 'n_records']):,} ({int(cnt.loc[sc, 'n_events'])})"
                              if sc in cnt.index else "--") for sc, col in zip(schemes, cols)}})
    return Table("tableS4_station_rmse", pd.DataFrame(rows, columns=["Model"] + cols),
                 f"Test RMSE (ln units) at stations seen and unseen in train_fit ({inp.cfg.profile} profile), for "
                 "models that use station information: estimate [event-clustered bootstrap "
                 f"{inp.cfg.bootstrap_level:.0%} interval].",
                 "(site) = NGA-West2 model plus train_fit station terms; the hybrids encode stations from "
                 "train_fit only. Both therefore have the same site information, and unseen stations receive "
                 "no station term in either. RMSE is averaged over folds and intensity measures as in Table 3.",
                 align="l" + "c" * len(cols), panels=panels, stacked=True)


MAIN_TABLES: dict[str, Callable[[ReportInputs], Table]] = {
    "table1_attrition": table1_attrition, "table2_splits": table2_splits, "table3_accuracy": table3_accuracy,
    "table4_conformal_ablation": table4_conformal_ablation,
    "table5_conditional_coverage": table5_conditional_coverage,
}
SUPPLEMENT_TABLES: dict[str, Callable[[ReportInputs], Table]] = {
    "tableS1_per_im_metrics": tableS1_per_im_metrics, "tableS2_sp23_comparison": tableS2_sp23_comparison,
    "tableS3_bootstrap_cis": tableS3_bootstrap_cis, "tableS4_station_rmse": tableS4_station_rmse,
}


def build_all_tables(inp: ReportInputs, log: Callable[[str], None] = print
                     ) -> tuple[dict[str, list[Path]], dict[str, str], dict[str, Table]]:
    """Write every table; returns (paths, skipped reasons, Table objects)."""
    out = inp.cfg.paper_dir / "tables"
    paths: dict[str, list[Path]] = {}
    skipped: dict[str, str] = {}
    tables: dict[str, Table] = {}
    for name, fn in {**MAIN_TABLES, **SUPPLEMENT_TABLES}.items():
        try:
            t = fn(inp)
            paths[name] = write_table(t, out)
            tables[name] = t
            log(f"table  {name}: {len(t.frame)} rows")
        except (MissingInput, KeyError) as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"[:240]
            log(f"table  {name}: SKIPPED ({skipped[name]})")
    return paths, skipped, tables
