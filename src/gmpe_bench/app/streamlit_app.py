"""gmpe-bench explorer: Streamlit front end for the published NGA-West2 benchmark.

Run with ``gmpe-bench app`` (or ``streamlit run app.py`` from the repository root). The app needs neither
the PEER flatfiles nor ``full.parquet``: it reads the paper results (repository or packaged copy).
Tab 3 runs the full audited evaluation only when the curated profile exists locally.
"""

from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st

from gmpe_bench import __version__
from gmpe_bench.app import data as D
from gmpe_bench.app import validation as V

REPO_URL = "https://github.com/PavanMohanN/gmpe-benchmarking"
DEFAULT_MODELS = ["ASK14_cal", "ASK14_site", "Hybrid_ASK14_MLP", "Hybrid_BSSA14_XGB"]
STATUS_ICON = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL", "SKIP": "⏭️ SKIP"}
REGIME_ORDER = [D.label(s) for s in D.SCHEMES]


def _streamlit_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except ValueError:  # pragma: no cover - unusual version strings
        return (1, 0)


# Streamlit >= 1.50 sizes elements with width="stretch"/"content"; older releases use use_container_width.
_NEW_WIDTH = _streamlit_version() >= (1, 50)
STRETCH = {"width": "stretch"} if _NEW_WIDTH else {"use_container_width": True}
CONTENT = {"width": "content"} if _NEW_WIDTH else {"use_container_width": False}


@st.cache_data(show_spinner=False)
def _paper_data() -> D.PaperData:
    return D.load_paper_data()


@st.cache_data(show_spinner=False)
def _splits(profile: str) -> dict | None:
    return V.load_splits_payload(profile)


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =============================================================================== tab 1
def tab_benchmark(d: D.PaperData) -> None:
    st.markdown("#### Leakage inflation: random record split vs event-grouped split")
    st.caption("How much a random split flatters each model: the rise in test RMSE when whole earthquakes "
               "are held out. Fixed-coefficient published models barely move; flexible learners do.")
    leak = d.leakage()
    c1, c2 = st.columns([3, 2])
    with c1:
        chart = alt.Chart(leak).mark_bar().encode(
            x=alt.X("inflation_pct:Q", title="RMSE increase, Random → Event (%)"),
            y=alt.Y("label:N", sort=leak["label"].tolist(), title=None),
            color=alt.Color("family:N", title="Family"),
            tooltip=["label", "family", alt.Tooltip("rmse_random:Q", format=".3f"),
                     alt.Tooltip("rmse_event:Q", format=".3f"), alt.Tooltip("inflation_pct:Q", format="+.1f")])
        st.altair_chart(chart, **STRETCH)
    with c2:
        st.dataframe(leak[["label", "rmse_random", "rmse_event", "inflation_pct"]].rename(
            columns={"label": "Model", "rmse_random": "RMSE random", "rmse_event": "RMSE event",
                     "inflation_pct": "Inflation %"}).style.format(precision=3), hide_index=True,
            **STRETCH)

    st.markdown("#### Equal-information accuracy across deployment regimes")
    st.caption("ASK14 (site) adds train-fit station terms, the same station information the learned models "
               "receive. Bars are 95% event-clustered bootstrap intervals (1,000 earthquake resamples).")
    acc = d.accuracy()
    models_all = list(dict.fromkeys(acc["model"]))
    c1, c2 = st.columns(2)
    regimes = c1.multiselect("Regimes", D.SCHEMES, default=list(D.SCHEMES), format_func=D.label, key="t1_reg")
    models = c2.multiselect("Models", models_all, default=[m for m in DEFAULT_MODELS if m in models_all],
                            format_func=D.label, key="t1_mod")
    sel = acc[acc["scheme"].isin(regimes) & acc["model"].isin(models)].copy()
    sel["regime"] = sel["regime"].astype(str)
    if sel.empty:
        st.info("Choose at least one regime and one model.")
    else:
        base = alt.Chart(sel).encode(
            x=alt.X("regime:N", sort=REGIME_ORDER, title=None),
            xOffset=alt.XOffset("label:N", sort=[D.label(m) for m in models]),
            color=alt.Color("label:N", sort=[D.label(m) for m in models], title="Model"))
        points = base.mark_point(filled=True, size=70).encode(
            y=alt.Y("rmse:Q", title="Test RMSE (ln units)", scale=alt.Scale(zero=False)),
            tooltip=["regime", "label", alt.Tooltip("rmse:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"),
                     alt.Tooltip("hi:Q", format=".3f")])
        bars = base.mark_rule().encode(y="lo:Q", y2="hi:Q")
        st.altair_chart((bars + points).properties(height=360), **STRETCH)

    diff = d.differences()
    if not diff.empty:
        dsel = diff[diff["scheme"].isin(regimes)]
        with st.expander("Paired RMSE differences with 95% bootstrap intervals (Table S3)", expanded=True):
            show = dsel.assign(**{"95% CI": dsel.apply(lambda r: f"[{r['lo']:+.3f}, {r['hi']:+.3f}]", axis=1),
                                  "Resolved": dsel["resolved"].map({True: "yes", False: "tie"})})
            piv = show.pivot_table(index="comparison", columns="regime", values="estimate", sort=False)
            st.dataframe(show[["regime", "comparison", "estimate", "95% CI", "Resolved"]].rename(
                columns={"regime": "Regime", "comparison": "Difference (A − B)", "estimate": "Estimate"})
                .style.format({"Estimate": "{:+.3f}"}), hide_index=True, **STRETCH)
            st.caption("Negative values favor the first model. 'tie' means the interval contains zero. "
                       f"{len(piv)} comparisons shown.")

    st.markdown("#### Calibration: coverage vs mean width")
    c1, c2 = st.columns([1, 3])
    level = c1.radio("Nominal level", ["0.90", "0.95"], format_func=lambda x: f"{float(x):.0%}", key="t1_lvl")
    ablations = c1.checkbox("Show HSCP ablations", value=False, key="t1_abl")
    cal = d.calibration(level)
    cal = cal[cal["scheme"].isin(regimes or list(D.SCHEMES))]
    if not ablations:
        cal = cal[cal["main"]]
    nominal = float(level)
    pts = alt.Chart(cal).mark_point(filled=True, size=80).encode(
        x=alt.X("width:Q", title="Mean interval width (ln units)", scale=alt.Scale(zero=False)),
        y=alt.Y("coverage:Q", title="Empirical coverage", scale=alt.Scale(zero=False)),
        color=alt.Color("label:N", title="Method"),
        tooltip=["regime", "label", alt.Tooltip("coverage:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"),
                 alt.Tooltip("hi:Q", format=".3f"), alt.Tooltip("width:Q", format=".2f")])
    err = alt.Chart(cal).mark_rule().encode(x="width:Q", y="lo:Q", y2="hi:Q", color="label:N")
    ref = alt.Chart(pd.DataFrame({"y": [nominal]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    facet = (err + pts + ref).properties(width=230, height=200).facet(
        facet=alt.Facet("regime:N", sort=REGIME_ORDER, title=None), columns=3)
    c2.altair_chart(facet, **CONTENT)
    mace = d.mace(level)
    if not ablations:
        mace = mace[mace["method"].isin(D.MAIN_METHODS)]
    c1.markdown("**Mean absolute coverage error** (6 regimes)")
    c1.dataframe(mace[["label", "mace"]].rename(columns={"label": "Method", "mace": "MACE"})
                 .style.format({"MACE": "{:.3f}"}), hide_index=True, **STRETCH)
    st.caption("Values are means over eight models (four published, four learned). Event-clustered intervals "
               "resample whole earthquakes; the Mw ≥ 7 regime rests on 20 test earthquakes.")


# =============================================================================== tab 2
ACC_METRICS = {"rmse": "Test RMSE (ln units)", "cov90": "Gaussian 90% coverage (model σ)",
               "width90": "Gaussian 90% width", "tau": "τ between-event", "phi_s2s": "φS2S site-to-site",
               "phi_0": "φ0 within-event"}


def tab_period(d: D.PaperData) -> None:
    st.markdown("#### Period-by-period accuracy and variance components")
    pa = d.period_accuracy
    c1, c2, c3 = st.columns([1, 1, 2])
    regime = c1.selectbox("Regime", D.SCHEMES, format_func=D.label, key="t2_reg")
    metric = c2.selectbox("Metric", list(ACC_METRICS), format_func=ACC_METRICS.get, key="t2_met")
    models_all = list(dict.fromkeys(pa["model"]))
    defaults = ["ASK14_cal", "ASK14_site", "Hybrid_BSSA14_XGB", "XGBoost"]
    models = c3.multiselect("Models", models_all, default=[m for m in defaults if m in models_all],
                            format_func=D.label, key="t2_mod")
    sub = pa[(pa["scheme"] == regime) & pa["model"].isin(models)].assign(label=lambda x: x["model"].map(D.label))
    if metric in ("tau", "phi_s2s", "phi_0") and sub["model"].str.endswith("_site").any():
        st.caption("Variance components are computed on ergodic residuals, so they are not shown for (site) models.")
    sa = D.period_axis(sub).dropna(subset=[metric])
    if not sa.empty:
        line = alt.Chart(sa).mark_line(point=True).encode(
            x=alt.X("period:Q", scale=alt.Scale(type="log"), title="Period T (s)"),
            y=alt.Y(f"{metric}:Q", title=ACC_METRICS[metric], scale=alt.Scale(zero=False)),
            color=alt.Color("label:N", title="Model"),
            tooltip=["label", "im", alt.Tooltip(f"{metric}:Q", format=".3f")])
        st.altair_chart(line.properties(height=340), **STRETCH)
    pg = sub[sub["im"].isin(["PGA", "PGV"])].pivot_table(index="label", columns="im", values=metric)
    if not pg.empty:
        st.dataframe(pg.style.format("{:.3f}"), **STRETCH)

    st.markdown("#### Calibrated bands per period")
    pc = d.period_calibration
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    regime_c = c1.selectbox("Regime", D.SCHEMES, format_func=D.label, key="t2c_reg")
    level = c2.radio("Nominal", [0.9, 0.95], format_func=lambda x: f"{x:.0%}", key="t2c_lvl", horizontal=True)
    methods = c3.multiselect("Methods", list(D.METHOD_LABELS), default=["NaiveSplitConformal",
                             "ClusteredEventConformal", "HSCP"], format_func=D.label, key="t2c_meth")
    model = c4.selectbox("Model", ["mean"] + list(dict.fromkeys(pc["model"])),
                         format_func=lambda m: "Mean of 8 models" if m == "mean" else D.label(m), key="t2c_model")
    sub = pc[(pc["scheme"] == regime_c) & pc["method"].isin(methods) & (pc["nominal"].round(3) == level)]
    if model != "mean":
        sub = sub[sub["model"] == model]
    sub = (sub.groupby(["method", "im"], sort=False).agg(period=("period", "first"), coverage=("coverage", "mean"),
                                                         width=("width", "mean")).reset_index())
    sub["label"] = sub["method"].map(D.label)
    sa = D.period_axis(sub)
    if not sa.empty:
        x = alt.X("period:Q", scale=alt.Scale(type="log"), title="Period T (s)")
        cov = alt.Chart(sa).mark_line(point=True).encode(
            x=x, y=alt.Y("coverage:Q", title="Coverage", scale=alt.Scale(zero=False)),
            color=alt.Color("label:N", title="Method"), tooltip=["label", "im", alt.Tooltip("coverage:Q", format=".3f")])
        nom = alt.Chart(pd.DataFrame({"y": [level]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        wid = alt.Chart(sa).mark_line(point=True).encode(
            x=x, y=alt.Y("width:Q", title="Mean width (ln units)", scale=alt.Scale(zero=False)),
            color=alt.Color("label:N", title="Method"), tooltip=["label", "im", alt.Tooltip("width:Q", format=".2f")])
        c1, c2 = st.columns(2)
        c1.altair_chart((cov + nom).properties(height=300), **STRETCH)
        c2.altair_chart(wid.properties(height=300), **STRETCH)

    st.markdown("#### Whole-spectrum coverage: HSCP simultaneous band vs marginal bands")
    st.caption("Share of test records whose entire usable spectrum lies inside the band. Marginal bands are "
               "valid one period at a time and enclose far fewer whole spectra.")
    lvl = st.radio("Nominal level", ["0.90", "0.95"], format_func=lambda x: f"{float(x):.0%}", key="t2j_lvl",
                   horizontal=True)
    cal = d.calibration(lvl)
    rows = []
    for _, r in cal[cal["scheme"].isin(D.SCHEMES)].iterrows():
        if r["method"] == "HSCP":
            rows += [{"regime": r["regime"], "quantity": "HSCP per-period (marginal) coverage", "value": r["coverage"]},
                     {"regime": r["regime"], "quantity": "HSCP marginal bands: whole spectrum", "value": r["joint_marginal"]},
                     {"regime": r["regime"], "quantity": "HSCP simultaneous band: whole spectrum", "value": r["joint_band"]}]
        if r["method"] == "NaiveSplitConformal":
            rows.append({"regime": r["regime"], "quantity": "Split CP marginal bands: whole spectrum",
                         "value": r["joint_marginal"]})
    jdf = pd.DataFrame(rows)
    order = ["HSCP per-period (marginal) coverage", "Split CP marginal bands: whole spectrum",
             "HSCP marginal bands: whole spectrum", "HSCP simultaneous band: whole spectrum"]
    bars = alt.Chart(jdf).mark_bar().encode(
        x=alt.X("regime:N", sort=REGIME_ORDER, title=None), xOffset=alt.XOffset("quantity:N", sort=order),
        y=alt.Y("value:Q", title="Coverage", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("quantity:N", sort=order, title=None, legend=alt.Legend(orient="bottom", columns=2)),
        tooltip=["regime", "quantity", alt.Tooltip("value:Q", format=".3f")])
    nom = alt.Chart(pd.DataFrame({"y": [float(lvl)]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    st.altair_chart((bars + nom).properties(height=340), **STRETCH)

    with st.expander("Table S1: per-IM metrics under the Event split (paper)"):
        st.dataframe(d.tables.get("tableS1_per_im_metrics", pd.DataFrame()), hide_index=True, **STRETCH)


# =============================================================================== tab 3
def tab_upload() -> None:
    st.markdown("#### Evaluate your own predictions")
    st.caption("Long-format CSV with columns rsn, im, mu (ln units) and optionally sigma, scheme and fold. "
               "With scheme and fold, each fold's calibration and test records are scored separately "
               "(recommended); without them the file holds out-of-fold predictions for one scheme.")
    c1, c2, c3 = st.columns(3)
    profile = c1.selectbox("Profile", ["full", "sp23"], key="t3_prof")
    payload = _splits(profile)
    schemes = list(payload["schemes"]) if payload else list(D.SCHEMES) + ["random"]
    scheme = c2.selectbox("Scheme (for files without scheme/fold)", schemes, format_func=D.label, key="t3_scheme")
    local = D.local_profile_available(profile)
    c3.metric("Mode", "Local: full evaluation" if local else "Hosted: audit only")

    with st.expander("Templates", expanded=False):
        tpl = V.sample_template(payload, scheme if scheme != "random" else "event_grouped")
        st.download_button("Download sample CSV (10 records × 27 IMs)", _csv(tpl), "gmpe_bench_sample_predictions.csv",
                           "text/csv", key="t3_tpl")
        if payload:
            fold = st.selectbox("Fold skeleton", [f["fold"] for f in payload["schemes"][scheme]], key="t3_fold")
            if st.checkbox("Prepare skeleton of all calibration and test records of this fold", key="t3_skel"):
                sk = V.fold_skeleton(payload, scheme, fold)
                st.download_button(f"Download skeleton ({len(sk):,} rows)", _csv(sk),
                                   f"skeleton_{profile}_{scheme}_{fold}.csv", "text/csv", key="t3_skel_dl")
        st.dataframe(tpl.head(8), hide_index=True, **STRETCH)

    up = st.file_uploader("Predictions CSV", type=["csv"], key="t3_upload")
    if up is None:
        if not payload:
            st.info(f"Frozen splits_{profile}.json not found: uploads get schema checks only.")
        return
    try:
        df = V.read_predictions_csv(io.BytesIO(up.getvalue()))
    except Exception as exc:  # noqa: BLE001 - any parse error is shown to the user
        st.error(f"Could not read the file as CSV: {exc}")
        return
    res = V.audit_predictions(df, payload, scheme, profile)
    st.markdown(f"**Audit** ({res.mode} mode, {len(df):,} rows)")
    st.dataframe(res.table().assign(status=lambda t: t["status"].map(STATUS_ICON)), hide_index=True,
                 **STRETCH)
    if len(res.folds):
        st.dataframe(res.folds.style.format({"test_coverage": "{:.1%}", "cal_coverage": "{:.1%}",
                                             "ims_per_test_record": "{:.1f}"}), hide_index=True,
                     **STRETCH)
    if not res.ok:
        st.error("Fix the failing checks and upload again.")
        return
    if local:
        n_boot = st.slider("Bootstrap replicates", 0, 1000, 200, 100, key="t3_boot")
        with st.spinner("Running gmpe_bench.evaluate_predictions on the curated NGA-West2 profile ..."):
            try:
                report = V.run_local_evaluation(res, D.data_dir(), n_boot)
            except Exception as exc:  # noqa: BLE001 - surfaced in the page
                st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
                return
        st.success("Audited NGA-West2 evaluation complete.")
        st.code(report.summary(), language=None)
        st.dataframe(report.accuracy, hide_index=True, **STRETCH)
        st.dataframe(report.calibration, hide_index=True, **STRETCH)
        st.download_button("Download report (long CSV)", _csv(report.to_dataframe()), "gmpe_bench_report.csv",
                           "text/csv", key="t3_rep")
    else:
        st.info("Hosted mode: the NGA-West2 targets are not available here (PEER data cannot be redistributed), "
                "so the file was audited against the frozen splits only. Run the full evaluation locally:")
        st.code(V.local_snippet(up.name, profile, None if res.mode == "per-fold" else scheme), language="python")


# =============================================================================== page
def main() -> None:
    st.set_page_config(page_title="gmpe-bench explorer", layout="wide")
    d = _paper_data()
    st.title("gmpe-bench: NGA-West2 benchmark explorer")
    st.caption("Leakage-audited benchmark of ground-motion models and calibrated uncertainty (HSCP) on NGA-West2. "
               f"All charts use the published results; source: {d.source}.")
    with st.sidebar:
        st.markdown(f"**gmpe-bench {__version__}**")
        st.markdown(f"[Repository]({REPO_URL})")
        local = D.local_profile_available("full")
        st.markdown("**Curated profile:** " + ("found locally (`full.parquet`)" if local else
                                               "not found (hosted mode)"))
        split = d.splits()
        if not split.empty:
            st.markdown("**Regimes**")
            st.dataframe(split[["regime", "question"]].rename(columns={"regime": "Regime", "question": "Question"}),
                         hide_index=True, **STRETCH)
        for note in d.notes:
            st.caption(note)
    t1, t2, t3 = st.tabs(["Benchmark explorer", "Period & spectral bands", "Evaluate your predictions"])
    with t1:
        tab_benchmark(d)
    with t2:
        tab_period(d)
    with t3:
        tab_upload()


if __name__ == "__main__":
    main()
