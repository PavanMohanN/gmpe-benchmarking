"""Paper figures (vector PDF + 800 dpi PNG) and their captions.

Main figures (``results/paper/figures``)
    fig1_data_splits            data coverage, site classes, usable periods, leakage audit
    fig2_variance_components    tau(T), phi_S2S(T), phi_0(T) on test residuals
    fig3_regime_accuracy        RMSE and Gaussian 90 % coverage by split scheme
    fig4_conformal_ablation     coverage and width by uncertainty method
    fig5_spectral_bands         example response spectra with split-CP and HSCP bands
    fig6_conditional_coverage   coverage by Mw, R_rup, VS30 class and station status
Supplement (``results/paper/supplement``)
    figS1_random_leakage_inflation   RMSE under random vs event-grouped splits
    figS2_period_coverage_heatmap    coverage per IM and scheme, split CP vs HSCP

No titles are drawn; captions go to ``captions.md`` and ``captions.tex``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmpe_bench.conformal.evaluate import MW_LABELS, R_LABELS, VS_LABELS
from gmpe_bench.reporting.inputs import MissingInput, ReportInputs
from gmpe_bench.reporting.style import (
    DOUBLE_COL_IN,
    METHOD_ORDER,
    MODEL_ORDER,
    OKABE_ITO,
    SCHEME_ORDER,
    apply_style,
    label,
    ordered,
    panel_label,
    save,
    scheme_definitions,
    series_style,
    short_scheme,
    style_for,
)
from gmpe_bench.schema import IM_NAMES, period_of
from gmpe_bench.splits.group_splits import assign_region

SA_IMS = [n for n in IM_NAMES if period_of(n) is not None]
SA_T = np.array([period_of(n) for n in SA_IMS])
NEHRP = ((180.0, "E|D"), (360.0, "D|C"), (760.0, "C|B"), (1500.0, "B|A"))
BIN_ORDER: dict[str, tuple[str, ...]] = {
    "mw": MW_LABELS, "rrup": R_LABELS,
    "vs30_class": tuple(reversed(VS_LABELS)),          # stiff rock -> soft soil: A/B, C, D, E
    "station": ("seen", "unseen")}
BIN_LABELS: dict[str, str] = {
    "Mw<5": "<5", "5<=Mw<6.5": "5–6.5", "Mw>=6.5": "≥6.5", "Rrup<30": "<30", "30<=Rrup<100": "30–100",
    "Rrup>=100": "≥100", "E (<180)": "E", "D (180-360)": "D", "C (360-760)": "C", "A/B (>=760)": "A/B",
    "seen": "Seen", "unseen": "Unseen"}
MAIN_METHODS = ("UncalibratedGaussian", "NaiveSplitConformal", "ClusteredEventConformal",
                "WeightedShiftConformal", "HSCP")


@dataclass
class Figure:
    stem: str
    paths: list[Path]
    caption: str
    supplement: bool = False


def _sub(df: pd.DataFrame, **eq: object) -> pd.DataFrame:
    for k, v in eq.items():
        df = df[df[k] == v]
    return df


def _alpha(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return df[df["alpha"].round(4) == round(alpha, 4)]


# ------------------------------------------------------------------ fig 1
def fig1_data_splits(inp: ReportInputs) -> Figure:
    ds = inp.dataset()
    d = ds.df
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 5.2))
    ax = axes[0, 0]
    region = assign_region(d).to_numpy()
    for i, r in enumerate(ordered(set(region), ("California_WUS", "Japan", "Taiwan", "China",
                                               "Europe_MiddleEast", "Other"))):
        m = region == r
        st = series_style(i)
        ax.scatter(d["rrup"][m], d["mw"][m], s=6.0, marker=st["marker"], color=st["color"], linewidths=0,
                   alpha=0.75, label=f"{r.replace('_', ' ')} ({m.sum():,})", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"$R_{\mathrm{rup}}$ (km)")
    ax.set_ylabel(r"$M_w$")
    leg = ax.legend(loc="lower left", markerscale=2, handletextpad=0.2, fontsize=6, frameon=True,
                    framealpha=0.85, edgecolor="none")
    leg.get_frame().set_linewidth(0)
    panel_label(ax, "a")

    ax = axes[0, 1]
    st_vs = d.groupby("ssn")["vs30"].first().to_numpy(dtype=float)
    bins = np.logspace(np.log10(np.nanmin(st_vs) * 0.95), np.log10(np.nanmax(st_vs) * 1.05), 40)
    ax.hist(st_vs, bins=bins, color=OKABE_ITO[3], edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    for v, lab in NEHRP:
        if bins[0] < v < bins[-1]:
            ax.axvline(v, color="black", linestyle=":", linewidth=0.7)
            ax.text(v * 1.02, ax.get_ylim()[1] * 0.98, lab, ha="left", va="top", fontsize=6)
    ax.set_xlabel(r"$V_{S30}$ (m/s)")
    ax.set_ylabel("Stations")
    panel_label(ax, "b")

    ax = axes[1, 0]
    comps = [c for c in ("H", "V") if f"{c}_PGA" in d.columns]
    for i, comp in enumerate(comps):
        n = ds.mask(comp)[SA_IMS].sum().to_numpy()
        st = series_style(i)
        ax.plot(SA_T, n, color=st["color"], marker=st["marker"], linestyle=st["linestyle"], markevery=2,
                label="Horizontal (RotD50)" if comp == "H" else "Vertical")
    ax.set_xscale("log")
    ax.set_xlabel("Period, $T$ (s)")
    ax.set_ylabel("Usable records ($T \\leq 1/\\mathrm{LUF}$)")
    ax.set_ylim(0 if ax.get_ylim()[0] < 0 else ax.get_ylim()[0], ax.get_ylim()[1] * 1.12)
    ax.legend(loc="lower left")
    panel_label(ax, "c")

    ax = axes[1, 1]
    s = inp.split_summary
    s = s[s["profile"] == inp.cfg.profile]
    g = s.groupby("scheme")[["test_records_seen_event", "test_records_seen_station"]].mean() * 100
    g = g.loc[ordered(g.index, SCHEME_ORDER)]
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["test_records_seen_event"], 0.4, color=OKABE_ITO[4], edgecolor="black", linewidth=0.4,
           hatch="///", label="Event seen in training")
    ax.bar(x + 0.2, g["test_records_seen_station"], 0.4, color=OKABE_ITO[1], edgecolor="black", linewidth=0.4,
           label="Station seen in training")
    ax.set_xticks(x, [short_scheme(k) for k in g.index], rotation=0, ha="center")
    ax.set_ylabel("Test records (%)")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper right", ncol=2, fontsize=6)
    panel_label(ax, "d")
    fig.tight_layout()
    paths = save(fig, inp.cfg.paper_dir / "figures", "fig1_data_splits")
    cap = (f"Dataset and benchmark splits ({inp.cfg.profile} profile: {len(d):,} records, "
           f"{d['eqid'].nunique():,} events, {d['ssn'].nunique():,} stations). (a) Magnitude versus rupture "
           "distance by hypocentre region (record counts in the legend). (b) Station $V_{S30}$; dotted lines mark "
           "the NEHRP site-class boundaries E|D at 180 m/s, D|C at 360 m/s, C|B at 760 m/s and B|A at 1500 m/s "
           "(only boundaries within the data range are drawn). (c) Usable records per period after masking "
           "$T > 1/\\mathrm{LUF}$. (d) Share of test records whose event or station also occurs in training, "
           "averaged over the folds of each split scheme. " + scheme_definitions(g.index))
    return Figure("fig1_data_splits", paths, cap)


# ------------------------------------------------------------------ fig 2
def fig2_variance_components(inp: ReportInputs) -> Figure:
    acc = inp.accuracy()
    acc = acc[(acc["scheme"] == inp.cfg.accuracy_scheme) & acc["im"].isin(SA_IMS)]
    models = [m for m in ordered(acc["model"].unique(), MODEL_ORDER)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 2.5), sharey=True)
    comps = (("tau_test", r"$\tau$"), ("phi_s2s_test", r"$\phi_{S2S}$"), ("phi_0_test", r"$\phi_0$"))
    for k, (col, sym) in enumerate(comps):
        ax = axes[k]
        for i, m in enumerate(models):
            v = acc[acc["model"] == m].groupby("im")[col].mean().reindex(SA_IMS).to_numpy()
            st = style_for(m, "model", i)
            ax.plot(SA_T, v, color=st["color"], marker=st["marker"], linestyle=st["linestyle"], markevery=3,
                    label=label(m))
        ax.set_xscale("log")
        ax.set_xlabel("Period, $T$ (s)")
        ax.text(0.97, 0.97, sym, transform=ax.transAxes, ha="right", va="top", fontsize=10)
        panel_label(ax, "abc"[k])
    axes[0].set_ylabel("Standard deviation (ln units)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(5, len(lab)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12 if len(lab) <= 5 else 0.18, 1, 1))
    paths = save(fig, inp.cfg.paper_dir / "figures", "fig2_variance_components")
    cap = (r"Period-dependent variance components of test residuals: (a) between-event $\tau(T)$, "
           r"(b) site-to-site $\phi_{S2S}(T)$ and (c) single-station within-event $\phi_0(T)$, estimated by "
           f"crossed random effects on held-out events ({label(inp.cfg.accuracy_scheme, 'scheme')} split, "
           f"{inp.cfg.profile} profile), averaged over folds.")
    return Figure("fig2_variance_components", paths, cap)


# ------------------------------------------------------------------ fig 3
def _dotplot(ax: plt.Axes, frame: pd.DataFrame, value: str, series: list[str], kind: str,
             schemes: list[str]) -> None:
    width = 0.75
    for i, s in enumerate(series):
        v = frame[frame[kind] == s].groupby("scheme")[value].mean().reindex(schemes).to_numpy()
        x = np.arange(len(schemes)) - width / 2 + width * (i + 0.5) / len(series)
        st = style_for(s, kind, i)
        hollow = "mfc" in st          # site-corrected variants: parent colour, hollow marker
        ax.plot(x, v, linestyle="none", marker=st["marker"], color=st["color"], markersize=4,
                markerfacecolor=st.get("mfc", st["color"]),
                markeredgecolor=st["color"] if hollow else "black", markeredgewidth=0.9 if hollow else 0.3,
                label=label(s, kind))
    ax.set_xticks(np.arange(len(schemes)), [short_scheme(s) for s in schemes], rotation=0, ha="center")
    for k in range(1, len(schemes)):
        ax.axvline(k - 0.5, color="0.85", linewidth=0.5, zorder=0)


def fig3_regime_accuracy(inp: ReportInputs) -> Figure:
    acc = inp.accuracy(include_site=True)
    models = ordered(acc["model"].unique(), MODEL_ORDER)
    schemes = ordered(acc["scheme"].unique(), SCHEME_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.9))
    _dotplot(axes[0], acc, "rmse", models, "model", schemes)
    axes[0].set_ylabel("Test RMSE (ln units)")
    panel_label(axes[0], "a")
    _dotplot(axes[1], acc, "cov90", models, "model", schemes)
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel(r"Coverage of $\hat\mu \pm 1.645\hat\sigma$")
    panel_label(axes[1], "b")
    h, lab = axes[0].get_legend_handles_labels()
    ncol = 5 if len(lab) <= 10 else 6
    rows = -(-len(lab) // ncol)
    fig.legend(h, lab, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.01), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.05 + 0.05 * rows, 1, 1))
    paths = save(fig, inp.cfg.paper_dir / "figures", "fig3_regime_accuracy")
    site = [m for m in models if m.endswith("_site")]
    cap = (f"Accuracy across distribution-shift regimes ({inp.cfg.profile} profile): (a) test RMSE and "
           "(b) empirical coverage of nominal 90% Gaussian intervals (dashed line), averaged over folds and "
           f"{len(IM_NAMES)} intensity measures, for classical NGA-West2 models (train_fit bias term), the "
           "mixed-effects baseline and machine-learning models. "
           + ("Hollow markers (site) are the NGA-West2 models with train_fit station terms added (zero for "
              "stations unseen in train_fit) and the matching single-station sigma. " if site else "")
           + scheme_definitions(schemes))
    return Figure("fig3_regime_accuracy", paths, cap)


# ------------------------------------------------------------------ fig 4
def fig4_conformal_ablation(inp: ReportInputs) -> Figure:
    cm = inp.conformal_metrics
    cm = _alpha(cm[(cm["profile"] == inp.cfg.profile) & (cm["im"] == "ALL")], inp.cfg.alpha)
    methods = [m for m in ordered(cm["method"].unique(), METHOD_ORDER) if m in MAIN_METHODS]
    schemes = ordered(cm["scheme"].unique(), SCHEME_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.9))
    for k, value in enumerate(("coverage", "width")):
        ax = axes[k]
        per_model = cm.groupby(["method", "scheme", "model"])[value].mean().reset_index()
        width = 0.75
        for i, meth in enumerate(methods):
            sub = per_model[per_model["method"] == meth].groupby("scheme")[value]
            mean = sub.mean().reindex(schemes).to_numpy()
            lo, hi = sub.min().reindex(schemes).to_numpy(), sub.max().reindex(schemes).to_numpy()
            x = np.arange(len(schemes)) - width / 2 + width * (i + 0.5) / len(methods)
            st = style_for(meth, "method", i)
            ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt=st["marker"], color=st["color"], markersize=4,
                        markeredgecolor="black", markeredgewidth=0.3, elinewidth=0.6, capsize=1.5,
                        label=label(meth, "method"))
        ax.set_xticks(np.arange(len(schemes)), [short_scheme(s) for s in schemes], rotation=0, ha="center")
        for j in range(1, len(schemes)):
            ax.axvline(j - 0.5, color="0.85", linewidth=0.5, zorder=0)
        panel_label(ax, "ab"[k])
    nominal_line = axes[0].axhline(1 - inp.cfg.alpha, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Mean interval width (ln units)")
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend([nominal_line], [f"Nominal {1 - inp.cfg.alpha:.0%} target"], loc="lower left",
                   handlelength=2.2, borderaxespad=0.4)
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    paths = save(fig, inp.cfg.paper_dir / "figures", "fig4_conformal_ablation")
    cap = (f"Uncertainty-method ablation at nominal {1 - inp.cfg.alpha:.0%} ({inp.cfg.profile} profile): "
           "(a) empirical coverage (dashed line: nominal) and (b) mean interval width, averaged over folds and "
           f"{len(IM_NAMES)} intensity measures. Markers are the mean over {cm['model'].nunique()} models; "
           f"bars span the model range. The dashed line in (a) is the nominal {1 - inp.cfg.alpha:.0%} target. "
           + scheme_definitions(schemes))
    return Figure("fig4_conformal_ablation", paths, cap)


# ------------------------------------------------------------------ fig 5
def _example_bands(inp: ReportInputs, scheme: str, model: str) -> dict[str, object]:
    """Recompute exact split-CP and HSCP bands for one fold (largest test set) of ``scheme``."""
    from gmpe_bench.conformal.evaluate import (
        _load_ml,
        _ml_arrays,
        _y_matrix,
        classical_predictions,
    )
    from gmpe_bench.conformal.methods import (
        HSCP,
        CalSet,
        NaiveSplitConformal,
        TestInputs,
    )
    from gmpe_bench.conformal.shift import density_ratio_weights, shift_features
    from gmpe_bench.splits import load_splits

    ds = inp.dataset()
    split_path = inp.cfg.data_dir / f"splits_{inp.cfg.profile}.json"
    if not split_path.exists():
        raise MissingInput(str(split_path))
    splits = load_splits(split_path, ds)
    if scheme not in splits:
        raise MissingInput(f"scheme {scheme} not in {split_path.name}")
    fold = max(splits[scheme], key=lambda f: f.test.size)
    cal_df, test_df = ds.df.iloc[fold.cal], ds.df.iloc[fold.test]
    rsn = ds.df["rsn"].to_numpy(dtype=np.int64)
    if model.endswith("_cal"):
        cache = inp.cfg.results / "baselines" / "cache"
        if not (cache / f"{model[:-4]}.parquet").exists():
            raise MissingInput(str(cache / f"{model[:-4]}.parquet"))
        p = classical_predictions(ds, fold, model[:-4], cache)
        (mu_c, sd_c), (mu_t, sd_t) = p["cal"], p["test"]
    else:
        ml = _load_ml(inp.cfg.results / "ml", inp.cfg.profile)
        if "cal" not in ml or "test" not in ml:
            raise MissingInput(str(inp.cfg.results / "ml"))
        a = _ml_arrays(ml["cal"], scheme, fold.fold, model, rsn[fold.cal])
        b = _ml_arrays(ml["test"], scheme, fold.fold, model, rsn[fold.test])
        if a is None or b is None:
            raise MissingInput(f"{model} predictions for {scheme}/{fold.fold}")
        (mu_c, sd_c), (mu_t, sd_t) = a, b
    Xc, Xt = shift_features(cal_df), shift_features(test_df)
    wc, wt = density_ratio_weights(Xc, Xt)
    train_st = np.unique(ds.df["ssn"].to_numpy()[fold.train_fit])
    cal = CalSet(_y_matrix(cal_df), mu_c, sd_c, cal_df["eqid"].to_numpy(), cal_df["ssn"].to_numpy(), Xc,
                 train_st, wc)
    test = TestInputs(mu_t, sd_t, test_df["eqid"].to_numpy(), test_df["ssn"].to_numpy(), Xt, wt)
    a_ = inp.cfg.alpha
    return {"fold": fold.fold, "test_df": test_df, "y": _y_matrix(test_df), "mu": mu_t,
            "naive": NaiveSplitConformal().calibrate(cal, test, a_), "hscp": HSCP().calibrate(cal, test, a_)}


def pretty_fold(scheme: str, fold: str) -> str:
    """Readable fold description for captions, e.g. 'Event split, fold 1'."""
    name = label(scheme, "scheme") + " split"
    if fold.startswith("fold") and fold[4:].isdigit():
        return f"{name}, fold {int(fold[4:]) + 1}"
    if fold.startswith("test_"):
        return f"{name}, held-out region {fold[5:].replace('_', ' ')}"
    return name


def _pick_records(test_df: pd.DataFrame, y: np.ndarray, n: int = 3) -> list[int]:
    sa = [IM_NAMES.index(s) for s in SA_IMS]
    full = np.isfinite(y[:, sa]).mean(axis=1) >= 0.9
    idx = np.flatnonzero(full) if full.sum() >= n else np.argsort(-np.isfinite(y[:, sa]).mean(axis=1))[:max(n, 1)]
    mw = test_df["mw"].to_numpy()[idx]
    order = idx[np.argsort(mw, kind="stable")]
    picks = [order[int(round(q * (order.size - 1)))] for q in np.linspace(0.1, 0.9, n)]
    return list(dict.fromkeys(int(p) for p in picks))


def _available_model(inp: ReportInputs) -> str:
    preferred = inp.cfg.example_model
    try:
        present = set(inp.conformal_metrics["model"])
    except MissingInput:
        present = set()
    if preferred in present or not present:
        return preferred
    return ordered(present, MODEL_ORDER)[-1]


def _example_sets(inp: ReportInputs, model: str) -> list[tuple[str, dict[str, object]]]:
    """Requested example schemes; a scheme without predictions is replaced by another available one."""
    try:
        cm = inp.conformal_metrics
        available = ordered(cm[cm["model"] == model]["scheme"].unique(), SCHEME_ORDER)
    except MissingInput:
        available = []
    candidates = list(dict.fromkeys(list(inp.cfg.example_schemes) + available))
    out: list[tuple[str, dict[str, object]]] = []
    for scheme in candidates:
        if len(out) == len(inp.cfg.example_schemes):
            break
        try:
            out.append((scheme, _example_bands(inp, scheme, model)))
        except MissingInput:
            continue
    if not out:
        raise MissingInput(f"no fold with {model} predictions for the example figure")
    return out


def fig5_spectral_bands(inp: ReportInputs) -> Figure:
    model = _available_model(inp)
    sets = _example_sets(inp, model)
    sa = [IM_NAMES.index(s) for s in SA_IMS]
    fig, axes = plt.subplots(len(sets), 3, figsize=(DOUBLE_COL_IN, 2.3 * len(sets)), squeeze=False)
    schemes = [s for s, _ in sets]
    letters = iter("abcdefghijkl")
    folds = []
    for r, (scheme, ex) in enumerate(sets):
        folds.append(pretty_fold(scheme, str(ex["fold"])))
        picks = _pick_records(ex["test_df"], ex["y"])  # type: ignore[arg-type]
        for c in range(3):
            ax = axes[r, c]
            if c >= len(picks):
                ax.axis("off")
                continue
            i = picks[c]
            y, mu = ex["y"][i, sa], ex["mu"][i, sa]  # type: ignore[index]
            nv, hs = ex["naive"], ex["hscp"]
            ax.fill_between(SA_T, np.exp(hs.lower[i, sa]), np.exp(hs.upper[i, sa]), color=OKABE_ITO[0],
                            alpha=0.35, linewidth=0, label=f"HSCP {1 - inp.cfg.alpha:.0%}")
            ax.plot(SA_T, np.exp(nv.lower[i, sa]), color=OKABE_ITO[3], linestyle=":", linewidth=1.0,
                    label=f"Split CP {1 - inp.cfg.alpha:.0%}")
            ax.plot(SA_T, np.exp(nv.upper[i, sa]), color=OKABE_ITO[3], linestyle=":", linewidth=1.0)
            if hs.joint_lower is not None:
                ax.plot(SA_T, np.exp(hs.joint_lower[i, sa]), color=OKABE_ITO[4], linestyle="-.", linewidth=0.8,
                        label="HSCP simultaneous")
                ax.plot(SA_T, np.exp(hs.joint_upper[i, sa]), color=OKABE_ITO[4], linestyle="-.", linewidth=0.8)
            ax.plot(SA_T, np.exp(mu), color="black", linestyle="--", linewidth=0.9, label="Model median")
            ax.plot(SA_T, np.exp(y), color="black", marker="o", markersize=2, linewidth=0.7, label="Observed")
            ax.set_xscale("log")
            ax.set_yscale("log")
            row = ex["test_df"].iloc[i]  # type: ignore[union-attr]
            ax.text(0.03, 0.03, f"$M_w$ {row['mw']:.1f}\n$R_{{\\mathrm{{rup}}}}$ {row['rrup']:.0f} km\n"
                    f"$V_{{S30}}$ {row['vs30']:.0f} m/s", transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})
            panel_label(ax, next(letters))
            if r == len(schemes) - 1:
                ax.set_xlabel("Period, $T$ (s)")
            if c == 0:
                ax.set_ylabel(f"$Sa$ (g), {label(scheme, 'scheme')} split")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    paths = save(fig, inp.cfg.paper_dir / "figures", "fig5_spectral_bands")
    cap = (f"Example test response spectra (RotD50) with nominal {1 - inp.cfg.alpha:.0%} bands from "
           f"{label(model)} ({inp.cfg.profile} profile; rows: {'; '.join(folds)}). Shaded: HSCP marginal band; "
           "dotted: split conformal band; dash-dot: HSCP simultaneous band over all usable periods; dashed: model "
           "median (HSCP re-centres records at stations with a calibration-set site term). Records are chosen at "
           "the 10th, 50th and 90th magnitude percentiles among test records with at least 90% usable periods.")
    return Figure("fig5_spectral_bands", paths, cap)


# ------------------------------------------------------------------ fig 6
def fig6_conditional_coverage(inp: ReportInputs) -> Figure:
    c = inp.conditional
    c = _alpha(c[c["profile"] == inp.cfg.profile], inp.cfg.alpha)
    methods = [m for m in ordered(c["method"].unique(), METHOD_ORDER) if m in MAIN_METHODS]
    strata = [s for s in ("mw", "rrup", "vs30_class", "station") if s in set(c["stratum"])]
    xlabels = {"mw": r"$M_w$", "rrup": r"$R_{\mathrm{rup}}$ (km)", "vs30_class": r"$V_{S30}$ class (NEHRP)",
               "station": "Station"}
    fig, axes = plt.subplots(1, len(strata), figsize=(DOUBLE_COL_IN, 2.6), sharey=True, squeeze=False)
    for k, stratum in enumerate(strata):
        ax = axes[0, k]
        sub = c[c["stratum"] == stratum]
        present = set(sub["bin"])
        bins = [b for b in BIN_ORDER[stratum] if b in present] + sorted(present - set(BIN_ORDER[stratum]))
        g = sub.groupby(["method", "bin"])[["covered", "n"]].sum()
        cov = (g["covered"] / g["n"]).unstack("bin")
        for i, meth in enumerate(methods):
            st = style_for(meth, "method", i)
            ax.plot(np.arange(len(bins)), cov.loc[meth].reindex(bins).to_numpy(), color=st["color"],
                    marker=st["marker"], linestyle=st["linestyle"], markeredgecolor="black",
                    markeredgewidth=0.3, label=label(meth, "method"))
        ax.axhline(1 - inp.cfg.alpha, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(np.arange(len(bins)), [BIN_LABELS.get(b, b) for b in bins], rotation=0, ha="center")
        ax.set_xlabel(xlabels[stratum])
        panel_label(ax, "abcd"[k])
    axes[0, 0].set_ylabel("Empirical coverage")
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo, hi + 0.15 * (hi - lo))  # headroom for panel letters
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    paths = save(fig, inp.cfg.paper_dir / "figures", "fig6_conditional_coverage")
    cap = (f"Conditional coverage at nominal {1 - inp.cfg.alpha:.0%} (dashed) by (a) magnitude, (b) rupture "
           r"distance (km), (c) NEHRP $V_{S30}$ class, ordered from stiff rock to soft soil (A/B: $\geq$760 m/s; "
           r"C: 360–760 m/s; D: 180–360 m/s; E: <180 m/s), and (d) whether the test station was seen in training or "
           f"calibration ({inp.cfg.profile} profile), pooled over split schemes, models, folds and intensity measures.")
    return Figure("fig6_conditional_coverage", paths, cap)


# ------------------------------------------------------------------ supplement
def figS1_random_leakage_inflation(inp: ReportInputs) -> Figure:
    rnd = inp.random_accuracy()
    ev = inp.accuracy()
    ev = ev[ev["scheme"] == inp.cfg.accuracy_scheme]
    a = rnd.groupby("model")["rmse"].mean()
    b = ev.groupby("model")["rmse"].mean()
    models = [m for m in ordered(a.index, MODEL_ORDER) if m in b.index]
    if not models:
        raise MissingInput("no model has both random and event-grouped results")
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * 0.6, 0.3 * len(models) + 1.0))
    y = np.arange(len(models))
    for j, m in enumerate(models):
        ax.plot([a[m], b[m]], [j, j], color="0.6", linewidth=1.0, zorder=1)
    ax.plot(a[models], y, "o", color=OKABE_ITO[4], markeredgecolor="black", markeredgewidth=0.3,
            label="Random split (leaky)")
    ax.plot(b[models], y, "s", color=OKABE_ITO[2], markeredgecolor="black", markeredgewidth=0.3,
            label=f"{label(inp.cfg.accuracy_scheme, 'scheme')}-grouped split")
    span = max(float(np.nanmax(np.r_[a[models], b[models]]) - np.nanmin(np.r_[a[models], b[models]])), 0.02)
    for j, m in enumerate(models):
        ax.text(max(a[m], b[m]) + 0.04 * span, j, f"{(b[m] / a[m] - 1) * 100:+.1f}%", va="center", fontsize=6)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.25 * span)
    ax.set_yticks(y, [label(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (ln units)")
    ax.legend(loc="lower right")
    paths = save(fig, inp.cfg.paper_dir / "supplement", "figS1_random_leakage_inflation")
    cap = ("Leakage inflation: test RMSE under a record-level random split versus the event-grouped split "
           f"({inp.cfg.profile} profile), averaged over folds and IMs. Labels give the RMSE increase when events "
           "are held out, i.e. the optimism of random-split evaluation.")
    return Figure("figS1_random_leakage_inflation", paths, cap, supplement=True)


def figS2_period_coverage_heatmap(inp: ReportInputs) -> Figure:
    cm = inp.conformal_metrics
    cm = _alpha(cm[(cm["profile"] == inp.cfg.profile) & cm["im"].isin(IM_NAMES)], inp.cfg.alpha)
    methods = [m for m in ("NaiveSplitConformal", "HSCP") if m in set(cm["method"])]
    schemes = ordered(cm["scheme"].unique(), SCHEME_ORDER)
    fig, axes = plt.subplots(1, len(methods), figsize=(DOUBLE_COL_IN, 5.4), sharey=True, squeeze=False)
    nominal = 1 - inp.cfg.alpha
    im = None
    for k, meth in enumerate(methods):
        ax = axes[0, k]
        piv = cm[cm["method"] == meth].groupby(["im", "scheme"])["coverage"].mean().unstack("scheme")
        piv = piv.reindex(index=IM_NAMES, columns=schemes)
        im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="cividis", vmin=nominal - 0.15, vmax=1.0)
        for (r, cidx), v in np.ndenumerate(piv.to_numpy()):
            if np.isfinite(v):
                ax.text(cidx, r, f"{v:.2f}", ha="center", va="center", fontsize=4.5,
                        color="white" if v < nominal - 0.05 else "black")
        ax.set_xticks(np.arange(len(schemes)), [short_scheme(s) for s in schemes], rotation=0, ha="center")
        ax.set_yticks(np.arange(len(IM_NAMES)), [n.replace("SA_", "SA ") for n in IM_NAMES], fontsize=5.5)
        ax.text(0.0, 1.01, f"({'ab'[k]}) {label(meth, 'method')}", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=8)
    if im is not None:
        cb = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.6, pad=0.02)
        cb.set_label("Empirical coverage")
    paths = save(fig, inp.cfg.paper_dir / "supplement", "figS2_period_coverage_heatmap")
    cap = (f"Coverage at nominal {nominal:.0%} per intensity measure and split scheme ({inp.cfg.profile} profile), "
           "averaged over models and folds: (a) split conformal, (b) HSCP. The colour scale is monotonic and "
           "colour-blind safe (cividis); values are printed in each cell. " + scheme_definitions(schemes))
    return Figure("figS2_period_coverage_heatmap", paths, cap, supplement=True)


MAIN_FIGURES: dict[str, Callable[[ReportInputs], Figure]] = {
    "fig1_data_splits": fig1_data_splits, "fig2_variance_components": fig2_variance_components,
    "fig3_regime_accuracy": fig3_regime_accuracy, "fig4_conformal_ablation": fig4_conformal_ablation,
    "fig5_spectral_bands": fig5_spectral_bands, "fig6_conditional_coverage": fig6_conditional_coverage,
}
SUPPLEMENT_FIGURES: dict[str, Callable[[ReportInputs], Figure]] = {
    "figS1_random_leakage_inflation": figS1_random_leakage_inflation,
    "figS2_period_coverage_heatmap": figS2_period_coverage_heatmap,
}


def caption_to_tex(text: str) -> str:
    """Escape LaTeX specials outside $...$ math (so '90%' and 'train_fit' are safe)."""
    parts = text.split("$")
    for i in range(0, len(parts), 2):          # even parts are outside math
        for ch, rep in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_")):
            parts[i] = parts[i].replace(ch, rep)
    return "$".join(parts)


def write_captions(figures: dict[str, Figure], tables: dict[str, object], out_dir: Path) -> list[Path]:
    """captions.md (review), captions*.tex and one <stem>.tex per figure (environments ready to \\input).

    LaTeX preamble needs ``\\usepackage{graphicx,adjustbox}``; graphics paths are relative to results/paper/.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Figure and table captions", ""]
    tex = ["% Figure environments generated by gmpe-bench; \\input{captions.tex}", ""]
    main = [f for f in figures.values() if not f.supplement]
    supp = [f for f in figures.values() if f.supplement]
    for heading, group, prefix in (("Main figures", main, "Figure "), ("Supplementary figures", supp, "Figure S")):
        if not group:
            continue
        md += [f"## {heading}", ""]
        for n, f in enumerate(group, 1):
            md += [f"**{prefix}{n}** (`{f.stem}`). {f.caption}", ""]
            tex += figure_environment(f) + [""]
    if tables:
        md += ["## Tables", ""]
        for name, t in tables.items():
            md += [f"**{name}**. {getattr(t, 'caption', '')}", ""]
    paths = [out_dir / "captions.md", out_dir / "captions.tex"]
    paths[0].write_text("\n".join(md), encoding="utf-8")
    paths[1].write_text("\n".join(tex), encoding="utf-8")
    for name, group in (("captions_main.tex", main), ("captions_supplement.tex", supp)):
        body = [f"% {name}: generated by gmpe-bench; graphics paths are relative to results/paper/", ""]
        for f in group:
            body += figure_environment(f) + [""]
        path = out_dir / name
        path.write_text("\n".join(body), encoding="utf-8")
        paths.append(path)
    # One file per figure, so a manuscript can place each figure next to its first citation:
    #   \input{../results/paper/figures/fig1_data_splits.tex}
    for f in figures.values():
        folder = out_dir / ("supplement" if f.supplement else "figures")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{f.stem}.tex"
        path.write_text(f"% {f.stem}.tex: generated by gmpe-bench; graphics path relative to results/paper/\n"
                        + "\n".join(figure_environment(f)) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def figure_environment(f: Figure) -> list[str]:
    """``figure`` environment lines; the graphics path is relative to results/paper/."""
    folder = "supplement" if f.supplement else "figures"
    # Figures are drawn at print size (3.5 or 7 in wide); \adjustimage only scales *down* to the text width,
    # so 7-8 pt figure text is never enlarged. Needs \usepackage{adjustbox} (also required by the tables).
    return [r"\begin{figure}[htbp]", r"\centering",
            rf"\adjustimage{{max width=\textwidth}}{{{folder}/{f.stem}.pdf}}",
            rf"\caption{{{caption_to_tex(f.caption)}}}", rf"\label{{fig:{f.stem}}}", r"\end{figure}"]


def build_all_figures(inp: ReportInputs, log: Callable[[str], None] = print
                      ) -> tuple[dict[str, Figure], dict[str, str]]:
    apply_style()
    figures: dict[str, Figure] = {}
    skipped: dict[str, str] = {}
    for name, fn in {**MAIN_FIGURES, **SUPPLEMENT_FIGURES}.items():
        try:
            figures[name] = fn(inp)
            log(f"figure {name}: {', '.join(p.name for p in figures[name].paths)}")
        except (MissingInput, KeyError) as exc:
            plt.close("all")
            skipped[name] = f"{type(exc).__name__}: {exc}"[:240]
            log(f"figure {name}: SKIPPED ({skipped[name]})")
    return figures, skipped
