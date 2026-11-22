"""Phase 6: tables, LaTeX formatting, figures, captions and manifest on a real mini pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmpe_bench import io
from gmpe_bench.reporting import ReportConfig, ReportInputs, build_paper_artifacts, to_booktabs
from gmpe_bench.reporting.figures import caption_to_tex
from gmpe_bench.reporting.inputs import MissingInput
from gmpe_bench.reporting.tables import Table, latex_escape

sys.path.insert(0, str(Path(__file__).parent))
from test_models import _sim  # noqa: E402

TABLES = ("table1_attrition", "table2_splits", "table3_accuracy", "table4_conformal_ablation",
          "table5_conditional_coverage", "tableS1_per_im_metrics", "tableS2_sp23_comparison",
          "tableS3_bootstrap_cis", "tableS4_station_rmse")
FIGURES = ("fig1_data_splits", "fig2_variance_components", "fig3_regime_accuracy", "fig4_conformal_ablation",
           "fig5_spectral_bands", "fig6_conditional_coverage")


def _attrition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prof, n in (("full", len(df)), ("sp23", int(0.8 * len(df)))):
        for rule, desc in (("R1", "Read H and V flatfiles"), ("R5", "Mask SA beyond T_max = 1/LUF"),
                           ("R9b", "sp23: Mw >= 3"), ("R12", f"Freeze {prof}.parquet")):
            if prof == "full" and rule == "R9b":
                continue
            rows.append({"profile": prof, "rule": rule, "description": desc, "n_before": n, "n_after": n,
                         "removed": 0, "n_events": df["eqid"].nunique(), "n_stations": df["ssn"].nunique(),
                         "note": ""})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run a small but real Phase 1-5 pipeline and return the repository-like root."""
    from gmpe_bench.baselines import evaluate_baselines
    from gmpe_bench.conformal import evaluate_conformal
    from gmpe_bench.models import evaluate_ml_models
    from gmpe_bench.splits import freeze_all

    r = tmp_path_factory.mktemp("repo")
    data = r / "data" / "processed"
    df = _sim(n_events=40, per_event=15, n_stations=60)
    df.loc[df["eqid"] > 30, ["hypo_lat", "hypo_lon"]] = (37.0, 140.0)  # a second region for spatial_block
    df.loc[df["eqid"] % 4 == 0, "cl_class"] = "CL2"
    io.write_parquet(df, data / "full.parquet")
    io.write_parquet(df[df["eqid"] % 5 != 0], data / "sp23.parquet")
    _attrition(df).to_csv(data / "attrition.csv", index=False)
    schemes = ["event_grouped", "spatial_block", "cl1_to_cl2"]
    freeze_all(data, ("full", "sp23"), seed=1, schemes=schemes)
    quiet = lambda _: None  # noqa: E731
    evaluate_baselines(data, r / "results" / "baselines", profiles=("full", "sp23"), schemes=tuple(schemes),
                       models=("BSSA14", "MixedEffects"), log=quiet)
    evaluate_ml_models(data, r / "results" / "ml", profiles=("full", "sp23"), schemes=("event_grouped",),
                       models=("SpectralMLP",), fast=True, log=quiet)
    evaluate_conformal(data, r / "results" / "conformal", ml_dir=r / "results" / "ml",
                       baseline_cache=r / "results" / "baselines" / "cache", profiles=("full",),
                       schemes=tuple(schemes), log=quiet)
    return r


@pytest.fixture(scope="module")
def artifacts(root: Path):
    cfg = ReportConfig(root=root, example_model="SpectralMLP", example_schemes=("event_grouped",),
                       site_models=("BSSA14_cal",), pair_models=("SpectralMLP", "BSSA14_cal", "BSSA14_site"),
                       station_models=("BSSA14_site", "SpectralMLP"), bootstrap_b=200)
    return cfg, build_paper_artifacts(cfg, log=lambda _: None)


# ------------------------------------------------------------------ LaTeX helpers
def test_latex_escape_relations_and_specials() -> None:
    assert latex_escape("Mw >= 7") == r"Mw $\geq$ 7"
    assert latex_escape("a<b") == r"a$<$b"
    assert latex_escape("90% & train_fit") == r"90\% \& train\_fit"
    assert latex_escape("$\\tau$") == "$\\tau$"


def test_booktabs_has_no_vertical_rules() -> None:
    t = Table("t", pd.DataFrame({"A": ["x", "y"], "B": ["1", "2"]}), "Caption 90%", "Note", bold={(0, 1)})
    tex = to_booktabs(t)
    spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", tex).group(1)
    assert "|" not in spec
    for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
        assert rule in tex
    assert r"\hline" not in tex and "cellcolor" not in tex and "rowcolor" not in tex
    assert "max width=6.25in" in tex and r"\textbf{1}" in tex and r"90\%" in tex


def test_caption_to_tex_keeps_math() -> None:
    assert caption_to_tex(r"90% of $\phi_{S2S}$ in train_fit") == r"90\% of $\phi_{S2S}$ in train\_fit"


# ------------------------------------------------------------------ end to end
def test_all_tables_written(artifacts) -> None:
    cfg, art = artifacts
    for name in TABLES:
        assert name in art.tables, art.skipped.get(name)
        csv, tex = cfg.paper_dir / "tables" / f"{name}.csv", cfg.paper_dir / "tables" / f"{name}.tex"
        assert csv.exists() and tex.exists()
        assert len(pd.read_csv(csv)) > 0
        text = tex.read_text(encoding="utf-8")
        assert "|" not in re.search(r"\\begin\{tabular\}\{([^}]*)\}", text).group(1)
        assert r"\toprule" in text and r"\bottomrule" in text and r"\hline" not in text


def test_tables_come_from_results(artifacts, root: Path) -> None:
    cfg, _ = artifacts
    t3 = pd.read_csv(cfg.paper_dir / "tables" / "table3_accuracy.csv")
    b = pd.read_csv(root / "results" / "baselines" / "baseline_metrics.csv")
    ref = b[(b["profile"] == "full") & (b["mode"] == "ergodic") & (b["model"] == "BSSA14_cal")
            & (b["scheme"] == "event_grouped")]["rmse"].mean()
    assert float(t3.loc[t3["Model"] == "BSSA14", "Event"].iloc[0]) == pytest.approx(ref, abs=5e-4)
    t2 = pd.read_csv(cfg.paper_dir / "tables" / "table2_splits.csv")
    assert set(t2["Audit"]) == {"pass"}


def test_main_figures_rendered(artifacts) -> None:
    cfg, art = artifacts
    for name in FIGURES:
        assert name in art.figures, art.skipped.get(name)
        pdf, png = cfg.paper_dir / "figures" / f"{name}.pdf", cfg.paper_dir / "figures" / f"{name}.png"
        assert pdf.stat().st_size > 1000 and png.stat().st_size > 1000
        assert pdf.read_bytes()[:4] == b"%PDF"


def test_png_resolution(artifacts) -> None:
    from PIL import Image

    cfg, _ = artifacts
    with Image.open(cfg.paper_dir / "figures" / "fig4_conformal_ablation.png") as im:
        dpi = im.info.get("dpi", (0, 0))[0]
        assert round(dpi) == 800


def test_supplement_and_skips(artifacts) -> None:
    cfg, art = artifacts
    assert (cfg.paper_dir / "supplement" / "figS2_period_coverage_heatmap.pdf").exists()
    assert "figS1_random_leakage_inflation" in art.skipped        # no random-split results in fixture
    assert "random" in art.skipped["figS1_random_leakage_inflation"]


def test_captions_and_manifest(artifacts) -> None:
    cfg, art = artifacts
    md = (cfg.paper_dir / "captions.md").read_text(encoding="utf-8")
    tex = (cfg.paper_dir / "captions.tex").read_text(encoding="utf-8")
    for name in FIGURES:
        assert name in md and f"fig:{name}" in tex
    assert "Figure 1" in md
    assert re.search(r"(?<!\\)%", tex.split("\n", 1)[1]) is None    # every % escaped after the header comment
    m = json.loads(art.manifest.read_text(encoding="utf-8"))
    assert m["outputs"] and m["inputs"]
    for rel, digest in list(m["outputs"].items())[:5]:
        assert io.sha256_file(cfg.root / rel) == digest
    assert "figS1_random_leakage_inflation" in m["skipped"]


def test_missing_results_raise_cleanly(tmp_path: Path) -> None:
    inp = ReportInputs(ReportConfig(root=tmp_path))
    with pytest.raises(MissingInput):
        _ = inp.conformal_metrics
    art = build_paper_artifacts(ReportConfig(root=tmp_path), log=lambda _: None)
    assert not art.tables and not art.figures and len(art.skipped) == len(TABLES) + len(FIGURES) + 2
    assert art.manifest.exists()
    assert np.isfinite(len(art.skipped))


# ------------------------------------------------------------------ polish pass
def test_series_styles_are_distinct() -> None:
    from gmpe_bench.reporting.style import METHOD_STYLE, MODEL_STYLE

    for table in (MODEL_STYLE, METHOD_STYLE):
        filled = [v for v in table.values() if "mfc" not in v]       # hollow = site-corrected variant
        markers = [v["marker"] for v in filled]
        colors = [v["color"] for v in filled]
        assert len(set(markers)) == len(markers) and len(set(colors)) == len(colors)
        looks = [(v["color"], v["marker"], v.get("mfc")) for v in table.values()]
        assert len(set(looks)) == len(looks)
    assert MODEL_STYLE["ASK14_cal"]["marker"] != MODEL_STYLE["Hybrid_BSSA14_XGB"]["marker"]
    # a site-corrected model is its parent's colour and marker, drawn hollow
    for parent in ("ASK14_cal", "BSSA14_cal"):
        site = MODEL_STYLE[parent.replace("_cal", "_site")]
        assert (site["color"], site["marker"], site["mfc"]) == (MODEL_STYLE[parent]["color"],
                                                                 MODEL_STYLE[parent]["marker"], "white")


def test_short_scheme_labels_and_definitions() -> None:
    from gmpe_bench.reporting.style import SCHEME_ORDER, scheme_definitions, short_scheme

    assert [short_scheme(s) for s in ("random", "event_grouped", "station_grouped", "spatial_block",
                                      "temporal")] == ["Rand", "Evt", "Stn", "Reg", "Temp"]
    text = scheme_definitions(SCHEME_ORDER)
    for code in ("Rand =", "Evt =", "Stn =", "Reg =", "Temp =", "CL1$\\rightarrow$2 ="):
        assert code in text


def test_pretty_fold() -> None:
    from gmpe_bench.reporting.figures import pretty_fold

    assert pretty_fold("event_grouped", "fold0") == "Event split, fold 1"
    assert pretty_fold("spatial_block", "test_California_WUS").endswith("held-out region California WUS")


def test_polished_captions(artifacts) -> None:
    cfg, art = artifacts
    md = (cfg.paper_dir / "captions.md").read_text(encoding="utf-8")
    assert "E|D at 180 m/s" in md and "B|A at 1500 m/s" in md          # fig1 NEHRP boundaries defined
    assert "Evt = event-grouped" in md                                   # short scheme codes defined
    assert "dashed line in (a) is the nominal 90% target" in md          # fig4 reference line explained
    for name in ("captions_main.tex", "captions_supplement.tex"):
        assert (cfg.paper_dir / name).exists()
    main = (cfg.paper_dir / "captions_main.tex").read_text(encoding="utf-8")
    assert "fig:fig1_data_splits" in main and "figS" not in main


def test_vs30_classes_stiff_to_soft(artifacts) -> None:
    cfg, _ = artifacts
    t5 = pd.read_csv(cfg.paper_dir / "tables" / "table5_conditional_coverage.csv")
    classes = [b for b in t5["Bin"] if str(b)[:1] in "ACDE" and "m/s" in str(b)]
    order = ["A", "C", "D", "E"]
    ranks = [order.index(str(b)[0]) for b in classes]
    assert ranks == sorted(ranks)


# ------------------------------------------------------------------ prose numbers and claims (manuscript)
def _gb_values(path: Path) -> dict[str, str]:
    return dict(re.findall(r"\\@namedef\{GB@([^}]+)\}\{(.*)\}", path.read_text(encoding="utf-8")))


def test_numbers_files_written(artifacts) -> None:
    cfg, art = artifacts
    names = {p.name for p in art.numbers}
    assert {"numbers.tex", "claims_check.csv", "numbers_missing.txt"} <= names
    tex = (cfg.paper_dir / "numbers.tex").read_text(encoding="utf-8")
    assert r"\providecommand{\GB}" in tex and tex.rstrip().endswith(r"\makeatother")
    claims = pd.read_csv(cfg.paper_dir / "claims_check.csv", keep_default_na=False)   # "NA" is a status
    assert list(claims.columns) == ["cid", "statement", "status", "detail"]
    assert set(claims["status"]) <= {"PASS", "FAIL", "NA"} and len(claims) >= 15


def test_numbers_copy_table_cells_exactly(artifacts) -> None:
    cfg, _ = artifacts
    gb = _gb_values(cfg.paper_dir / "numbers.tex")
    t3 = pd.read_csv(cfg.paper_dir / "tables" / "table3_accuracy.csv", dtype=str)
    row = t3[t3["Model"] == "BSSA14"].iloc[0]
    assert gb["rmse.bssa14.evt"] == row["Event"]                         # identical string, not re-rounded
    t1 = pd.read_csv(cfg.paper_dir / "tables" / "table1_attrition.csv", dtype=str)
    assert gb["n.full.records"] == t1[t1["Profile"] == "full"].iloc[-1]["Records"]
    t4 = pd.read_csv(cfg.paper_dir / "tables" / "table4_conformal_ablation.csv", dtype=str)
    cell = t4[t4["Method"] == "HSCP"].iloc[0]["Event"]                  # first panel = 90 %
    assert cell.startswith(gb["cov.hscp.evt"]) and f"({gb['wid.hscp.evt']})" in cell


def test_manuscript_keys_are_generated(artifacts) -> None:
    """Every GB key used in paper/main.tex and paper/supplement.tex is produced by the generator."""
    cfg, _ = artifacts
    paper = Path(__file__).resolve().parents[1] / "paper"
    if not (paper / "main.tex").exists():
        pytest.skip("paper/ not present")
    gb = _gb_values(cfg.paper_dir / "numbers.tex")
    for doc in ("main.tex", "supplement.tex"):
        text = "\n".join(ln.split("%", 1)[0] if not ln.lstrip().startswith("%") else ""
                         for ln in (paper / doc).read_text(encoding="utf-8").splitlines())
        used = set(re.findall(r"\\GB\{([^}]+)\}", text))
        assert used and not (used - set(gb)), f"{doc}: {sorted(used - set(gb))}"


def test_number_formatting_rules() -> None:
    from gmpe_bench.reporting.numbers import NumberBook, pct, sci

    assert pct(-0.04, signed=True) == r"0.0\%"                           # never -0.0 %
    assert pct(37.06, signed=True) == r"+37.1\%"
    assert pct(-2.5, signed=True) == r"\textminus{}2.5\%"
    assert sci(0.00123).startswith("$1.23")
    b = NumberBook()
    with pytest.raises(ValueError):
        b.put("bad key{", "1")
    b.claim("X1", "always fails on missing data", lambda: (b.num("nope") > 0, ""))
    assert b.claims[0].status == "NA"


def test_one_tex_file_per_figure(artifacts) -> None:
    cfg, art = artifacts
    for name in art.figures:
        folder = "supplement" if name.startswith("figS") else "figures"
        snippet = (cfg.paper_dir / folder / f"{name}.tex").read_text(encoding="utf-8")
        assert rf"\label{{fig:{name}}}" in snippet and r"\adjustimage{max width=\textwidth}" in snippet


# ------------------------------------------------------------------ revision: site baselines, bootstrap, summary
def test_site_corrected_baseline_in_table3_and_figure3(artifacts) -> None:
    cfg, art = artifacts
    t3 = pd.read_csv(cfg.paper_dir / "tables" / "table3_accuracy.csv", dtype=str)
    assert "BSSA14 (site)" in set(t3["Model"]) and "BSSA14" in set(t3["Model"])
    assert "(site) =" in (cfg.paper_dir / "tables" / "table3_accuracy.tex").read_text(encoding="utf-8")
    assert "Hollow markers (site)" in art.figures["fig3_regime_accuracy"].caption


def test_site_rows_match_phase3_site_predictions(root: Path) -> None:
    """Table 3 '(site)' values are the Phase 3 site_corrected metrics, recomputed here from mu_site/sigma_site."""
    from gmpe_bench.schema import IM_NAMES

    oof = pd.read_parquet(root / "results" / "baselines" / "oof_predictions.parquet")
    oof = oof[(oof["profile"] == "full") & (oof["model"] == "BSSA14_cal") & (oof["scheme"] == "event_grouped")]
    vals = []
    for (_, g), im in [((f, g), im) for f, g in oof.groupby("fold") for im in IM_NAMES]:
        y, mu, sd = (g[f"{p}_{im}"].to_numpy(float) for p in ("y", "mu_site", "sigma_site"))
        ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
        if ok.any():
            vals.append(np.sqrt(np.mean((y[ok] - mu[ok]) ** 2)))
    inp = ReportInputs(ReportConfig(root=root, site_models=("BSSA14_cal",)))
    acc = inp.accuracy(include_site=True)
    table = acc[(acc["model"] == "BSSA14_site") & (acc["scheme"] == "event_grouped")]["rmse"].mean()
    assert np.mean(vals) == pytest.approx(table, abs=1e-5)


def test_bootstrap_reproduces_tables_and_is_deterministic(root: Path, artifacts) -> None:
    from gmpe_bench.reporting.bootstrap import run_bootstrap

    cfg, _ = artifacts
    manifest = json.loads((cfg.paper_dir / "paper_manifest.json").read_text(encoding="utf-8"))
    checks = manifest["bootstrap_checks"]
    assert checks["rmse_max_abs_diff_vs_table3"] < 1e-5          # float32 predictions vs float64 metrics
    assert checks["coverage_max_abs_diff_vs_table4"] < 1e-12      # integer counts: exact
    again = run_bootstrap(ReportInputs(cfg), log=lambda _: None)
    first = pd.read_csv(cfg.paper_dir / "tables" / "tableS3_bootstrap_cis.csv", dtype=str)
    from gmpe_bench.reporting.tables import tableS3_bootstrap_cis

    inp = ReportInputs(cfg, log=lambda _: None)
    inp.__dict__["bootstrap"] = again
    assert tableS3_bootstrap_cis(inp).frame.astype(str).equals(first)
    for _, r in again.coverage.iterrows():
        assert r["lo"] <= r["estimate"] <= r["hi"]
    assert set(again.station["status"]) <= {"seen", "unseen"} and (again.station["n_records"] > 0).all()


def test_bootstrap_tables_written(artifacts) -> None:
    cfg, _ = artifacts
    s3 = (cfg.paper_dir / "tables" / "tableS3_bootstrap_cis.tex").read_text(encoding="utf-8")
    assert r"\begin{array}" in s3 and "Event-clustered bootstrap" in s3 and r"\scriptstyle[" in s3
    csv = pd.read_csv(cfg.paper_dir / "tables" / "tableS3_bootstrap_cis.csv", dtype=str)
    cell = csv.iloc[0, 1]
    assert re.match(r"^[+-]\d\.\d{3} \[[+-]\d\.\d{3}, [+-]\d\.\d{3}\]$", cell)       # plain text in CSV
    s4 = pd.read_csv(cfg.paper_dir / "tables" / "tableS4_station_rmse.csv", dtype=str)
    assert {"BSSA14 (site)", "Spectral MLP", "Test records (earthquakes)"} <= set(s4["Model"])


def test_paper_summary_json(artifacts) -> None:
    cfg, art = artifacts
    assert art.summary is not None and art.summary.name == "paper_summary_numbers.json"
    d = json.loads(art.summary.read_text(encoding="utf-8"))
    for key in ("meta", "data", "splits", "accuracy", "rmse_differences", "station_rmse", "conformal",
                "conditional", "numbers", "claims", "tables"):
        assert key in d, key
    assert d["meta"]["bootstrap"]["b"] == 200 and d["meta"]["bootstrap"]["seed"] == cfg.bootstrap_seed
    hscp = d["conformal"]["0.90"]["event_grouped"]["HSCP"]
    assert {"coverage", "width", "coverage_ci"} <= set(hscp) and hscp["coverage_ci"]["lo"] <= hscp["coverage"]
    assert "rmse_ci" in d["accuracy"]["event_grouped"]["BSSA14_site"]
    t4 = pd.read_csv(cfg.paper_dir / "tables" / "table4_conformal_ablation.csv", dtype=str)
    cell = t4[t4["Method"] == "HSCP"].iloc[0]["Event"]
    assert cell.startswith(f"{hscp['coverage']:.3f}")                 # JSON agrees with the printed table
    assert "tableS3_bootstrap_cis" in d["tables"] and "ci.diff.mlp.bssa14.evt.lo" in d["numbers"]
