"""Publication style: typography, Okabe-Ito colours, markers, labels and export.

Every series gets a colour *and* a marker *and* a line style, so figures stay
readable in greyscale print. Titles are never drawn inside figures; captions
live in ``captions.md`` / ``captions.tex``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, identical output on every OS
import matplotlib.pyplot as plt  # noqa: E402

DPI = 800
SINGLE_COL_IN = 3.5
DOUBLE_COL_IN = 7.0

OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#F0E442")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), "-", "--")

MODEL_LABELS: dict[str, str] = {
    "ASK14_cal": "ASK14", "BSSA14_cal": "BSSA14", "CB14_cal": "CB14", "CY14_cal": "CY14",
    "ASK14": "ASK14", "BSSA14": "BSSA14", "CB14": "CB14", "CY14": "CY14",
    "ASK14_site": "ASK14 (site)", "BSSA14_site": "BSSA14 (site)", "CB14_site": "CB14 (site)",
    "CY14_site": "CY14 (site)",
    "MixedEffects": "Mixed effects", "XGBoost": "XGBoost", "SpectralMLP": "Spectral MLP",
    "Hybrid_ASK14_MLP": "Hybrid ASK14+MLP", "Hybrid_BSSA14_XGB": "Hybrid BSSA14+XGB",
}
METHOD_LABELS: dict[str, str] = {
    "UncalibratedGaussian": "Gaussian", "NaiveSplitConformal": "Split CP",
    "ClusteredEventConformal": "Event-clustered CP", "WeightedShiftConformal": "Shift-weighted CP",
    "HSCP": "HSCP", "HSCP_noSite": "HSCP w/o site", "HSCP_noShift": "HSCP w/o shift",
    "HSCP_noSmooth": "HSCP w/o smoothing", "HSCP_abs": "HSCP unnormalised",
}
SCHEME_LABELS: dict[str, str] = {
    "event_grouped": "Event", "station_grouped": "Station", "spatial_block": "Region",
    "temporal": "Temporal", "magnitude_holdout": r"$M_w \geq 7$", "cl1_to_cl2": "CL1$\\rightarrow$CL2",
    "random": "Random",
}
SCHEME_LABELS_PLAIN: dict[str, str] = {
    "event_grouped": "Event", "station_grouped": "Station", "spatial_block": "Region",
    "temporal": "Temporal", "magnitude_holdout": "Mw >= 7", "cl1_to_cl2": "CL1 to CL2", "random": "Random",
}
# Short horizontal tick labels (defined in every caption that uses them)
SCHEME_SHORT: dict[str, str] = {
    "random": "Rand", "event_grouped": "Evt", "station_grouped": "Stn", "spatial_block": "Reg",
    "temporal": "Temp", "magnitude_holdout": r"$M_w{\geq}7$", "cl1_to_cl2": r"CL1$\rightarrow$2",
}
SCHEME_DEFINITIONS: dict[str, str] = {
    "random": "Rand = record-level random split (leaky reference)",
    "event_grouped": "Evt = event-grouped 5-fold",
    "station_grouped": "Stn = station-grouped 5-fold",
    "spatial_block": "Reg = leave-one-region-out",
    "temporal": "Temp = temporal (train before 2008, test 2008 onward)",
    "magnitude_holdout": r"$M_w{\geq}7$ = magnitude hold-out (test events with $M_w \geq 7$)",
    "cl1_to_cl2": r"CL1$\rightarrow$2 = mainshocks (CL1) for training, aftershocks (CL2) for testing",
}
SCHEME_ORDER = ("random", "event_grouped", "station_grouped", "spatial_block", "temporal",
                "magnitude_holdout", "cl1_to_cl2")
METHOD_ORDER = ("UncalibratedGaussian", "NaiveSplitConformal", "ClusteredEventConformal",
                "WeightedShiftConformal", "HSCP", "HSCP_noSite", "HSCP_noShift", "HSCP_noSmooth", "HSCP_abs")
MODEL_ORDER = ("ASK14_cal", "BSSA14_cal", "CB14_cal", "CY14_cal", "ASK14_site", "BSSA14_site", "CB14_site",
               "CY14_site", "MixedEffects", "XGBoost", "SpectralMLP", "Hybrid_ASK14_MLP", "Hybrid_BSSA14_XGB")


# Fixed style per model and per method: every series differs in colour AND marker AND line,
# so no two series collide even in greyscale (9 models > 8 Okabe-Ito colours).
MODEL_STYLE: dict[str, dict[str, object]] = {
    "ASK14_cal": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "BSSA14_cal": {"color": "#56B4E9", "marker": "s", "linestyle": "--"},
    "CB14_cal": {"color": "#009E73", "marker": "^", "linestyle": "-."},
    "CY14_cal": {"color": "#0072B2", "marker": "D", "linestyle": ":"},
    "MixedEffects": {"color": "#D55E00", "marker": "v", "linestyle": (0, (5, 1))},
    "XGBoost": {"color": "#CC79A7", "marker": "P", "linestyle": (0, (3, 1, 1, 1))},
    "SpectralMLP": {"color": "#F0E442", "marker": "*", "linestyle": (0, (1, 1))},
    "Hybrid_ASK14_MLP": {"color": "#000000", "marker": "X", "linestyle": "-"},
    "Hybrid_BSSA14_XGB": {"color": "#7F7F7F", "marker": "p", "linestyle": "--"},
    # Site-corrected NGA-West2 models: parent colour and marker, hollow face ("mfc"), dotted line.
    "ASK14_site": {"color": "#E69F00", "marker": "o", "linestyle": ":", "mfc": "white"},
    "BSSA14_site": {"color": "#56B4E9", "marker": "s", "linestyle": ":", "mfc": "white"},
    "CB14_site": {"color": "#009E73", "marker": "^", "linestyle": ":", "mfc": "white"},
    "CY14_site": {"color": "#0072B2", "marker": "D", "linestyle": ":", "mfc": "white"},
}
METHOD_STYLE: dict[str, dict[str, object]] = {
    "UncalibratedGaussian": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "NaiveSplitConformal": {"color": "#56B4E9", "marker": "s", "linestyle": "--"},
    "ClusteredEventConformal": {"color": "#009E73", "marker": "^", "linestyle": "-."},
    "WeightedShiftConformal": {"color": "#0072B2", "marker": "D", "linestyle": ":"},
    "HSCP": {"color": "#D55E00", "marker": "v", "linestyle": (0, (5, 1))},
}


def style_for(name: str, kind: str, i: int) -> dict[str, object]:
    """Fixed style for a known model/method, else the i-th generic style."""
    table = MODEL_STYLE if kind == "model" else METHOD_STYLE if kind == "method" else {}
    return dict(table.get(name, series_style(i)))


def short_scheme(name: str) -> str:
    return SCHEME_SHORT.get(name, name)


def scheme_definitions(schemes: object) -> str:
    """Caption sentence defining the short scheme labels that appear in a figure."""
    parts = [SCHEME_DEFINITIONS[s] for s in schemes if s in SCHEME_DEFINITIONS]  # type: ignore[union-attr]
    return ("Split schemes: " + "; ".join(parts) + ".") if parts else ""


def label(name: str, kind: str = "model") -> str:
    table = {"model": MODEL_LABELS, "method": METHOD_LABELS, "scheme": SCHEME_LABELS}[kind]
    return table.get(name, name)


def ordered(names: object, order: tuple[str, ...]) -> list[str]:
    names = list(dict.fromkeys(names))  # type: ignore[call-overload]
    return [n for n in order if n in names] + sorted(n for n in names if n not in order)


def series_style(i: int) -> dict[str, object]:
    """Colour, marker and line style for the i-th series (greyscale-safe)."""
    return {"color": OKABE_ITO[i % len(OKABE_ITO)], "marker": MARKERS[i % len(MARKERS)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)]}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": False, "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType fonts (editable, accepted by journals)
        "axes.titlesize": 8,
    })


def panel_label(ax: plt.Axes, letter: str) -> None:
    """(a), (b) ... in the top-left corner, inside the axes."""
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=8)


def save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Write ``stem.pdf`` (vector) and ``stem.png`` (800 dpi); close the figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=DPI)
    plt.close(fig)
    return paths
