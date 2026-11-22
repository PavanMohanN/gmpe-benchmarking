"""Phase 6: build every paper table, figure, caption, bootstrap interval and the SHA-256 manifest.

Usage (from the repository root):
    python scripts/05_make_paper_artifacts.py
    python scripts/05_make_paper_artifacts.py --example-model BSSA14_cal --alpha 0.05
    python scripts/05_make_paper_artifacts.py --bootstrap-b 2000 --bootstrap-seed 7

Reads data/processed and results/{baselines,ml,conformal}; writes results/paper/:
    tables/      table1..5 and tableS1..S4 (.csv + booktabs .tex); table3 includes ASK14 (site) and
                 BSSA14 (site); tableS3 = event-clustered bootstrap CIs, tableS4 = seen/unseen stations
    figures/     fig1..fig6 (.pdf vector + .png 800 dpi)
    supplement/  figS1..figS2
    captions.md, captions.tex, captions_main.tex, captions_supplement.tex, paper_manifest.json
    figures/<stem>.tex, supplement/<stem>.tex   one figure environment per file
    numbers.tex          every number quoted in paper/main.tex, as GB{key} macros
    claims_check.csv     every directional statement in paper/main.tex: PASS / FAIL / NA
    numbers_missing.txt  keys the results could not supply (printed as -- in the PDF)
    paper_summary_numbers.json  every metric, CI and count (upload results/paper/ alone for writing)

Bootstrap: B = 1,000 replicates (fixed seed) resampling whole earthquakes within each scheme.
RMSE intervals use results/baselines/oof_predictions.parquet and results/ml/oof_ml_predictions.parquet.
Coverage intervals need results/conformal/event_counts.parquet; add it to an existing Phase 5 run
without refitting anything:
    gmpe-bench conformal --event-counts-only --profiles full

figS1 (leakage inflation) needs results for the 'random' split, kept apart from the main runs:
    gmpe-bench baselines --schemes random --out results/baselines_random
    gmpe-bench train     --schemes random --out results/ml_random
Exit code 1 if any *main* table or figure is skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gmpe_bench.reporting import ReportConfig, build_paper_artifacts
from gmpe_bench.reporting.figures import MAIN_FIGURES
from gmpe_bench.reporting.tables import MAIN_TABLES

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="Repository folder (contains data/ and results/)")
    ap.add_argument("--out", type=Path, default=None, help="Default: <root>/results/paper")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--compare-profile", default="sp23")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--accuracy-scheme", default="event_grouped")
    ap.add_argument("--example-model", default="Hybrid_ASK14_MLP")
    ap.add_argument("--example-schemes", default="event_grouped,spatial_block")
    ap.add_argument("--bootstrap-b", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261122)
    a = ap.parse_args()

    cfg = ReportConfig(root=a.root, profile=a.profile, compare_profile=a.compare_profile, alpha=a.alpha,
                       accuracy_scheme=a.accuracy_scheme, example_model=a.example_model,
                       example_schemes=tuple(s.strip() for s in a.example_schemes.split(",") if s.strip()),
                       bootstrap_b=a.bootstrap_b, bootstrap_seed=a.bootstrap_seed, out_dir=a.out)
    art = build_paper_artifacts(cfg)
    print(f"\nTables : {len(art.tables)} written to {cfg.paper_dir / 'tables'}")
    print(f"Figures: {len(art.figures)} written to {cfg.paper_dir / 'figures'} and /supplement")
    print(f"Captions: {', '.join(p.name for p in art.captions)}")
    print(f"Manifest: {art.manifest}")
    print(f"Numbers : {', '.join(p.name for p in art.numbers)}")
    print(f"Summary : {art.summary}")
    import json

    checks = json.loads(art.manifest.read_text(encoding="utf-8")).get("bootstrap_checks", {})
    for k, v in checks.items():
        print(f"Check   : {k} = {v:.1e}" if v is not None else f"Check   : {k} = n/a")
    bad = [c for c in art.claims if c.status != "PASS"]
    print(f"Claims  : {len(art.claims) - len(bad)} of {len(art.claims)} PASS")
    for c in bad:
        print(f"  {c.cid} {c.status}: {c.statement}\n      {c.detail}")
    if art.skipped:
        print("\nSkipped:")
        for name, why in art.skipped.items():
            print(f"  {name}: {why}")
    main_missing = [n for n in art.skipped if n in MAIN_TABLES or n in MAIN_FIGURES]
    return 1 if main_missing else 0


if __name__ == "__main__":
    sys.exit(main())
