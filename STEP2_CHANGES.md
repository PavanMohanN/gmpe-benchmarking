# Step 2: plug-in API (on top of commit 5424d45)

Unzip into the repository root (paths are repo-relative), then:

    pip install -e .
    gmpe-bench selftest
    pytest -q

New files
- src/gmpe_bench/api.py                              BaseGMM, LinearGMM, evaluate(), evaluate_predictions(), iter_folds(), BenchmarkReport
- src/gmpe_bench/_smoke.py                           in-memory synthetic fixture, used only by `gmpe-bench selftest` and tests
- src/gmpe_bench/reference/reference_benchmarks.json extract of results/paper/paper_summary_numbers.json + split fingerprints
- tests/test_api.py                                  34 tests (2 run only when data/processed/full.parquet exists)

Changed files (additions only)
- src/gmpe_bench/__init__.py   lazy exports of the API (import stays fast)
- src/gmpe_bench/cli.py        `selftest` gains the labelled smoke test (`--no-smoke` skips it)
- pyproject.toml               package-data for reference/*.json
- README.md                    Python quickstart, PEER data setup, selftest

Untouched: splits, data rules, baselines, models, conformal methods, reporting, data/, results/.
