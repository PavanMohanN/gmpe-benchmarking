from typer.testing import CliRunner
from gmpe_bench.cli import app

runner = CliRunner()


def test_selftest_passes() -> None:
    res = runner.invoke(app, ["selftest"])
    assert res.exit_code == 0
    assert "selftest passed" in res.stdout


def test_info_runs() -> None:
    res = runner.invoke(app, ["info"])
    assert res.exit_code == 0
    assert "gmpe-bench 0.1.0" in res.stdout