from __future__ import annotations
from pathlib import Path
from platformdirs import user_data_dir, user_cache_dir

APP_NAME = "gmpe-bench"


def data_dir() -> Path:
    p = Path(user_data_dir(APP_NAME))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = Path(user_cache_dir(APP_NAME))
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = Path.cwd() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p
