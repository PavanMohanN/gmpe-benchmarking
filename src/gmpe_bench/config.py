from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json

PERIODS: list[float] = [
    0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.5, 8.0, 10.0,
]

IM_NAMES: list[str] = ["PGA", "PGV"] + [f"SA_{t:g}s" for t in PERIODS]


@dataclass
class Config:
    periods: list[float] = field(default_factory=lambda: list(PERIODS))
    im_names: list[str] = field(default_factory=lambda: list(IM_NAMES))
    seed: int = 42
    n_folds: int = 5
    cal_frac: float = 0.25
    alphas: list[float] = field(default_factory=lambda: [0.10, 0.05])
    luf_factor_included: bool = True

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


def default_config() -> Config:
    return Config()


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return default_config()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(**data)
