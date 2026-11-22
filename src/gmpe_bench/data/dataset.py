"""GMDataset: the single object every later phase works with."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gmpe_bench import io
from gmpe_bench.data import features, targets
from gmpe_bench.schema import IM_NAMES


@dataclass
class GMDataset:
    """One cleaned NGA-West2 profile: one row per record."""

    df: pd.DataFrame
    profile: str = "full"

    @classmethod
    def from_parquet(cls, path: Path, profile: str | None = None) -> GMDataset:
        path = Path(path)
        return cls(io.read_parquet(path), profile or path.stem)

    def __len__(self) -> int:
        return len(self.df)

    def X(self, tier: str = "A") -> pd.DataFrame:
        """Input matrix for Tier ``A`` or ``B``."""
        tier = tier.upper()
        if tier == "A":
            return features.tier_a_features(self.df)
        if tier == "B":
            return features.tier_b_features(self.df)
        raise ValueError(f"tier must be 'A' or 'B', got {tier!r}")

    def Y(self, component: str = "H") -> pd.DataFrame:
        """Natural-log targets; columns are IM names."""
        return targets.get_targets(self.df, component)

    def mask(self, component: str = "H") -> pd.DataFrame:
        """True where the target is usable."""
        return targets.target_mask(self.df, component)

    @property
    def event_groups(self) -> np.ndarray:
        return self.df["eqid"].to_numpy(dtype=np.int64)

    @property
    def station_groups(self) -> np.ndarray:
        return self.df["ssn"].to_numpy(dtype=np.int64)

    def subset(self, idx: Sequence[int] | np.ndarray) -> GMDataset:
        """Positional subset; returns a new dataset."""
        return GMDataset(self.df.iloc[np.asarray(idx)].reset_index(drop=True), self.profile)

    def describe(self) -> dict[str, object]:
        d = self.df
        usable = {c: int(self.mask(c).any(axis=1).sum()) for c in targets.TARGET_COMPONENTS}
        per_period = {c: self.mask(c).sum().astype(int).to_dict() for c in ("H", "V")}
        return {
            "profile": self.profile,
            "n_records": len(d),
            "n_events": int(d["eqid"].nunique()),
            "n_stations": int(d["ssn"].nunique()),
            "mw_range": [float(d["mw"].min()), float(d["mw"].max())],
            "rjb_range_km": [float(d["rjb"].min()), float(d["rjb"].max())],
            "vs30_range": [float(d["vs30"].min()), float(d["vs30"].max())],
            "records_with_usable": usable,
            "usable_per_im": per_period,
            "ims": list(IM_NAMES),
        }
