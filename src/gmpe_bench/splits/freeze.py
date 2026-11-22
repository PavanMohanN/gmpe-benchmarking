"""Freeze splits to JSON (by RSN, so files survive row re-ordering) and write Table 2.

Files written to ``out_dir``:

* ``splits_<profile>.json``  every scheme and fold as sorted RSN lists
* ``split_summary.csv``      leakage audit rows for all profiles (Paper Table 2)
* ``splits_manifest.json``   SHA-256 of each file above
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gmpe_bench import io
from gmpe_bench.data import GMDataset
from gmpe_bench.splits.group_splits import SplitFold
from gmpe_bench.splits.leakage import LeakageReport, leakage_table

FORMAT_VERSION = 1


def rsn_fingerprint(ds: GMDataset) -> str:
    """SHA-256 of the sorted RSN set: ties a split file to one dataset."""
    rsn = np.sort(ds.df["rsn"].to_numpy(dtype=np.int64))
    return hashlib.sha256(rsn.tobytes()).hexdigest()


def save_splits(ds: GMDataset, splits: dict[str, list[SplitFold]], path: Path,
                seed: int, cal_frac: float) -> Path:
    rsn = ds.df["rsn"].to_numpy(dtype=np.int64)
    payload = {
        "format_version": FORMAT_VERSION,
        "profile": ds.profile,
        "n_records": len(ds),
        "rsn_sha256": rsn_fingerprint(ds),
        "seed": seed,
        "cal_frac": cal_frac,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schemes": {
            name: [
                {"fold": f.fold, "group_key": f.group_key,
                 **{p: np.sort(rsn[idx]).tolist() for p, idx in f.parts().items()}}
                for f in folds
            ]
            for name, folds in splits.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def load_splits(path: Path, ds: GMDataset) -> dict[str, list[SplitFold]]:
    """Rebuild positional folds for ``ds``; refuses a file made for another dataset."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["rsn_sha256"] != rsn_fingerprint(ds):
        raise ValueError(f"{Path(path).name} was frozen for a different dataset "
                         f"(profile {payload['profile']}, {payload['n_records']} records)")
    pos = pd.Series(np.arange(len(ds)), index=ds.df["rsn"].to_numpy(dtype=np.int64))
    out: dict[str, list[SplitFold]] = {}
    for name, folds in payload["schemes"].items():
        out[name] = [
            SplitFold(name, f["fold"],
                      *(np.sort(pos.loc[f[p]].to_numpy(dtype=np.int64))
                        for p in ("train_fit", "cal", "test", "excluded")),
                      group_key=f["group_key"])
            for f in folds
        ]
    return out


def write_summary(reports: list[LeakageReport], path: Path) -> Path:
    path = Path(path)
    leakage_table(reports).to_csv(path, index=False)
    return path


def write_split_manifest(files: list[Path], path: Path) -> Path:
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {Path(f).name: io.sha256_file(Path(f)) for f in files},
    }
    return io.write_manifest(info, path)


def freeze_all(data_dir: Path, profiles: tuple[str, ...] = ("full", "sp23"), seed: int = 42,
               cal_frac: float = 0.25, schemes: list[str] | None = None
               ) -> tuple[list[LeakageReport], dict[str, Path]]:
    """Make, audit and save splits for every profile in ``data_dir``. Used by CLI and script."""
    from gmpe_bench.splits.group_splits import make_splits
    from gmpe_bench.splits.leakage import audit_splits

    data_dir = Path(data_dir)
    reports: list[LeakageReport] = []
    files: dict[str, Path] = {}
    for prof in profiles:
        src = data_dir / f"{prof}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found. Run `gmpe-bench build` first.")
        ds = GMDataset.from_parquet(src, prof)
        splits = make_splits(ds, schemes, seed=seed, cal_frac=cal_frac)
        if schemes is None and not ("cl_class" in ds.df and (ds.df["cl_class"] == "CL2").any()):
            splits.pop("cl1_to_cl2", None)
        reports += audit_splits(ds, splits)
        files[prof] = save_splits(ds, splits, data_dir / f"splits_{prof}.json", seed, cal_frac)


    files["summary"] = write_summary(reports, data_dir / "split_summary.csv")
    files["manifest"] = write_split_manifest(list(files.values()), data_dir / "splits_manifest.json")
    return reports, files
