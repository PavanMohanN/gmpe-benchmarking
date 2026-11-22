"""File input/output: flatfiles, Parquet, checksums and the build manifest."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_flatfile(path: Path, cache_dir: Path | None = None) -> pd.DataFrame:
    """Read an NGA-West2 flatfile (.xlsx, .xls or .csv).

    Excel reads are slow, so with ``cache_dir`` the parsed frame is pickled
    under a name that includes the file's checksum; a changed file is re-read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cache: Path | None = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{path.stem}_{sha256_file(path)[:16]}.pkl"
        if cache.exists():
            return pd.read_pickle(cache)

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # openpyxl extension warnings
            df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported flatfile type: {path.suffix}")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Cast mixed-type object columns to strings so pyarrow can write them."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe(df).to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path), engine="pyarrow")


def write_manifest(info: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path
