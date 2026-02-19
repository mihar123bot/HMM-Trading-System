"""
external_data/storage.py
Parquet-based persistence for raw and derived external data.

Storage layout
--------------
data/
  raw/
    binance/
      funding/BTCUSDT.parquet        ← funding rate history
      open_interest/BTCUSDT.parquet  ← OI history
    defillama/
      stablecoins.parquet            ← stablecoin supply (daily)
  features/
    hourly_merged/BTC-USD.parquet    ← merged hourly OHLCV + external features
  reports/
    data_quality_YYYYMMDD.json       ← data quality snapshots

Each parquet file has a sidecar metadata dict stored in its parquet metadata:
  last_updated_utc, source, symbol, row_count, schema_hash

Usage
-----
    from external_data.storage import PATHS, save, load

    df = load(PATHS.funding("BTCUSDT"))          # returns empty DF if missing
    save(df, PATHS.funding("BTCUSDT"))           # writes parquet + updates metadata
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Root of the project (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = _PROJECT_ROOT / "data"


@dataclass(frozen=True)
class _Paths:
    """Centralised path resolver for all external data artefacts."""

    def funding(self, symbol: str = "BTCUSDT") -> Path:
        return DATA_ROOT / "raw" / "binance" / "funding" / f"{symbol}.parquet"

    def open_interest(self, symbol: str = "BTCUSDT") -> Path:
        return DATA_ROOT / "raw" / "binance" / "open_interest" / f"{symbol}.parquet"

    def stablecoins(self) -> Path:
        return DATA_ROOT / "raw" / "defillama" / "stablecoins.parquet"

    def merged(self, ticker: str = "BTC-USD") -> Path:
        return DATA_ROOT / "features" / "hourly_merged" / f"{ticker}.parquet"

    def quality_report(self, date_str: Optional[str] = None) -> Path:
        date_str = date_str or datetime.now(timezone.utc).strftime("%Y%m%d")
        return DATA_ROOT / "reports" / f"data_quality_{date_str}.json"


PATHS = _Paths()


def save(df: pd.DataFrame, path: Path, source: str = "", symbol: str = "") -> None:
    """
    Write *df* to *path* as Parquet.

    Creates parent directories automatically.
    Metadata (last_updated_utc, source, symbol, row_count, schema_hash) is
    embedded in the parquet file metadata via pandas attrs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    schema_hash = hashlib.md5(
        str(sorted(df.dtypes.to_dict().items())).encode()
    ).hexdigest()[:8]

    meta = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "source":           source,
        "symbol":           symbol,
        "row_count":        len(df),
        "schema_hash":      schema_hash,
    }
    df.attrs["external_meta"] = meta

    df.to_parquet(str(path), engine="pyarrow", compression="snappy")
    logger.info("Saved %d rows → %s", len(df), path)


def load(path: Path) -> pd.DataFrame:
    """
    Load a Parquet file; return an empty DataFrame if the file does not exist.
    """
    if not path.exists():
        logger.debug("File not found: %s — returning empty DataFrame.", path)
        return pd.DataFrame()

    df = pd.read_parquet(str(path), engine="pyarrow")
    logger.info("Loaded %d rows ← %s", len(df), path)
    return df


def get_meta(path: Path) -> dict:
    """Return metadata dict for a parquet file, or {} if missing."""
    if not path.exists():
        return {}
    df = pd.read_parquet(str(path), engine="pyarrow")
    return df.attrs.get("external_meta", {})


def append_or_replace(
    new_df: pd.DataFrame,
    path: Path,
    dedup_on: str = "ts",
    source: str = "",
    symbol: str = "",
) -> pd.DataFrame:
    """
    Load existing data, concatenate *new_df*, deduplicate on *dedup_on*,
    sort, and save. Returns the merged DataFrame.
    """
    existing = load(path)
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df])
        if dedup_on in combined.columns:
            combined = combined.drop_duplicates(dedup_on)
        elif dedup_on == combined.index.name:
            combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

    save(combined, path, source=source, symbol=symbol)
    return combined
