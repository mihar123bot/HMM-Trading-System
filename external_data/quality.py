"""
external_data/quality.py
Data quality checks for external (Binance + DefiLlama) data.

Reports coverage gaps, staleness, and schema issues.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def check_funding(df: pd.DataFrame, expected_interval_h: int = 8) -> dict:
    """
    Quality check for funding rate data.

    Checks:
    - total rows
    - date range
    - % missing 8-hour slots
    - extreme values (|funding_rate| > 1%)
    - NaN count
    """
    if df.empty:
        return {"status": "empty", "issues": ["No funding data available"]}

    n = len(df)
    expected_idx = pd.date_range(df.index[0], df.index[-1], freq=f"{expected_interval_h}h")
    n_expected   = len(expected_idx)
    n_gaps       = max(n_expected - n, 0)
    pct_missing  = round(n_gaps / max(n_expected, 1) * 100, 2)

    n_extreme    = int((df["funding_rate"].abs() > 0.01).sum()) if "funding_rate" in df.columns else 0
    n_nan        = int(df["funding_rate"].isna().sum()) if "funding_rate" in df.columns else 0

    issues = []
    if pct_missing > 5:
        issues.append(f"{pct_missing:.1f}% of expected 8h funding slots missing ({n_gaps} gaps)")
    if n_extreme > 0:
        issues.append(f"{n_extreme} bars with |funding_rate| > 1% (extreme values)")
    if n_nan > 0:
        issues.append(f"{n_nan} NaN values in funding_rate")

    return {
        "n_rows":       n,
        "date_range":   f"{df.index[0]} → {df.index[-1]}",
        "pct_missing":  pct_missing,
        "n_extreme":    n_extreme,
        "n_nan":        n_nan,
        "issues":       issues,
        "status":       "ok" if not issues else "warnings",
    }


def check_oi(df: pd.DataFrame) -> dict:
    """Quality check for open interest data."""
    if df.empty:
        return {
            "status": "empty",
            "issues": ["No OI data available — Binance OI history limited to ~30 days"],
        }

    n = len(df)
    n_nan = int(df["open_interest"].isna().sum()) if "open_interest" in df.columns else 0

    issues = []
    if n_nan > 0:
        issues.append(f"{n_nan} NaN values in open_interest")
    if n < 24:
        issues.append(f"Only {n} OI rows — insufficient for z-score. Expect ~30 days / 720+ rows.")

    return {
        "n_rows":    n,
        "date_range": f"{df.index[0]} → {df.index[-1]}" if n > 0 else "N/A",
        "n_nan":     n_nan,
        "issues":    issues,
        "status":    "ok" if not issues else "warnings",
    }


def check_stablecoins(df: pd.DataFrame) -> dict:
    """Quality check for stablecoin supply data."""
    if df.empty:
        return {"status": "empty", "issues": ["No stablecoin data available"]}

    n = len(df)
    n_nan  = int(df["stablecoin_supply_usd"].isna().sum()) if "stablecoin_supply_usd" in df.columns else 0
    n_zero = int((df["stablecoin_supply_usd"] == 0).sum()) if "stablecoin_supply_usd" in df.columns else 0

    issues = []
    if n_nan > 0:
        issues.append(f"{n_nan} NaN values in stablecoin_supply_usd")
    if n_zero > 0:
        issues.append(f"{n_zero} zero-supply rows")

    return {
        "n_rows":    n,
        "date_range": f"{df.index[0]} → {df.index[-1]}" if n > 0 else "N/A",
        "n_nan":     n_nan,
        "n_zero":    n_zero,
        "issues":    issues,
        "status":    "ok" if not issues else "warnings",
    }


def check_merged(df: pd.DataFrame) -> dict:
    """Quality check for the merged hourly feature DataFrame."""
    if df.empty:
        return {"status": "empty", "issues": ["Merged DataFrame is empty"]}

    ext_cols = [
        "funding_rate", "funding_z",
        "open_interest", "oi_change_1h", "oi_z",
        "stablecoin_supply_usd", "stablecoin_supply_change_30d",
        "ext_overheat", "ext_low_liquidity",
    ]
    coverage = {}
    for col in ext_cols:
        if col in df.columns:
            pct = round((1 - df[col].isna().mean()) * 100, 1)
            coverage[col] = pct
        else:
            coverage[col] = 0.0

    issues = [
        f"{col}: {pct:.0f}% coverage"
        for col, pct in coverage.items()
        if pct < 50
    ]

    return {
        "n_rows":   len(df),
        "date_range": f"{df.index[0]} → {df.index[-1]}",
        "coverage": coverage,
        "issues":   issues,
        "status":   "ok" if not issues else "partial_coverage",
    }


def save_quality_report(report: dict, path: Path) -> None:
    """Save a quality report dict as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report["generated_utc"] = datetime.now(timezone.utc).isoformat()
    with open(str(path), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Quality report saved → %s", path)
