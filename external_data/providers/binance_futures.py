"""
external_data/providers/binance_futures.py
Binance USD-M Futures public REST API — no authentication required.

Endpoints used
--------------
Funding rate history (8h cadence, 00:00 / 08:00 / 16:00 UTC):
    GET https://fapi.binance.com/fapi/v1/fundingRate
    Params: symbol, startTime (ms), endTime (ms), limit (max 1000)
    Returns: [{fundingTime, symbol, fundingRate, markPrice}]

Historical open interest (1h resolution):
    GET https://fapi.binance.com/futures/data/openInterestHist
    Params: symbol, period ("1h"), startTime (ms), endTime (ms), limit (max 500)
    Returns: [{symbol, sumOpenInterest, sumOpenInterestValue, timestamp}]

Data schemas produced
---------------------
Funding DataFrame:
    index : UTC DatetimeTZDtype (UTC)
    ts    : UTC datetime (same as index)
    funding_rate : float (raw decimal, e.g. 0.0001 = 0.01%)
    symbol       : str
    source       : "binance"

Open Interest DataFrame:
    index         : UTC DatetimeTZDtype (UTC)
    ts            : UTC datetime
    open_interest : float (sumOpenInterest in BTC)
    oi_value_usd  : float (sumOpenInterestValue in USDT)
    symbol        : str
    source        : "binance"
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"

# Max items per Binance API page
FUNDING_PAGE_LIMIT = 1000
OI_PAGE_LIMIT      = 500

# Request timeout and retry settings
REQUEST_TIMEOUT = 30   # seconds
MAX_RETRIES     = 3
RETRY_DELAY     = 2.0  # seconds between retries


def _get(url: str, params: dict) -> dict | list:
    """GET with simple retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError("Unreachable")


def _dt_to_ms(dt: datetime) -> int:
    """Convert a UTC-aware datetime to Unix milliseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_funding_rates(
    symbol: str = "BTCUSDT",
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch historical funding rates for *symbol* from Binance USD-M.

    Funding is settled every 8 hours (00:00, 08:00, 16:00 UTC).
    Backfills up to ~730 days in paginated 1000-row chunks.

    Parameters
    ----------
    symbol   : Binance USD-M symbol (default "BTCUSDT")
    start_dt : start of range (UTC); defaults to 730 days ago
    end_dt   : end of range (UTC); defaults to now

    Returns
    -------
    pd.DataFrame with columns: ts, funding_rate, symbol, source
    indexed by UTC datetime.
    """
    end_dt   = end_dt   or datetime.now(timezone.utc)
    start_dt = start_dt or (end_dt - timedelta(days=730))

    rows = []
    chunk_start = _dt_to_ms(start_dt)
    end_ms      = _dt_to_ms(end_dt)

    while chunk_start < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": chunk_start,
            "endTime":   end_ms,
            "limit":     FUNDING_PAGE_LIMIT,
        }
        data = _get(FUNDING_URL, params)
        if not data:
            break

        for row in data:
            rows.append({
                "ts":           pd.Timestamp(row["fundingTime"], unit="ms", tz="UTC"),
                "funding_rate": float(row["fundingRate"]),
                "symbol":       symbol,
                "source":       "binance",
            })

        # Advance to next chunk (last fundingTime + 1ms)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts >= end_ms or len(data) < FUNDING_PAGE_LIMIT:
            break
        chunk_start = last_ts + 1

    if not rows:
        logger.warning("No funding rate data returned for %s", symbol)
        return pd.DataFrame(columns=["ts", "funding_rate", "symbol", "source"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("ts").sort_values("ts")
    df = df.set_index("ts")
    df.index.name = "ts"
    return df


def fetch_open_interest_hist(
    symbol: str = "BTCUSDT",
    period: str = "1h",
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch historical open interest for *symbol* from Binance.

    Uses the /futures/data/openInterestHist endpoint which provides
    aggregated OI snapshots (1h, 2h, 4h, 6h, 12h, 1d periods).

    Parameters
    ----------
    symbol   : Binance USD-M symbol (default "BTCUSDT")
    period   : time bucket ("1h" recommended for hourly alignment)
    start_dt : start of range (UTC); defaults to 730 days ago
    end_dt   : end of range (UTC); defaults to now

    Returns
    -------
    pd.DataFrame with columns: ts, open_interest, oi_value_usd, symbol, source
    indexed by UTC datetime.

    Notes
    -----
    Binance only provides ~30 days of historical OI via this endpoint.
    For older history the data will be absent; the feature engineering
    module forward-fills from the earliest available point.
    """
    end_dt   = end_dt   or datetime.now(timezone.utc)
    start_dt = start_dt or (end_dt - timedelta(days=730))

    rows = []
    chunk_start = _dt_to_ms(start_dt)
    end_ms      = _dt_to_ms(end_dt)

    while chunk_start < end_ms:
        params = {
            "symbol":    symbol,
            "period":    period,
            "startTime": chunk_start,
            "endTime":   end_ms,
            "limit":     OI_PAGE_LIMIT,
        }
        try:
            data = _get(OI_HIST_URL, params)
        except Exception as e:
            logger.warning("OI history fetch failed: %s. Returning partial data.", e)
            break

        if not data:
            break

        for row in data:
            rows.append({
                "ts":           pd.Timestamp(row["timestamp"], unit="ms", tz="UTC"),
                "open_interest": float(row["sumOpenInterest"]),
                "oi_value_usd":  float(row["sumOpenInterestValue"]),
                "symbol":        symbol,
                "source":        "binance",
            })

        last_ts = int(data[-1]["timestamp"])
        if last_ts >= end_ms or len(data) < OI_PAGE_LIMIT:
            break
        chunk_start = last_ts + 1

    if not rows:
        logger.warning(
            "No OI history returned for %s. Binance OI history is limited to ~30 days. "
            "Forward-going snapshots will be used when available.",
            symbol,
        )
        return pd.DataFrame(columns=["ts", "open_interest", "oi_value_usd", "symbol", "source"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("ts").sort_values("ts")
    df = df.set_index("ts")
    df.index.name = "ts"
    return df
