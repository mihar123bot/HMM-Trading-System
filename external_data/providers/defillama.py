"""
external_data/providers/defillama.py
DefiLlama stablecoins API — public, no authentication required.

Endpoint used
-------------
Historical stablecoin circulating supply (daily aggregated):
    GET https://stablecoins.llama.fi/stablecoincharts/all
    Returns: [{date: unix_timestamp, totalCirculating: {peggedUSD: float, ...}}]

Data schema produced
--------------------
Stablecoin supply DataFrame:
    index                  : UTC date (DatetimeTZDtype UTC, midnight)
    date                   : UTC date
    stablecoin_supply_usd  : float — total USD-pegged stablecoin supply
    source                 : "defillama"
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://stablecoins.llama.fi"
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3
RETRY_DELAY     = 2.0


def _get(url: str) -> list | dict:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning("DefiLlama request failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError("Unreachable")


def fetch_stablecoin_supply(
    start_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch historical total USD-pegged stablecoin circulating supply (daily).

    The endpoint returns the sum across all chains and all stablecoins pegged
    to USD. This gives a macro measure of USD liquidity in crypto markets.

    A declining 30-day change in supply signals capital leaving crypto
    (used as a liquidity risk gate in the backtester).

    Parameters
    ----------
    start_dt : optional start filter (UTC); data before this date is dropped.

    Returns
    -------
    pd.DataFrame with columns: date, stablecoin_supply_usd, source
    indexed by UTC midnight datetime.
    """
    url  = f"{BASE_URL}/stablecoincharts/all"
    data = _get(url)

    if not data:
        logger.warning("DefiLlama returned empty stablecoin data.")
        return pd.DataFrame(columns=["date", "stablecoin_supply_usd", "source"])

    rows = []
    for entry in data:
        ts = entry.get("date")
        if ts is None:
            continue
        # totalCirculating may contain peggedUSD, peggedEUR, etc.
        circulating = entry.get("totalCirculating", {})
        usd_supply  = float(circulating.get("peggedUSD", 0.0))
        rows.append({
            "date":                 pd.Timestamp(int(ts), unit="s", tz="UTC"),
            "stablecoin_supply_usd": usd_supply,
            "source":               "defillama",
        })

    if not rows:
        return pd.DataFrame(columns=["date", "stablecoin_supply_usd", "source"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("date").sort_values("date")
    df = df.set_index("date")
    df.index.name = "date"

    if start_dt is not None:
        start_ts = pd.Timestamp(start_dt, tz="UTC")
        df = df[df.index >= start_ts]

    return df
