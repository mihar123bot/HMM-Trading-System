"""
data_loader.py
Fetches BTC-USD hourly OHLCV data for the last 730 days using yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_btc_data(days: int = 730, interval: str = "1h") -> pd.DataFrame:
    """
    Download BTC-USD OHLCV hourly data.

    yfinance limits: hourly data is only available for the last 730 days.
    We fetch in 60-day chunks to stay within API limits and concatenate.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume
    indexed by UTC datetime.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    # yfinance caps 1h data at ~730 days; fetch in chunks of 59 days
    chunk_size = 59
    frames = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_size), end)
        df = yf.download(
            "BTC-USD",
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if not df.empty:
            frames.append(df)
        chunk_start = chunk_end

    if not frames:
        raise RuntimeError("yfinance returned no data. Check your internet connection.")

    data = pd.concat(frames)
    data = data[~data.index.duplicated(keep="first")]
    data.sort_index(inplace=True)

    # Flatten MultiIndex columns if present (yfinance ≥0.2 behavior)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Keep only OHLCV
    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.dropna(inplace=True)

    # Ensure numeric types
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data.dropna(inplace=True)

    return data


if __name__ == "__main__":
    df = fetch_btc_data()
    print(f"Fetched {len(df)} hourly candles from {df.index[0]} to {df.index[-1]}")
    print(df.tail())
