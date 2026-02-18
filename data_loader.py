"""
data_loader.py
Fetches hourly OHLCV data for any supported asset using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ── Supported asset registry ─────────────────────────────────────────────────
ASSETS: dict[str, str] = {
    "Bitcoin (BTC-USD)":                              "BTC-USD",
    "Ethereum (ETH-USD)":                             "ETH-USD",
    "Solana (SOL-USD)":                               "SOL-USD",
    "XRP (XRP-USD)":                                  "XRP-USD",
    "NVIDIA (NVDA)":                                  "NVDA",
    "Broadcom (AVGO)":                                "AVGO",
    "Taiwan Semiconductor (TSM)":                     "TSM",
    "Microsoft (MSFT)":                               "MSFT",
    "ServiceNow (NOW)":                               "NOW",
    "Coinbase (COIN)":                                "COIN",
}

# Reverse lookup: ticker -> display name
TICKER_TO_NAME: dict[str, str] = {v: k for k, v in ASSETS.items()}


def fetch_asset_data(ticker: str, days: int = 730, interval: str = "1h") -> pd.DataFrame:
    """
    Download OHLCV hourly data for *ticker* covering the last *days* days.

    yfinance caps 1h data at ~730 days; fetches in 59-day chunks.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume
    indexed by UTC datetime.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    chunk_size = 59
    frames = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_size), end)
        df = yf.download(
            ticker,
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
        raise RuntimeError(
            f"yfinance returned no data for {ticker}. Check your internet connection."
        )

    data = pd.concat(frames)
    data = data[~data.index.duplicated(keep="first")]
    data.sort_index(inplace=True)

    # Flatten MultiIndex columns if present (yfinance ≥0.2 behaviour)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.dropna(inplace=True)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data.dropna(inplace=True)

    return data


if __name__ == "__main__":
    for name, ticker in ASSETS.items():
        try:
            df = fetch_asset_data(ticker)
            print(f"{name}: {len(df)} bars  {df.index[0]} → {df.index[-1]}")
        except Exception as e:
            print(f"{name}: ERROR — {e}")
