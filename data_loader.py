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


def run_data_sanity_checks(df: pd.DataFrame) -> dict:
    """
    Run pre-flight data quality checks on an OHLCV DataFrame.

    Returns a dict with:
        n_rows              : total bars
        date_range          : first → last timestamp string
        pct_missing_rows    : % of expected hourly slots that are absent
        n_gaps              : count of missing hourly slots
        has_negative_close  : True if any Close <= 0
        has_negative_volume : True if any Volume < 0
        n_zero_volume       : count of bars with Volume == 0
        timezone_utc        : tz label of the index (None if naive)
        range_outlier_pct   : % of bars where (High-Low)/Close > 10%
        close_min           : minimum Close
        close_max           : maximum Close
        close_mean          : mean Close
        issues              : list of human-readable warning strings (empty = clean)
    """
    n = len(df)
    if n == 0:
        return {"n_rows": 0, "issues": ["DataFrame is empty"]}

    # Expected hourly index
    try:
        expected_idx = pd.date_range(df.index[0], df.index[-1], freq="h")
        n_expected = len(expected_idx)
        n_gaps     = n_expected - n
        pct_missing = round(n_gaps / max(n_expected, 1) * 100, 2)
    except Exception:
        n_gaps      = -1
        pct_missing = -1.0

    has_neg_close  = bool((df["Close"] <= 0).any())
    has_neg_vol    = bool((df["Volume"] < 0).any())
    n_zero_vol     = int((df["Volume"] == 0).sum())

    tz_label = str(df.index.tz) if df.index.tz is not None else "naive (no tz)"

    range_ratio    = (df["High"] - df["Low"]) / df["Close"].replace(0, float("nan"))
    n_outliers     = int((range_ratio > 0.10).sum())
    outlier_pct    = round(n_outliers / max(n, 1) * 100, 2)

    # Build issue list
    issues: list[str] = []
    if pct_missing > 1.0:
        issues.append(f"{pct_missing:.1f}% of expected hourly bars are missing ({n_gaps} gaps)")
    if has_neg_close:
        issues.append("Close <= 0 detected — data quality problem")
    if has_neg_vol:
        issues.append("Volume < 0 detected — data quality problem")
    if n_zero_vol > 0:
        issues.append(f"{n_zero_vol} bars with zero volume")
    if outlier_pct > 5.0:
        issues.append(f"{outlier_pct:.1f}% of bars have (High-Low)/Close > 10% — possible outliers")
    if "naive" in tz_label:
        issues.append("Index has no timezone — expected UTC-aware timestamps")

    return {
        "n_rows":              n,
        "date_range":          f"{df.index[0]} → {df.index[-1]}",
        "pct_missing_rows":    pct_missing,
        "n_gaps":              n_gaps,
        "has_negative_close":  has_neg_close,
        "has_negative_volume": has_neg_vol,
        "n_zero_volume":       n_zero_vol,
        "timezone_utc":        tz_label,
        "range_outlier_pct":   outlier_pct,
        "close_min":           round(float(df["Close"].min()), 2),
        "close_max":           round(float(df["Close"].max()), 2),
        "close_mean":          round(float(df["Close"].mean()), 2),
        "issues":              issues,
    }


if __name__ == "__main__":
    for name, ticker in ASSETS.items():
        try:
            df = fetch_asset_data(ticker)
            report = run_data_sanity_checks(df)
            status = "✓ CLEAN" if not report["issues"] else f"⚠ {len(report['issues'])} issue(s)"
            print(f"{name}: {len(df)} bars  {df.index[0]} → {df.index[-1]}  [{status}]")
            for issue in report["issues"]:
                print(f"    ! {issue}")
        except Exception as e:
            print(f"{name}: ERROR — {e}")
