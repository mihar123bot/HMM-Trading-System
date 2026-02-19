"""
external_data/features.py
Feature engineering for external data signals.

All computations are leakage-safe:
- Rolling statistics use only past observations (shift(1) where needed).
- No future data bleeds into any feature.

Features produced
-----------------
From funding data (8h → forward-filled hourly):
    funding_rate             : raw funding rate
    funding_z                : 90-day rolling z-score of funding_rate

From open interest data (hourly):
    open_interest            : raw OI in BTC
    oi_change_1h             : % change from previous hour
    oi_z                     : 90-day rolling z-score of open_interest

From stablecoin supply data (daily → forward-filled hourly):
    stablecoin_supply_usd    : total USD stablecoin supply
    stablecoin_supply_change_30d : 30-day % change in supply

Risk gate columns (derived from features):
    ext_overheat      : True when |funding_z| > overheat_z_threshold
    ext_low_liquidity : True when stablecoin_supply_change_30d < liquidity_change_min
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default config
FUNDING_Z_WINDOW        = 90 * 3       # 90 days × 3 funding events/day ≈ 270 observations
OI_Z_WINDOW             = 90 * 24      # 90 days × 24 hours
STABLECOIN_30D_WINDOW   = 30           # 30 calendar days (daily data)
OVERHEAT_Z_THRESHOLD    = 2.0          # |funding_z| > 2 → overheat gate
LIQUIDITY_CHANGE_MIN    = 0.0          # stablecoin_supply_change_30d < 0 → liquidity gate


def _z_score(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score (uses observations up to and including current bar)."""
    mu  = series.rolling(window, min_periods=max(window // 4, 10)).mean()
    sig = series.rolling(window, min_periods=max(window // 4, 10)).std()
    return (series - mu) / sig.replace(0, np.nan)


def build_funding_features(funding_df: pd.DataFrame, z_window: int = FUNDING_Z_WINDOW) -> pd.DataFrame:
    """
    Compute funding features from raw funding rate series.

    Parameters
    ----------
    funding_df : DataFrame with 'funding_rate' column, UTC DatetimeTZDtype index
    z_window   : rolling window for z-score (default 270 = ~90 days of 8h data)

    Returns
    -------
    DataFrame with: funding_rate, funding_z
    """
    df = funding_df[["funding_rate"]].copy()
    df["funding_z"] = _z_score(df["funding_rate"], z_window)
    return df


def build_oi_features(oi_df: pd.DataFrame, z_window: int = OI_Z_WINDOW) -> pd.DataFrame:
    """
    Compute open interest features from raw OI series.

    Parameters
    ----------
    oi_df    : DataFrame with 'open_interest' column, UTC DatetimeTZDtype index (1h)
    z_window : rolling window for z-score (default 2160 = 90 days × 24h)

    Returns
    -------
    DataFrame with: open_interest, oi_change_1h, oi_z
    """
    df = oi_df[["open_interest"]].copy()
    df["oi_change_1h"] = df["open_interest"].pct_change() * 100
    df["oi_z"]         = _z_score(df["open_interest"], z_window)
    return df


def build_stablecoin_features(
    stablecoin_df: pd.DataFrame,
    window_30d: int = STABLECOIN_30D_WINDOW,
) -> pd.DataFrame:
    """
    Compute stablecoin supply features from raw daily supply series.

    Parameters
    ----------
    stablecoin_df : DataFrame with 'stablecoin_supply_usd' column, daily UTC index
    window_30d    : rolling window in days for 30d change (default 30)

    Returns
    -------
    DataFrame with: stablecoin_supply_usd, stablecoin_supply_change_30d
    Daily resolution — caller must forward-fill to hourly.
    """
    df = stablecoin_df[["stablecoin_supply_usd"]].copy()
    lagged = df["stablecoin_supply_usd"].shift(window_30d)
    df["stablecoin_supply_change_30d"] = (
        (df["stablecoin_supply_usd"] - lagged) / lagged.replace(0, np.nan) * 100
    )
    return df


def build_merged_hourly(
    hourly_df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    stablecoin_df: Optional[pd.DataFrame] = None,
    overheat_z_threshold: float = OVERHEAT_Z_THRESHOLD,
    liquidity_change_min: float = LIQUIDITY_CHANGE_MIN,
) -> pd.DataFrame:
    """
    Merge all external features into the canonical hourly OHLCV DataFrame.

    Alignment rules (leakage-safe):
    - Funding (8h events) → forward-filled to hourly
    - OI (hourly) → direct join (or forward-fill if gaps exist)
    - Stablecoins (daily) → forward-filled to hourly

    Gate columns added:
    - ext_overheat      : |funding_z| > overheat_z_threshold
    - ext_low_liquidity : stablecoin_supply_change_30d < liquidity_change_min

    Parameters
    ----------
    hourly_df      : canonical hourly OHLCV DataFrame (UTC index, required)
    funding_df     : optional funding rate DataFrame from binance_futures provider
    oi_df          : optional OI DataFrame from binance_futures provider
    stablecoin_df  : optional stablecoin DataFrame from defillama provider
    overheat_z_threshold : |funding_z| threshold for overheat gate
    liquidity_change_min : stablecoin 30d change % minimum for liquidity gate

    Returns
    -------
    hourly_df with external feature columns appended.
    All new columns default to NaN if source data is absent.
    """
    result = hourly_df.copy()

    # Ensure index is UTC
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")

    # ── Funding features ──────────────────────────────────────────────────────
    if funding_df is not None and not funding_df.empty:
        try:
            f_feat = build_funding_features(funding_df)
            # Resample to hourly if not already; forward-fill (8h events → every hour)
            f_hourly = f_feat.resample("1h").last().reindex(result.index, method="ffill")
            result = result.join(f_hourly[["funding_rate", "funding_z"]], how="left")
        except Exception as e:
            logger.warning("Failed to merge funding features: %s", e)
    else:
        result["funding_rate"] = np.nan
        result["funding_z"]    = np.nan

    # ── Open interest features ────────────────────────────────────────────────
    if oi_df is not None and not oi_df.empty:
        try:
            oi_feat = build_oi_features(oi_df)
            oi_hourly = oi_feat.resample("1h").last().reindex(result.index, method="ffill")
            result = result.join(oi_hourly[["open_interest", "oi_change_1h", "oi_z"]], how="left")
        except Exception as e:
            logger.warning("Failed to merge OI features: %s", e)
    else:
        result["open_interest"] = np.nan
        result["oi_change_1h"]  = np.nan
        result["oi_z"]          = np.nan

    # ── Stablecoin features ───────────────────────────────────────────────────
    if stablecoin_df is not None and not stablecoin_df.empty:
        try:
            sc_feat = build_stablecoin_features(stablecoin_df)
            # Daily → forward-fill to hourly
            sc_hourly = sc_feat.resample("1h").last().reindex(result.index, method="ffill")
            result = result.join(
                sc_hourly[["stablecoin_supply_usd", "stablecoin_supply_change_30d"]],
                how="left",
            )
        except Exception as e:
            logger.warning("Failed to merge stablecoin features: %s", e)
    else:
        result["stablecoin_supply_usd"]         = np.nan
        result["stablecoin_supply_change_30d"]  = np.nan

    # ── Risk gate columns ─────────────────────────────────────────────────────
    fz = result.get("funding_z", pd.Series(np.nan, index=result.index))
    sc = result.get("stablecoin_supply_change_30d", pd.Series(np.nan, index=result.index))

    result["ext_overheat"] = (
        fz.abs() > overheat_z_threshold
    ).where(fz.notna(), other=False).astype(bool)

    result["ext_low_liquidity"] = (
        sc < liquidity_change_min
    ).where(sc.notna(), other=False).astype(bool)

    return result
