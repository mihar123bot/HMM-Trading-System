"""
external_data/update.py
CLI entry point for fetching and updating all external data sources.

Usage
-----
  # Full backfill (up to 730 days):
  python -m external_data.update --backfill_days 730

  # Incremental update (append new data since last fetch):
  python -m external_data.update --incremental

  # Specific symbol / custom days:
  python -m external_data.update --symbol BTCUSDT --backfill_days 90

  # Build merged hourly feature file only (assumes raw data already fetched):
  python -m external_data.update --merge_only

Behaviour
---------
1. Fetch funding rates (Binance, paginated, up to 730d)
2. Fetch open interest history (Binance, ~30d max; documented limitation)
3. Fetch stablecoin supply (DefiLlama daily, no date limit)
4. Build features and merge into hourly parquet
5. Save data quality report to data/reports/
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as a module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from external_data.providers.binance_futures import (
    fetch_funding_rates,
    fetch_open_interest_hist,
)
from external_data.providers.defillama import fetch_stablecoin_supply
from external_data.storage import PATHS, save, load, append_or_replace
from external_data.features import build_merged_hourly
from external_data.quality import (
    check_funding, check_oi, check_stablecoins, check_merged, save_quality_report
)
from data_loader import fetch_asset_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch and update HMM Trading System external data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",         default="BTCUSDT",   help="Binance USD-M symbol")
    p.add_argument("--ticker",         default="BTC-USD",   help="yfinance ticker (for OHLCV base)")
    p.add_argument("--backfill_days",  type=int, default=730, help="Days to backfill")
    p.add_argument("--incremental",    action="store_true",  help="Only fetch data newer than last saved")
    p.add_argument("--merge_only",     action="store_true",  help="Skip fetch; only rebuild merged feature file")
    p.add_argument("--overheat_z",     type=float, default=2.0, help="|funding_z| threshold for overheat gate")
    p.add_argument("--liquidity_min",  type=float, default=0.0, help="Stablecoin 30d change %% min for liquidity gate")
    return p.parse_args()


def _incremental_start(path: Path) -> datetime:
    """Return the last timestamp in an existing parquet (or 730d ago if absent)."""
    df = load(path)
    if df.empty or not isinstance(df.index, pd.DatetimeTZDtype.__class__):
        return datetime.now(timezone.utc) - timedelta(days=730)
    try:
        return df.index.max().to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=730)


def main(args: argparse.Namespace) -> None:
    import pandas as pd  # local import to avoid circular at module level

    now      = datetime.now(timezone.utc)
    end_dt   = now
    start_dt = now - timedelta(days=args.backfill_days)

    if args.merge_only:
        logger.info("--merge_only: skipping provider fetches, rebuilding merged file only.")
        _build_merged(args, start_dt)
        return

    # ── 1. Funding rates ──────────────────────────────────────────────────────
    logger.info("=== Funding Rates ===")
    if args.incremental:
        start_dt_funding = _incremental_start(PATHS.funding(args.symbol))
        logger.info("Incremental: fetching since %s", start_dt_funding)
    else:
        start_dt_funding = start_dt

    try:
        funding_df = fetch_funding_rates(args.symbol, start_dt=start_dt_funding, end_dt=end_dt)
        if not funding_df.empty:
            funding_df = append_or_replace(
                funding_df, PATHS.funding(args.symbol),
                dedup_on="ts", source="binance", symbol=args.symbol,
            )
            logger.info("Funding: %d rows saved.", len(funding_df))
        else:
            logger.warning("Funding: no data returned.")
            funding_df = load(PATHS.funding(args.symbol))
    except Exception as e:
        logger.error("Funding fetch failed: %s", e)
        funding_df = load(PATHS.funding(args.symbol))

    qr_funding = check_funding(funding_df)
    logger.info("Funding quality: %s  (issues: %s)", qr_funding["status"], qr_funding["issues"])

    # ── 2. Open Interest ──────────────────────────────────────────────────────
    logger.info("=== Open Interest ===")
    logger.info(
        "NOTE: Binance OI history (/futures/data/openInterestHist) is limited to ~30 days. "
        "Older history will not be available — this is a documented API limitation."
    )

    try:
        oi_df = fetch_open_interest_hist(args.symbol, period="1h", start_dt=start_dt, end_dt=end_dt)
        if not oi_df.empty:
            oi_df = append_or_replace(
                oi_df, PATHS.open_interest(args.symbol),
                dedup_on="ts", source="binance", symbol=args.symbol,
            )
            logger.info("OI: %d rows saved.", len(oi_df))
        else:
            logger.warning("OI: no data returned (expected for historical backfill > 30d).")
            oi_df = load(PATHS.open_interest(args.symbol))
    except Exception as e:
        logger.error("OI fetch failed: %s", e)
        oi_df = load(PATHS.open_interest(args.symbol))

    qr_oi = check_oi(oi_df)
    logger.info("OI quality: %s  (issues: %s)", qr_oi["status"], qr_oi["issues"])

    # ── 3. Stablecoin supply ──────────────────────────────────────────────────
    logger.info("=== Stablecoin Supply ===")
    try:
        sc_df = fetch_stablecoin_supply(start_dt=start_dt)
        if not sc_df.empty:
            save(sc_df, PATHS.stablecoins(), source="defillama")
            logger.info("Stablecoins: %d daily rows saved.", len(sc_df))
        else:
            logger.warning("Stablecoins: no data returned.")
            sc_df = load(PATHS.stablecoins())
    except Exception as e:
        logger.error("Stablecoin fetch failed: %s", e)
        sc_df = load(PATHS.stablecoins())

    qr_sc = check_stablecoins(sc_df)
    logger.info("Stablecoins quality: %s  (issues: %s)", qr_sc["status"], qr_sc["issues"])

    # ── 4. Build merged feature file ──────────────────────────────────────────
    _build_merged(args, start_dt)

    # ── 5. Save quality report ────────────────────────────────────────────────
    quality_report = {
        "funding":     qr_funding,
        "oi":          qr_oi,
        "stablecoins": qr_sc,
    }
    save_quality_report(quality_report, PATHS.quality_report())
    logger.info("Quality report saved.")
    logger.info("=== Update complete ===")


def _build_merged(args: argparse.Namespace, start_dt: datetime) -> None:
    """Fetch OHLCV and merge all external features into hourly parquet."""
    import pandas as pd

    logger.info("=== Building Merged Hourly Feature File ===")
    try:
        hourly_df = fetch_asset_data(args.ticker)
    except Exception as e:
        logger.error("OHLCV fetch failed: %s", e)
        return

    funding_df   = load(PATHS.funding(args.symbol))
    oi_df        = load(PATHS.open_interest(args.symbol))
    stablecoin_df = load(PATHS.stablecoins())

    merged = build_merged_hourly(
        hourly_df,
        funding_df     = funding_df    if not funding_df.empty    else None,
        oi_df          = oi_df         if not oi_df.empty         else None,
        stablecoin_df  = stablecoin_df if not stablecoin_df.empty else None,
        overheat_z_threshold = args.overheat_z,
        liquidity_change_min = args.liquidity_min,
    )

    save(merged, PATHS.merged(args.ticker), source="merged", symbol=args.ticker)
    logger.info("Merged hourly: %d rows saved → %s", len(merged), PATHS.merged(args.ticker))

    qr_merged = check_merged(merged)
    logger.info("Merged quality: %s  (issues: %s)", qr_merged["status"], qr_merged["issues"])


if __name__ == "__main__":
    args = _parse_args()
    main(args)
