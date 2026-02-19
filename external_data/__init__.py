"""
external_data — Phase 4 external signal ingestion.

Providers
---------
- binance_futures: funding rate history, open interest (hourly) from Binance USD-M
- defillama:       stablecoin total circulating supply (daily) from DefiLlama

Storage
-------
Parquet files under data/raw/ and data/features/hourly_merged/

Feature Engineering
-------------------
- funding_z             : 90-day rolling z-score of funding rate
- oi_change_1h          : % change in open interest from previous hour
- oi_z                  : 90-day rolling z-score of open interest
- stablecoin_supply_change_30d : 30-day % change in total stablecoin supply

Risk Gates (used in backtester)
--------------------------------
- ext_overheat      : True when |funding_z| > threshold (market over-/under-heated)
- ext_low_liquidity : True when stablecoin supply declining over 30d (capital leaving)

CLI
---
python -m external_data.update --backfill_days 730
python -m external_data.update --incremental
"""

from .storage  import PATHS
from .features import build_merged_hourly

__all__ = ["PATHS", "build_merged_hourly"]
