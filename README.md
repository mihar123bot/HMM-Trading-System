# HMM Regime-Based Trading System

A professional algorithmic trading research dashboard that uses a **Hidden Markov Model (HMM)** to detect BTC market regimes and combines them with a 4-bucket technical confirmation gate, a full risk management system, and external on-chain / derivatives data to generate high-conviction entry signals.

> **Scope:** BTC-USD spot only · hourly bars · no leverage · no derivatives execution

---

## Overview

The system classifies every hourly candle into one of three regimes — **Bull**, **Bear**, or **Neutral** — by fitting a 7-state Gaussian HMM to three market features. A long position is only taken when:

1. The HMM regime has been **Bull for at least N consecutive bars** (configurable)
2. **All 4 bucket minimums** of the technical confirmation gate are met
3. The 48-hour cooldown after the last exit has elapsed
4. No active **Phase 3 risk gates** (kill switch, stress filter, external data gates)

Exits are triggered by the first of: Force-Flat (stress spike) → Kill Switch → Stop Loss → Take Profit → Regime Flip.

---

## Project Structure

```
HMM-Trading-System/
├── data_loader.py           # OHLCV ingestion (yfinance) + data sanity checks + merged loader
├── hmm_engine.py            # GaussianHMM fitting and regime labelling
├── indicators.py            # Technical indicators + CONFIG dict + bucket voting gate + risk gates
├── backtester.py            # Event-driven backtest: v1_next_open fills, attribution, waterfall,
│                            #   vol targeting, kill switch, stress gates, tail metrics
├── walk_forward.py          # Rolling walk-forward engine + lockbox OOS evaluation
├── app.py                   # Streamlit dashboard (charts, metrics, trade log, diagnostics)
├── requirements.txt         # Python dependencies
├── external_data/           # Phase 4: external on-chain + derivatives data package
│   ├── __init__.py
│   ├── update.py            # CLI entry point (fetch + build merged hourly parquet)
│   ├── features.py          # Feature engineering (funding_z, oi_z, stablecoin_change_30d)
│   ├── quality.py           # Data quality checks + JSON report
│   ├── storage.py           # Parquet storage layer (PATHS, save, load, append_or_replace)
│   └── providers/
│       ├── binance_futures.py   # Funding rates + OI history (Binance USD-M Futures API)
│       └── defillama.py         # Stablecoin supply (DefiLlama API)
├── data/                    # Auto-created data directory (gitignored)
│   ├── raw/binance/         # funding/<symbol>.parquet, open_interest/<symbol>.parquet
│   ├── raw/defillama/       # stablecoins.parquet
│   ├── features/hourly_merged/  # <ticker>.parquet (OHLCV + external features, hourly)
│   └── reports/             # data_quality_YYYYMMDD.json
└── README.md                # This file
```

---

## Architecture

### `data_loader.py`

- Downloads **hourly** OHLCV data for the last **730 days** via `yfinance` (59-day chunks)
- Deduplicates and forward-sorts the index, flattens MultiIndex columns
- `run_data_sanity_checks(df)` — pre-flight validation returning:
  - % missing hourly bars / gap count
  - Negative-close and negative-volume flags
  - Zero-volume bar count
  - Timezone label
  - Range-outlier % `(High-Low)/Close > 10%`
  - Close min / max / mean
  - Human-readable `issues` list (empty = clean)
- `load_btc_merged_hourly(ticker, use_external, ...)` — loads the Phase 4 merged parquet (OHLCV + external features) and falls back to OHLCV-only if the parquet is missing

**Supported Assets:**

| Display Name | Ticker |
|---|---|
| Bitcoin | BTC-USD |
| Ethereum | ETH-USD |
| Solana | SOL-USD |
| XRP | XRP-USD |
| NVIDIA | NVDA |
| Broadcom | AVGO |
| Taiwan Semiconductor | TSM |
| Microsoft | MSFT |
| ServiceNow | NOW |
| Coinbase | COIN |

---

### `hmm_engine.py`

- Fits a **`hmmlearn.GaussianHMM`** with **7 hidden states** using 200 EM iterations
- Three training features:
  1. **Log Return** — `log(Close_t / Close_{t-1})`
  2. **Range Ratio** — `(High − Low) / Close` (intrabar volatility proxy)
  3. **Volume Volatility** — 5-bar rolling std of `log(Volume)`
- All features are z-score scaled (`StandardScaler`) before training
- **Auto-labelling** by mean log-return: highest → Bull, lowest → Bear, rest → Neutral
- Viterbi decoding assigns a regime label; forward-backward gives `p_bull` confidence

---

### `indicators.py`

All thresholds and periods are controlled by a **single `CONFIG` dict**. Key Phase 2 additions:

```python
CONFIG: dict = {
    # Periods
    "rsi_period": 14, "momentum_period": 10, "volume_sma_period": 20,
    "volatility_period": 24, "adx_period": 14, "atr_period": 14,
    "ema_fast": 50, "ema_slow": 200,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    # Thresholds
    "rsi_max": 90, "momentum_min_pct": 1.0,
    "volatility_max_pct": 80.0,   # Phase 2 fix: was 6.0 (too tight for BTC)
    "adx_min": 25, "p_bull_min": 0.55,
    # Bucket voting
    "trend_min": 2, "strength_min": 1,
    "participation_min": 1, "risk_min": 2,
    # Phase 2: per-signal enable/disable toggles
    "sig_rsi_on": True, "sig_momentum_on": True, "sig_volume_on": True,
    "sig_volatility_on": True, "sig_adx_on": True,
    "sig_ema_fast_on": True, "sig_ema_slow_on": True,
    "sig_macd_on": True, "sig_confidence_on": True,
    # Phase 3: risk gate parameters
    "stress_range_threshold": 0.03, "stress_cooldown_hours": 12,
    "stress_force_flat": False, "market_quality_filter": False,
    "kill_switch_enabled": False, "kill_switch_dd_pct": 10.0,
    "kill_switch_cooldown_h": 48,
    "vol_targeting_enabled": False, "vol_target_pct": 30.0,
    "vol_target_min_mult": 0.25, "vol_target_max_mult": 1.0,
}
```

**4-Bucket Voting Gate — ALL bucket minimums must be met to enable entry:**

| Bucket | Signals | Default Min | Phase 2 |
|--------|---------|-------------|---------|
| **Trend** | EMA Fast, EMA Slow, MACD | ≥ 2 | Per-signal toggles; max recalculated dynamically |
| **Strength** | ADX | ≥ 1 | Toggle; auto-passes if 0 active signals |
| **Participation** | Volume > SMA | ≥ 1 | Toggle; auto-passes if 0 active signals |
| **Risk/Cond.** | RSI, Volatility, Momentum, HMM Confidence | ≥ 2 | Per-signal toggles; dynamic max |

Phase 2 also adds:
- `df["range_1h"] = (High - Low) / Close` — intrabar range ratio
- `df["stress_spike"] = range_1h >= stress_range_threshold` — stress bar flag
- `df.attrs["bucket_maxes"]` and `df.attrs["bucket_active_signals"]` — UI introspection
- Historical pass rates per signal (displayed in dashboard scorecard)

---

### `backtester.py`

**Capital:** $10,000 starting · **Leverage:** 1× (spot only)

**Execution model (`v1_next_open`):**

Signal fires at bar `i` (based on `Close[i]`); fill executes at bar `i+1`'s `Open`.

**Cost model — slippage and fees are separated:**

| Component | Model |
|-----------|-------|
| Slippage | Embedded in fill price: `entry_fill = Open × (1 + slippage_bps/10_000)` |
| Fee | Dollar debit from equity: `fee_usd = cash × fee_bps / 10_000` per side |
| Notional | `cash` (spot, 1×) × `pos_size_mult` (vol targeting, Phase 3) |

**Per-trade cost breakdown (all logged):**

`Fee Entry ($)` · `Fee Exit ($)` · `Slippage Entry ($)` · `Slippage Exit ($)` · `Total Cost ($)` · `Notional ($)` · `Sanity Pass` ✓

**Entry Rules (all must hold):**
1. HMM regime == Bull
2. Bull streak ≥ `min_regime_bars` consecutive bars (default 3)
3. All 4 bucket minimums met (`signal_ok == True`)
4. Not in 48-hour post-exit cooldown
5. Not in post-stress cooldown (`stress_cooldown_hours` after a spike bar)
6. Market quality filter: not a stress spike bar (if `use_market_quality_filter`)
7. Kill switch not active (if `kill_switch_enabled`)

**Exit Rules (first triggered wins):**

| Priority | Rule | Condition |
|---|---|---|
| 1 (highest) | **Force Flat** | Stress spike bar detected (if `stress_force_flat`) |
| 2 | **Kill Switch** | Rolling drawdown from HWM ≥ `kill_switch_dd_pct` |
| 3 | **Stop Loss** | Price drops ≥ stop % from entry fill |
| 4 | **Take Profit** | Price rises ≥ take profit % from entry fill |
| 5 | **Regime Flip** | HMM exits Bull state |

**Phase 3 — Risk Gates:**

| Gate | Parameter | Description |
|------|-----------|-------------|
| **Kill Switch** | `kill_switch_dd_pct`, `kill_switch_cooldown_h` | Halts trading N hours after rolling drawdown exceeds threshold |
| **Market Quality Filter** | `stress_range_threshold`, `use_market_quality_filter` | Blocks entries on bars where `(H-L)/C ≥ threshold` |
| **Stress Force-Flat** | `stress_force_flat`, `stress_range_threshold` | Force-exits open position on stress spike bars |
| **Post-Stress Cooldown** | `stress_cooldown_hours` | Always blocks new entries for N hours after any spike bar |
| **Vol Targeting** | `vol_target_pct`, `vol_target_min/max_mult` | Scales position size: `mult = clip(target_vol / realized_vol, min, max)` |

**Tail Metrics (Phase 3):**

| Metric | Computation |
|--------|-------------|
| Sortino Ratio | Mean hourly return / downside std × √8760 |
| CVaR 95% | 5th percentile of hourly returns × √8760 × 100 (annualised %) |
| Max Consecutive Losses | Longest losing trade streak |
| Worst Decile Trade | 10th percentile trade P&L |
| Large Loss Trades | Trades with return < 2× stop loss |
| Time-to-Recovery | Bars from equity trough to new HWM |

**Attribution & Waterfall (returned in `metrics`):**

Every run produces:
- `metrics["attribution"]` — per-bar gate counts: `bars_bull_regime`, `bars_signal_ok_while_bull`, `bars_eligible`, `entries_blocked_cooldown`, `entries_blocked_gate`, `entries_blocked_external`, `exits_force_flat`, `exits_kill_switch`, and `pct_time_*` percentages
- `metrics["waterfall"]` — 6-step eligibility funnel (total → bull → signal_ok → eligible → not_blocked → entries_taken)
- `metrics["config_hash"]` — 8-char MD5 of key params (for run reproducibility)
- `metrics["execution_rule_version"]` — `"v1_next_open"` (tagged on every trade record too)

---

### `walk_forward.py`

**Methodology:**
1. Reserve the last `lockbox_pct` of bars as a never-touched Lockbox OOS set
2. Roll training/test folds: fit fresh HMM per fold, trade on out-of-sample test window
3. Concatenate all test equity curves → composite OOS equity curve
4. Evaluate Lockbox using the final fold's model (evaluated once at the end)
5. Save a JSON snapshot to `wf_results/` with full attribution and waterfall per fold

**Boundary guarantee (PDR-D):**
`fold_train.index[-1] < fold_test.index[0]` is asserted at runtime — no index overlap possible.

**Snapshot schema includes:**
`config_hash` · `execution_rule_version` · `fold_metrics[].attribution` · `fold_metrics[].waterfall`

---

### `external_data/` — Phase 4: External Data Ingestion

External on-chain and derivatives data is fetched, stored in Parquet, and merged with OHLCV for use as risk gates.

**Providers:**

| Provider | Data | Endpoint | Depth |
|----------|------|----------|-------|
| Binance USD-M | Funding rates (8h) | `GET /fapi/v1/fundingRate` | Up to 730 days |
| Binance USD-M | Open interest history (1h) | `GET /futures/data/openInterestHist` | ~30 days (API limit) |
| DefiLlama | Total stablecoin supply (daily) | `GET /stablecoincharts/all` | Full history |

**Features computed:**

| Feature | Source | Description |
|---------|--------|-------------|
| `funding_rate` | Binance | Raw 8h funding rate, forward-filled to hourly |
| `funding_z` | Binance | 90-day rolling z-score of funding rate |
| `open_interest` | Binance | Raw OI in BTC |
| `oi_change_1h` | Binance | % change from previous hour |
| `oi_z` | Binance | 90-day rolling z-score of OI |
| `stablecoin_supply_usd` | DefiLlama | Total USD stablecoin supply, forward-filled |
| `stablecoin_supply_change_30d` | DefiLlama | 30-day % change in supply |
| `ext_overheat` | Derived | `True` when `|funding_z| > overheat_z_threshold` |
| `ext_low_liquidity` | Derived | `True` when `stablecoin_change_30d < liquidity_change_min` |

All rolling statistics use only past observations (no future leakage).

**Storage layout:**
```
data/
  raw/
    binance/
      funding/BTCUSDT.parquet
      open_interest/BTCUSDT.parquet
    defillama/
      stablecoins.parquet
  features/
    hourly_merged/BTC-USD.parquet    ← merged OHLCV + all external features
  reports/
    data_quality_YYYYMMDD.json
```

**CLI usage:**
```bash
# Full backfill (up to 730 days):
python -m external_data.update --backfill_days 730

# Incremental update (append since last fetch):
python -m external_data.update --incremental

# Rebuild merged feature file only (raw data already fetched):
python -m external_data.update --merge_only

# Custom symbol / thresholds:
python -m external_data.update --symbol BTCUSDT --overheat_z 2.0 --liquidity_min 0.0
```

---

### `app.py`

**Sidebar — Strategy Parameters:**

1. **Asset Selection** — dropdown of supported assets
2. **Voting Gate** — per-bucket minimum sliders + per-signal enable/disable checkboxes
3. **HMM Confidence** — `p_bull` threshold slider
4. **Risk Management** — Stop Loss %, Take Profit %, Min Regime Bars
5. **Execution Costs** — Fee bps/side, Slippage bps/side
6. **Stop Mode** — Fixed % or ATR-Scaled (with ATR multiples)
7. **Entry Thresholds** — RSI max, Momentum min, Volatility max, ADX min
8. **Indicator Periods** — all lookback windows
9. **MACD Settings** — Fast / Slow / Signal periods
10. **Risk Gates (Phase 3):**
    - Kill Switch (enable, DD threshold %, cooldown hours)
    - Market Quality / Stress Filter (enable filter, enable force-flat, range threshold, cooldown hours)
    - Vol Targeting (enable, target vol %, min/max multiplier)

**Main Dashboard panels:**

- **Data Quality expander** — sanity report (missing rows, zero-volume, outliers, timezone); warns if issues found
- **Signal pill** — `LONG` (green) or `CASH` (red)
- **Regime badge** — Bull / Bear / Neutral / Neutral-HighVol
- **Bull Confidence bar** — `p_bull` vs threshold
- **Bucket scorecard** — live pass/fail for all 4 buckets + individual signals with historical pass rates
- **Candlestick chart** — regime-shaded background; EMA overlays; entry/exit markers
- **RSI and Volume subplots**
- **Performance metrics** — Total Return, Alpha, Win Rate, Max Drawdown, Sharpe, Final Equity
- **Exit breakdown row** — Stop Loss, Take Profit, Regime Flip, Force Flat, Kill Switch exits; Total Fees; Stop Mode; Exec Rule
- **Tail metrics row** — Sortino, CVaR 95%, Max Consec Losses, Worst Decile Trade, Large Loss Trades, Time-to-Recovery
- **Config hash & execution rule** caption
- **Equity curve** vs Buy-and-Hold
- **Enhanced trade log** — cost breakdown columns (`Fee Entry`, `Fee Exit`, `Slippage Entry/Exit`, `Total Cost`, `Notional`, `Sanity ✓`)
- **HMM state summary** — mean/std return per hidden state

**Walk-Forward Analysis tab:**

- Per-fold metrics table
- **Constraint Attribution panel** — aggregated across folds: % time Bull, % signal OK, % eligible, blocked by cooldown, gate, and external; Force Flat + Kill Switch exits
- **Eligibility Waterfall chart** — horizontal bar chart + table for the last fold
- OOS equity curve (concatenated test windows)
- Lockbox OOS metrics
- JSON download

**Caching strategy:**
- `fetch_raw_data(ticker)` — `ttl=3600`, `persist="disk"` — survives app restarts
- `fit_hmm_cached(ticker)` — `ttl=3600`, `persist="disk"` — HMM fit is preserved across sessions
- `run_pipeline(ticker, cfg, risk)` — `ttl=300` — full pipeline keyed by all active settings
- **"Refresh Data & Refit Model"** button clears all caches and reruns from scratch

---

## Installation

```bash
cd HMM-Trading-System
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The dashboard opens in your browser at `http://localhost:8501`.

---

## Fetching External Data (Phase 4)

Before the app can use external gates, fetch the data at least once:

```bash
python -m external_data.update --backfill_days 730
```

This fetches ~730 days of funding rates, ~30 days of OI history, and full DefiLlama stablecoin supply history, then builds the merged hourly feature parquet. Subsequent runs can use `--incremental` to append new data only.

---

## Running the Backtest Directly

```bash
python3 backtester.py
```

Prints metrics, attribution, eligibility waterfall, tail metrics, full trade log, and fee sanity audit for BTC-USD with default settings.

---

## Running Walk-Forward

```bash
python3 walk_forward.py BTC-USD
```

Runs a rolling walk-forward (180d train / 14d test / 20% lockbox) and saves a JSON snapshot to `wf_results/`.

---

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — Streamlit installs `requirements.txt` automatically

> Note: External data (Phase 4) requires a persistent volume or scheduled job to stay fresh on cloud deployments. The app degrades gracefully to OHLCV-only signals if the parquet files are absent.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Multi-asset OHLCV data download |
| `hmmlearn` | GaussianHMM regime detection |
| `scikit-learn` | StandardScaler feature normalisation |
| `pandas` / `numpy` | Data processing |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard framework |
| `pyarrow` | Parquet storage for external data (Phase 4) |
| `requests` | HTTP client for Binance + DefiLlama APIs (Phase 4) |

---

## Changelog

### Phase 4 — External Data Ingestion (PDR-H)

New `external_data/` package providing on-chain and derivatives signals:

- **Binance USD-M API**: paginated funding rate history (730d), open interest history (~30d)
- **DefiLlama API**: total stablecoin supply (full history, daily)
- **Feature engineering**: `funding_z` (90d rolling z-score), `oi_change_1h`, `oi_z`, `stablecoin_supply_change_30d`
- **Risk gate columns**: `ext_overheat` (`|funding_z| > threshold`) and `ext_low_liquidity` (`stablecoin_change_30d < min`)
- **Parquet storage** with metadata sidecars (`last_updated_utc`, `source`, `schema_hash`, `row_count`)
- **Data quality reports** — JSON snapshots with coverage statistics per data source
- **CLI update tool** (`python -m external_data.update`) with `--backfill_days`, `--incremental`, `--merge_only` modes
- **Graceful degradation** — app and backtester work without external data; gates default to `False` (pass) when data absent

### Phase 3 — Risk Controls & Tail Metrics (PDR-G, J, K, L, M)

- **Kill Switch** — triggers N-hour trading halt when rolling drawdown from HWM exceeds threshold
- **Stress Force-Flat** — force-exits open positions when `(H-L)/C ≥ threshold`; always sets post-stress cooldown
- **Market Quality Filter** — blocks new entries on stress spike bars
- **Post-Stress Cooldown** — always active after any spike bar, independent of entry filter setting
- **Volatility Targeting** — scales position size: `mult = clip(vol_target_pct / realized_vol, min, max)`; idle cash tracked and returned at exit
- **Tail Metrics** — Sortino ratio, CVaR 95% (annualised), max consecutive losses/wins, worst decile trade, large-loss count, time-to-recovery
- **Extended Attribution** — tracks `entries_blocked_gate`, `entries_blocked_external`, `exits_force_flat`, `exits_kill_switch`
- **Sidebar Risk Gates section** — Kill Switch, Market Quality/Stress Filter, and Vol Targeting expanders in Streamlit sidebar
- **Tail metrics row** added to main dashboard performance panel

### Phase 2 — Bucket Gate Refactor (PDR-F)

- **Critical fix**: `volatility_max_pct` corrected from `6.0` → `80.0` — the previous threshold blocked virtually all BTC entries (BTC annualised hourly volatility is typically 40–100%)
- **Per-signal enable/disable toggles** (`sig_*_on` keys in CONFIG and sidebar checkboxes) — each of the 9 signals can be individually disabled without removing it from the scorecard
- **Dynamic bucket max** — disabled signals are excluded from the bucket's max count; buckets with 0 active signals auto-pass
- **Historical pass rates** — each signal displays its historical pass rate (% of bars where condition was true) next to the live value in the scorecard
- **Intrabar range features** — `range_1h` and `stress_spike` columns added for Phase 3 gates

### Phase 1 — Diagnostics, Realism, Cost Integrity (PDR-A–E, I)

**Execution (`v1_next_open`)** — fills now execute at the next bar's `Open` instead of the current bar's `Close`. More realistic; all results reflect this change.

**Cost model** — slippage and fees are now strictly separated:
- Slippage modelled as a price adjustment to the fill
- Fees modelled as an explicit dollar debit from equity (based on notional, not BTC price)
- Per-trade audit columns: `Fee Entry ($)`, `Fee Exit ($)`, `Slippage Entry ($)`, `Slippage Exit ($)`, `Total Cost ($)`, `Sanity Pass`

**Leverage removed** — system is spot-only (1×); the former 2.5× PnL multiplier has been removed.

**Attribution & Waterfall** — every backtest (and every walk-forward fold) now returns:
- Constraint attribution: % time in Bull, % signal OK, % eligible, entries blocked by cooldown, exit type counts
- Eligibility waterfall: 6-step funnel from total bars to entries taken

**Train/test boundary assertion** — `assert fold_train.index[-1] < fold_test.index[0]` on every fold (PDR-D).

**Data sanity checks** — `run_data_sanity_checks(df)` validates OHLCV integrity before every run.

**Config hash + execution version** — every trade record and report snapshot is tagged with `config_hash` and `execution_rule_version` for run reproducibility.

---

## Disclaimer

This project is for **research and educational purposes only**. It is not financial advice and should not be used for live trading without independent risk assessment. Past simulated performance does not guarantee future results.
