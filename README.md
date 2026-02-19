# HMM Regime-Based Trading System

A professional algorithmic trading research dashboard that uses a **Hidden Markov Model (HMM)** to detect BTC market regimes and combines them with a 4-bucket technical confirmation gate and a full risk management system to generate high-conviction entry signals.

> **Scope:** BTC-USD spot only · hourly bars · no leverage · no derivatives execution

---

## Overview

The system classifies every hourly candle into one of three regimes — **Bull**, **Bear**, or **Neutral** — by fitting a 7-state Gaussian HMM to three market features. A long position is only taken when:

1. The HMM regime has been **Bull for at least N consecutive bars** (configurable)
2. **All 4 bucket minimums** of the technical confirmation gate are met
3. The 48-hour cooldown after the last exit has elapsed

Exits are triggered by the first of: Stop Loss → Take Profit → Regime Flip.

---

## Project Structure

```
HMM-Trading-System/
├── data_loader.py      # OHLCV ingestion (yfinance) + data sanity checks
├── hmm_engine.py       # GaussianHMM fitting and regime labelling
├── indicators.py       # Technical indicators + CONFIG dict + bucket voting gate
├── backtester.py       # Event-driven backtest: v1_next_open fills, attribution, waterfall
├── walk_forward.py     # Rolling walk-forward engine + lockbox OOS evaluation
├── app.py              # Streamlit dashboard (charts, metrics, trade log, diagnostics)
├── requirements.txt    # Python dependencies
└── README.md           # This file
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

All thresholds and periods are controlled by a **single `CONFIG` dict**:

```python
CONFIG: dict = {
    # Periods
    "rsi_period": 14, "momentum_period": 10, "volume_sma_period": 20,
    "volatility_period": 24, "adx_period": 14, "atr_period": 14,
    "ema_fast": 50, "ema_slow": 200,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    # Thresholds
    "rsi_max": 90, "momentum_min_pct": 1.0,
    "volatility_max_pct": 6.0, "adx_min": 25, "p_bull_min": 0.55,
    # Bucket voting
    "trend_min": 2, "strength_min": 1,
    "participation_min": 1, "risk_min": 2,
}
```

**4-Bucket Voting Gate** — ALL bucket minimums must be met to enable entry:

| Bucket | Signals (max) | Default Min |
|--------|--------------|-------------|
| **Trend** | EMA Fast, EMA Slow, MACD (3) | ≥ 2 |
| **Strength** | ADX (1) | ≥ 1 |
| **Participation** | Volume > SMA (1) | ≥ 1 |
| **Risk/Cond.** | RSI, Volatility, Momentum, HMM Confidence (4) | ≥ 2 |

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
| Notional | `cash` (spot, 1×) |

**Per-trade cost breakdown (all logged):**

`Fee Entry ($)` · `Fee Exit ($)` · `Slippage Entry ($)` · `Slippage Exit ($)` · `Total Cost ($)` · `Notional ($)` · `Sanity Pass` ✓

**Entry Rules (all must hold):**
1. HMM regime == Bull
2. Bull streak ≥ `min_regime_bars` consecutive bars (default 3)
3. All 4 bucket minimums met (`signal_ok == True`)
4. Not in 48-hour post-exit cooldown

**Exit Rules (first triggered wins):**

| Priority | Rule | Default |
|---|---|---|
| 1 (highest) | **Stop Loss** | Price drops ≥ 3% from entry fill |
| 2 | **Take Profit** | Price rises ≥ 4% from entry fill |
| 3 | **Regime Flip** | HMM exits Bull state |

**Attribution & Waterfall (returned in `metrics`):**

Every run produces:
- `metrics["attribution"]` — per-bar gate counts: `bars_bull_regime`, `bars_signal_ok_while_bull`, `bars_eligible`, `entries_blocked_cooldown`, exit type counts, and `pct_time_*` percentages
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

### `app.py`

**Sidebar — Strategy Parameters:**

1. **Asset Selection** — dropdown of supported assets
2. **Voting Gate** — per-bucket minimum sliders
3. **HMM Confidence** — `p_bull` threshold slider
4. **Risk Management** — Stop Loss %, Take Profit %, Min Regime Bars
5. **Execution Costs** — Fee bps/side, Slippage bps/side
6. **Stop Mode** — Fixed % or ATR-Scaled (with ATR multiples)
7. **Entry Thresholds** — RSI max, Momentum min, Volatility max, ADX min
8. **Indicator Periods** — all lookback windows
9. **MACD Settings** — Fast / Slow / Signal periods

**Main Dashboard panels:**

- **Data Quality expander** — sanity report (missing rows, zero-volume, outliers, timezone); warns if issues found
- **Signal pill** — `LONG` (green) or `CASH` (red)
- **Regime badge** — Bull / Bear / Neutral / Neutral-HighVol
- **Bull Confidence bar** — `p_bull` vs threshold
- **Bucket scorecard** — live pass/fail for all 4 buckets + 9 individual signals
- **Candlestick chart** — regime-shaded background; EMA overlays; entry/exit markers
- **RSI and Volume subplots**
- **Performance metrics** — Total Return, Alpha, Win Rate, Max Drawdown, Sharpe, Final Equity
- **Exit breakdown row** — Stop Loss, Take Profit, Regime Flip exits; Total Fees; Stop Mode; Exec Rule
- **Config hash & execution rule** caption
- **Equity curve** vs Buy-and-Hold
- **Enhanced trade log** — cost breakdown columns (`Fee Entry`, `Fee Exit`, `Slippage Entry/Exit`, `Total Cost`, `Notional`, `Sanity ✓`)
- **HMM state summary** — mean/std return per hidden state

**Walk-Forward Analysis tab:**

- Per-fold metrics table
- **Constraint Attribution panel** — aggregated across folds: % time Bull, % signal OK, % eligible, blocked by cooldown, exit type counts
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

## Running the Backtest Directly

```bash
python3 backtester.py
```

Prints metrics, attribution, eligibility waterfall, full trade log, and fee sanity audit for BTC-USD with default settings.

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

---

## Changelog

### Phase 1 — Diagnostics, Realism, Cost Integrity

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
