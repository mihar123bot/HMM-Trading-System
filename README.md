# HMM Regime-Based Trading System

A professional algorithmic trading research dashboard that uses a **Hidden Markov Model (HMM)** to detect market regimes across 10 assets and combines them with 8 technical confirmations and a full risk management system to generate high-conviction entry signals.

---

## Overview

The system classifies every hourly candle into one of three regimes — **Bull**, **Bear**, or **Neutral** — by fitting a 7-state Gaussian HMM to three market features. A long position is only taken when:

1. The HMM regime has been **Bull for at least N consecutive bars** (configurable)
2. At least **7 out of 8** technical confirmation signals are satisfied simultaneously
3. The 48-hour cooldown after the last exit has elapsed

Exits are triggered by the first of: Stop Loss → Take Profit → Regime Flip.

---

## Project Structure

```
HMM-Trading-System/
├── data_loader.py      # Multi-asset OHLCV data ingestion (yfinance)
├── hmm_engine.py       # GaussianHMM fitting and regime labelling
├── indicators.py       # Technical indicators + CONFIG dict + 8-vote system
├── backtester.py       # Event-driven backtest with SL/TP/regime-flip exits
├── app.py              # Streamlit dashboard (Plotly charts, metrics, trade log)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Architecture

### `data_loader.py`
- Supports **10 assets** via a central `ASSETS` registry (see table below)
- Downloads **hourly** OHLCV data for the last **730 days** via `yfinance`
- Fetches in 59-day chunks to comply with the yfinance 1-hour data limit
- Deduplicates and forward-sorts the index, flattens MultiIndex columns

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
- **Auto-labelling** by mean log-return:
  - Highest mean return state → **Bull**
  - Lowest mean return state → **Bear**
  - All other states → **Neutral**
- Viterbi decoding assigns a regime label to every candle

---

### `indicators.py`

All thresholds and periods are controlled by a **single `CONFIG` dict** at the top of the file — no need to hunt through code to change assumptions:

```python
CONFIG: dict = {
    # Indicator periods
    "rsi_period":          14,
    "momentum_period":     10,
    "volume_sma_period":   20,
    "volatility_period":   24,
    "adx_period":          14,
    "ema_fast":            50,
    "ema_slow":            200,
    "macd_fast":           12,
    "macd_slow":           26,
    "macd_signal":         9,
    # Entry thresholds
    "rsi_max":             90,
    "momentum_min_pct":    1.0,
    "volatility_max_pct":  6.0,
    "adx_min":             25,
    # Voting gate
    "votes_required":      7,
}
```

The Streamlit sidebar overrides these values at runtime — changes take effect immediately without editing any file.

**8 Confirmation Signals (voting system):**

| # | Signal | Default Condition | Indicator |
|---|--------|-----------|-----------|
| 1 | RSI | RSI(14) < 90 | Wilder RSI |
| 2 | Momentum | 10-bar momentum > 1% | Price momentum |
| 3 | Volume | Volume > 20-period SMA | Volume SMA |
| 4 | Volatility | 24-bar annualised vol < 6% | Rolling σ × √8760 |
| 5 | ADX | ADX(14) > 25 | Wilder ADX |
| 6 | EMA Fast | Close > EMA(50) | Exponential MA |
| 7 | EMA Slow | Close > EMA(200) | Exponential MA |
| 8 | MACD | MACD Line > Signal Line | MACD(12,26,9) |

Entry is **only permitted** when `vote_count ≥ votes_required` (default 7).

---

### `backtester.py`

**Capital & Leverage:**
- Starting capital: **$10,000**
- Leverage: **2.5×** (applied to PnL; not a margin-call simulation)

**Entry Rules (all must be true):**
1. HMM regime == Bull
2. Bull regime held for ≥ `min_regime_bars` consecutive bars (default 3)
3. `vote_count ≥ votes_required`
4. Not in 48-hour cooldown

**Exit Rules (first triggered wins, in priority order):**

| Priority | Rule | Default |
|---|---|---|
| 1 (highest) | **Stop Loss** | Price drops ≥ 3% from entry |
| 2 | **Take Profit** | Price rises ≥ 4% from entry |
| 3 | **Regime Flip** | HMM exits Bull state |

**Cooldown:** Hard **48-hour** lock after any exit — prevents re-entry during choppy transitions.

**Metrics computed:**
- Total Return, Buy-and-Hold Return, Alpha
- Win Rate, Max Drawdown, Sharpe Ratio (annualised, ×√8760 for hourly data)
- Exit breakdown: Stop Loss count, Take Profit count, Regime Flip count
- Final Equity

---

### `app.py`

**Sidebar — Strategy Parameters (in order):**

1. **Asset Selection** — dropdown of 10 supported assets
2. **Voting Gate** — `votes_required` slider (1–8)
3. **Risk Management** — Stop Loss %, Take Profit %, Min Regime Bars
4. **Entry Thresholds** — RSI max, Momentum min, Volatility max, ADX min
5. **Indicator Periods** — RSI, Momentum, Volume SMA, Volatility, ADX periods; EMA fast/slow
6. **MACD Settings** — Fast period, Slow period, Signal period

**Dashboard panels:**
- **Signal pill** — `LONG` (green) or `CASH` (red)
- **Regime badge** — Bull / Bear / Neutral
- **Vote scorecard** — live pass/fail status for all 8 signals
- **Candlestick chart** — regime-coloured background (green = Bull, red = Bear, grey = Neutral); EMA overlays; entry/exit markers
- **RSI and Volume subplots**
- **Equity curve** vs Buy-and-Hold
- **Performance metrics** — Total Return, Alpha, Win Rate, Max Drawdown, Sharpe, Final Equity
- **Exit breakdown row** — Stop Loss exits, Take Profit exits, Regime Flip exits, Min Regime Bars used
- **Trade log** — colour-coded by profit/loss
- **HMM state summary** — mean/std return per hidden state

**Caching strategy:**
- `fetch_raw_data(ticker)` — `ttl=3600`, `persist="disk"` — survives app restarts
- `fit_hmm_cached(ticker)` — `ttl=3600`, `persist="disk"` — HMM fit is preserved across sessions
- `run_pipeline(ticker, cfg, risk)` — `ttl=300` — full pipeline with indicators and backtest; keyed by all active settings
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

- Data and HMM model are cached to disk — no repeated API calls or refits on page refresh
- Sidebar changes rerun only the indicator + backtest pipeline (fast)
- Click **"Refresh Data & Refit Model"** to force a full fresh fetch and refit

---

## Running the Backtest Directly

```bash
python3 backtester.py
```

Prints all metrics and the full trade log to the terminal using BTC-USD with default settings.

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

## Disclaimer

This project is for **research and educational purposes only**. It is not financial advice and should not be used for live trading without independent risk assessment. Past simulated performance does not guarantee future results.
