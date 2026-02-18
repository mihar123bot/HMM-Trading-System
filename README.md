# HMM Regime-Based BTC Trading System

A professional algorithmic trading research dashboard that uses a **Hidden Markov Model (HMM)** to detect market regimes on Bitcoin and combines them with 8 technical confirmations to generate high-conviction entry signals.

---

## Overview

The system classifies every hourly candle into one of three regimes — **Bull**, **Bear**, or **Neutral** — by fitting a 7-state Gaussian HMM to three market features. A long position is only taken when the HMM regime is Bullish **and** at least 7 out of 8 technical confirmation signals are satisfied simultaneously.

---

## Project Structure

```
HMM-Trading-System/
├── data_loader.py      # BTC-USD OHLCV data ingestion (yfinance)
├── hmm_engine.py       # GaussianHMM fitting and regime labelling
├── indicators.py       # Technical indicators + 8-vote confirmation system
├── backtester.py       # Event-driven backtest simulation
├── app.py              # Streamlit dashboard (Plotly charts, metrics, trade log)
└── requirements.txt    # Python dependencies
```

---

## Architecture

### `data_loader.py`
- Downloads BTC-USD **hourly** OHLCV data for the last **730 days** via `yfinance`
- Fetches in 59-day chunks to comply with the yfinance 1-hour data limit
- Deduplicates and forward-sorts the index, normalises column names

### `hmm_engine.py`
- Fits a **`hmmlearn.GaussianHMM`** with **7 hidden states** using 200 EM iterations
- Three training features:
  1. **Log Return** — `log(Close_t / Close_{t-1})`
  2. **Range Ratio** — `(High − Low) / Close` (intrabar volatility proxy)
  3. **Volume Volatility** — 5-bar rolling std of `log(Volume)`
- All features are z-score scaled before training
- **Auto-labelling** by mean log-return:
  - Highest mean return state → **Bull Run**
  - Lowest mean return state → **Bear / Crash**
  - All other states → **Neutral**
- Viterbi decoding assigns a regime label to every candle

### `indicators.py`
**8 Confirmation Signals (voting system):**

| # | Signal | Condition | Indicator |
|---|--------|-----------|-----------|
| 1 | RSI | RSI(14) < 90 | Wilder RSI |
| 2 | Momentum | 10-bar momentum > 1% | Price momentum |
| 3 | Volume | Volume > 20-period SMA | Volume SMA |
| 4 | Volatility | 24-bar annualised vol < 6% | Rolling σ × √8760 |
| 5 | ADX | ADX(14) > 25 | Wilder ADX |
| 6 | EMA 50 | Close > EMA(50) | Exponential MA |
| 7 | EMA 200 | Close > EMA(200) | Exponential MA |
| 8 | MACD | MACD Line > Signal Line | MACD(12,26,9) |

An entry is **only permitted** when `vote_count ≥ 7`.

### `backtester.py`
- Starting capital: **$10,000**
- Leverage: **2.5×** (applied to PnL, not margin-call simulation)
- **Entry**: Regime = Bull AND vote_count ≥ 7
- **Exit**: Regime flips to Bear or Crash (any non-Bull state)
- **Cooldown**: Hard **48-hour** lock after every exit — prevents re-entry during chop
- Logs every trade with: entry/exit time, prices, PnL, return %, exit reason, and running equity
- Computes: Total Return, Buy-and-Hold Return, Alpha, Win Rate, Max Drawdown, Sharpe Ratio

### `app.py`
Streamlit dashboard with:
- **Signal pill** — `LONG` (green) or `CASH` (red)
- **Regime badge** — Bull / Bear / Neutral
- **Vote scorecard** — live pass/fail status for all 8 signals
- **Candlestick chart** — regime-coloured background (green = Bull, red = Bear, grey = Neutral); EMA 50/200 overlaid; entry/exit markers
- **RSI and Volume subplots**
- **Equity curve** vs Buy-and-Hold
- **Performance metrics** — Total Return, Alpha, Win Rate, Max Drawdown, Sharpe, Final Equity
- **Trade log** — colour-coded by profit/loss
- **HMM state summary** — mean/std return per state

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

- Data is cached for **1 hour** (no repeated API calls on refresh)
- Click **"Refresh Data & Refit Model"** to force a fresh fetch and refit

---

## Running the Backtest Directly

```bash
python3 backtester.py
```

This prints all metrics and the full trade log to the terminal.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | BTC-USD data download |
| `hmmlearn` | GaussianHMM regime detection |
| `scikit-learn` | StandardScaler feature normalisation |
| `pandas` / `numpy` | Data processing |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard framework |

---

## Disclaimer

This project is for **research and educational purposes only**. It is not financial advice and should not be used for live trading without independent risk assessment. Past simulated performance does not guarantee future results.
