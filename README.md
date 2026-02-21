# HMM Regime-Based Trading System

Current-state documentation (live app behavior only).

## What this app does

- Detects market regime with HMM (Bull / Bear / Neutral)
- Uses a **2-bucket voting gate** for entries:
  - Trend bucket
  - Risk/Conditioning bucket
- Runs event-driven backtests with risk controls
- Supports calibration workflow with multi-seed robustness checks

---

## Run

```bash
cd HMM-Trading-System
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

---

## Current UI behavior

### Sidebar

- **Asset Selection**
- **Refresh Data & Refit Model** button
- **Mode selector**:
  - `Locked` (default): no manual strategy tweaking
  - `Research`: exposes a compact R&D control set

### Locked mode (default)

- Strategy and risk controls are fixed to defaults in sidebar
- Parameter tuning is intended to happen in **Calibration Lab**

### Research mode (compact controls)

- Entry style: `Trend` / `Mean Reversion` / `Hybrid`
- Core gate thresholds:
  - `p_bull_min`
  - `trend_min`
  - `risk_min`
- Mean reversion controls:
  - `mr_down_bars`
  - `mr_short_drop_pct`
  - `mr_bounce_rsi_max`
- Core risk controls:
  - Stop Loss
  - Take Profit
  - Kill-switch DD
  - Stress threshold

---

## Entry logic (current)

### Entry styles

- **Trend**: standard gate signal path
- **Mean Reversion**: pullback/bounce path
- **Hybrid**: allows either Trend or Mean Reversion qualification

### Mean reversion inputs

- Consecutive down bars (`mr_down_bars`)
- Pullback magnitude (`mr_short_drop_pct`)
- RSI cap (`mr_bounce_rsi_max`)

---

## Calibration Lab (current)

- Trials control
- Up to **3 seeds**, each selectable (run 1, 2, or 3 seeds)
- Progress bar + live run status
- "This can take several minutes" warning during runs

### Outputs

- Before/after key metrics
- Multi-seed summary table
- Seed agreement score + per-parameter agreement
- Executive commentary
- Parameter-change table with categories:
  - Signal
  - Risk

### Apply behavior

- **Apply Calibrated Parameters** applies:
  - tuned signal config
  - tuned risk config
- Applied settings are overlaid into active runtime

---

## Backtest/risk model (current)

- Next-open execution model
- Cost model includes fee + slippage
- Risk controls include:
  - stop loss / take profit
  - trailing stop
  - regime-flip grace
  - kill switch
  - stress filter / force-flat
  - volatility targeting

---

## Notes

- Streamlit cache warnings about TTL with `persist="disk"` are expected.
- If UI looks stale after changes, hard refresh the browser.
