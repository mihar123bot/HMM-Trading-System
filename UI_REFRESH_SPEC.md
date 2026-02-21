# HMM Trading App — UI Refresh Spec (v1)

## Goal
Make the app feel like a **decision cockpit**, not a research notebook.

Primary user question:
> "Can I trade right now, and if yes, how?"

---

## Design Principles
1. **Decision-first:** surface action and risk before raw analytics.
2. **Progressive disclosure:** keep advanced controls hidden by default.
3. **Consistency:** same metric cards and naming across pages.
4. **Fast scan:** user should understand state in <5 seconds.
5. **Honest state:** always show data freshness and quality.

---

## Information Architecture
Use 4 top-level tabs only:

1. **Live**
2. **Backtest**
3. **Walk-Forward**
4. **Settings**

Remove/merge any extra tabs that duplicate these functions.

---

## Global Layout (applies to all tabs)

### Header Bar (fixed)
- App title + symbol/timeframe selector
- Data freshness: `Updated Xs ago`
- Data health badge: `Healthy / Degraded / Stale`
- Run mode badge: `Paper / Live` (if relevant)

### Left Sidebar (compact)
- Symbol
- Timeframe
- Date range (for backtest/WF)
- `Run` button (context-aware)
- Advanced settings collapsed by default

### Metric Card Style
All metrics rendered as identical cards:
- Label
- Value
- Delta vs baseline (if available)
- Tooltip with formula

---

## Tab 1 — Live (most important)

### Section A: Trade Decision Card (top, full width)
Show in this order:
1. **Current Regime** (Trend / Range / Volatile)
2. **Regime Confidence** (e.g., 78%, plus High/Med/Low badge)
3. **Trade Gate** (Allowed / Blocked)
4. **Recommended Action** (Long / Short / Flat)
5. **Position Size Multiplier** (e.g., 0.6x)
6. **Risk Context** (Stop distance, max loss/trade)

This card is the app’s north star.

### Section B: Price + Regime Timeline
- Main chart: price candles + entries/exits
- Underlay strip: colored regime bands over time
- Optional moving average overlays (toggle)

### Section C: Why This Signal? (explainability box)
- Regime probability vector (3–4 states)
- Top 3 feature contributors
- Gate reason text:
  - e.g., `Blocked: confidence < threshold` or `Allowed: trend regime + momentum confirmation`

### Section D: Current Risk Panel
- Exposure
- Estimated slippage/spread (if available)
- Max drawdown guard status
- Cooldown status

---

## Tab 2 — Backtest

### Section A: Run Controls (minimal)
- Symbol, timeframe, date range
- Strategy preset selector
- `Run Baseline` primary CTA
- `Advanced` accordion for detailed parameters

### Section B: Core Performance Scorecard (fixed set)
- CAGR
- Sharpe
- Max Drawdown
- Win Rate
- Profit Factor
- Trades

### Section C: Curves
- Equity curve
- Drawdown curve

### Section D: Trade Diagnostics
- PnL by regime
- PnL distribution histogram
- Trade duration distribution

### Section E: Run Artifacts
- Config hash
- Run timestamp
- Save snapshot button (`Save as baseline`)
- Compare mode toggle

---

## Tab 3 — Walk-Forward

### Section A: WF Summary
- Number of folds
- Median Sharpe
- Median Max DD
- Stability score (custom composite)

### Section B: Fold Table
Columns:
- Fold ID
- Train window
- Test window
- Return
- Sharpe
- Max DD
- Pass/Fail flag

### Section C: Stability Visuals
- Sharpe per fold line chart
- Return dispersion boxplot
- Regime consistency heatmap

### Section D: Decision Banner
- `Deployable` if stability above threshold
- `Not deployable` with exact failing criteria

---

## Tab 4 — Settings

### Section A: Presets
- Conservative
- Balanced
- Aggressive

### Section B: HMM Core
- Number of states
- Covariance type
- Refit frequency

### Section C: Feature Set
- Feature pack checkboxes
- Standardization toggle
- Missing value handling

### Section D: Execution & Risk
- Entry thresholds
- Exit logic mode
- Position sizing mode
- Max daily loss / trade cap

### Section E: Save/Load
- Export config JSON
- Import config JSON
- Reset to defaults

---

## UX Details

### Loading States
- Skeleton cards while computing
- Progress bar for backtest/WF
- Explicit status text (`Loading data`, `Fitting HMM`, `Running simulation`)

### Empty/Error States
- No data: actionable suggestion + retry
- API failure: show source + fallback used
- Invalid params: inline field-level error

### Terminology cleanup
Use one term consistently:
- `Regime` (not state/mode alternately)
- `Trade Gate`
- `Confidence`
- `Baseline`

---

## Component Map (Streamlit)

Suggested reusable components:
- `render_header_bar(context)`
- `render_trade_decision_card(signal_state)`
- `render_metric_scorecard(metrics)`
- `render_regime_timeline(df_price, df_regime)`
- `render_signal_explainability(explain_obj)`
- `render_backtest_summary(results)`
- `render_wf_fold_table(wf_results)`
- `render_data_health_badge(health)`

If keeping `ui_components.py`, migrate toward this component contract.

---

## Implementation Plan (small, fast iterations)

### Phase 1 (1–2 sessions)
- Add new nav structure (4 tabs)
- Implement header bar + data health
- Implement Trade Decision Card on Live tab

### Phase 2
- Standardize scorecards and metric naming
- Rework Backtest page with baseline/save/compare controls

### Phase 3
- Rework Walk-Forward stability page and deployability banner
- Add explainability panel + gate reasons

### Phase 4
- Settings cleanup (presets + advanced accordion)
- Final polish (loading/error states, wording)

---

## Acceptance Criteria
1. User can answer “trade or not” in <5s on Live tab.
2. Backtest and WF both expose fixed, comparable metrics.
3. Advanced knobs are hidden by default.
4. Every run can be traced with config hash and timestamp.
5. Data quality/freshness is visible at all times.

---

## Nice-to-Have (later)
- Dark/light theme toggle
- Keyboard quick actions (run/reset/save)
- Mobile-friendly compact mode
- Regime change alerts
