"""
backtester.py
Runs the regime-based HMM strategy simulation.

Entry Rules
-----------
  - HMM regime == Bull for at least MIN_REGIME_BARS consecutive bars
  - vote_count >= votes_required

Exit Rules (first condition that triggers wins)
-----------
  1. Stop Loss   : price drops STOP_LOSS_PCT from entry  (fastest, highest priority)
  2. Take Profit : price rises TAKE_PROFIT_PCT from entry
  3. Regime Flip : HMM regime flips to any non-Bull state

Cooldown : 48-hour hard lock after ANY exit
Leverage : 2.5× applied to PnL calculation
Capital  : $10,000 starting
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STARTING_CAPITAL  = 10_000.0
LEVERAGE          = 2.5
COOLDOWN_HOURS    = 48

# Risk management defaults (overridable via run_backtest kwargs)
STOP_LOSS_PCT     = -3.0   # % move from entry price that triggers stop  (negative)
TAKE_PROFIT_PCT   =  4.0   # % move from entry price that triggers TP    (positive)
MIN_REGIME_BARS   =  3     # consecutive Bull bars required before entry

REGIME_BULL = "Bull"


def run_backtest(
    data: pd.DataFrame,
    stop_loss_pct:   float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    min_regime_bars: int   = MIN_REGIME_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parameters
    ----------
    data            : DataFrame with Close, regime, signal_ok columns
    stop_loss_pct   : exit if price return from entry <= this value  (e.g. -3.0)
    take_profit_pct : exit if price return from entry >= this value  (e.g.  4.0)
    min_regime_bars : minimum consecutive Bull bars before entry allowed

    Returns
    -------
    result_df, trades_df, metrics
    """
    df = data.copy()
    df = df.dropna(subset=["Close", "regime", "signal_ok"])

    n = len(df)
    equity   = np.full(n, STARTING_CAPITAL, dtype=float)
    position = np.zeros(n, dtype=int)

    trades = []

    cash        = STARTING_CAPITAL
    in_trade    = False
    entry_price = 0.0
    entry_time  = None
    exit_time   = None
    bull_streak = 0        # consecutive Bull bars counter

    sl_factor = stop_loss_pct   / 100.0
    tp_factor = take_profit_pct / 100.0

    for i in range(1, n):
        row       = df.iloc[i]
        regime    = row["regime"]
        signal_ok = bool(row["signal_ok"])
        price     = float(row["Close"])
        ts        = df.index[i]

        # ── Track consecutive Bull bars ──────────────────────────────────
        if regime == REGIME_BULL:
            bull_streak += 1
        else:
            bull_streak = 0

        # ── Cooldown ─────────────────────────────────────────────────────
        in_cooldown = False
        if exit_time is not None:
            hours_since_exit = (ts - exit_time).total_seconds() / 3600
            in_cooldown = hours_since_exit < COOLDOWN_HOURS

        # ── Exit logic ───────────────────────────────────────────────────
        if in_trade:
            price_ret = (price - entry_price) / entry_price

            if price_ret <= sl_factor:
                exit_reason = "Stop Loss"
            elif price_ret >= tp_factor:
                exit_reason = "Take Profit"
            elif regime != REGIME_BULL:
                exit_reason = "Regime Flip"
            else:
                exit_reason = None

            if exit_reason:
                leveraged_pnl = price_ret * cash * LEVERAGE
                exit_equity   = cash + leveraged_pnl
                ret_pct       = price_ret * LEVERAGE * 100

                trades.append({
                    "Entry Time":       entry_time,
                    "Exit Time":        ts,
                    "Entry Price":      round(entry_price, 2),
                    "Exit Price":       round(price, 2),
                    "Return (%)":       round(ret_pct, 3),
                    "PnL ($)":          round(leveraged_pnl, 2),
                    "Exit Reason":      exit_reason,
                    "Equity After ($)": round(exit_equity, 2),
                })

                cash        = exit_equity
                in_trade    = False
                exit_time   = ts
                entry_price = 0.0

        # ── Entry logic ──────────────────────────────────────────────────
        if (
            not in_trade
            and not in_cooldown
            and regime == REGIME_BULL
            and bull_streak >= min_regime_bars
            and signal_ok
        ):
            in_trade    = True
            entry_price = price
            entry_time  = ts

        # ── Mark-to-market equity ─────────────────────────────────────────
        if in_trade:
            unrealised = (price - entry_price) / entry_price * cash * LEVERAGE
            equity[i]  = cash + unrealised
        else:
            equity[i] = cash

        position[i] = 1 if in_trade else 0

    # ── Close any open position at last bar ──────────────────────────────
    if in_trade:
        last_price    = float(df["Close"].iloc[-1])
        price_ret     = (last_price - entry_price) / entry_price
        leveraged_pnl = price_ret * cash * LEVERAGE
        exit_equity   = cash + leveraged_pnl
        trades.append({
            "Entry Time":       entry_time,
            "Exit Time":        df.index[-1],
            "Entry Price":      round(entry_price, 2),
            "Exit Price":       round(last_price, 2),
            "Return (%)":       round(price_ret * LEVERAGE * 100, 3),
            "PnL ($)":          round(leveraged_pnl, 2),
            "Exit Reason":      "End of Data",
            "Equity After ($)": round(exit_equity, 2),
        })
        equity[-1] = exit_equity

    df["equity"]   = equity
    df["position"] = position

    trades_df = pd.DataFrame(trades)
    metrics   = _compute_metrics(df, trades_df, stop_loss_pct, take_profit_pct, min_regime_bars)

    return df, trades_df, metrics


# ──────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────

def _compute_metrics(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    stop_loss_pct: float,
    take_profit_pct: float,
    min_regime_bars: int,
) -> dict:
    equity = df["equity"].values
    close  = df["Close"].values

    total_return = (equity[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    bah_return   = (close[-1] - close[0]) / close[0] * 100
    alpha        = total_return - bah_return

    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd   = float(drawdown.min())

    if len(trades_df) > 0:
        wins       = (trades_df["Return (%)"] > 0).sum()
        win_rate   = wins / len(trades_df) * 100
        sl_exits   = (trades_df["Exit Reason"] == "Stop Loss").sum()
        tp_exits   = (trades_df["Exit Reason"] == "Take Profit").sum()
        reg_exits  = (trades_df["Exit Reason"] == "Regime Flip").sum()
    else:
        win_rate = sl_exits = tp_exits = reg_exits = 0

    eq_series  = pd.Series(equity)
    ret_series = eq_series.pct_change().dropna()
    sharpe = (
        ret_series.mean() / ret_series.std() * np.sqrt(8760)
        if ret_series.std() > 0 else 0.0
    )

    return {
        "Total Return (%)":       round(total_return, 2),
        "Buy & Hold Return (%)":  round(bah_return, 2),
        "Alpha (%)":              round(alpha, 2),
        "Win Rate (%)":           round(win_rate, 2),
        "Max Drawdown (%)":       round(max_dd, 2),
        "Sharpe Ratio":           round(sharpe, 3),
        "Total Trades":           len(trades_df),
        "Stop Loss Exits":        int(sl_exits),
        "Take Profit Exits":      int(tp_exits),
        "Regime Flip Exits":      int(reg_exits),
        "Final Equity ($)":       round(float(equity[-1]), 2),
        "Starting Capital ($)":   STARTING_CAPITAL,
        # Active settings (for display)
        "Stop Loss (%)":          stop_loss_pct,
        "Take Profit (%)":        take_profit_pct,
        "Min Regime Bars":        min_regime_bars,
    }


if __name__ == "__main__":
    from data_loader import fetch_asset_data
    from hmm_engine import fit_hmm, predict_regimes
    from indicators import compute_indicators

    print("Fetching data …")
    raw = fetch_asset_data("BTC-USD")

    print("Fitting HMM …")
    model, scaler, feat_df, state_map, bull_state, bear_state = fit_hmm(raw)
    with_regimes = predict_regimes(model, scaler, feat_df, state_map, raw)

    print("Computing indicators …")
    full = compute_indicators(with_regimes)

    print("Running backtest …")
    result, trades, metrics = run_backtest(full)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if not trades.empty:
        print(f"\n=== Trades ({len(trades)}) ===")
        print(trades.to_string(index=False))
