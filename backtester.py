"""
backtester.py
Runs the regime-based HMM strategy simulation.

Rules
-----
Entry  : HMM regime == Bull AND vote_count >= 7
Exit   : HMM regime flips to Bear OR Crash (any non-Bull)
Cooldown: 48-hour hard lock after ANY exit
Leverage: 2.5× applied to PnL calculation
Capital : $10,000 starting

Returns
-------
result_df  : full data DataFrame with position/equity columns
trades_df  : log of every trade (entry time, exit time, PnL, …)
metrics    : dict of performance stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta

STARTING_CAPITAL = 10_000.0
LEVERAGE = 2.5
COOLDOWN_HOURS = 48

REGIME_BULL = "Bull"
REGIME_BEAR = "Bear"


def run_backtest(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parameters
    ----------
    data : DataFrame with columns including
           Close, regime, signal_ok  (from hmm_engine + indicators)

    Returns
    -------
    result_df, trades_df, metrics
    """
    df = data.copy()
    df = df.dropna(subset=["Close", "regime", "signal_ok"])

    n = len(df)
    equity = np.full(n, STARTING_CAPITAL, dtype=float)
    position = np.zeros(n, dtype=int)   # 1 = long, 0 = flat

    trades = []

    cash = STARTING_CAPITAL
    in_trade = False
    entry_price = 0.0
    entry_time = None
    exit_time = None          # tracks last exit for cooldown
    units = 0.0               # number of BTC units held (leveraged)

    for i in range(1, n):
        row = df.iloc[i]
        prev_equity = equity[i - 1]

        regime = row["regime"]
        signal_ok = bool(row["signal_ok"])
        price = float(row["Close"])
        ts = df.index[i]

        # ── Cooldown check ──────────────────────────────────────────────
        in_cooldown = False
        if exit_time is not None:
            hours_since_exit = (ts - exit_time).total_seconds() / 3600
            in_cooldown = hours_since_exit < COOLDOWN_HOURS

        # ── Exit logic ──────────────────────────────────────────────────
        if in_trade and regime != REGIME_BULL:
            # Regime flipped — exit immediately
            leveraged_pnl = (price - entry_price) / entry_price * cash * LEVERAGE
            exit_equity = cash + leveraged_pnl
            ret_pct = (price - entry_price) / entry_price * LEVERAGE * 100

            trades.append(
                {
                    "Entry Time": entry_time,
                    "Exit Time": ts,
                    "Entry Price": round(entry_price, 2),
                    "Exit Price": round(price, 2),
                    "Return (%)": round(ret_pct, 3),
                    "PnL ($)": round(leveraged_pnl, 2),
                    "Exit Reason": "Regime Flip",
                    "Equity After ($)": round(exit_equity, 2),
                }
            )

            cash = exit_equity
            in_trade = False
            exit_time = ts
            entry_price = 0.0
            units = 0.0

        # ── Entry logic ─────────────────────────────────────────────────
        if (
            not in_trade
            and not in_cooldown
            and regime == REGIME_BULL
            and signal_ok
        ):
            in_trade = True
            entry_price = price
            entry_time = ts
            # units are notional; PnL computed at exit on cash×leverage
            units = (cash * LEVERAGE) / price

        # ── Mark-to-market equity ────────────────────────────────────────
        if in_trade:
            unrealised = (price - entry_price) / entry_price * cash * LEVERAGE
            equity[i] = cash + unrealised
        else:
            equity[i] = cash

        position[i] = 1 if in_trade else 0

    # ── Close any open position at last bar ─────────────────────────────
    if in_trade:
        last_price = float(df["Close"].iloc[-1])
        leveraged_pnl = (last_price - entry_price) / entry_price * cash * LEVERAGE
        exit_equity = cash + leveraged_pnl
        ret_pct = (last_price - entry_price) / entry_price * LEVERAGE * 100
        trades.append(
            {
                "Entry Time": entry_time,
                "Exit Time": df.index[-1],
                "Entry Price": round(entry_price, 2),
                "Exit Price": round(last_price, 2),
                "Return (%)": round(ret_pct, 3),
                "PnL ($)": round(leveraged_pnl, 2),
                "Exit Reason": "End of Data",
                "Equity After ($)": round(exit_equity, 2),
            }
        )
        equity[-1] = exit_equity

    df["equity"] = equity
    df["position"] = position

    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(df, trades_df)

    return df, trades_df, metrics


# ──────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────

def _compute_metrics(df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    equity = df["equity"].values
    close = df["Close"].values

    total_return = (equity[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    # Buy-and-hold return
    bah_return = (close[-1] - close[0]) / close[0] * 100
    alpha = total_return - bah_return

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd = float(drawdown.min())

    # Win rate
    if len(trades_df) > 0:
        wins = (trades_df["Return (%)"] > 0).sum()
        win_rate = wins / len(trades_df) * 100
    else:
        win_rate = 0.0

    # Sharpe (hourly, annualised ×sqrt(8760))
    eq_series = pd.Series(equity)
    ret_series = eq_series.pct_change().dropna()
    sharpe = (
        ret_series.mean() / ret_series.std() * np.sqrt(8760)
        if ret_series.std() > 0
        else 0.0
    )

    return {
        "Total Return (%)": round(total_return, 2),
        "Buy & Hold Return (%)": round(bah_return, 2),
        "Alpha (%)": round(alpha, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Total Trades": len(trades_df),
        "Final Equity ($)": round(float(equity[-1]), 2),
        "Starting Capital ($)": STARTING_CAPITAL,
    }


if __name__ == "__main__":
    # Quick smoke-test
    from data_loader import fetch_btc_data
    from hmm_engine import fit_hmm, predict_regimes
    from indicators import compute_indicators

    print("Fetching data …")
    raw = fetch_btc_data()

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
