import numpy as np
import pandas as pd

from backtester import run_backtest


def _sample_df(n=240):
    idx = pd.date_range("2025-01-01", periods=n, freq="H", tz="UTC")
    base = 100 + np.cumsum(np.random.default_rng(7).normal(0, 0.25, n))
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])

    df = pd.DataFrame(index=idx)
    df["Open"] = open_.values
    df["Close"] = close.values
    df["regime"] = "Bull"
    df.loc[df.index[::11], "regime"] = "Neutral"
    df["signal_ok"] = True
    df.loc[df.index[::7], "signal_ok"] = False
    df["range_1h"] = 0.02
    df.loc[df.index[::19], "range_1h"] = 0.06
    df["volatility_pct"] = 55.0
    df["p_bull"] = 0.62
    df["rsi"] = 42.0
    return df


def test_indicator_gate_logic_parity_baseline():
    df = _sample_df()

    _, trades_old, metrics_old = run_backtest(
        df,
        use_indicator_gate_logic=False,
        use_market_quality_filter=True,
        stress_force_flat=True,
        stress_range_threshold=0.04,
        kill_switch_enabled=True,
        use_vol_targeting=True,
        vol_target_pct=30.0,
        vol_target_min_mult=0.25,
        vol_target_max_mult=1.0,
    )

    _, trades_new, metrics_new = run_backtest(
        df,
        use_indicator_gate_logic=True,
        use_market_quality_filter=True,
        stress_force_flat=True,
        stress_range_threshold=0.04,
        kill_switch_enabled=True,
        use_vol_targeting=True,
        vol_target_pct=30.0,
        vol_target_min_mult=0.25,
        vol_target_max_mult=1.0,
    )

    assert metrics_old["Total Trades"] == metrics_new["Total Trades"]
    assert metrics_old["Stop Loss Exits"] == metrics_new["Stop Loss Exits"]
    assert metrics_old["Take Profit Exits"] == metrics_new["Take Profit Exits"]
    assert len(trades_old) == len(trades_new)
