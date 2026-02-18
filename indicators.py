"""
indicators.py
Computes all 8 technical confirmation signals used by the voting system.

──────────────────────────────────────────────
  CONFIGURATION — edit thresholds and periods here.
  Everything flows from CONFIG; no magic numbers elsewhere.
──────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — single source of truth for every threshold and lookback period.
#
#  Periods
#  -------
#  rsi_period          : RSI lookback (bars)
#  momentum_period     : number of bars for price momentum
#  volume_sma_period   : rolling window for volume baseline
#  volatility_period   : rolling window for annualised-vol calculation
#  adx_period          : ADX / DI smoothing period
#  ema_fast            : fast EMA period (EMA 50 by default)
#  ema_slow            : slow EMA period (EMA 200 by default)
#  macd_fast           : MACD fast EMA
#  macd_slow           : MACD slow EMA
#  macd_signal         : MACD signal EMA
#
#  Thresholds (entry conditions)
#  ----------
#  rsi_max             : RSI must be BELOW this value         (default 90)
#  momentum_min_pct    : momentum must be ABOVE this %        (default 1.0)
#  volatility_max_pct  : annualised vol must be BELOW this %  (default 6.0)
#  adx_min             : ADX must be ABOVE this value         (default 25)
#
#  Voting
#  ------
#  votes_required      : minimum passing votes (out of 8) to enable entry
# ══════════════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    # ── Periods ──────────────────────────────────────────────────────────────
    "rsi_period":         14,
    "momentum_period":    10,
    "volume_sma_period":  20,
    "volatility_period":  24,   # hours; annualised via ×√8760
    "adx_period":         14,
    "ema_fast":           50,
    "ema_slow":           200,
    "macd_fast":          12,
    "macd_slow":          26,
    "macd_signal":        9,

    # ── Thresholds ───────────────────────────────────────────────────────────
    "rsi_max":            90,    # RSI < rsi_max
    "momentum_min_pct":   1.0,   # momentum > momentum_min_pct %
    "volatility_max_pct": 6.0,   # annualised vol < volatility_max_pct %
    "adx_min":            25,    # ADX > adx_min

    # ── Voting ────────────────────────────────────────────────────────────────
    "votes_required":     7,     # out of 8 signals
}


# ──────────────────────────────────────────────
# Helper primitives — all respect CONFIG
# ──────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed ADX."""
    prev_close = close.shift(1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    plus_dm_s  = pd.Series(plus_dm,  index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    atr      = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di  = 100 * plus_dm_s.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return adx


# ──────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────

def compute_indicators(data: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Compute all indicators and boolean signals, driven entirely by *cfg*
    (defaults to the module-level CONFIG).

    Parameters
    ----------
    data : OHLCV DataFrame (must have Open, High, Low, Close, Volume)
    cfg  : optional override dict — pass only the keys you want to change,
           the rest fall back to CONFIG defaults.
           Example: compute_indicators(data, cfg={"rsi_max": 75, "adx_min": 20})

    Raw value columns added:
        rsi, momentum_pct, vol_{n}_sma, volatility_pct,
        adx, ema{fast}, ema{slow}, macd_line, macd_signal

    Boolean signal columns added:
        sig_rsi, sig_momentum, sig_volume, sig_volatility,
        sig_adx, sig_ema_fast, sig_ema_slow, sig_macd

    Summary columns added:
        vote_count  : integer 0-8 (number of passing signals)
        signal_ok   : True when vote_count >= cfg["votes_required"]
    """
    # Merge caller overrides into a working copy of CONFIG
    c = {**CONFIG, **(cfg or {})}

    df     = data.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Raw indicators ───────────────────────────────────────────────────────
    df["rsi"] = _rsi(close, c["rsi_period"])

    df["momentum_pct"] = (close / close.shift(c["momentum_period"]) - 1) * 100

    vol_sma_col = f"vol_{c['volume_sma_period']}_sma"
    df[vol_sma_col] = volume.rolling(c["volume_sma_period"]).mean()

    log_ret = np.log(close / close.shift(1))
    df["volatility_pct"] = log_ret.rolling(c["volatility_period"]).std() * np.sqrt(8760) * 100

    df["adx"] = _adx(high, low, close, c["adx_period"])

    ema_fast_col = f"ema{c['ema_fast']}"
    ema_slow_col = f"ema{c['ema_slow']}"
    df[ema_fast_col] = _ema(close, c["ema_fast"])
    df[ema_slow_col] = _ema(close, c["ema_slow"])

    df["macd_line"], df["macd_signal"] = _macd(
        close, c["macd_fast"], c["macd_slow"], c["macd_signal"]
    )

    # ── Boolean signals ──────────────────────────────────────────────────────
    df["sig_rsi"]        = df["rsi"] < c["rsi_max"]
    df["sig_momentum"]   = df["momentum_pct"] > c["momentum_min_pct"]
    df["sig_volume"]     = volume > df[vol_sma_col]
    df["sig_volatility"] = df["volatility_pct"] < c["volatility_max_pct"]
    df["sig_adx"]        = df["adx"] > c["adx_min"]
    df["sig_ema_fast"]   = close > df[ema_fast_col]
    df["sig_ema_slow"]   = close > df[ema_slow_col]
    df["sig_macd"]       = df["macd_line"] > df["macd_signal"]

    signal_cols = [
        "sig_rsi", "sig_momentum", "sig_volume", "sig_volatility",
        "sig_adx", "sig_ema_fast", "sig_ema_slow", "sig_macd",
    ]

    df["vote_count"] = df[signal_cols].sum(axis=1)
    df["signal_ok"]  = df["vote_count"] >= c["votes_required"]

    # Stash effective config so callers can inspect it
    df.attrs["indicator_config"] = c

    return df


def get_current_signals(df: pd.DataFrame) -> dict:
    """
    Return a snapshot dict of the most recent bar's signal values.

    Each entry: signal_label -> (passed: bool, current_value)
    The labels reflect the thresholds actually used (from df.attrs["indicator_config"]).
    """
    last = df.iloc[-1]
    c    = df.attrs.get("indicator_config", CONFIG)

    vol_sma_col  = f"vol_{c['volume_sma_period']}_sma"
    ema_fast_col = f"ema{c['ema_fast']}"
    ema_slow_col = f"ema{c['ema_slow']}"

    return {
        f"RSI < {c['rsi_max']}": (
            bool(last["sig_rsi"]),
            round(float(last["rsi"]), 2),
        ),
        f"Momentum > {c['momentum_min_pct']}%": (
            bool(last["sig_momentum"]),
            round(float(last["momentum_pct"]), 2),
        ),
        f"Volume > {c['volume_sma_period']}-SMA": (
            bool(last["sig_volume"]),
            round(float(last["Volume"]), 0),
        ),
        f"Volatility < {c['volatility_max_pct']}%": (
            bool(last["sig_volatility"]),
            round(float(last["volatility_pct"]), 2),
        ),
        f"ADX > {c['adx_min']}": (
            bool(last["sig_adx"]),
            round(float(last["adx"]), 2),
        ),
        f"Price > EMA {c['ema_fast']}": (
            bool(last["sig_ema_fast"]),
            round(float(last[ema_fast_col]), 2),
        ),
        f"Price > EMA {c['ema_slow']}": (
            bool(last["sig_ema_slow"]),
            round(float(last[ema_slow_col]), 2),
        ),
        "MACD > Signal": (
            bool(last["sig_macd"]),
            round(float(last["macd_line"]), 4),
        ),
    }
