"""
indicators.py
Computes all technical confirmation signals used by the bucket voting system.

──────────────────────────────────────────────
  CONFIGURATION — edit thresholds and periods here.
  Everything flows from CONFIG; no magic numbers elsewhere.
──────────────────────────────────────────────

Bucket Voting System (replaces flat 7-of-8 gate)
-------------------------------------------------
Signals are grouped into 4 buckets. Each bucket has a minimum passing score.
All four bucket minimums must be met to enable entry.

  Trend         (3 signals): EMA Fast, EMA Slow, MACD          → need ≥ trend_min
  Strength      (1 signal) : ADX                               → need ≥ strength_min
  Participation (1 signal) : Volume > SMA                      → need ≥ participation_min
  Risk/Cond.    (3+1 sigs) : RSI, Volatility, Momentum,        → need ≥ risk_min
                             [+ HMM Confidence if p_bull avail]
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
#  atr_period          : ATR smoothing period (Wilder)
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
#  p_bull_min          : HMM Bull confidence must be ABOVE this (default 0.55)
#                        Only active when p_bull column exists in data.
#
#  Bucket Voting
#  -------------
#  trend_min           : min Trend signals   (0-3, default 2)
#  strength_min        : min Strength signals (0-1, default 1)
#  participation_min   : min Participation signals (0-1, default 1)
#  risk_min            : min Risk signals    (0-3 or 0-4, default 2)
# ══════════════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    # ── Periods ──────────────────────────────────────────────────────────────
    "rsi_period":         14,
    "momentum_period":    10,
    "volume_sma_period":  20,
    "volatility_period":  24,   # hours; annualised via ×√8760
    "adx_period":         14,
    "atr_period":         14,   # Wilder ATR period
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
    "p_bull_min":         0.55,  # HMM Bull confidence >= p_bull_min

    # ── Bucket Voting ─────────────────────────────────────────────────────────
    "trend_min":          2,     # of 3 (ema_fast, ema_slow, macd)
    "strength_min":       1,     # of 1 (adx)
    "participation_min":  1,     # of 1 (volume)
    "risk_min":           2,     # of 3–4 (rsi, volatility, momentum [, confidence])
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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ──────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────

def compute_indicators(data: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Compute all indicators and bucket-based boolean signals, driven entirely
    by *cfg* (defaults to the module-level CONFIG).

    Parameters
    ----------
    data : OHLCV DataFrame with regime/hmm columns (must have Open, High, Low, Close, Volume)
    cfg  : optional override dict — pass only the keys you want to change,
           the rest fall back to CONFIG defaults.

    Raw value columns added:
        rsi, momentum_pct, vol_{n}_sma, volatility_pct,
        adx, atr, ema{fast}, ema{slow}, macd_line, macd_signal

    Boolean signal columns added:
        sig_rsi, sig_momentum, sig_volume, sig_volatility,
        sig_adx, sig_ema_fast, sig_ema_slow, sig_macd, sig_confidence

    Bucket score columns added:
        trend_score         : 0–3 (ema_fast + ema_slow + macd)
        strength_score      : 0–1 (adx)
        participation_score : 0–1 (volume)
        risk_score          : 0–3 or 0–4 (rsi + volatility + momentum [+ confidence])

    Summary columns added:
        vote_count  : total of all bucket scores
        signal_ok   : True when all bucket minimums are met
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
    df["atr"] = _atr(high, low, close, c["atr_period"])

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

    # HMM confidence signal (uses p_bull column if available from hmm_engine)
    if "p_bull" in df.columns:
        df["sig_confidence"] = df["p_bull"] >= c["p_bull_min"]
    else:
        df["sig_confidence"] = True  # graceful fallback when p_bull not present

    # ── Bucket scores ─────────────────────────────────────────────────────────
    df["trend_score"]         = df[["sig_ema_fast", "sig_ema_slow", "sig_macd"]].sum(axis=1)
    df["strength_score"]      = df[["sig_adx"]].sum(axis=1)
    df["participation_score"] = df[["sig_volume"]].sum(axis=1)

    # Risk bucket includes confidence signal (4 signals total when p_bull present)
    risk_signals = ["sig_rsi", "sig_volatility", "sig_momentum", "sig_confidence"]
    df["risk_score"] = df[risk_signals].sum(axis=1)

    # ── Bucket gate ───────────────────────────────────────────────────────────
    df["signal_ok"] = (
        (df["trend_score"]         >= c["trend_min"])
        & (df["strength_score"]      >= c["strength_min"])
        & (df["participation_score"] >= c["participation_min"])
        & (df["risk_score"]          >= c["risk_min"])
    )

    # vote_count = total signals passing across all buckets (for display)
    all_signal_cols = [
        "sig_rsi", "sig_momentum", "sig_volume", "sig_volatility",
        "sig_adx", "sig_ema_fast", "sig_ema_slow", "sig_macd", "sig_confidence",
    ]
    df["vote_count"] = df[all_signal_cols].sum(axis=1)

    # Stash effective config so callers can inspect it
    df.attrs["indicator_config"] = c

    return df


def get_current_signals(df: pd.DataFrame) -> dict:
    """
    Return a snapshot dict of the most recent bar's signal values.

    Returns two sections:
      "signals"  : individual signal label -> (passed: bool, current_value)
      "buckets"  : bucket name -> (score: int, required: int, passed: bool)
      "p_bull"   : float (HMM Bull confidence, or None if unavailable)

    The labels reflect the thresholds actually used (from df.attrs["indicator_config"]).
    """
    last = df.iloc[-1]
    c    = df.attrs.get("indicator_config", CONFIG)

    vol_sma_col  = f"vol_{c['volume_sma_period']}_sma"
    ema_fast_col = f"ema{c['ema_fast']}"
    ema_slow_col = f"ema{c['ema_slow']}"

    has_confidence = "p_bull" in df.columns

    signals = {
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

    if has_confidence:
        signals[f"HMM Confidence ≥ {c['p_bull_min']}"] = (
            bool(last["sig_confidence"]),
            round(float(last["p_bull"]), 3),
        )

    risk_max = 4 if has_confidence else 3
    buckets = {
        "Trend":         (int(last["trend_score"]),         3,        c["trend_min"]),
        "Strength":      (int(last["strength_score"]),      1,        c["strength_min"]),
        "Participation": (int(last["participation_score"]), 1,        c["participation_min"]),
        "Risk/Cond.":    (int(last["risk_score"]),          risk_max, c["risk_min"]),
    }

    p_bull_val = round(float(last["p_bull"]), 3) if has_confidence else None

    return {
        "signals": signals,
        "buckets": buckets,
        "p_bull":  p_bull_val,
    }
