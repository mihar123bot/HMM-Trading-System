"""
indicators.py
Computes all technical confirmation signals used by the bucket voting system.

──────────────────────────────────────────────
  CONFIGURATION — edit thresholds and periods here.
  Everything flows from CONFIG; no magic numbers elsewhere.
──────────────────────────────────────────────

Bucket Voting System
--------------------
Signals are grouped into 4 buckets. Each bucket has a minimum passing score.
All four bucket minimums must be met to enable entry.

  Trend         (3 signals): EMA Fast, EMA Slow, MACD          → need ≥ trend_min
  Strength      (1 signal) : ADX                               → need ≥ strength_min
  Participation (1 signal) : Volume > SMA                      → need ≥ participation_min
  Risk/Cond.    (3+1 sigs) : RSI, Volatility, Momentum,        → need ≥ risk_min
                             [+ HMM Confidence if p_bull avail]

Phase 2 — per-signal enable/disable:
  Each signal has a corresponding `sig_*_on` config key (default True).
  When a signal is disabled, it is excluded from its bucket's total possible
  score; the bucket minimum is capped at the number of enabled signals.
  If all signals in a bucket are disabled, that bucket auto-passes.

Phase 3 — stress / risk gates / vol targeting:
  - range_1h = (High-Low)/Close is computed per bar
  - stress_spike flag emitted when range_1h >= stress_range_threshold
  - apply_btc_risk_gates() evaluates row-level gate decisions
  - compute_vol_size_multiplier() returns vol-targeting position size
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — single source of truth for every threshold and lookback period.
# ══════════════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    # ── Periods ──────────────────────────────────────────────────────────────
    "rsi_period":           14,
    "momentum_period":       5,
    "volume_sma_period":    10,
    "volatility_period":    24,   # hours; annualised via ×√8760
    "adx_period":           14,
    "atr_period":           14,   # Wilder ATR period
    "ema_fast":             20,
    "ema_slow":             50,
    "macd_fast":            12,
    "macd_slow":            26,
    "macd_signal":           9,
    "roc_period":            8,   # rate-of-change lookback (bars) — faster than EMA
    "p_bull_slope_period":   3,   # p_bull slope lookback (bars) — HMM confidence trend

    # ── Thresholds (entry conditions) ────────────────────────────────────────
    "rsi_max":            90,    # RSI < rsi_max
    "momentum_min_pct":   1.0,   # momentum > momentum_min_pct %
    # NOTE: BTC annualised hourly vol is typically 40-100%.
    # The previous default of 6% was too restrictive and nearly always blocked entry.
    "volatility_max_pct": 80.0,  # annualised vol < volatility_max_pct %
    "adx_min":            25,    # ADX > adx_min
    "p_bull_min":         0.55,  # HMM Bull confidence >= p_bull_min

    # ── Per-signal enable/disable (Phase 2) ───────────────────────────────────
    # Set to False to exclude a signal from its bucket (and reduce bucket max by 1).
    "sig_rsi_on":           True,
    "sig_momentum_on":      True,
    "sig_volume_on":        False,
    "sig_volatility_on":    True,
    "sig_adx_on":           False,
    "sig_ema_fast_on":      True,
    "sig_ema_slow_on":      True,
    "sig_macd_on":          True,
    "sig_confidence_on":    True,
    "sig_roc_on":           True,   # rate-of-change > 0 over roc_period bars
    "sig_pbull_slope_on":   True,   # p_bull increasing over p_bull_slope_period bars

    # ── Bucket voting ─────────────────────────────────────────────────────────
    "trend_min":          2,     # of enabled trend signals
    "strength_min":       0,     # disabled by default
    "participation_min":  0,     # disabled by default
    "risk_min":           2,     # of enabled risk signals

    # ── Stress / market quality gates (Phase 3 G, K) ─────────────────────────
    # range_1h = (High-Low)/Close; a spike bar is one where range_1h is large.
    "stress_range_threshold": 0.04,   # (H-L)/C > 4% → stress spike
    "stress_cooldown_hours":  24,     # hours to block entries after a stress spike
    "stress_force_flat":      True,   # True → force-flat open position on stress spike
    "market_quality_filter":  True,   # True → block entries on stress spikes

    # ── Kill switch (Phase 3 G) ───────────────────────────────────────────────
    "kill_switch_enabled":    True,
    "kill_switch_dd_pct":     9.0,    # rolling drawdown % from HWM → trigger
    "kill_switch_cooldown_h": 24,     # hours disabled after trigger

    # ── Volatility-targeted position sizing (Phase 3 J) ───────────────────────
    "vol_targeting_enabled":  False,
    "vol_target_pct":         30.0,   # target annualised vol %
    "vol_target_min_mult":    0.25,   # minimum size multiplier
    "vol_target_max_mult":    1.0,    # maximum size multiplier
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
# Main indicators function
# ──────────────────────────────────────────────

def compute_indicators(data: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Compute all indicators and bucket-based boolean signals, driven entirely
    by *cfg* (defaults to the module-level CONFIG).

    Phase 2 change: per-signal enable/disable (sig_*_on config keys).
    When a signal is disabled it is excluded from its bucket's possible max.
    A bucket with 0 enabled signals auto-passes (never blocks entry).

    Phase 3 additions: range_1h, stress_spike columns.

    Parameters
    ----------
    data : OHLCV DataFrame with regime/hmm columns
    cfg  : optional override dict — only changed keys need be passed.

    Raw value columns added:
        rsi, momentum_pct, vol_{n}_sma, volatility_pct,
        adx, atr, ema{fast}, ema{slow}, macd_line, macd_signal,
        range_1h

    Boolean signal columns added:
        sig_rsi, sig_momentum, sig_volume, sig_volatility,
        sig_adx, sig_ema_fast, sig_ema_slow, sig_macd, sig_confidence,
        stress_spike

    Bucket score columns:
        trend_score, strength_score, participation_score, risk_score

    Summary columns:
        vote_count  : total of all active signal booleans
        signal_ok   : True when all bucket minimums are met
    """
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

    # Phase 3: range_1h = (High-Low)/Close (intrabar range ratio)
    df["range_1h"] = (high - low) / close.replace(0, np.nan)

    # Fast momentum: rate of change over roc_period bars
    df["roc_pct"] = (close / close.shift(c["roc_period"]) - 1) * 100

    # HMM confidence slope: p_bull increasing over p_bull_slope_period bars
    if "p_bull" in df.columns:
        df["p_bull_slope"] = df["p_bull"] - df["p_bull"].shift(c["p_bull_slope_period"])
    else:
        df["p_bull_slope"] = 0.0

    # ── Boolean signals ──────────────────────────────────────────────────────
    df["sig_rsi"]        = df["rsi"] < c["rsi_max"]
    df["sig_momentum"]   = df["momentum_pct"] > c["momentum_min_pct"]
    df["sig_volume"]     = volume > df[vol_sma_col]
    df["sig_volatility"] = df["volatility_pct"] < c["volatility_max_pct"]
    df["sig_adx"]        = df["adx"] > c["adx_min"]
    df["sig_ema_fast"]   = close > df[ema_fast_col]
    df["sig_ema_slow"]   = close > df[ema_slow_col]
    df["sig_macd"]       = df["macd_line"] > df["macd_signal"]

    # HMM confidence signal (uses p_bull column if available)
    if "p_bull" in df.columns:
        df["sig_confidence"] = df["p_bull"] >= c["p_bull_min"]
    else:
        df["sig_confidence"] = True

    # Rate-of-change signal: positive ROC over roc_period bars
    df["sig_roc"] = df["roc_pct"] > 0

    # p_bull slope signal: HMM confidence is increasing
    df["sig_p_bull_slope"] = df["p_bull_slope"] > 0

    # Phase 3: stress spike flag
    df["stress_spike"] = df["range_1h"] >= c.get("stress_range_threshold", 0.03)

    # ── Phase 2: Dynamic bucket scores respecting enable/disable flags ────────

    # Trend bucket — includes fast alternatives (ROC, p_bull slope) alongside EMA/MACD
    trend_signal_map = [
        ("sig_ema_fast",     "sig_ema_fast_on"),
        ("sig_ema_slow",     "sig_ema_slow_on"),
        ("sig_macd",         "sig_macd_on"),
        ("sig_roc",          "sig_roc_on"),
        ("sig_p_bull_slope", "sig_pbull_slope_on"),
    ]
    trend_active = [col for col, key in trend_signal_map if c.get(key, True)]
    trend_max    = len(trend_active)
    df["trend_score"] = df[trend_active].sum(axis=1) if trend_active else pd.Series(0, index=df.index)

    # Strength bucket
    strength_active = ["sig_adx"] if c.get("sig_adx_on", True) else []
    strength_max    = len(strength_active)
    df["strength_score"] = df[strength_active].sum(axis=1) if strength_active else pd.Series(0, index=df.index)

    # Participation bucket
    participation_active = ["sig_volume"] if c.get("sig_volume_on", True) else []
    participation_max    = len(participation_active)
    df["participation_score"] = df[participation_active].sum(axis=1) if participation_active else pd.Series(0, index=df.index)

    # Risk/Conditioning bucket
    risk_signal_map = [
        ("sig_rsi",        "sig_rsi_on"),
        ("sig_volatility", "sig_volatility_on"),
        ("sig_momentum",   "sig_momentum_on"),
    ]
    risk_active = [col for col, key in risk_signal_map if c.get(key, True)]
    if "p_bull" in df.columns and c.get("sig_confidence_on", True):
        risk_active.append("sig_confidence")
    risk_max = len(risk_active)
    df["risk_score"] = df[risk_active].sum(axis=1) if risk_active else pd.Series(0, index=df.index)

    # ── Bucket gate ───────────────────────────────────────────────────────────
    # If a bucket has 0 active signals → it auto-passes (| max=0 condition).
    # Effective min is capped at the number of active signals.
    def _bucket_pass(score_col, required, max_sigs):
        if max_sigs == 0:
            return pd.Series(True, index=df.index)
        return df[score_col] >= min(required, max_sigs)

    df["signal_ok"] = (
        _bucket_pass("trend_score",         c["trend_min"],         trend_max)
        & _bucket_pass("strength_score",    c["strength_min"],      strength_max)
        & _bucket_pass("participation_score", c["participation_min"], participation_max)
        & _bucket_pass("risk_score",        c["risk_min"],          risk_max)
    )

    # vote_count = total of all currently-active signal booleans
    all_active_signals = (
        trend_active + strength_active + participation_active
        + [s for s in risk_active if s != "sig_confidence"]
        + (["sig_confidence"] if "sig_confidence" in risk_active else [])
    )
    all_active_signals = list(dict.fromkeys(all_active_signals))  # dedup, preserve order
    df["vote_count"] = df[all_active_signals].sum(axis=1) if all_active_signals else 0

    # Stash effective config and bucket structure for callers (e.g. get_current_signals)
    df.attrs["indicator_config"] = c
    df.attrs["bucket_maxes"] = {
        "Trend":         trend_max,
        "Strength":      strength_max,
        "Participation": participation_max,
        "Risk/Cond.":    risk_max,
    }
    df.attrs["bucket_active_signals"] = {
        "Trend":         trend_active,
        "Strength":      strength_active,
        "Participation": participation_active,
        "Risk/Cond.":    risk_active,
    }

    return df


def get_current_signals(df: pd.DataFrame) -> dict:
    """
    Return a snapshot dict of the most recent bar's signal values.

    Returns:
      "signals"  : label -> (passed: bool, current_value)
      "buckets"  : bucket name -> (score: int, max: int, required: int)
      "p_bull"   : float | None
      "pass_rates": signal label -> % of all bars where signal = True
    """
    last = df.iloc[-1]
    c    = df.attrs.get("indicator_config", CONFIG)
    bucket_maxes = df.attrs.get("bucket_maxes", {
        "Trend": 3, "Strength": 1, "Participation": 1, "Risk/Cond.": 4
    })

    vol_sma_col  = f"vol_{c['volume_sma_period']}_sma"
    ema_fast_col = f"ema{c['ema_fast']}"
    ema_slow_col = f"ema{c['ema_slow']}"

    has_confidence = "p_bull" in df.columns

    # Build signal entries — only include signals that are enabled
    signals = {}
    if c.get("sig_rsi_on", True):
        signals[f"RSI < {c['rsi_max']}"] = (bool(last["sig_rsi"]), round(float(last["rsi"]), 2))
    if c.get("sig_momentum_on", True):
        signals[f"Momentum > {c['momentum_min_pct']}%"] = (bool(last["sig_momentum"]), round(float(last["momentum_pct"]), 2))
    if c.get("sig_volume_on", True):
        signals[f"Volume > {c['volume_sma_period']}-SMA"] = (bool(last["sig_volume"]), round(float(last["Volume"]), 0))
    if c.get("sig_volatility_on", True):
        signals[f"Volatility < {c['volatility_max_pct']}%"] = (bool(last["sig_volatility"]), round(float(last["volatility_pct"]), 2))
    if c.get("sig_adx_on", True):
        signals[f"ADX > {c['adx_min']}"] = (bool(last["sig_adx"]), round(float(last["adx"]), 2))
    if c.get("sig_ema_fast_on", True):
        signals[f"Price > EMA {c['ema_fast']}"] = (bool(last["sig_ema_fast"]), round(float(last[ema_fast_col]), 2))
    if c.get("sig_ema_slow_on", True):
        signals[f"Price > EMA {c['ema_slow']}"] = (bool(last["sig_ema_slow"]), round(float(last[ema_slow_col]), 2))
    if c.get("sig_macd_on", True):
        signals["MACD > Signal"] = (bool(last["sig_macd"]), round(float(last["macd_line"]), 4))
    if has_confidence and c.get("sig_confidence_on", True):
        signals[f"HMM Confidence ≥ {c['p_bull_min']}"] = (bool(last["sig_confidence"]), round(float(last["p_bull"]), 3))
    if c.get("sig_roc_on", True):
        signals[f"ROC({c['roc_period']}b) > 0"] = (bool(last["sig_roc"]), round(float(last["roc_pct"]), 3))
    if c.get("sig_pbull_slope_on", True):
        slope_val = round(float(last["p_bull_slope"]), 4) if "p_bull_slope" in last.index else 0.0
        signals[f"p_bull Slope({c['p_bull_slope_period']}b) > 0"] = (bool(last["sig_p_bull_slope"]), slope_val)

    buckets = {
        "Trend":         (int(last["trend_score"]),         bucket_maxes.get("Trend", 3),         c["trend_min"]),
        "Strength":      (int(last["strength_score"]),      bucket_maxes.get("Strength", 1),      c["strength_min"]),
        "Participation": (int(last["participation_score"]), bucket_maxes.get("Participation", 1), c["participation_min"]),
        "Risk/Cond.":    (int(last["risk_score"]),          bucket_maxes.get("Risk/Cond.", 4),    c["risk_min"]),
    }

    p_bull_val = round(float(last["p_bull"]), 3) if has_confidence else None

    # Historical pass rates for each signal column
    sig_cols = {
        f"RSI < {c['rsi_max']}":                 "sig_rsi",
        f"Momentum > {c['momentum_min_pct']}%":  "sig_momentum",
        f"Volume > {c['volume_sma_period']}-SMA": "sig_volume",
        f"Volatility < {c['volatility_max_pct']}%": "sig_volatility",
        f"ADX > {c['adx_min']}":                 "sig_adx",
        f"Price > EMA {c['ema_fast']}":          "sig_ema_fast",
        f"Price > EMA {c['ema_slow']}":          "sig_ema_slow",
        "MACD > Signal":                          "sig_macd",
        f"ROC({c['roc_period']}b) > 0":          "sig_roc",
        f"p_bull Slope({c['p_bull_slope_period']}b) > 0": "sig_p_bull_slope",
    }
    if has_confidence:
        sig_cols[f"HMM Confidence ≥ {c['p_bull_min']}"] = "sig_confidence"

    n = max(len(df), 1)
    pass_rates = {}
    for label, col in sig_cols.items():
        if col in df.columns:
            pass_rates[label] = round(float(df[col].sum() / n * 100), 1)

    return {
        "signals":    signals,
        "buckets":    buckets,
        "p_bull":     p_bull_val,
        "pass_rates": pass_rates,
    }


# ──────────────────────────────────────────────
# Phase 3: Risk gate evaluation (row-level)
# ──────────────────────────────────────────────

def apply_btc_risk_gates(row: pd.Series, config: dict, state: dict) -> dict:
    """
    Evaluate row-level risk gates and return a GateDecision dict.

    Parameters
    ----------
    row    : single DataFrame row (pd.Series) with indicator columns
    config : effective CONFIG dict
    state  : mutable strategy state dict with keys:
               kill_switch_active (bool)
               kill_switch_until  (pd.Timestamp | None)
               stress_cooldown_until (pd.Timestamp | None)

    Returns
    -------
    dict with keys:
        allow_entry     : bool
        force_flat      : bool
        size_multiplier : float (1.0 = full size)
        reasons         : list[str]
    """
    allow_entry = True
    force_flat  = False
    size_mult   = 1.0
    reasons     = []

    ts = row.name  # timestamp index

    # 1. Kill switch check
    if config.get("kill_switch_enabled", False) and state.get("kill_switch_active", False):
        kill_until = state.get("kill_switch_until")
        if kill_until is None or ts < kill_until:
            allow_entry = False
            force_flat  = True
            reasons.append("kill_switch_active")

    # 2. Stress spike check
    range_val = float(row["range_1h"]) if "range_1h" in row.index and not pd.isna(row["range_1h"]) else 0.0
    thresh = config.get("stress_range_threshold", 0.03)
    is_stress = range_val >= thresh

    if is_stress:
        if config.get("market_quality_filter", False):
            allow_entry = False
            reasons.append(f"stress_spike: range_1h={range_val:.3f}")
        if config.get("stress_force_flat", False):
            force_flat = True
            reasons.append(f"force_flat_on_stress")

    # 3. Stress cooldown
    cooldown_until = state.get("stress_cooldown_until")
    if cooldown_until is not None and ts < cooldown_until:
        allow_entry = False
        reasons.append("stress_cooldown")

    return {
        "allow_entry":     allow_entry,
        "force_flat":      force_flat,
        "size_multiplier": size_mult,
        "reasons":         reasons,
    }


# ──────────────────────────────────────────────
# Phase 3: Volatility-targeted position sizing
# ──────────────────────────────────────────────

def compute_vol_size_multiplier(row: pd.Series, config: dict) -> float:
    """
    Return a position size multiplier based on vol targeting (Phase 3 J).

    size_mult = clip(vol_target_pct / current_vol_pct, min_mult, max_mult)

    Parameters
    ----------
    row    : single DataFrame row with 'volatility_pct' column
    config : effective CONFIG dict

    Returns
    -------
    float in [vol_target_min_mult, vol_target_max_mult]
    """
    if not config.get("vol_targeting_enabled", False):
        return 1.0

    if "volatility_pct" not in row.index:
        return 1.0

    current_vol = float(row["volatility_pct"])
    if current_vol <= 0 or np.isnan(current_vol):
        return float(config.get("vol_target_max_mult", 1.0))

    target   = float(config.get("vol_target_pct",      30.0))
    raw_mult = target / current_vol

    return float(np.clip(
        raw_mult,
        config.get("vol_target_min_mult", 0.25),
        config.get("vol_target_max_mult", 1.0),
    ))
