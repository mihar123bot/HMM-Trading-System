"""
backtester.py
Runs the regime-based HMM strategy simulation (spot-only, no leverage).

Entry Rules
-----------
  - HMM regime == Bull for at least MIN_REGIME_BARS consecutive bars
  - entry_signal_ok == True (built upstream; includes bucket voting via signal_ok and/or MR mode logic)
  - Not in 12-hour post-exit cooldown
  - Not blocked by risk gates (kill switch, market quality, stress cooldown)

Exit Rules (first condition that triggers wins)
-----------
  1. Stop Loss      : price drops to SL level
  2. Trailing Stop  : price drops N% from its highest close since entry (optional)
  3. Take Profit    : price rises to TP level
  4. Regime Flip    : HMM regime flips to non-Bull for > regime_flip_grace_bars bars
  5. Force Flat     : risk gate demands immediate exit (stress spike / kill switch)

Regime-flip grace period
------------------------
  If regime_flip_grace_bars > 0, a regime flip to non-Bull does NOT immediately
  trigger exit. The strategy waits up to N bars; if the regime returns to Bull
  within the grace window the trade continues. Only exits after N+1 consecutive
  non-Bull bars.

Stop modes
----------
  Fixed %   : SL = entry_fill × (1 − |stop_loss_pct|/100)
              TP = entry_fill × (1 + take_profit_pct/100)
  ATR-based : SL = entry_fill − k_stop × ATR_at_entry
              TP = entry_fill + k_tp   × ATR_at_entry

Execution model (v1_next_open)
-------------------------------
  Signal fires at bar i (based on Close[i] data); fill executes at bar i+1's Open.

  Cost model — slippage and fees are treated separately:
    slippage_factor  = slippage_bps / 10_000
    entry_fill       = Open[i+1] × (1 + slippage_factor)
    exit_fill        = Open[i+1] × (1 − slippage_factor)

    notional         = cash × position_size   (spot; position_size ≤ 1.0)
    fee_entry_usd    = notional × fee_bps / 10_000
    fee_exit_usd     = notional × fee_bps / 10_000

Phase 3 additions
-----------------
  Vol targeting (J): size_mult = clip(target_vol / realized_vol, min_mult, max_mult)
  Kill switch  (G): if rolling drawdown > threshold → halt trading N hours
  Market quality filter (K): block entry on stress spike bars
  Stress force-flat (G): exit immediately when stress spike detected
  Tail metrics (L): Sortino, CVaR 95%, max consecutive losses, time-to-recovery

Cooldown  : 12-hour hard lock after ANY exit
Leverage  : 1× (spot only)
Capital   : $10,000 starting
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

STARTING_CAPITAL     = 10_000.0
LEVERAGE             = 1.0   # spot-only
COOLDOWN_HOURS       = 12
EXECUTION_RULE_VER   = "v1_next_open"

# Risk management defaults (overridable via run_backtest kwargs)
STOP_LOSS_PCT     = -2.5
TAKE_PROFIT_PCT   =  4.0
MIN_REGIME_BARS   =  2

REGIME_BULL = "Bull"

# Per-asset execution cost defaults (bps per side)
EXECUTION_COSTS: dict[str, dict] = {
    "BTC-USD": {"fee_bps": 5,  "slippage_bps": 3},
    "ETH-USD": {"fee_bps": 5,  "slippage_bps": 3},
    "SOL-USD": {"fee_bps": 10, "slippage_bps": 5},
    "XRP-USD": {"fee_bps": 10, "slippage_bps": 5},
    "COIN":    {"fee_bps": 8,  "slippage_bps": 4},
    "NVDA":    {"fee_bps": 2,  "slippage_bps": 1},
    "AVGO":    {"fee_bps": 2,  "slippage_bps": 1},
    "TSM":     {"fee_bps": 2,  "slippage_bps": 1},
    "MSFT":    {"fee_bps": 1,  "slippage_bps": 1},
    "NOW":     {"fee_bps": 2,  "slippage_bps": 1},
}
DEFAULT_COSTS = {"fee_bps": 5, "slippage_bps": 3}


def _make_config_hash(stop_loss_pct, take_profit_pct, min_regime_bars,
                      fee_bps, slippage_bps,
                      use_trailing_stop=False, trailing_stop_pct=2.0,
                      trail_atr_mult=1.25, trail_activation_pct=1.5,
                      regime_flip_grace_bars=0, use_pbull_sizing=False) -> str:
    payload = json.dumps({
        "stop_loss_pct":          stop_loss_pct,
        "take_profit_pct":        take_profit_pct,
        "min_regime_bars":        min_regime_bars,
        "fee_bps":                fee_bps,
        "slippage_bps":           slippage_bps,
        "use_trailing_stop":      use_trailing_stop,
        "trailing_stop_pct":      trailing_stop_pct,
        "trail_atr_mult":         trail_atr_mult,
        "trail_activation_pct":   trail_activation_pct,
        "regime_flip_grace_bars": regime_flip_grace_bars,
        "use_pbull_sizing":       use_pbull_sizing,
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def run_backtest(
    data: pd.DataFrame,
    stop_loss_pct:    float = STOP_LOSS_PCT,
    take_profit_pct:  float = TAKE_PROFIT_PCT,
    min_regime_bars:  int   = MIN_REGIME_BARS,
    # Execution cost params
    ticker:           str   = "",
    fee_bps:          float = None,
    slippage_bps:     float = None,
    # ATR stop params
    use_atr_stops:    bool  = False,
    k_stop:           float = 2.0,
    k_tp:             float = 3.0,
    # Trailing stop
    use_trailing_stop:    bool  = True,
    trailing_stop_pct:    float = 2.0,   # fixed-% fallback distance (when no ATR)
    trail_atr_mult:       float = 2.5,   # ATR multiples for trailing stop level
    trail_activation_pct: float = 1.5,   # only activate trailing stop after this % gain
    # Regime-flip grace period
    regime_flip_grace_bars: int = 2,
    # p_bull position sizing (E)
    use_pbull_sizing:     bool  = True,
    # Phase 3 J — vol targeting
    use_vol_targeting:  bool  = False,
    vol_target_pct:     float = 30.0,
    vol_target_min_mult: float = 0.25,
    vol_target_max_mult: float = 1.0,
    # Phase 3 G — kill switch
    kill_switch_enabled:   bool  = True,
    kill_switch_dd_pct:    float = 9.0,
    kill_switch_cooldown_h: int  = 24,
    # Phase 3 K — market quality filter
    use_market_quality_filter: bool  = True,
    stress_range_threshold:    float = 0.04,
    # Phase 3 G — stress force-flat
    stress_force_flat:     bool  = True,
    stress_cooldown_hours: int   = 24,
    # External gate columns (Phase 4 — if present in data)
    use_external_gates: bool = False,
    # Mean-reversion entry controls
    entry_mode: str = "Hybrid",  # Trend | Mean Reversion | Hybrid
    mr_down_bars: int = 2,
    mr_bounce_rsi_max: float = 45.0,
    mr_short_drop_pct: float = 0.4,
    p_bull_min: float = 0.55,
    # Optional: use indicators.py gate/sizing helpers (parity path)
    use_indicator_gate_logic: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parameters
    ----------
    data              : DataFrame with Open, Close, regime, signal_ok, [atr, p_bull,
                        range_1h, volatility_pct, ext_overheat, ext_low_liquidity] columns
    stop_loss_pct     : exit if return <= this (e.g. -3.0 = -3%)
    take_profit_pct   : exit if return >= this (e.g. 4.0 = +4%)
    min_regime_bars   : minimum consecutive Bull bars before entry
    ticker            : for auto-selecting execution cost defaults
    fee_bps           : commission fee (basis points per side); None = auto from ticker
    slippage_bps      : slippage (basis points per side); None = auto from ticker
    use_atr_stops     : if True, use ATR-scaled SL/TP
    k_stop / k_tp     : ATR multiples for SL / TP
    use_trailing_stop    : if True, trail the SL below the highest close since entry.
                           Level = trade_high − trail_atr_mult × ATR (when ATR available)
                           or trade_high × (1 − trailing_stop_pct/100) as fallback.
                           Only activates once the trade is up >= trail_activation_pct %.
    trailing_stop_pct    : fixed-% fallback trailing distance (when ATR not available)
    trail_atr_mult       : ATR multiples for the ATR-based trailing stop level
    trail_activation_pct : gain % required before the trailing stop arms itself
    regime_flip_grace_bars : allow up to N non-Bull bars before regime-flip exit fires
                             (0 = exit on first non-Bull bar, matching prior behaviour)
    use_pbull_sizing     : if True, scale position size by p_bull (e.g. p_bull=0.85 → 85% size)
    use_vol_targeting : (Phase 3 J) scale position by vol_target_pct / realized_vol
    vol_target_pct    : target annualised vol % for sizing
    vol_target_min/max_mult : clamp bounds for size multiplier
    kill_switch_enabled : (Phase 3 G) halt trading after deep drawdown
    kill_switch_dd_pct  : max rolling drawdown % before kill switch fires
    kill_switch_cooldown_h : hours to remain halted after kill switch
    use_market_quality_filter : (Phase 3 K) skip entries on stress spike bars
    stress_range_threshold    : (H-L)/C threshold for stress spike
    stress_force_flat  : (Phase 3 G) force-flat open position on stress spike
    stress_cooldown_hours : post-stress entry cooldown
    use_external_gates : honour ext_overheat / ext_low_liquidity columns if present

    Returns
    -------
    result_df, trades_df, metrics
    """
    df = data.copy()
    if use_indicator_gate_logic:
        from indicators import apply_btc_risk_gates, compute_vol_size_multiplier

    required_cols = ["Open", "Close", "regime", "signal_ok"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # ── Resolve execution costs ───────────────────────────────────────────────
    defaults      = EXECUTION_COSTS.get(ticker, DEFAULT_COSTS)
    _fee_bps      = fee_bps      if fee_bps      is not None else defaults["fee_bps"]
    _slippage_bps = slippage_bps if slippage_bps is not None else defaults["slippage_bps"]
    slip_factor   = _slippage_bps / 10_000.0
    fee_factor    = _fee_bps      / 10_000.0

    has_atr    = "atr"          in df.columns and not df["atr"].isna().all()
    has_p_bull = "p_bull"       in df.columns
    has_range  = "range_1h"     in df.columns
    has_vol    = "volatility_pct" in df.columns
    # Phase 4 external gate columns
    has_ext_overheat  = use_external_gates and "ext_overheat"    in df.columns
    has_ext_liquidity = use_external_gates and "ext_low_liquidity" in df.columns

    _use_atr = use_atr_stops and has_atr

    config_hash = _make_config_hash(
        stop_loss_pct, take_profit_pct, min_regime_bars, _fee_bps, _slippage_bps,
        use_trailing_stop, trailing_stop_pct, trail_atr_mult, trail_activation_pct,
        regime_flip_grace_bars, use_pbull_sizing,
    )

    n = len(df)
    equity   = np.full(n, STARTING_CAPITAL, dtype=float)
    position = np.zeros(n, dtype=int)

    trades = []

    cash            = STARTING_CAPITAL
    in_trade        = False
    pending_entry   = False
    pending_exit    = False
    pending_exit_reason = None

    entry_price    = 0.0
    entry_fill     = 0.0
    entry_time     = None
    entry_equity   = 0.0   # notional used for this trade
    idle_cash      = 0.0   # cash NOT invested (vol targeting)
    exit_time      = None
    bull_streak    = 0
    atr_at_entry   = 0.0
    sl_price       = 0.0
    tp_price       = 0.0
    total_fees     = 0.0
    pos_size_mult  = 1.0   # vol targeting multiplier stored for trade record

    sl_factor = stop_loss_pct   / 100.0
    tp_factor = take_profit_pct / 100.0

    # Peak equity for kill switch HWM tracking
    hwm = STARTING_CAPITAL

    # Phase 3: kill switch state
    kill_switch_active = False
    kill_switch_until  = None

    # Phase 3: stress cooldown state
    stress_cooldown_until = None

    # ── Attribution counters ──────────────────────────────────────────────────
    bars_bull_regime         = 0
    bars_p_bull_pass         = 0
    bars_signal_ok           = 0
    bars_eligible            = 0
    entries_blocked_cooldown = 0
    entries_blocked_gate     = 0   # kill switch / market quality / stress cooldown
    entries_blocked_external = 0   # Phase 4 external gate blocks
    entry_attempts_base      = 0   # attempts before cooldown/gate/external blocking
    exits_sl = exits_tp = exits_regime = exits_eod = exits_force_flat = exits_kill_switch = exits_trailing_sl = 0

    # Trailing stop and regime-flip grace state
    trade_high         = 0.0   # highest close since entry (for trailing stop)
    bear_streak_trade  = 0     # consecutive non-Bull bars while in trade (for grace period)
    trailing_activated = False  # True once trade gains >= trail_activation_pct %
    trailing_sl_factor = trailing_stop_pct / 100.0

    for i in range(1, n):
        row       = df.iloc[i]
        prev_row  = df.iloc[i - 1]
        regime    = row["regime"]
        signal_ok = bool(row["signal_ok"])
        close_i   = float(row["Close"])
        open_i    = float(row["Open"])
        ts        = df.index[i]

        # Mean-reversion signal (pullback then bounce)
        mr_bars = max(int(mr_down_bars), 1)
        has_window = i >= mr_bars
        pullback_ok = False
        bounce_ok = False
        short_drop_ok = False
        if has_window:
            recent_closes = df["Close"].iloc[i - mr_bars:i + 1].astype(float).values
            # pullback into bar i-1
            pullback_ok = bool(np.all(np.diff(recent_closes[:-1]) < 0)) if len(recent_closes) > 2 else True
            # bounce on current bar i
            bounce_ok = bool(recent_closes[-1] > recent_closes[-2])
            start_px = float(recent_closes[0])
            trough_px = float(recent_closes[-2])
            short_drop_ok = ((trough_px - start_px) / max(start_px, 1e-9) * 100.0) <= -abs(float(mr_short_drop_pct))
        has_rsi = "rsi" in row.index and not pd.isna(row.get("rsi", np.nan))
        rsi_ok = (float(row["rsi"]) <= float(mr_bounce_rsi_max)) if has_rsi else True
        mr_signal_ok = pullback_ok and bounce_ok and short_drop_ok and rsi_ok

        mode = str(entry_mode).strip().lower()
        if mode == "trend":
            entry_signal_ok = signal_ok
        elif mode in ("mean reversion", "mean_reversion", "mr"):
            entry_signal_ok = mr_signal_ok
        else:
            entry_signal_ok = signal_ok or mr_signal_ok
        p_bull_i  = float(row["p_bull"]) if has_p_bull else 1.0
        range_i   = float(row["range_1h"]) if has_range else 0.0
        vol_i     = float(row["volatility_pct"]) if has_vol else 0.0

        # ── Kill switch check ──────────────────────────────────────────────
        if in_trade and entry_fill > 0:
            m2m_equity = idle_cash + (close_i / entry_fill) * entry_equity
            hwm = max(hwm, m2m_equity)
            effective_equity = m2m_equity
        else:
            hwm = max(hwm, cash)
            effective_equity = cash

        if kill_switch_enabled:
            dd_pct = (hwm - effective_equity) / hwm * 100 if hwm > 0 else 0.0
            if not kill_switch_active and dd_pct >= kill_switch_dd_pct:
                kill_switch_active = True
                kill_switch_until  = ts + pd.Timedelta(hours=kill_switch_cooldown_h)
            if kill_switch_active and kill_switch_until is not None and ts >= kill_switch_until:
                kill_switch_active = False
                kill_switch_until  = None

        # ── Stress cooldown update ─────────────────────────────────────────
        is_stress_bar = has_range and range_i >= stress_range_threshold
        if is_stress_bar:
            stress_cooldown_until = ts + pd.Timedelta(hours=stress_cooldown_hours)

        # ── Attribution: regime / p_bull / signal ─────────────────────────
        if regime == REGIME_BULL:
            bars_bull_regime += 1
            if (not has_p_bull) or (p_bull_i >= float(p_bull_min)):
                bars_p_bull_pass += 1
            if entry_signal_ok:
                bars_signal_ok += 1

        # ── Track consecutive Bull bars ────────────────────────────────────
        if regime == REGIME_BULL:
            bull_streak += 1
        else:
            bull_streak = 0

        # ── Cooldown check ────────────────────────────────────────────────
        in_cooldown = False
        if exit_time is not None:
            hours_since_exit = (ts - exit_time).total_seconds() / 3600
            in_cooldown = hours_since_exit < COOLDOWN_HOURS

        # ── Step 1: Execute any pending fills at this bar's Open ───────────
        if pending_exit and in_trade:
            exit_open     = open_i
            exit_fill_val = exit_open * (1 - slip_factor)
            notional      = entry_equity
            fee_entry_usd   = notional * fee_factor
            slip_entry_usd  = notional * slip_factor
            fee_exit_usd    = notional * fee_factor
            slip_exit_usd   = notional * slip_factor
            expected_rt_cost = notional * (_fee_bps + _slippage_bps) / 10_000 * 2
            total_cost_usd   = fee_entry_usd + fee_exit_usd + slip_entry_usd + slip_exit_usd
            sanity_pass      = abs(total_cost_usd - expected_rt_cost) < 0.01

            price_ret   = (exit_fill_val - entry_fill) / entry_fill
            gross_pnl   = price_ret * notional
            net_pnl     = gross_pnl - fee_entry_usd - fee_exit_usd
            exit_equity = entry_equity + net_pnl
            ret_pct     = net_pnl / notional * 100 if notional > 0 else 0.0

            total_fees += fee_entry_usd + fee_exit_usd

            _exit_reason = pending_exit_reason
            if _exit_reason == "Stop Loss":
                exits_sl += 1
            elif _exit_reason == "Trailing Stop":
                exits_trailing_sl += 1
            elif _exit_reason == "Take Profit":
                exits_tp += 1
            elif _exit_reason == "Regime Flip":
                exits_regime += 1
            elif _exit_reason == "Force Flat":
                exits_force_flat += 1
            elif _exit_reason == "Kill Switch":
                exits_kill_switch += 1

            trade_record = {
                "Entry Time":           entry_time,
                "Exit Time":            ts,
                "Entry Price":          round(entry_price, 2),
                "Exit Price":           round(exit_open, 2),
                "Entry Fill":           round(entry_fill, 2),
                "Exit Fill":            round(exit_fill_val, 2),
                "Return (%)":           round(ret_pct, 3),
                "PnL ($)":              round(net_pnl, 2),
                "Fee ($)":              round(fee_entry_usd + fee_exit_usd, 2),
                "Fee Entry ($)":        round(fee_entry_usd, 2),
                "Fee Exit ($)":         round(fee_exit_usd, 2),
                "Slippage Entry ($)":   round(slip_entry_usd, 2),
                "Slippage Exit ($)":    round(slip_exit_usd, 2),
                "Total Cost ($)":       round(total_cost_usd, 2),
                "Notional ($)":         round(notional, 2),
                "Size Mult":            round(pos_size_mult, 3),
                "Sanity Pass":          sanity_pass,
                "Exit Reason":          _exit_reason,
                "Equity After ($)":     round(idle_cash + exit_equity, 2),
                "execution_rule_version": EXECUTION_RULE_VER,
                "config_hash":          config_hash,
            }
            if _use_atr:
                trade_record["ATR Stop Level"] = round(sl_price, 2)
                trade_record["ATR TP Level"]   = round(tp_price, 2)

            trades.append(trade_record)
            cash     = idle_cash + exit_equity
            idle_cash = 0.0
            in_trade  = False
            exit_time = ts
            entry_price = 0.0; entry_fill = 0.0; entry_equity = 0.0
            pending_exit = False; pending_exit_reason = None

        if pending_entry and not in_trade:
            # Phase 3 J: vol targeting size multiplier
            if use_indicator_gate_logic:
                _size_cfg = {
                    "vol_targeting_enabled": use_vol_targeting,
                    "vol_target_pct": vol_target_pct,
                    "vol_target_min_mult": vol_target_min_mult,
                    "vol_target_max_mult": vol_target_max_mult,
                }
                pos_size_mult = float(compute_vol_size_multiplier(row, _size_cfg))
            elif use_vol_targeting and has_vol and vol_i > 0:
                pos_size_mult = float(np.clip(
                    vol_target_pct / vol_i,
                    vol_target_min_mult,
                    vol_target_max_mult,
                ))
            else:
                pos_size_mult = 1.0

            # E: p_bull-based position scaling — higher confidence → larger position
            if use_pbull_sizing and has_p_bull:
                pos_size_mult = float(np.clip(pos_size_mult * p_bull_i, 0.05, 1.0))

            entry_open   = open_i
            entry_fill   = entry_open * (1 + slip_factor)
            entry_time   = ts
            notional     = cash * pos_size_mult
            idle_cash    = cash - notional
            entry_equity = notional
            in_trade          = True
            entry_price       = entry_open
            trade_high        = entry_fill   # reset trailing stop high water mark
            bear_streak_trade = 0            # reset grace period counter
            trailing_activated = False       # reset trailing stop activation

            # Always capture ATR at entry — used for ATR-based trailing stop
            # even when use_atr_stops = False
            if has_atr and "atr" in prev_row.index:
                atr_val = prev_row["atr"]
                atr_at_entry = float(atr_val) if not np.isnan(atr_val) else 0.0
            else:
                atr_at_entry = 0.0

            if _use_atr and atr_at_entry > 0:
                sl_price = entry_fill - k_stop * atr_at_entry
                tp_price = entry_fill + k_tp   * atr_at_entry
            else:
                sl_price = entry_fill * (1 + sl_factor)
                tp_price = entry_fill * (1 + tp_factor)

            pending_entry = False

        # ── Update trade-level trackers (trailing stop high, grace period) ──
        if in_trade:
            trade_high = max(trade_high, close_i)
            if regime == REGIME_BULL:
                bear_streak_trade = 0
            else:
                bear_streak_trade += 1
            # Arm trailing stop once the trade is sufficiently in profit
            if use_trailing_stop and not trailing_activated and entry_fill > 0:
                if (close_i - entry_fill) / entry_fill >= trail_activation_pct / 100.0:
                    trailing_activated = True

        # ── Step 2: Check exit conditions for current bar ──────────────────
        # ATR-based trailing level (armed only after trail_activation_pct gain)
        _trail_hit = False
        if use_trailing_stop and trailing_activated:
            if atr_at_entry > 0:
                _trail_hit = close_i <= trade_high - trail_atr_mult * atr_at_entry
            else:
                _trail_hit = close_i <= trade_high * (1 - trailing_sl_factor)

        if in_trade and not pending_exit:
            # Phase 3 G: stress force-flat (highest priority)
            if stress_force_flat and is_stress_bar:
                pending_exit = True
                pending_exit_reason = "Force Flat"
            elif kill_switch_active and kill_switch_enabled:
                pending_exit = True
                pending_exit_reason = "Kill Switch"
            elif _use_atr:
                if close_i <= sl_price:
                    pending_exit = True; pending_exit_reason = "Stop Loss"
                elif _trail_hit:
                    pending_exit = True; pending_exit_reason = "Trailing Stop"
                elif close_i >= tp_price:
                    pending_exit = True; pending_exit_reason = "Take Profit"
                elif regime != REGIME_BULL and bear_streak_trade > regime_flip_grace_bars:
                    pending_exit = True; pending_exit_reason = "Regime Flip"
            else:
                price_ret_check = (close_i - entry_fill) / entry_fill
                if price_ret_check <= sl_factor:
                    pending_exit = True; pending_exit_reason = "Stop Loss"
                elif _trail_hit:
                    pending_exit = True; pending_exit_reason = "Trailing Stop"
                elif price_ret_check >= tp_factor:
                    pending_exit = True; pending_exit_reason = "Take Profit"
                elif regime != REGIME_BULL and bear_streak_trade > regime_flip_grace_bars:
                    pending_exit = True; pending_exit_reason = "Regime Flip"

        # ── Step 3: Check entry conditions ────────────────────────────────
        if (
            not in_trade and not pending_entry and not pending_exit
            and regime == REGIME_BULL
            and bull_streak >= min_regime_bars
            and entry_signal_ok
        ):
            entry_attempts_base += 1
            blocked = False
            if in_cooldown:
                entries_blocked_cooldown += 1
                blocked = True

            if not blocked:
                if use_indicator_gate_logic:
                    _gate_cfg = {
                        "kill_switch_enabled": kill_switch_enabled,
                        "use_market_quality_filter": use_market_quality_filter,
                        "stress_range_threshold": stress_range_threshold,
                        "stress_force_flat": stress_force_flat,
                    }
                    _gate_state = {
                        "kill_switch_active": kill_switch_active,
                        "kill_switch_until": kill_switch_until,
                        "stress_cooldown_until": stress_cooldown_until,
                    }
                    gate_decision = apply_btc_risk_gates(row, _gate_cfg, _gate_state)
                    if not gate_decision.get("allow_entry", True):
                        entries_blocked_gate += 1
                        blocked = True
                else:
                    # Kill switch gate
                    if kill_switch_active and kill_switch_enabled:
                        entries_blocked_gate += 1
                        blocked = True

            if not blocked and not use_indicator_gate_logic:
                # Market quality / stress filter
                if use_market_quality_filter and is_stress_bar:
                    entries_blocked_gate += 1
                    blocked = True

            if not blocked and not use_indicator_gate_logic:
                # Stress cooldown
                if stress_cooldown_until is not None and ts < stress_cooldown_until:
                    entries_blocked_gate += 1
                    blocked = True

            if not blocked:
                # Phase 4: external gates
                if has_ext_overheat and bool(row["ext_overheat"]):
                    entries_blocked_external += 1
                    blocked = True
                elif has_ext_liquidity and bool(row["ext_low_liquidity"]):
                    entries_blocked_external += 1
                    blocked = True

            if not blocked and i < n - 1:
                pending_entry = True

        # ── Attribution: eligible bar ─────────────────────────────────────
        if (
            regime == REGIME_BULL
            and entry_signal_ok
            and not in_cooldown
            and bull_streak >= min_regime_bars
        ):
            bars_eligible += 1

        # ── Mark-to-market equity ─────────────────────────────────────────
        if in_trade:
            unrealised = (close_i - entry_fill) / entry_fill * entry_equity
            equity[i]  = idle_cash + entry_equity + unrealised
        else:
            equity[i] = cash

        position[i] = 1 if in_trade else 0

    # ── Close any open position at last bar ───────────────────────────────────
    if in_trade:
        last_close     = float(df["Close"].iloc[-1])
        exit_fill_last = last_close * (1 - slip_factor)
        notional       = entry_equity
        fee_entry_usd  = notional * fee_factor
        slip_entry_usd = notional * slip_factor
        fee_exit_usd   = notional * fee_factor
        slip_exit_usd  = notional * slip_factor
        total_cost_usd = fee_entry_usd + fee_exit_usd + slip_entry_usd + slip_exit_usd
        expected_rt    = notional * (_fee_bps + _slippage_bps) / 10_000 * 2

        price_ret   = (exit_fill_last - entry_fill) / entry_fill
        gross_pnl   = price_ret * notional
        net_pnl     = gross_pnl - fee_entry_usd - fee_exit_usd
        exit_equity = entry_equity + net_pnl

        total_fees += fee_entry_usd + fee_exit_usd
        exits_eod  += 1

        trade_record = {
            "Entry Time":           entry_time,
            "Exit Time":            df.index[-1],
            "Entry Price":          round(entry_price, 2),
            "Exit Price":           round(last_close, 2),
            "Entry Fill":           round(entry_fill, 2),
            "Exit Fill":            round(exit_fill_last, 2),
            "Return (%)":           round(net_pnl / notional * 100 if notional > 0 else 0, 3),
            "PnL ($)":              round(net_pnl, 2),
            "Fee ($)":              round(fee_entry_usd + fee_exit_usd, 2),
            "Fee Entry ($)":        round(fee_entry_usd, 2),
            "Fee Exit ($)":         round(fee_exit_usd, 2),
            "Slippage Entry ($)":   round(slip_entry_usd, 2),
            "Slippage Exit ($)":    round(slip_exit_usd, 2),
            "Total Cost ($)":       round(total_cost_usd, 2),
            "Notional ($)":         round(notional, 2),
            "Size Mult":            round(pos_size_mult, 3),
            "Sanity Pass":          abs(total_cost_usd - expected_rt) < 0.01,
            "Exit Reason":          "End of Data",
            "Equity After ($)":     round(idle_cash + exit_equity, 2),
            "execution_rule_version": EXECUTION_RULE_VER,
            "config_hash":          config_hash,
        }
        if _use_atr:
            trade_record["ATR Stop Level"] = round(sl_price, 2)
            trade_record["ATR TP Level"]   = round(tp_price, 2)

        trades.append(trade_record)
        equity[-1] = idle_cash + exit_equity
        cash       = idle_cash + exit_equity

    df["equity"]   = equity
    df["position"] = position

    trades_df = pd.DataFrame(trades)
    n_trades  = len(trades_df)

    # ── Waterfall ─────────────────────────────────────────────────────────────
    attempts_after_cooldown = max(entry_attempts_base - entries_blocked_cooldown, 0)
    attempts_after_gate = max(attempts_after_cooldown - entries_blocked_gate, 0)
    attempts_after_external = max(attempts_after_gate - entries_blocked_external, 0)

    waterfall = {
        "1_attempt_base":          entry_attempts_base,
        "2_blocked_cooldown":      entries_blocked_cooldown,
        "3_after_cooldown":        attempts_after_cooldown,
        "4_blocked_gate":          entries_blocked_gate,
        "5_after_gate":            attempts_after_gate,
        "6_blocked_external":      entries_blocked_external,
        "7_not_blocked":           attempts_after_external,
        "8_entries_taken":         n_trades,
    }

    attribution = {
        "bars_bull_regime":          bars_bull_regime,
        "bars_p_bull_pass":          bars_p_bull_pass,
        "bars_signal_ok_while_bull": bars_signal_ok,
        "bars_eligible":             bars_eligible,
        "entry_attempts_base":       entry_attempts_base,
        "entries_blocked_cooldown":  entries_blocked_cooldown,
        "entries_blocked_gate":      entries_blocked_gate,
        "entries_blocked_external":  entries_blocked_external,
        "exits_stop_loss":           exits_sl,
        "exits_trailing_stop":       exits_trailing_sl,
        "exits_take_profit":         exits_tp,
        "exits_regime_flip":         exits_regime,
        "exits_end_of_data":         exits_eod,
        "exits_force_flat":          exits_force_flat,
        "exits_kill_switch":         exits_kill_switch,
        "pct_time_bull":             round(bars_bull_regime / max(n, 1) * 100, 1),
        "pct_time_p_bull_pass":      round(bars_p_bull_pass / max(n, 1) * 100, 1),
        "pct_time_signal_ok":        round(bars_signal_ok   / max(n, 1) * 100, 1),
        "pct_time_eligible":         round(bars_eligible    / max(n, 1) * 100, 1),
    }

    metrics = _compute_metrics(
        df, trades_df,
        stop_loss_pct, take_profit_pct, min_regime_bars,
        _fee_bps, _slippage_bps, total_fees,
        use_atr_stops=_use_atr, k_stop=k_stop, k_tp=k_tp,
    )
    metrics["attribution"]              = attribution
    metrics["waterfall"]                = waterfall
    metrics["execution_rule_version"]   = EXECUTION_RULE_VER
    metrics["config_hash"]              = config_hash
    metrics["exits_stop_loss"]          = exits_sl
    metrics["exits_trailing_stop"]      = exits_trailing_sl
    metrics["exits_take_profit"]        = exits_tp
    metrics["exits_regime_flip"]        = exits_regime
    metrics["exits_force_flat"]         = exits_force_flat
    metrics["exits_kill_switch"]        = exits_kill_switch
    metrics["entries_blocked_gate"]     = entries_blocked_gate
    metrics["entries_blocked_external"] = entries_blocked_external

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
    fee_bps: float,
    slippage_bps: float,
    total_fees: float,
    use_atr_stops: bool = False,
    k_stop: float = 2.0,
    k_tp: float = 3.0,
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
        wins      = (trades_df["Return (%)"] > 0).sum()
        win_rate  = wins / len(trades_df) * 100
        sl_exits  = (trades_df["Exit Reason"] == "Stop Loss").sum()
        tp_exits  = (trades_df["Exit Reason"] == "Take Profit").sum()
        reg_exits = (trades_df["Exit Reason"] == "Regime Flip").sum()
        rets_arr  = trades_df["Return (%)"].values
    else:
        win_rate = sl_exits = tp_exits = reg_exits = 0
        rets_arr = np.array([])

    eq_series  = pd.Series(equity)
    ret_series = eq_series.pct_change().dropna()

    sharpe = (
        ret_series.mean() / ret_series.std() * np.sqrt(8760)
        if ret_series.std() > 0 else 0.0
    )

    # ── Phase 3 L: Tail metrics ───────────────────────────────────────────────
    # Sortino ratio (uses downside std only)
    neg_rets = ret_series[ret_series < 0]
    downside_std = neg_rets.std() if len(neg_rets) > 1 else ret_series.std()
    sortino = (
        ret_series.mean() / downside_std * np.sqrt(8760)
        if downside_std > 0 else 0.0
    )

    # CVaR 95% (hourly): average of worst 5% of hourly equity returns
    if len(ret_series) > 20:
        var_05   = ret_series.quantile(0.05)
        cvar_95  = float(ret_series[ret_series <= var_05].mean() * 100)
    else:
        cvar_95 = 0.0

    # Max consecutive losses / wins (trade-level)
    max_consec_losses = 0; cl = 0
    max_consec_wins   = 0; cw = 0
    for r in rets_arr:
        if r > 0:
            cw += 1; cl = 0
            max_consec_wins = max(max_consec_wins, cw)
        else:
            cl += 1; cw = 0
            max_consec_losses = max(max_consec_losses, cl)

    # Worst decile trade return
    worst_decile = float(np.percentile(rets_arr, 10)) if len(rets_arr) >= 10 else (
        float(rets_arr.min()) if len(rets_arr) > 0 else 0.0
    )

    # Large-loss trades (worse than 2× stop loss)
    large_loss_threshold = stop_loss_pct * 2  # e.g. -6% if SL = -3%
    n_large_losses = int((rets_arr < large_loss_threshold).sum()) if len(rets_arr) > 0 else 0

    # Time-to-recovery from max drawdown (hours)
    if max_dd < 0:
        dd_arr = drawdown
        trough_idx = int(np.argmin(dd_arr))
        # Recovery: first bar after trough where equity >= peak at trough
        peak_val = peak[trough_idx]
        recovery_idx = None
        for k in range(trough_idx + 1, len(equity)):
            if equity[k] >= peak_val:
                recovery_idx = k
                break
        time_to_recovery_h = (recovery_idx - trough_idx) if recovery_idx is not None else None
    else:
        time_to_recovery_h = 0

    metrics = {
        "Total Return (%)":        round(total_return, 2),
        "Buy & Hold Return (%)":   round(bah_return, 2),
        "Alpha (%)":               round(alpha, 2),
        "Win Rate (%)":            round(win_rate, 2),
        "Max Drawdown (%)":        round(max_dd, 2),
        "Sharpe Ratio":            round(sharpe, 3),
        "Sortino Ratio":           round(sortino, 3),
        "CVaR 95% (hourly %)":     round(cvar_95, 3),
        "Max Consec Losses":       max_consec_losses,
        "Max Consec Wins":         max_consec_wins,
        "Worst Decile Trade (%)":  round(worst_decile, 2),
        "Large Loss Trades":       n_large_losses,
        "Time-to-Recovery (h)":    time_to_recovery_h,
        "Total Trades":            len(trades_df),
        "Stop Loss Exits":         int(sl_exits),
        "Take Profit Exits":       int(tp_exits),
        "Regime Flip Exits":       int(reg_exits),
        "Min Regime Bars":         min_regime_bars,
        "Final Equity ($)":        round(float(equity[-1]), 2),
        "Starting Capital ($)":    STARTING_CAPITAL,
        "Total Fees ($)":          round(total_fees, 2),
        "Exit Mode":               "ATR-Scaled" if use_atr_stops else "Fixed %",
        "Stop Loss (%)":           stop_loss_pct,
        "Take Profit (%)":         take_profit_pct,
        "Fee (bps/side)":          fee_bps,
        "Slippage (bps/side)":     slippage_bps,
    }

    if use_atr_stops:
        metrics["ATR k_stop"] = k_stop
        metrics["ATR k_tp"]   = k_tp

    return metrics


if __name__ == "__main__":
    from data_loader import fetch_asset_data
    from hmm_engine import fit_hmm, predict_regimes
    from indicators import compute_indicators

    print("Fetching data …")
    raw = fetch_asset_data("BTC-USD")

    print("Fitting HMM …")
    model, scaler, feat_df, state_map, bull_states, bear_state = fit_hmm(raw)
    with_regimes = predict_regimes(model, scaler, feat_df, state_map, raw, bull_states)

    print("Computing indicators …")
    full = compute_indicators(with_regimes)

    print("Running backtest …")
    result, trades, metrics = run_backtest(full, ticker="BTC-USD")

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        if k not in ("attribution", "waterfall"):
            print(f"  {k}: {v}")

    print("\n=== Attribution ===")
    for k, v in metrics["attribution"].items():
        print(f"  {k}: {v}")

    print("\n=== Eligibility Waterfall ===")
    for k, v in metrics["waterfall"].items():
        print(f"  {k}: {v}")

    if not trades.empty:
        print(f"\n=== Trades ({len(trades)}) ===")
        display_cols = [
            "Entry Time", "Exit Time", "Entry Fill", "Exit Fill",
            "Return (%)", "PnL ($)", "Total Cost ($)",
            "Notional ($)", "Size Mult", "Sanity Pass", "Exit Reason",
        ]
        display_cols = [c for c in display_cols if c in trades.columns]
        print(trades[display_cols].to_string(index=False))

        all_pass = trades["Sanity Pass"].all() if "Sanity Pass" in trades.columns else None
        print(f"\nFee sanity check — all trades pass: {all_pass}")

    # Tail metrics
    print("\n=== Tail Metrics ===")
    for k in ["Sortino Ratio", "CVaR 95% (hourly %)", "Max Consec Losses",
              "Worst Decile Trade (%)", "Large Loss Trades", "Time-to-Recovery (h)"]:
        print(f"  {k}: {metrics.get(k)}")
