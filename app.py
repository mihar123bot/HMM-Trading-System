"""
app.py
Streamlit dashboard for the HMM Regime-Based Trading System.

Tabs
----
Main Dashboard    : live signal, chart, metrics, trade log, tail metrics
Walk-Forward      : rolling OOS validation + lockbox evaluation

Sidebar
-------
Asset Selection → Voting Gate (buckets + per-signal toggles) →
HMM Confidence → Risk Management → Execution Costs → Stop Mode →
Phase 3 Risk Gates (kill switch, vol targeting, market quality) →
Entry Thresholds → Indicator Periods → MACD Settings
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="HMM Regime Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from project modules ────────────────────────────────────────────
from data_loader import fetch_asset_data, run_data_sanity_checks, ASSETS, TICKER_TO_NAME
from hmm_engine import fit_hmm, predict_regimes, get_state_stats
from indicators import compute_indicators, get_current_signals, CONFIG as DEFAULT_CONFIG
from backtester import run_backtest, STARTING_CAPITAL, EXECUTION_COSTS, DEFAULT_COSTS

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .signal-long {
        display:inline-block; padding:8px 28px;
        background:linear-gradient(135deg,#00c853,#69f0ae);
        color:#000; font-weight:800; font-size:1.4rem;
        border-radius:50px; letter-spacing:1px;
    }
    .signal-cash {
        display:inline-block; padding:8px 28px;
        background:linear-gradient(135deg,#ff5252,#ff8a80);
        color:#000; font-weight:800; font-size:1.4rem;
        border-radius:50px; letter-spacing:1px;
    }
    .regime-bull {
        display:inline-block; padding:6px 20px;
        background:#1a472a; color:#69f0ae;
        font-weight:700; border-radius:8px;
        border:1px solid #00c853;
    }
    .regime-bear {
        display:inline-block; padding:6px 20px;
        background:#4a1010; color:#ff8a80;
        font-weight:700; border-radius:8px;
        border:1px solid #ff5252;
    }
    .regime-neutral {
        display:inline-block; padding:6px 20px;
        background:#1c2128; color:#8b949e;
        font-weight:700; border-radius:8px;
        border:1px solid #30363d;
    }
    .regime-neutral-highvol {
        display:inline-block; padding:6px 20px;
        background:#2d2008; color:#ffa657;
        font-weight:700; border-radius:8px;
        border:1px solid #f0a500;
    }
    .section-title {
        color:#8b949e; font-size:0.75rem;
        text-transform:uppercase; letter-spacing:2px;
        margin-bottom:4px;
    }
    hr { border-color:#21262d; }
    [data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
    .bucket-card {
        background:#161b22; border:1px solid #30363d;
        border-radius:10px; padding:10px 14px; margin-bottom:6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — live CONFIG editor
# ──────────────────────────────────────────────────────────────────────────────

def build_sidebar() -> tuple[str, dict, dict]:
    """Render sidebar controls and return (ticker, indicator_cfg, risk_cfg)."""

    st.sidebar.markdown("## Asset Selection")
    asset_name = st.sidebar.selectbox("Select Asset", options=list(ASSETS.keys()), index=0)
    ticker = ASSETS[asset_name]
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Strategy Parameters")
    st.sidebar.markdown("All changes trigger an instant backtest rerun.")
    st.sidebar.markdown("---")

    cfg  = {}
    risk = {}

    # ── Voting Gate (bucket system) ───────────────────────────────────────────
    st.sidebar.markdown("### Voting Gate")
    st.sidebar.caption(
        "Signals are split into 4 buckets. ALL bucket minimums must be met to enter."
    )

    with st.sidebar.expander("Trend Bucket (EMA + MACD + ROC + p_bull Slope)", expanded=True):
        cfg["trend_min"] = st.slider(
            "Min Trend signals (of enabled)", 0, 5, DEFAULT_CONFIG["trend_min"], 1,
            key="trend_min",
        )
        st.caption("Enable/disable individual trend signals:")
        cfg["sig_ema_fast_on"]    = st.checkbox("EMA Fast",          DEFAULT_CONFIG.get("sig_ema_fast_on",    True), key="sig_ema_fast_on")
        cfg["sig_ema_slow_on"]    = st.checkbox("EMA Slow",          DEFAULT_CONFIG.get("sig_ema_slow_on",    True), key="sig_ema_slow_on")
        cfg["sig_macd_on"]        = st.checkbox("MACD > Signal",     DEFAULT_CONFIG.get("sig_macd_on",        True), key="sig_macd_on")
        cfg["sig_roc_on"]         = st.checkbox("ROC > 0",           DEFAULT_CONFIG.get("sig_roc_on",         True), key="sig_roc_on")
        cfg["sig_pbull_slope_on"] = st.checkbox("p_bull Slope > 0",  DEFAULT_CONFIG.get("sig_pbull_slope_on", True), key="sig_pbull_slope_on")

    with st.sidebar.expander("Strength + Participation"):
        cfg["strength_min"] = st.slider(
            "ADX required (0 = skip)", 0, 1, DEFAULT_CONFIG["strength_min"], 1,
            key="strength_min",
        )
        cfg["sig_adx_on"] = st.checkbox("ADX Signal", DEFAULT_CONFIG.get("sig_adx_on", True), key="sig_adx_on")

        cfg["participation_min"] = st.slider(
            "Volume > SMA required (0 = skip)", 0, 1, DEFAULT_CONFIG["participation_min"], 1,
            key="participation_min",
        )
        cfg["sig_volume_on"] = st.checkbox("Volume Signal", DEFAULT_CONFIG.get("sig_volume_on", True), key="sig_volume_on")

    with st.sidebar.expander("Risk/Conditioning Bucket"):
        cfg["risk_min"] = st.slider(
            "Min Risk/Cond signals (of enabled)", 0, 4, DEFAULT_CONFIG["risk_min"], 1,
            key="risk_min",
        )
        st.caption("Enable/disable individual risk signals:")
        cfg["sig_rsi_on"]        = st.checkbox("RSI Signal",        DEFAULT_CONFIG.get("sig_rsi_on",        True), key="sig_rsi_on")
        cfg["sig_volatility_on"] = st.checkbox("Volatility Signal", DEFAULT_CONFIG.get("sig_volatility_on", True), key="sig_volatility_on")
        cfg["sig_momentum_on"]   = st.checkbox("Momentum Signal",   DEFAULT_CONFIG.get("sig_momentum_on",   True), key="sig_momentum_on")
        cfg["sig_confidence_on"] = st.checkbox("HMM Confidence",    DEFAULT_CONFIG.get("sig_confidence_on", True), key="sig_confidence_on")

    st.sidebar.markdown("---")

    # ── HMM Confidence ────────────────────────────────────────────────────────
    st.sidebar.markdown("### HMM Confidence")
    with st.sidebar.expander("Bull State Probability Gate"):
        cfg["p_bull_min"] = st.slider(
            "Min Bull confidence (p_bull ≥)", 0.0, 1.0, float(DEFAULT_CONFIG["p_bull_min"]), 0.05,
            help="Minimum posterior probability of the Bull HMM state.",
            key="p_bull_min",
        )

    st.sidebar.markdown("---")

    # ── Risk Management ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Risk Management")
    sl_display = st.sidebar.slider(
        "Stop Loss (%)", 0.5, 15.0, 3.0, 0.5,
        help="Exit if price drops this % from entry fill price",
    )
    risk["stop_loss_pct"]   = -sl_display
    risk["take_profit_pct"] = st.sidebar.slider(
        "Take Profit (%)", 0.5, 30.0, 4.0, 0.5,
        help="Exit if price rises this % from entry fill price",
    )
    risk["min_regime_bars"] = st.sidebar.slider(
        "Min Bull Regime Bars", 1, 24, 3, 1,
        help="Bull regime must persist N consecutive bars before entry",
    )
    risk["regime_flip_grace_bars"] = st.sidebar.slider(
        "Regime-Flip Grace Period (bars)", 0, 6, 2, 1,
        help="Allow up to N non-Bull bars before regime-flip exit fires. "
             "0 = exit immediately on first non-Bull bar.",
    )
    risk["use_pbull_sizing"] = st.sidebar.toggle(
        "Scale Size by p_bull",
        value=False,
        help="Multiply position size by p_bull (e.g. p_bull=0.85 → 85% size). "
             "Composes with vol targeting.",
    )

    st.sidebar.markdown("---")

    # ── Execution Costs ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Execution Costs")
    asset_defaults = EXECUTION_COSTS.get(ticker, DEFAULT_COSTS)
    with st.sidebar.expander(f"Fees & Slippage (defaults for {ticker})"):
        risk["fee_bps"] = st.slider(
            "Commission fee (bps/side)", 0, 50, asset_defaults["fee_bps"], 1,
            help="1 bps = 0.01%. Applied at entry and exit.",
        )
        risk["slippage_bps"] = st.slider(
            "Slippage (bps/side)", 0, 50, asset_defaults["slippage_bps"], 1,
            help="Market impact / half-spread estimate.",
        )

    st.sidebar.markdown("---")

    # ── Stop Mode ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("### Stop Mode")
    risk["use_atr_stops"] = st.sidebar.toggle(
        "Use ATR-Scaled Stops",
        value=True,
        help="Replace fixed % SL/TP with ATR multiples (recommended). "
             "Adapts to current volatility; wider stops in high-vol, tighter in calm.",
    )
    if risk["use_atr_stops"]:
        with st.sidebar.expander("ATR Stop Parameters", expanded=True):
            risk["k_stop"] = st.slider("Stop Loss (ATR multiples)", 0.5, 6.0, 2.0, 0.25, key="k_stop")
            risk["k_tp"]   = st.slider("Take Profit (ATR multiples)", 0.5, 8.0, 3.0, 0.25, key="k_tp")
    else:
        risk["k_stop"] = 2.0
        risk["k_tp"]   = 3.0

    risk["use_trailing_stop"] = st.sidebar.toggle(
        "Use Trailing Stop",
        value=True,
        help="Trail the stop below the trade high using ATR distance (or fixed % fallback). "
             "Only arms after the trade gains trail_activation_pct%.",
    )
    if risk["use_trailing_stop"]:
        with st.sidebar.expander("Trailing Stop Parameters", expanded=True):
            risk["trail_atr_mult"] = st.slider(
                "Trailing Stop (ATR multiples)", 0.5, 4.0, 1.25, 0.25, key="trail_atr",
                help="Trail level = trade_high − N × ATR. Smaller = tighter trail.",
            )
            risk["trail_activation_pct"] = st.slider(
                "Activation Threshold (%)", 0.0, 5.0, 1.5, 0.25, key="trail_act",
                help="Trailing stop only arms after the trade is this % in profit.",
            )
            risk["trailing_stop_pct"] = st.slider(
                "Fixed % Trail Fallback (%)", 0.5, 15.0, 2.0, 0.25, key="trail_pct",
                help="Used only when ATR is unavailable.",
            )
    else:
        risk["trail_atr_mult"]       = 1.25
        risk["trail_activation_pct"] = 1.5
        risk["trailing_stop_pct"]    = 2.0

    st.sidebar.markdown("---")

    # ── Phase 3: Risk Gates ────────────────────────────────────────────────────
    st.sidebar.markdown("### Risk Gates (Phase 3)")

    with st.sidebar.expander("Kill Switch (Drawdown-Based Halt)"):
        risk["kill_switch_enabled"] = st.checkbox(
            "Enable Kill Switch", DEFAULT_CONFIG.get("kill_switch_enabled", False), key="kill_switch_on",
            help="Halt all trading when rolling drawdown exceeds threshold.",
        )
        if risk["kill_switch_enabled"]:
            risk["kill_switch_dd_pct"]    = st.slider("Max Drawdown Before Halt (%)",   2.0, 30.0, float(DEFAULT_CONFIG.get("kill_switch_dd_pct", 10.0)),    0.5, key="ks_dd")
            risk["kill_switch_cooldown_h"] = st.slider("Halt Duration (hours)",          6,   168,  int(DEFAULT_CONFIG.get("kill_switch_cooldown_h", 48)),     6,   key="ks_cd")
        else:
            risk["kill_switch_dd_pct"]     = 10.0
            risk["kill_switch_cooldown_h"] = 48

    with st.sidebar.expander("Market Quality / Stress Filter"):
        risk["use_market_quality_filter"] = st.checkbox(
            "Block Entries on Stress Spikes", DEFAULT_CONFIG.get("market_quality_filter", False),
            key="mq_filter",
            help="Skip entry when (High-Low)/Close exceeds spike threshold.",
        )
        risk["stress_force_flat"] = st.checkbox(
            "Force-Flat on Stress Spike", DEFAULT_CONFIG.get("stress_force_flat", False),
            key="stress_ff",
            help="Immediately exit open position when a stress spike is detected.",
        )
        risk["stress_range_threshold"] = st.slider(
            "Stress Spike Threshold (range ratio)", 0.01, 0.15,
            float(DEFAULT_CONFIG.get("stress_range_threshold", 0.03)), 0.005,
            key="stress_thresh",
            help="(High-Low)/Close threshold. Bars above this are 'stress spikes'.",
        )
        cfg["stress_range_threshold"] = risk["stress_range_threshold"]
        risk["stress_cooldown_hours"] = st.slider(
            "Post-Stress Entry Cooldown (hours)", 0, 72,
            int(DEFAULT_CONFIG.get("stress_cooldown_hours", 12)), 1,
            key="stress_cooldown",
        )

    with st.sidebar.expander("Volatility-Targeted Sizing (Phase 3 J)"):
        risk["use_vol_targeting"] = st.checkbox(
            "Enable Vol Targeting", DEFAULT_CONFIG.get("vol_targeting_enabled", False),
            key="vol_tgt_on",
            help="Scale position size so realised vol matches a target level.",
        )
        if risk["use_vol_targeting"]:
            risk["vol_target_pct"]      = st.slider("Target Annualised Vol (%)",  5.0, 150.0, float(DEFAULT_CONFIG.get("vol_target_pct",      30.0)), 1.0, key="vol_tgt")
            risk["vol_target_min_mult"] = st.slider("Min Size Multiplier",        0.1,   1.0, float(DEFAULT_CONFIG.get("vol_target_min_mult", 0.25)),  0.05, key="vol_min")
            risk["vol_target_max_mult"] = st.slider("Max Size Multiplier",        0.5,   1.0, float(DEFAULT_CONFIG.get("vol_target_max_mult", 1.0)),   0.05, key="vol_max")
        else:
            risk["vol_target_pct"]      = 30.0
            risk["vol_target_min_mult"] = 0.25
            risk["vol_target_max_mult"] = 1.0

    st.sidebar.markdown("---")

    # ── Entry Thresholds ──────────────────────────────────────────────────────
    st.sidebar.markdown("### Entry Thresholds")
    cfg["rsi_max"]            = st.sidebar.slider("RSI Max (< threshold)",  50, 100, DEFAULT_CONFIG["rsi_max"], 1)
    cfg["momentum_min_pct"]   = st.sidebar.slider("Min Momentum (%)", 0.0, 10.0, float(DEFAULT_CONFIG["momentum_min_pct"]), 0.1)
    # Phase 2 fix: BTC annualised vol is typically 40-100%. Default raised to 80%.
    cfg["volatility_max_pct"] = st.sidebar.slider(
        "Max Volatility (%)", 1.0, 150.0, float(DEFAULT_CONFIG["volatility_max_pct"]), 1.0,
        help="Annualised hourly vol must be below this. BTC typical range: 40–100%. "
             "The previous default of 6% was too restrictive for crypto.",
    )
    cfg["adx_min"]            = st.sidebar.slider("Min ADX", 10, 50, DEFAULT_CONFIG["adx_min"], 1)

    st.sidebar.markdown("---")

    # ── Indicator Periods ─────────────────────────────────────────────────────
    st.sidebar.markdown("### Indicator Periods")
    cfg["rsi_period"]          = st.sidebar.slider("RSI Period",                5,  50, DEFAULT_CONFIG["rsi_period"],          1)
    cfg["momentum_period"]     = st.sidebar.slider("Momentum Period (bars)",    2,  50, DEFAULT_CONFIG["momentum_period"],     1)
    cfg["volume_sma_period"]   = st.sidebar.slider("Volume SMA Period",         5, 100, DEFAULT_CONFIG["volume_sma_period"],   1)
    cfg["volatility_period"]   = st.sidebar.slider("Volatility Window (bars)",  6, 168, DEFAULT_CONFIG["volatility_period"],   1)
    cfg["adx_period"]          = st.sidebar.slider("ADX Period",                5,  50, DEFAULT_CONFIG["adx_period"],          1)
    cfg["atr_period"]          = st.sidebar.slider("ATR Period",                5,  50, DEFAULT_CONFIG["atr_period"],          1)
    cfg["ema_fast"]            = st.sidebar.slider("EMA Fast Period",           5, 100, DEFAULT_CONFIG["ema_fast"],            1)
    cfg["ema_slow"]            = st.sidebar.slider("EMA Slow Period",          20, 500, DEFAULT_CONFIG["ema_slow"],            5)
    cfg["roc_period"]          = st.sidebar.slider("ROC Period (bars)",         2,  48, DEFAULT_CONFIG["roc_period"],          1,
        help="Rate-of-change lookback for sig_roc signal (faster than EMA).")
    cfg["p_bull_slope_period"] = st.sidebar.slider("p_bull Slope Period (bars)",1,  12, DEFAULT_CONFIG["p_bull_slope_period"], 1,
        help="Lookback for p_bull slope signal — detects rising HMM confidence.")

    st.sidebar.markdown("---")

    # ── MACD Settings ─────────────────────────────────────────────────────────
    st.sidebar.markdown("### MACD Settings")
    cfg["macd_fast"]   = st.sidebar.slider("MACD Fast",   2,  50, DEFAULT_CONFIG["macd_fast"],   1)
    cfg["macd_slow"]   = st.sidebar.slider("MACD Slow",   5, 100, DEFAULT_CONFIG["macd_slow"],   1)
    cfg["macd_signal"] = st.sidebar.slider("MACD Signal", 2,  30, DEFAULT_CONFIG["macd_signal"], 1)

    st.sidebar.markdown("---")
    st.sidebar.caption("Indicator defaults defined in `indicators.py → CONFIG`.")
    return ticker, cfg, risk


# ──────────────────────────────────────────────────────────────────────────────
# Cached data fetch — keyed by ticker
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def fetch_raw_data(ticker: str):
    return fetch_asset_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def fit_hmm_cached(ticker: str):
    raw = fetch_raw_data(ticker)
    model, scaler, feat_df, state_map, bull_states, bear_state = fit_hmm(raw)
    with_regimes = predict_regimes(model, scaler, feat_df, state_map, raw, bull_states)
    state_stats  = get_state_stats(model, scaler, state_map, feat_df, raw)
    return with_regimes, state_stats


@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline(ticker: str, cfg: dict, risk: dict):
    raw = fetch_raw_data(ticker)
    sanity_report = run_data_sanity_checks(raw)

    with_regimes, state_stats = fit_hmm_cached(ticker)
    full = compute_indicators(with_regimes, cfg=cfg)
    result_df, trades_df, metrics = run_backtest(
        full,
        ticker                  = ticker,
        stop_loss_pct           = risk["stop_loss_pct"],
        take_profit_pct         = risk["take_profit_pct"],
        min_regime_bars         = risk["min_regime_bars"],
        fee_bps                 = risk.get("fee_bps"),
        slippage_bps            = risk.get("slippage_bps"),
        use_atr_stops           = risk.get("use_atr_stops", False),
        k_stop                  = risk.get("k_stop", 2.0),
        k_tp                    = risk.get("k_tp",   3.0),
        # Phase 3 risk gates
        use_vol_targeting        = risk.get("use_vol_targeting",       False),
        vol_target_pct           = risk.get("vol_target_pct",          30.0),
        vol_target_min_mult      = risk.get("vol_target_min_mult",     0.25),
        vol_target_max_mult      = risk.get("vol_target_max_mult",     1.0),
        kill_switch_enabled      = risk.get("kill_switch_enabled",     False),
        kill_switch_dd_pct       = risk.get("kill_switch_dd_pct",      10.0),
        kill_switch_cooldown_h   = risk.get("kill_switch_cooldown_h",  48),
        use_market_quality_filter= risk.get("use_market_quality_filter", False),
        stress_range_threshold   = risk.get("stress_range_threshold",  0.03),
        stress_force_flat        = risk.get("stress_force_flat",       False),
        stress_cooldown_hours    = risk.get("stress_cooldown_hours",   12),
        use_trailing_stop        = risk.get("use_trailing_stop",       True),
        trailing_stop_pct        = risk.get("trailing_stop_pct",       2.0),
        trail_atr_mult           = risk.get("trail_atr_mult",          1.25),
        trail_activation_pct     = risk.get("trail_activation_pct",    1.5),
        regime_flip_grace_bars   = risk.get("regime_flip_grace_bars",  2),
        use_pbull_sizing         = risk.get("use_pbull_sizing",        False),
    )
    current_signals = get_current_signals(full)
    return result_df, trades_df, metrics, state_stats, current_signals, sanity_report


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Bull":    "rgba(0,200,83,0.12)",
    "Bear":    "rgba(255,82,82,0.12)",
    "Neutral": "rgba(100,100,100,0.06)",
}


def _regime_shapes(df: pd.DataFrame) -> list:
    shapes = []
    if df.empty:
        return shapes
    current_regime = df["regime"].iloc[0]
    start_time = df.index[0]
    for i in range(1, len(df)):
        regime = df["regime"].iloc[i]
        if regime != current_regime or i == len(df) - 1:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=start_time, x1=df.index[i], y0=0, y1=1,
                fillcolor=REGIME_COLORS.get(current_regime, "rgba(0,0,0,0)"),
                line=dict(width=0), layer="below",
            ))
            current_regime = regime
            start_time = df.index[i]
    return shapes


def build_chart(df: pd.DataFrame, trades_df: pd.DataFrame, cfg: dict, ticker: str = "Asset") -> go.Figure:
    cutoff = df.index[-1] - pd.Timedelta(days=90)
    plot_df = df[df.index >= cutoff].copy()

    ema_fast_col = f"ema{cfg['ema_fast']}"
    ema_slow_col = f"ema{cfg['ema_slow']}"
    vol_sma_col  = f"vol_{cfg['volume_sma_period']}_sma"

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.02,
        subplot_titles=(
            f"{ticker} — Regime-Shaded Candlestick",
            f"RSI ({cfg['rsi_period']})",
            "Volume",
        ),
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],   close=plot_df["Close"],
        name=ticker,
        increasing_line_color="#26a641", decreasing_line_color="#f85149",
        increasing_fillcolor="#26a641",  decreasing_fillcolor="#f85149",
    ), row=1, col=1)

    if ema_fast_col in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ema_fast_col],
                                 name=f"EMA {cfg['ema_fast']}",
                                 line=dict(color="#f0a500", width=1), opacity=0.8), row=1, col=1)
    if ema_slow_col in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ema_slow_col],
                                 name=f"EMA {cfg['ema_slow']}",
                                 line=dict(color="#58a6ff", width=1), opacity=0.8), row=1, col=1)

    if not trades_df.empty:
        entries = trades_df[pd.to_datetime(trades_df["Entry Time"]) >= cutoff]
        exits   = trades_df[pd.to_datetime(trades_df["Exit Time"])  >= cutoff]
        if not entries.empty:
            fig.add_trace(go.Scatter(
                x=entries["Entry Time"], y=entries["Entry Price"], mode="markers",
                name="Entry",
                marker=dict(symbol="triangle-up", size=12, color="#69f0ae",
                            line=dict(color="#fff", width=1))), row=1, col=1)
        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=exits["Exit Time"], y=exits["Exit Price"], mode="markers",
                name="Exit",
                marker=dict(symbol="triangle-down", size=12, color="#ff5252",
                            line=dict(color="#fff", width=1))), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi"], name="RSI",
                             line=dict(color="#d2a8ff", width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,82,82,0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,200,83,0.5)",  row=2, col=1)
    fig.add_hline(y=cfg["rsi_max"], line_dash="dot",
                  line_color="rgba(255,255,255,0.35)", row=2, col=1)

    bar_colors = ["#26a641" if c >= o else "#f85149"
                  for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume",
                         marker_color=bar_colors, opacity=0.6), row=3, col=1)
    if vol_sma_col in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[vol_sma_col],
                                 name=f"Vol {cfg['volume_sma_period']}-SMA",
                                 line=dict(color="#ffa657", width=1)), row=3, col=1)

    fig.update_layout(
        shapes=_regime_shapes(plot_df),
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        xaxis3=dict(rangeselector=dict(
            buttons=[
                dict(count=7,  label="7d",  step="day", stepmode="backward"),
                dict(count=30, label="30d", step="day", stepmode="backward"),
                dict(count=90, label="90d", step="day", stepmode="backward"),
                dict(step="all"),
            ],
            bgcolor="#161b22", activecolor="#238636", font=dict(color="#e6edf3"),
        ), type="date"),
    )
    return fig


def build_equity_chart(result_df: pd.DataFrame, title: str = "Equity Curve vs Buy & Hold") -> go.Figure:
    close = result_df["Close"]
    bah   = STARTING_CAPITAL * (close / close.iloc[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df["equity"], name="Strategy",
                             line=dict(color="#69f0ae", width=2),
                             fill="tozeroy", fillcolor="rgba(105,240,174,0.07)"))
    fig.add_trace(go.Scatter(x=result_df.index, y=bah, name="Buy & Hold",
                             line=dict(color="#58a6ff", width=1.5, dash="dash")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=title, font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    return fig


def _regime_css_class(regime_detail: str) -> str:
    if regime_detail == "Bull":
        return "regime-bull"
    elif regime_detail == "Bear":
        return "regime-bear"
    elif regime_detail == "Neutral-HighVol":
        return "regime-neutral-highvol"
    else:
        return "regime-neutral"


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward Tab
# ──────────────────────────────────────────────────────────────────────────────

def render_walk_forward_tab(ticker: str, cfg: dict, risk: dict):
    st.markdown("### Walk-Forward Validation")
    st.markdown(
        '<p class="section-title">Prevents overfitting by testing on unseen data. '
        "Tune using WF OOS results — never on the Lockbox.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_cfg, col_run = st.columns([3, 1])
    with col_cfg:
        wf_col1, wf_col2, wf_col3 = st.columns(3)
        with wf_col1:
            train_window = st.slider("Train Window (days)", 60, 365, 180, 10, key="wf_train")
        with wf_col2:
            test_window  = st.slider("Test Window (days)",  7,  60,   14,  1,  key="wf_test")
        with wf_col3:
            lockbox_pct  = st.slider("Lockbox OOS (%)",    10,  40,   20,  5,  key="wf_lockbox") / 100.0

    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_wf = st.button("Run Walk-Forward", type="primary", use_container_width=True)

    if run_wf:
        from walk_forward import run_walk_forward

        progress_bar  = st.progress(0.0)
        status_holder = st.empty()

        def _cb(pct: float, msg: str):
            progress_bar.progress(min(pct, 1.0))
            status_holder.text(msg)

        try:
            oos_df, fold_metrics, lb_metrics, state_stats = run_walk_forward(
                ticker,
                train_window_days = train_window,
                test_window_days  = test_window,
                lockbox_pct       = lockbox_pct,
                cfg               = cfg,
                risk              = risk,
                progress_callback = _cb,
            )
            st.session_state["wf_results"] = {
                "oos_df":       oos_df,
                "fold_metrics": fold_metrics,
                "lb_metrics":   lb_metrics,
                "state_stats":  state_stats,
                "ticker":       ticker,
                "train_window": train_window,
                "test_window":  test_window,
                "lockbox_pct":  lockbox_pct,
            }
            progress_bar.empty()
            status_holder.empty()
        except Exception as e:
            progress_bar.empty()
            status_holder.empty()
            st.error(f"Walk-forward failed: {e}")
            st.exception(e)
            return

    wf = st.session_state.get("wf_results")
    if wf is None:
        st.info(
            "Configure the parameters above and click **Run Walk-Forward** to start. "
            "This fits a fresh HMM per fold — expect 3–8 minutes for full history."
        )
        return

    fold_metrics = wf["fold_metrics"]
    lb_metrics   = wf["lb_metrics"]
    oos_df       = wf["oos_df"]

    valid_folds = [f for f in fold_metrics if "Error" not in f]
    n_folds     = len(valid_folds)

    st.markdown(f"#### Results — {wf['ticker']}  |  {n_folds} folds  |  "
                f"Train: {wf['train_window']}d  |  Test: {wf['test_window']}d  |  "
                f"Lockbox: {int(wf['lockbox_pct']*100)}%")

    if valid_folds:
        avg_ret    = np.mean([f["Total Return (%)"] for f in valid_folds])
        avg_sharpe = np.mean([f["Sharpe Ratio"]     for f in valid_folds])
        avg_dd     = np.mean([f["Max Drawdown (%)"] for f in valid_folds])
        avg_wr     = np.mean([f["Win Rate (%)"]     for f in valid_folds])
        wf_m1, wf_m2, wf_m3, wf_m4 = st.columns(4)
        for col, label, value in [
            (wf_m1, "Avg Return / Fold",   f"{avg_ret:+.2f}%"),
            (wf_m2, "Avg Sharpe / Fold",   f"{avg_sharpe:.3f}"),
            (wf_m3, "Avg Max DD / Fold",   f"{avg_dd:.2f}%"),
            (wf_m4, "Avg Win Rate / Fold", f"{avg_wr:.1f}%"),
        ]:
            with col:
                st.metric(label=label, value=value)

    st.markdown("---")

    # Per-fold table
    st.markdown("#### Per-Fold Metrics")
    table_cols = ["Fold", "Train Start", "Train End", "Test Start", "Test End",
                  "Trades", "Total Return (%)", "Win Rate (%)", "Max Drawdown (%)",
                  "Sharpe Ratio", "Total Fees ($)"]
    fold_df = pd.DataFrame([{k: v for k, v in f.items() if k in table_cols} for f in valid_folds])
    if not fold_df.empty:
        fmt_map = {c: v for c, v in {
            "Total Return (%)": "{:+.2f}%",
            "Win Rate (%)":     "{:.1f}%",
            "Max Drawdown (%)": "{:.2f}%",
            "Sharpe Ratio":     "{:.3f}",
            "Total Fees ($)":   "${:.2f}",
        }.items() if c in fold_df.columns}
        st.dataframe(
            fold_df.style.format(fmt_map).applymap(
                lambda v: "color: #69f0ae" if isinstance(v, (int, float)) and v > 0 else
                          "color: #ff5252" if isinstance(v, (int, float)) and v < 0 else "",
                subset=[c for c in ["Total Return (%)", "Sharpe Ratio"] if c in fold_df.columns],
            ),
            use_container_width=True, height=300,
        )

    st.markdown("---")

    # Attribution summary
    attr_list = [f.get("attribution", {}) for f in valid_folds if f.get("attribution")]
    if attr_list:
        st.markdown("#### Constraint Attribution (averaged across folds)")
        avg_bull  = np.mean([a.get("pct_time_bull",        0) for a in attr_list])
        avg_sig   = np.mean([a.get("pct_time_signal_ok",   0) for a in attr_list])
        avg_elig  = np.mean([a.get("pct_time_eligible",    0) for a in attr_list])
        tot_cd    = sum([a.get("entries_blocked_cooldown",  0) for a in attr_list])
        tot_gate  = sum([a.get("entries_blocked_gate",      0) for a in attr_list])
        tot_ext   = sum([a.get("entries_blocked_external",  0) for a in attr_list])
        tot_sl    = sum([a.get("exits_stop_loss",           0) for a in attr_list])
        tot_tp    = sum([a.get("exits_take_profit",         0) for a in attr_list])
        tot_rf    = sum([a.get("exits_regime_flip",         0) for a in attr_list])
        tot_ff    = sum([a.get("exits_force_flat",          0) for a in attr_list])
        tot_ks    = sum([a.get("exits_kill_switch",         0) for a in attr_list])

        ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8, ac9 = st.columns(9)
        for col, label, value, color in [
            (ac1, "% Bull",        f"{avg_bull:.1f}%",  "#69f0ae"),
            (ac2, "% Signal OK",   f"{avg_sig:.1f}%",   "#58a6ff"),
            (ac3, "% Eligible",    f"{avg_elig:.1f}%",  "#f0a500"),
            (ac4, "Blk Cooldown",  tot_cd,               "#ff5252"),
            (ac5, "Blk Gate",      tot_gate,             "#ff5252"),
            (ac6, "Blk External",  tot_ext,              "#ff5252"),
            (ac7, "SL Exits",      tot_sl,               "#ff8a80"),
            (ac8, "TP Exits",      tot_tp,               "#69f0ae"),
            (ac9, "FF/KS Exits",   tot_ff + tot_ks,      "#ffa657"),
        ]:
            with col:
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                    f'padding:8px 12px;text-align:center;">'
                    f'<div style="color:#8b949e;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;">{label}</div>'
                    f'<div style="color:{color};font-size:1.1rem;font-weight:800;">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Eligibility waterfall
    wf_list = [f.get("waterfall", {}) for f in valid_folds if f.get("waterfall")]
    if wf_list:
        st.markdown("#### Eligibility Waterfall (last fold)")
        last_wf = wf_list[-1]
        step_labels = {
            "1_total_bars":    "1. Total Bars",
            "2_bull_regime":   "2. Bull Regime",
            "3_signal_ok":     "3. Signal OK (while Bull)",
            "4_eligible":      "4. Eligible (not in cooldown)",
            "5_not_blocked":   "5. Not Blocked by Gates",
            "6_entries_taken": "6. Entries Taken",
        }
        wf_labels = [step_labels.get(k, k) for k in last_wf]
        wf_values = list(last_wf.values())
        total_bars = wf_values[0] if wf_values else 1

        wf_fig = go.Figure(go.Bar(
            x=wf_values, y=wf_labels, orientation="h",
            marker_color=["#58a6ff", "#69f0ae", "#f0a500", "#d2a8ff", "#ffa657", "#ff5252"],
            text=[f"{v:,}  ({v/max(total_bars,1)*100:.1f}%)" for v in wf_values],
            textposition="outside",
        ))
        wf_fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=320, margin=dict(l=10, r=80, t=20, b=10),
            xaxis_title="Bars", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(wf_fig, use_container_width=True)

        wf_table = pd.DataFrame({
            "Step": list(step_labels.get(k, k) for k in last_wf),
            "Bars": wf_values,
            "% of Total": [f"{v/max(total_bars,1)*100:.1f}%" for v in wf_values],
        })
        st.dataframe(wf_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    if not oos_df.empty and "equity" in oos_df.columns:
        st.markdown("#### OOS Equity Curve (All Test Windows Concatenated)")
        st.plotly_chart(build_equity_chart(oos_df, "OOS Equity vs Buy & Hold"), use_container_width=True)

    st.markdown("---")

    # Lockbox OOS
    st.markdown("#### Lockbox OOS (Gold-Standard — View Only, Never Tune To This)")
    if "Error" in lb_metrics:
        st.warning(f"Lockbox evaluation failed: {lb_metrics['Error']}")
    elif lb_metrics:
        lb_period = f"{lb_metrics.get('Period Start', '')} → {lb_metrics.get('Period End', '')}"
        st.caption(f"Period: {lb_period}")
        lb1, lb2, lb3, lb4, lb5, lb6 = st.columns(6)
        for col, label, value in [
            (lb1, "Lockbox Return",   f"{lb_metrics.get('Total Return (%)', 0):+.2f}%"),
            (lb2, "Alpha vs B&H",     f"{lb_metrics.get('Alpha (%)', 0):+.2f}%"),
            (lb3, "Win Rate",         f"{lb_metrics.get('Win Rate (%)', 0):.1f}%"),
            (lb4, "Max Drawdown",     f"{lb_metrics.get('Max Drawdown (%)', 0):.2f}%"),
            (lb5, "Sharpe",           f"{lb_metrics.get('Sharpe Ratio', 0):.3f}"),
            (lb6, "Total Fees",       f"${lb_metrics.get('Total Fees ($)', 0):,.2f}"),
        ]:
            with col:
                st.metric(label=label, value=value)
    else:
        st.info("Lockbox metrics unavailable.")

    st.markdown("---")

    # JSON download
    snapshot_data = {
        "ticker":          wf["ticker"],
        "train_window":    wf["train_window"],
        "test_window":     wf["test_window"],
        "lockbox_pct":     wf["lockbox_pct"],
        "fold_metrics":    fold_metrics,
        "lockbox_metrics": lb_metrics,
    }
    st.download_button(
        "Download JSON Snapshot",
        data=json.dumps(snapshot_data, indent=2, default=str),
        file_name=f"wf_{wf['ticker']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard layout
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ticker, cfg, risk = build_sidebar()
    asset_name = TICKER_TO_NAME.get(ticker, ticker)

    col_title, col_btn, col_time = st.columns([3, 1, 1])
    with col_title:
        st.markdown(f"## HMM Regime-Based Trading System — {asset_name}")
        exit_mode  = "ATR-Scaled" if risk.get("use_atr_stops") else "Fixed %"
        gates_active = []
        if risk.get("kill_switch_enabled"):    gates_active.append("Kill Switch")
        if risk.get("use_market_quality_filter"): gates_active.append("MQ Filter")
        if risk.get("use_vol_targeting"):      gates_active.append("Vol Target")
        gates_str = " · " + " · ".join(gates_active) if gates_active else ""
        st.markdown(
            '<p class="section-title">Hidden Markov Model · 5 States · '
            f'Spot Only · 12h Cooldown · Bucket Voting Gate · {exit_mode} Stops{gates_str}</p>',
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh Data & Refit Model", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_time:
        st.markdown(
            f'<p style="text-align:right;color:#8b949e;margin-top:18px;">'
            f'Last updated<br><b style="color:#e6edf3">'
            f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC</b></p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    tab_main, tab_wf = st.tabs(["Main Dashboard", "Walk-Forward Analysis"])

    with st.spinner(f"Loading {ticker} data · Fitting HMM · Running backtest…"):
        try:
            result_df, trades_df, metrics, state_stats, current_signals, sanity_report = run_pipeline(
                ticker, cfg, risk
            )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.exception(e)
            st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Main Dashboard
    # ─────────────────────────────────────────────────────────────────────────
    with tab_main:

        # Data sanity
        issues = sanity_report.get("issues", [])
        with st.expander(
            f"{'⚠️ Data Quality Warnings' if issues else '✅ Data Quality'} "
            f"— {sanity_report.get('n_rows', 0):,} bars  "
            f"{sanity_report.get('date_range', '')}",
            expanded=bool(issues),
        ):
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Missing rows", f"{sanity_report.get('pct_missing_rows', 0):.2f}%")
            sc2.metric("Zero-volume bars", sanity_report.get("n_zero_volume", 0))
            sc3.metric("Range outliers (>10%)", f"{sanity_report.get('range_outlier_pct', 0):.2f}%")
            sc4.metric("Timezone", sanity_report.get("timezone_utc", "–"))
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("All sanity checks passed.")
            sc5, sc6, sc7 = st.columns(3)
            sc5.metric("Close Min", f"${sanity_report.get('close_min', 0):,.0f}")
            sc6.metric("Close Max", f"${sanity_report.get('close_max', 0):,.0f}")
            sc7.metric("Close Mean", f"${sanity_report.get('close_mean', 0):,.0f}")

        st.markdown("---")

        # Current regime & signal
        current_regime = result_df["regime"].iloc[-1]
        regime_detail  = result_df.get("regime_detail", result_df["regime"]).iloc[-1]
        vote_count     = int(result_df["vote_count"].iloc[-1])
        signal_ok      = bool(result_df["signal_ok"].iloc[-1])
        is_long        = current_regime == "Bull" and signal_ok

        signal_label = "LONG" if is_long else "CASH"
        signal_cls   = "signal-long" if is_long else "signal-cash"
        regime_cls   = _regime_css_class(regime_detail)

        p_bull_val  = current_signals.get("p_bull")
        signals_map = current_signals.get("signals", {})
        buckets_map = current_signals.get("buckets", {})
        pass_rates  = current_signals.get("pass_rates", {})

        col_sig, col_reg, col_conf = st.columns([1.2, 1.5, 2])
        with col_sig:
            st.markdown('<p class="section-title">Current Signal</p>', unsafe_allow_html=True)
            st.markdown(f'<span class="{signal_cls}">{signal_label}</span>', unsafe_allow_html=True)
        with col_reg:
            st.markdown('<p class="section-title">Detected Regime</p>', unsafe_allow_html=True)
            st.markdown(f'<span class="{regime_cls}">{regime_detail}</span>', unsafe_allow_html=True)
        with col_conf:
            if p_bull_val is not None:
                st.markdown('<p class="section-title">Bull Confidence (p_bull)</p>', unsafe_allow_html=True)
                conf_color = "#69f0ae" if p_bull_val >= cfg.get("p_bull_min", 0.55) else "#ff5252"
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                    f'padding:8px 14px;">'
                    f'<span style="color:{conf_color};font-size:1.3rem;font-weight:800;">{p_bull_val:.3f}</span>'
                    f'<span style="color:#8b949e;font-size:0.8rem;margin-left:8px;">'
                    f'({int(p_bull_val*100)}% confidence · threshold {int(cfg.get("p_bull_min",0.55)*100)}%)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Bucket scorecard
        gate_ok    = signal_ok
        gate_color = "#69f0ae" if gate_ok else "#ff5252"
        gate_label = "✅ ENTRY GATE OPEN" if gate_ok else "❌ ENTRY GATE CLOSED"
        st.markdown(
            f'<p class="section-title">Bucket Voting Gate — '
            f'<b style="color:{gate_color}">{gate_label}</b></p>',
            unsafe_allow_html=True,
        )

        if buckets_map:
            bucket_cols = st.columns(4)
            bucket_icons = {"Trend": "📈", "Strength": "💪", "Participation": "📊", "Risk/Cond.": "🛡️"}
            for col_i, (bucket_name, (score, max_score, required)) in enumerate(buckets_map.items()):
                effective_min = min(required, max_score) if max_score > 0 else 0
                passed    = score >= effective_min if max_score > 0 else True
                b_color   = "#69f0ae" if passed else "#ff5252"
                b_icon    = "✅" if passed else "❌"
                icon_label = bucket_icons.get(bucket_name, "")
                with bucket_cols[col_i]:
                    st.markdown(
                        f'<div class="bucket-card">'
                        f'<div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">'
                        f'{icon_label} {bucket_name}</div>'
                        f'<div style="color:{b_color};font-size:1.6rem;font-weight:800;">'
                        f'{b_icon} {score}/{max_score}</div>'
                        f'<div style="color:#8b949e;font-size:0.75rem;">need ≥ {effective_min}'
                        + (' (auto-pass)' if max_score == 0 else '') + '</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Individual signals with historical pass rates
        st.markdown('<p class="section-title" style="margin-top:12px;">Individual Signals '
                    '(value · historical pass rate)</p>', unsafe_allow_html=True)
        sig_cols = st.columns(5)
        for idx, (name, (passed, value)) in enumerate(signals_map.items()):
            icon    = "✅" if passed else "❌"
            color   = "#69f0ae" if passed else "#ff5252"
            rate    = pass_rates.get(name)
            rate_str = f" · {rate:.0f}% hist" if rate is not None else ""
            with sig_cols[idx % 5]:
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;'
                    f'border-radius:8px;padding:7px 10px;margin-bottom:6px;">'
                    f'<span style="color:{color};font-weight:700;font-size:0.8rem;">{icon} {name}</span><br>'
                    f'<span style="color:#8b949e;font-size:0.75rem;">{value}{rate_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Candlestick chart
        st.markdown(f"### {ticker} Price Chart (Last 90 Days) — Regime-Shaded Background")
        st.markdown(
            '<p class="section-title">'
            '🟢 Green = Bull &nbsp;|&nbsp; 🔴 Red = Bear &nbsp;|&nbsp; ⬛ Grey = Neutral</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_chart(result_df, trades_df, cfg, ticker), use_container_width=True)

        st.markdown("---")

        # Performance metrics
        st.markdown("### Performance Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        for col, label, value, help_text in [
            (m1, "Total Return",  f"{metrics['Total Return (%)']:+.2f}%",   f"Start: ${STARTING_CAPITAL:,.0f}"),
            (m2, "Alpha vs B&H",  f"{metrics['Alpha (%)']:+.2f}%",          f"B&H: {metrics['Buy & Hold Return (%)']:+.2f}%"),
            (m3, "Win Rate",      f"{metrics['Win Rate (%)']:.1f}%",         f"{metrics['Total Trades']} trades"),
            (m4, "Max Drawdown",  f"{metrics['Max Drawdown (%)']:.2f}%",     "Peak-to-trough"),
            (m5, "Sharpe Ratio",  f"{metrics['Sharpe Ratio']:.3f}",          "Annualised (hourly)"),
            (m6, "Final Equity",  f"${metrics['Final Equity ($)']:,.0f}",    "Spot position, 1×"),
        ]:
            with col:
                st.metric(label=label, value=value, help=help_text)

        # Phase 3 L: Tail metrics row
        st.markdown('<p class="section-title" style="margin-top:12px;">Tail Risk Metrics (Phase 3)</p>',
                    unsafe_allow_html=True)
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        for col, label, value, color in [
            (t1, "Sortino Ratio",         f"{metrics.get('Sortino Ratio', 0):.3f}",         "#69f0ae"),
            (t2, "CVaR 95% (ann %)",      f"{metrics.get('CVaR 95% (ann %)', 0):.2f}%",     "#ff5252"),
            (t3, "Max Consec Losses",     metrics.get("Max Consec Losses", 0),               "#ff5252"),
            (t4, "Worst Decile Trade",    f"{metrics.get('Worst Decile Trade (%)', 0):.2f}%","#ffa657"),
            (t5, "Large Loss Trades",     metrics.get("Large Loss Trades", 0),               "#ff5252"),
            (t6, "Time to Recovery (h)",  metrics.get("Time-to-Recovery (h)", "N/A"),        "#8b949e"),
        ]:
            with col:
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                    f'padding:10px 14px;text-align:center;">'
                    f'<div style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;">{label}</div>'
                    f'<div style="color:{color};font-size:1.1rem;font-weight:800;">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Exit breakdown
        e1, e2, e3, e4, e5, e6, e7 = st.columns(7)
        stop_mode_label = metrics.get("Exit Mode", "Fixed %")
        exec_ver  = metrics.get("execution_rule_version", "–")
        cfg_hash  = metrics.get("config_hash", "–")
        for col, label, value, color in [
            (e1, "Stop Loss Exits",     metrics["Stop Loss Exits"],                  "#ff5252"),
            (e2, "Trailing Stop Exits", metrics.get("exits_trailing_stop", 0),       "#ffa657"),
            (e3, "Take Profit Exits",   metrics["Take Profit Exits"],                "#69f0ae"),
            (e4, "Regime Flip Exits",   metrics["Regime Flip Exits"],                "#f0a500"),
            (e5, "Total Fees ($)",      f"${metrics['Total Fees ($)']:,.2f}",        "#d2a8ff"),
            (e6, "Stop Mode",           stop_mode_label,                             "#58a6ff"),
            (e7, "Exec Rule",           exec_ver,                                    "#8b949e"),
        ]:
            with col:
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                    f'padding:10px 16px;text-align:center;">'
                    f'<div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">{label}</div>'
                    f'<div style="color:{color};font-size:1.35rem;font-weight:800;">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.caption(f"Config hash: `{cfg_hash}`  ·  Execution rule: `{exec_ver}`")

        st.markdown("---")

        st.plotly_chart(build_equity_chart(result_df), use_container_width=True)
        st.markdown("---")

        # Trade log + HMM state table
        col_trades, col_states = st.columns([3, 2])

        with col_trades:
            st.markdown("### Trade Log")
            if trades_df.empty:
                st.info("No trades executed in the backtest window.")
            else:
                disp = trades_df.copy()
                disp["Entry Time"] = pd.to_datetime(disp["Entry Time"]).dt.strftime("%Y-%m-%d %H:%M")
                disp["Exit Time"]  = pd.to_datetime(disp["Exit Time"]).dt.strftime("%Y-%m-%d %H:%M")

                if "Sanity Pass" in disp.columns:
                    disp["Sanity ✓"] = disp["Sanity Pass"].map({True: "✓", False: "✗"})

                preferred_cols = [
                    "Entry Time", "Exit Time", "Entry Price", "Exit Price",
                    "Return (%)", "PnL ($)",
                    "Fee Entry ($)", "Fee Exit ($)",
                    "Slippage Entry ($)", "Slippage Exit ($)",
                    "Total Cost ($)", "Notional ($)", "Size Mult", "Sanity ✓",
                    "Exit Reason", "Equity After ($)",
                ]
                fallback_cols = [
                    "Entry Time", "Exit Time", "Entry Price", "Exit Price",
                    "Return (%)", "PnL ($)", "Fee ($)", "Exit Reason", "Equity After ($)",
                ]
                has_cost_detail = "Fee Entry ($)" in disp.columns
                display_cols = preferred_cols if has_cost_detail else fallback_cols
                display_cols = [c for c in display_cols if c in disp.columns]
                disp = disp[display_cols]

                def _color_ret(val):
                    return f"color: {'#69f0ae' if val > 0 else '#ff5252'}; font-weight: bold"

                fmt = {
                    "Entry Price":        "${:,.2f}",
                    "Exit Price":         "${:,.2f}",
                    "PnL ($)":            "${:,.2f}",
                    "Equity After ($)":   "${:,.2f}",
                    "Return (%)":         "{:+.2f}%",
                    "Fee ($)":            "${:.2f}",
                    "Fee Entry ($)":      "${:.2f}",
                    "Fee Exit ($)":       "${:.2f}",
                    "Slippage Entry ($)": "${:.2f}",
                    "Slippage Exit ($)":  "${:.2f}",
                    "Total Cost ($)":     "${:.2f}",
                    "Notional ($)":       "${:,.0f}",
                    "Size Mult":          "{:.3f}",
                }
                fmt = {k: v for k, v in fmt.items() if k in disp.columns}

                styled = (
                    disp.style
                    .applymap(_color_ret, subset=["Return (%)", "PnL ($)"])
                    .format(fmt)
                )
                st.dataframe(styled, use_container_width=True, height=340)

                if has_cost_detail:
                    n_fail = (trades_df.get("Sanity Pass", pd.Series([True])) == False).sum()
                    if n_fail == 0:
                        st.caption("✓ Fee sanity checks passed for all trades")
                    else:
                        st.warning(f"⚠ {n_fail} trade(s) failed fee sanity check")

        with col_states:
            st.markdown("### HMM State Summary")
            st.dataframe(
                state_stats.style.format({
                    "Mean Return (%)": "{:+.4f}%",
                    "Std Return (%)":  "{:.4f}%",
                }),
                use_container_width=True, height=340,
            )

        st.markdown("---")
        st.markdown(
            '<p style="color:#484f58;font-size:0.75rem;text-align:center;">'
            "HMM Regime Trading System · Research & educational purposes only · Not financial advice."
            "</p>",
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — Walk-Forward Analysis
    # ─────────────────────────────────────────────────────────────────────────
    with tab_wf:
        render_walk_forward_tab(ticker, cfg, risk)


if __name__ == "__main__":
    main()
