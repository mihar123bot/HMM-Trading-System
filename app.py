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
import copy

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="HMM Regime Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from project modules ────────────────────────────────────────────
from data_loader import fetch_asset_data, run_data_sanity_checks, ASSETS
from hmm_engine import fit_hmm, predict_regimes, get_state_stats
from indicators import compute_indicators, get_current_signals, CONFIG as DEFAULT_CONFIG
from backtester import run_backtest, STARTING_CAPITAL
from ui_components import card, metric_card, pill, section_title, legend as render_legend

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --bg-0: #0d1117;
        --surface-1: #161b22;
        --surface-2: #1c2128;
        --border-1: #30363d;
        --border-2: #21262d;
        --text-1: #e6edf3;
        --text-2: #8b949e;
        --ok: #69f0ae;
        --ok-bg: #1a472a;
        --ok-border: #00c853;
        --bad: #ff8a80;
        --bad-bg: #4a1010;
        --bad-border: #ff5252;
        --warn: #ffa657;
        --warn-bg: #2d2008;
        --warn-border: #f0a500;
        --info: #58a6ff;
        --cost: #d2a8ff;
        --radius-sm: 8px;
        --radius-md: 10px;
        --radius-lg: 12px;
        --sp-1: 4px;
        --sp-2: 8px;
        --sp-3: 10px;
        --sp-4: 12px;
        --sp-5: 16px;
        --fs-xs: 0.72rem;
        --fs-sm: 0.82rem;
        --fs-md: 0.9rem;
        --fs-lg: 1.35rem;
        --fs-xl: 1.95rem;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-0);
        color: var(--text-1);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0.4rem;
    }
    [data-testid="metric-container"] {
        background: var(--surface-1);
        border: 1px solid var(--border-1);
        border-radius: var(--radius-lg);
        padding: var(--sp-5) 20px;
        text-align: center;
    }
    [data-testid="metric-container"] {
        text-align: center;
        align-items: center;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        justify-content: center;
        font-size: 0.86rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"],
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        justify-content: center;
        text-align: center;
        width: 100%;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        justify-content: center;
    }
    .signal-long {
        display:inline-block; padding:8px 28px;
        background:linear-gradient(135deg,var(--ok-border),var(--ok));
        color:#000; font-weight:800; font-size:1.4rem;
        border-radius:50px; letter-spacing:1px;
    }
    .signal-cash {
        display:inline-block;
        min-width:140px;
        text-align:center;
        padding:8px 24px;
        background:var(--bad-bg); color:var(--bad);
        font-weight:700; font-size:1rem;
        border-radius:var(--radius-sm); border:1px solid var(--bad-border);
    }
    .regime-bull {
        display:inline-block; padding:6px 20px;
        background:var(--ok-bg); color:var(--ok);
        font-weight:700; border-radius:var(--radius-sm);
        border:1px solid var(--ok-border);
    }
    .regime-bear {
        display:inline-block; padding:6px 20px;
        background:var(--bad-bg); color:var(--bad);
        font-weight:700; border-radius:var(--radius-sm);
        border:1px solid var(--bad-border);
    }
    .regime-neutral {
        display:inline-block; padding:6px 20px;
        background:var(--surface-2); color:var(--text-2);
        font-weight:700; border-radius:var(--radius-sm);
        border:1px solid var(--border-1);
    }
    .regime-neutral-highvol {
        display:inline-block; padding:6px 20px;
        background:var(--warn-bg); color:var(--warn);
        font-weight:700; border-radius:var(--radius-sm);
        border:1px solid var(--warn-border);
    }
    .section-title {
        color:var(--text-2); font-size:0.75rem;
        text-transform:uppercase; letter-spacing:2px;
        margin-bottom:4px;
    }
    hr { border-color:var(--border-2); }
    [data-testid="stDataFrame"] { border-radius:var(--radius-md); overflow:hidden; }
    [data-testid="stDataFrame"] thead th {
        font-size:14px !important;
        font-weight:700 !important;
    }
    [data-testid="stDataFrame"] tbody td {
        font-size:13px !important;
    }
    /* Streamlit grid renderer (non-table DOM) */
    [data-testid="stDataFrame"] [role="columnheader"] {
        font-size:14px !important;
        line-height:1.1 !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"] {
        font-size:13px !important;
        line-height:1.1 !important;
    }
    .bucket-card {
        background:var(--surface-1); border:1px solid var(--border-1);
        border-radius:var(--radius-md); padding:var(--sp-3) 14px; margin-bottom:6px;
    }
    .summary-metric-card {
        background:var(--surface-1);
        border:1px solid var(--border-1);
        border-radius:var(--radius-md);
        padding:10px 16px;
        height:92px;
        min-height:92px;
        max-height:92px;
        width:100%;
        box-sizing:border-box;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
    }
    .summary-metric-label {
        display:flex;
        align-items:center;
        justify-content:center;
        color:var(--text-2);
        font-size:0.86rem;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:1px;
        line-height:1.1;
        margin-bottom:6px;
    }
    .summary-metric-value {
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:1.35rem;
        font-weight:800;
        line-height:1;
    }
    .summary-metric-sub {
        color:var(--text-2);
        font-size:0.68rem;
        line-height:1.1;
        margin-top:6px;
        text-align:center;
    }
    .plain-metric-block {
        text-align:center;
        padding:4px 0;
    }
    .plain-metric-label {
        color:var(--text-2);
        font-size:0.86rem;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:1px;
        line-height:1.1;
        margin-bottom:6px;
    }
    .plain-metric-value {
        color:var(--text-1);
        font-size:2.15rem;
        font-weight:800;
        line-height:1;
    }
    .plain-metric-sub {
        color:var(--text-2);
        font-size:0.8rem;
        line-height:1.1;
        margin-top:6px;
    }
    .wf-metric-card {
        background:var(--surface-1);
        border:1px solid var(--border-1);
        border-radius:var(--radius-md);
        padding:10px 12px;
        height:92px;
        min-height:92px;
        max-height:92px;
        width:100%;
        box-sizing:border-box;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
    }
    .wf-metric-label {
        color:var(--text-2);
        font-size:0.86rem;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:1px;
        line-height:1.1;
        margin-bottom:6px;
    }
    .wf-metric-value {
        font-size:1.95rem;
        font-weight:800;
        line-height:1;
    }
    .main-takeaway-grid {
        display:grid;
        grid-template-columns:repeat(3, minmax(0, 1fr));
        justify-content:center;
        gap:10px;
        margin:0 auto 4px auto;
        width:100%;
    }
    .main-takeaway-card {
        background:var(--surface-1);
        border:1px solid var(--border-1);
        border-radius:var(--radius-lg);
        padding:10px 12px;
        min-height:88px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
    }
    .main-takeaway-title {
        color:var(--text-2);
        font-size:0.86rem;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:1px;
        margin-bottom:6px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.08rem;
        font-weight:700;
    }
    div[data-testid="stNumberInput"] {
        width: 100%;
    }
    @media (max-width: 900px) {
        .main-takeaway-grid { grid-template-columns:1fr; }
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
    st.sidebar.toggle(
        "Run quick calibration after refit",
        value=False,
        key="run_quick_cal_after_refit",
        help="Optional: after refresh/refit, run 20-trial 1-seed calibration and auto-apply.",
    )
    if st.sidebar.button("Refresh Data & Refit Model", use_container_width=True):
        st.session_state["force_auto_setup"] = True
        st.cache_data.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Strategy Parameters")
    st.sidebar.markdown("All changes trigger an instant backtest rerun.")
    st.sidebar.markdown("---")

    cfg  = {}
    risk = {}

    strategy_mode = st.sidebar.selectbox("Mode", ["Locked", "Research"], index=0, key="strategy_mode")
    if strategy_mode == "Locked":
        st.sidebar.caption("Strategy controls are locked. Use Calibration Lab to tune parameters.")
    else:
        st.sidebar.caption("Research mode: a small set of controls is unlocked.")

    # Strategy defaults (locked in sidebar)
    cfg["trend_min"] = int(DEFAULT_CONFIG["trend_min"])
    cfg["risk_min"] = int(DEFAULT_CONFIG["risk_min"])
    cfg["p_bull_min"] = float(DEFAULT_CONFIG["p_bull_min"])
    cfg["rsi_max"] = int(DEFAULT_CONFIG["rsi_max"])
    cfg["momentum_min_pct"] = float(DEFAULT_CONFIG["momentum_min_pct"])
    cfg["volatility_max_pct"] = float(DEFAULT_CONFIG["volatility_max_pct"])
    cfg["ema_fast"] = int(DEFAULT_CONFIG["ema_fast"])
    cfg["ema_slow"] = int(DEFAULT_CONFIG["ema_slow"])

    cfg["strength_min"] = 0
    cfg["participation_min"] = 0
    cfg["sig_adx_on"] = False
    cfg["sig_volume_on"] = False
    cfg["sig_ema_fast_on"] = DEFAULT_CONFIG.get("sig_ema_fast_on", True)
    cfg["sig_ema_slow_on"] = DEFAULT_CONFIG.get("sig_ema_slow_on", True)
    cfg["sig_macd_on"] = DEFAULT_CONFIG.get("sig_macd_on", True)
    cfg["sig_roc_on"] = DEFAULT_CONFIG.get("sig_roc_on", True)
    cfg["sig_pbull_slope_on"] = DEFAULT_CONFIG.get("sig_pbull_slope_on", True)
    cfg["sig_rsi_on"] = DEFAULT_CONFIG.get("sig_rsi_on", True)
    cfg["sig_volatility_on"] = DEFAULT_CONFIG.get("sig_volatility_on", True)
    cfg["sig_momentum_on"] = DEFAULT_CONFIG.get("sig_momentum_on", True)
    cfg["sig_confidence_on"] = DEFAULT_CONFIG.get("sig_confidence_on", True)

    cfg["adx_min"] = int(DEFAULT_CONFIG["adx_min"])
    cfg["rsi_period"] = int(DEFAULT_CONFIG["rsi_period"])
    cfg["momentum_period"] = int(DEFAULT_CONFIG["momentum_period"])
    cfg["volume_sma_period"] = int(DEFAULT_CONFIG["volume_sma_period"])
    cfg["volatility_period"] = int(DEFAULT_CONFIG["volatility_period"])
    cfg["adx_period"] = int(DEFAULT_CONFIG["adx_period"])
    cfg["atr_period"] = int(DEFAULT_CONFIG["atr_period"])
    cfg["roc_period"] = int(DEFAULT_CONFIG["roc_period"])
    cfg["p_bull_slope_period"] = int(DEFAULT_CONFIG["p_bull_slope_period"])
    cfg["macd_fast"] = int(DEFAULT_CONFIG["macd_fast"])
    cfg["macd_slow"] = int(DEFAULT_CONFIG["macd_slow"])
    cfg["macd_signal"] = int(DEFAULT_CONFIG["macd_signal"])

    # Entry style defaults
    cfg["entry_mode"] = "Hybrid"
    cfg["mr_down_bars"] = 2
    cfg["mr_short_drop_pct"] = 0.4
    cfg["mr_bounce_rsi_max"] = 45

    # Risk defaults
    risk["stop_loss_pct"] = -2.5
    risk["take_profit_pct"] = 4.0
    risk["min_regime_bars"] = 2
    risk["regime_flip_grace_bars"] = 2
    risk["use_pbull_sizing"] = True
    risk["fee_bps"] = int(DEFAULT_CONFIG.get("fee_bps", 5))
    risk["slippage_bps"] = int(DEFAULT_CONFIG.get("slippage_bps", 3))

    risk["use_atr_stops"] = False
    risk["k_stop"] = 2.0
    risk["k_tp"] = 3.0
    risk["use_trailing_stop"] = True
    risk["trail_atr_mult"] = 2.5
    risk["trail_activation_pct"] = 1.5
    risk["trailing_stop_pct"] = 2.0

    risk["kill_switch_enabled"] = DEFAULT_CONFIG.get("kill_switch_enabled", True)
    risk["kill_switch_dd_pct"] = float(DEFAULT_CONFIG.get("kill_switch_dd_pct", 9.0))
    risk["kill_switch_cooldown_h"] = int(DEFAULT_CONFIG.get("kill_switch_cooldown_h", 24))
    risk["use_market_quality_filter"] = DEFAULT_CONFIG.get("use_market_quality_filter", True)
    risk["stress_force_flat"] = DEFAULT_CONFIG.get("stress_force_flat", True)
    risk["stress_range_threshold"] = float(DEFAULT_CONFIG.get("stress_range_threshold", 0.04))
    cfg["stress_range_threshold"] = risk["stress_range_threshold"]
    risk["stress_cooldown_hours"] = int(DEFAULT_CONFIG.get("stress_cooldown_hours", 24))
    risk["use_vol_targeting"] = DEFAULT_CONFIG.get("vol_targeting_enabled", False)
    risk["vol_target_pct"] = float(DEFAULT_CONFIG.get("vol_target_pct", 30.0))
    risk["vol_target_min_mult"] = float(DEFAULT_CONFIG.get("vol_target_min_mult", 0.25))
    risk["vol_target_max_mult"] = float(DEFAULT_CONFIG.get("vol_target_max_mult", 1.0))

    if strategy_mode == "Research":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Research Controls")
        cfg["entry_mode"] = st.sidebar.selectbox("Entry Style", ["Trend", "Mean Reversion", "Hybrid"], index=2)
        cfg["p_bull_min"] = st.sidebar.slider("Min Bull confidence", 0.40, 0.90, float(cfg["p_bull_min"]), 0.01)
        cfg["trend_min"] = st.sidebar.slider("Min Trend signals", 0, 5, int(cfg["trend_min"]), 1)
        cfg["risk_min"] = st.sidebar.slider("Min Risk/Conditioning signals", 0, 4, int(cfg["risk_min"]), 1)
        with st.sidebar.expander("Mean Reversion", expanded=False):
            cfg["mr_down_bars"] = st.sidebar.slider("Consecutive down bars", 1, 3, int(cfg["mr_down_bars"]), 1)
            cfg["mr_short_drop_pct"] = st.sidebar.slider("Pullback size (%)", 0.1, 2.0, float(cfg["mr_short_drop_pct"]), 0.1)
            cfg["mr_bounce_rsi_max"] = st.sidebar.slider("RSI cap", 25, 60, int(cfg["mr_bounce_rsi_max"]), 1)
        with st.sidebar.expander("Risk", expanded=False):
            sl_abs = abs(float(risk["stop_loss_pct"]))
            risk["stop_loss_pct"] = -st.sidebar.slider("Stop Loss (%)", 0.5, 10.0, sl_abs, 0.5)
            risk["take_profit_pct"] = st.sidebar.slider("Take Profit (%)", 1.0, 12.0, float(risk["take_profit_pct"]), 0.5)
            risk["kill_switch_dd_pct"] = st.sidebar.slider("Kill Switch DD (%)", 4.0, 20.0, float(risk["kill_switch_dd_pct"]), 0.5)
            risk["stress_range_threshold"] = st.sidebar.slider("Stress threshold", 0.01, 0.10, float(risk["stress_range_threshold"]), 0.005)
            cfg["stress_range_threshold"] = risk["stress_range_threshold"]

    return ticker, cfg, risk


# ──────────────────────────────────────────────────────────────────────────────
# Cached data fetch — keyed by ticker
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def fetch_raw_data(ticker: str):
    return fetch_asset_data(ticker)


@st.cache_data(ttl=300, show_spinner=False, persist="disk")
def fetch_live_btc_price() -> float | None:
    try:
        df = fetch_asset_data("BTC-USD", days=3)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def fit_hmm_cached(ticker: str):
    raw = fetch_raw_data(ticker)
    model, scaler, feat_df, state_map, bull_states, bear_state = fit_hmm(raw)
    with_regimes = predict_regimes(model, scaler, feat_df, state_map, raw, bull_states)
    state_stats  = get_state_stats(model, scaler, state_map, feat_df, raw)
    return with_regimes, state_stats


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
        kill_switch_enabled      = risk.get("kill_switch_enabled",     True),
        kill_switch_dd_pct       = risk.get("kill_switch_dd_pct",      9.0),
        kill_switch_cooldown_h   = risk.get("kill_switch_cooldown_h",  24),
        use_market_quality_filter= risk.get("use_market_quality_filter", True),
        stress_range_threshold   = risk.get("stress_range_threshold",  0.04),
        stress_force_flat        = risk.get("stress_force_flat",       True),
        stress_cooldown_hours    = risk.get("stress_cooldown_hours",   24),
        use_trailing_stop        = risk.get("use_trailing_stop",       True),
        trailing_stop_pct        = risk.get("trailing_stop_pct",       2.0),
        trail_atr_mult           = risk.get("trail_atr_mult",          2.5),
        trail_activation_pct     = risk.get("trail_activation_pct",    1.5),
        regime_flip_grace_bars   = risk.get("regime_flip_grace_bars",  2),
        use_pbull_sizing         = risk.get("use_pbull_sizing",        False),
        entry_mode               = cfg.get("entry_mode", "Hybrid"),
        mr_down_bars             = cfg.get("mr_down_bars", 2),
        mr_bounce_rsi_max        = cfg.get("mr_bounce_rsi_max", 45.0),
        mr_short_drop_pct        = cfg.get("mr_short_drop_pct", 0.4),
        p_bull_min               = cfg.get("p_bull_min", 0.55),
    )
    current_signals = get_current_signals(full)
    return result_df, trades_df, metrics, state_stats, current_signals, sanity_report


def _run_backtest_from_regimes(with_regimes: pd.DataFrame, ticker: str, cfg: dict, risk: dict):
    full = compute_indicators(with_regimes, cfg=cfg)
    return run_backtest(
        full,
        ticker                   = ticker,
        stop_loss_pct            = risk["stop_loss_pct"],
        take_profit_pct          = risk["take_profit_pct"],
        min_regime_bars          = risk["min_regime_bars"],
        fee_bps                  = risk.get("fee_bps"),
        slippage_bps             = risk.get("slippage_bps"),
        use_atr_stops            = risk.get("use_atr_stops", False),
        k_stop                   = risk.get("k_stop", 2.0),
        k_tp                     = risk.get("k_tp",   3.0),
        use_vol_targeting        = risk.get("use_vol_targeting", False),
        vol_target_pct           = risk.get("vol_target_pct", 30.0),
        vol_target_min_mult      = risk.get("vol_target_min_mult", 0.25),
        vol_target_max_mult      = risk.get("vol_target_max_mult", 1.0),
        kill_switch_enabled      = risk.get("kill_switch_enabled", True),
        kill_switch_dd_pct       = risk.get("kill_switch_dd_pct", 9.0),
        kill_switch_cooldown_h   = risk.get("kill_switch_cooldown_h", 24),
        use_market_quality_filter= risk.get("use_market_quality_filter", True),
        stress_range_threshold   = risk.get("stress_range_threshold", 0.04),
        stress_force_flat        = risk.get("stress_force_flat", True),
        stress_cooldown_hours    = risk.get("stress_cooldown_hours", 24),
        use_trailing_stop        = risk.get("use_trailing_stop", True),
        trailing_stop_pct        = risk.get("trailing_stop_pct", 2.0),
        trail_atr_mult           = risk.get("trail_atr_mult", 2.5),
        trail_activation_pct     = risk.get("trail_activation_pct", 1.5),
        regime_flip_grace_bars   = risk.get("regime_flip_grace_bars", 2),
        use_pbull_sizing         = risk.get("use_pbull_sizing", False),
        entry_mode               = cfg.get("entry_mode", "Hybrid"),
        mr_down_bars             = cfg.get("mr_down_bars", 2),
        mr_bounce_rsi_max        = cfg.get("mr_bounce_rsi_max", 45.0),
        mr_short_drop_pct        = cfg.get("mr_short_drop_pct", 0.4),
        p_bull_min               = cfg.get("p_bull_min", 0.55),
    )


CALIBRATION_SPACE = {
    "p_bull_min": (0.45, 0.80, 0.01, "float"),
    "trend_min": (1, 5, 1, "int"),
    "risk_min": (1, 4, 1, "int"),
    "rsi_max": (45, 80, 1, "int"),
    "momentum_min_pct": (0.0, 5.0, 0.1, "float"),
    "volatility_max_pct": (20.0, 150.0, 1.0, "float"),
    "ema_fast": (5, 60, 1, "int"),
    "ema_slow": (40, 300, 5, "int"),
}
CALIBRATION_RISK_SPACE = {
    "stop_loss_pct": (-6.0, -1.0, 0.5, "float"),
    "take_profit_pct": (2.0, 8.0, 0.5, "float"),
    "min_regime_bars": (1, 6, 1, "int"),
    "regime_flip_grace_bars": (0, 4, 1, "int"),
    "kill_switch_dd_pct": (6.0, 15.0, 0.5, "float"),
    "stress_range_threshold": (0.02, 0.08, 0.005, "float"),
}

def _sample_cfg_risk(base_cfg: dict, base_risk: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    c = copy.deepcopy(base_cfg)
    r = copy.deepcopy(base_risk)
    for k, (lo, hi, step, typ) in CALIBRATION_SPACE.items():
        v = rng.choice(np.arange(lo, hi + step, step, dtype=int)) if typ == "int" else round(rng.uniform(lo, hi) / step) * step
        c[k] = int(v) if typ == "int" else float(v)
    for k, (lo, hi, step, typ) in CALIBRATION_RISK_SPACE.items():
        v = rng.choice(np.arange(lo, hi + step, step, dtype=int)) if typ == "int" else round(rng.uniform(lo, hi) / step) * step
        r[k] = int(v) if typ == "int" else float(v)
    if c["ema_fast"] >= c["ema_slow"]:
        c["ema_slow"] = min(300, c["ema_fast"] + 20)
    return c, r

def _score_metrics(m: dict, base_cfg: dict, cand_cfg: dict, base_risk: dict, cand_risk: dict, sample_bars: int) -> float:
    sharpe = float(m.get("Sharpe Ratio", 0.0)); total_return = float(m.get("Total Return (%)", 0.0)); max_dd = abs(float(m.get("Max Drawdown (%)", 0.0))); trades = float(m.get("Total Trades", 0.0))
    drift = 0.0
    for k, (lo, hi, *_rest) in CALIBRATION_SPACE.items():
        drift += abs(float(cand_cfg.get(k, base_cfg.get(k, lo))) - float(base_cfg.get(k, lo))) / max(float(hi - lo), 1e-9)
    for k, (lo, hi, *_rest) in CALIBRATION_RISK_SPACE.items():
        drift += abs(float(cand_risk.get(k, base_risk.get(k, lo))) - float(base_risk.get(k, lo))) / max(float(hi - lo), 1e-9)

    # Trade penalty normalized to sample length (target trades per 1000 bars)
    target_trades = max(8.0, (max(sample_bars, 1) / 1000.0) * 25.0)
    trade_penalty = ((target_trades - trades) * 0.05) if trades < target_trades else 0.0

    return (1.8 * sharpe) + (0.03 * total_return) - (0.05 * max_dd) - (0.30 * drift) - trade_penalty

def _calibrate_on_regimes(with_regimes: pd.DataFrame, ticker: str, cfg: dict, risk: dict, trials: int, seed: int, progress_cb=None):
    _, _, baseline_metrics = _run_backtest_from_regimes(with_regimes, ticker, cfg, risk)
    sample_bars = len(with_regimes)
    best_cfg, best_risk, best_metrics = copy.deepcopy(cfg), copy.deepcopy(risk), baseline_metrics
    best_score = _score_metrics(baseline_metrics, cfg, cfg, risk, risk, sample_bars)
    rng = np.random.default_rng(seed)
    for i in range(trials):
        cand_cfg, cand_risk = _sample_cfg_risk(cfg, risk, rng)
        _, _, cand_metrics = _run_backtest_from_regimes(with_regimes, ticker, cand_cfg, cand_risk)
        cand_score = _score_metrics(cand_metrics, cfg, cand_cfg, risk, cand_risk, sample_bars)
        if cand_score > best_score:
            best_score, best_cfg, best_risk, best_metrics = cand_score, cand_cfg, cand_risk, cand_metrics
        if progress_cb is not None:
            progress_cb(i + 1, trials, seed)
    return {"seed": int(seed), "best_cfg": best_cfg, "best_risk": best_risk, "best_metrics": best_metrics, "best_score": float(best_score)}

def _aggregate_cfg_risk_from_runs(base_cfg: dict, base_risk: dict, runs: list[dict]) -> tuple[dict, dict]:
    aligned_cfg = copy.deepcopy(base_cfg); aligned_risk = copy.deepcopy(base_risk)
    for k, (lo, hi, step, typ) in CALIBRATION_SPACE.items():
        vals = [float(r["best_cfg"].get(k, base_cfg.get(k, lo))) for r in runs]
        snapped = round(min(max(float(np.median(vals)), float(lo)), float(hi)) / step) * step
        aligned_cfg[k] = int(snapped) if typ == "int" else float(snapped)
    for k, (lo, hi, step, typ) in CALIBRATION_RISK_SPACE.items():
        vals = [float(r["best_risk"].get(k, base_risk.get(k, lo))) for r in runs]
        snapped = round(min(max(float(np.median(vals)), float(lo)), float(hi)) / step) * step
        aligned_risk[k] = int(snapped) if typ == "int" else float(snapped)
    if aligned_cfg["ema_fast"] >= aligned_cfg["ema_slow"]:
        aligned_cfg["ema_slow"] = int(min(300, aligned_cfg["ema_fast"] + 20))
    return aligned_cfg, aligned_risk

def _seed_agreement_stats(runs: list[dict]) -> tuple[float, list[dict]]:
    if not runs:
        return 0.0, []
    per_param_agreement = []
    agreement_values = []
    for k, (lo, hi, _step, _typ) in CALIBRATION_SPACE.items():
        vals = [float(r.get("best_cfg", {}).get(k, lo)) for r in runs]
        span = max(vals) - min(vals) if vals else 0.0
        width = max(float(hi - lo), 1e-9)
        disagreement = span / width
        agreement = max(0.0, min(1.0, 1.0 - disagreement))
        agreement_values.append(agreement)
        per_param_agreement.append({
            "Parameter": k,
            "Agreement": f"{agreement*100:.1f}%",
            "Seed Range": f"{min(vals):.3g} → {max(vals):.3g}",
        })
    overall_agreement = float(np.mean(agreement_values)) if agreement_values else 0.0
    return overall_agreement, per_param_agreement


def run_calibration_multi_seed(ticker: str, cfg: dict, risk: dict, trials: int = 60, seeds: list[int] | None = None, progress_cb=None):
    seeds = seeds or [42, 123, 777]
    with_regimes, _ = fit_hmm_cached(ticker)
    _, _, baseline_metrics = _run_backtest_from_regimes(with_regimes, ticker, cfg, risk)
    runs = []
    for seed_idx, s in enumerate(seeds):
        def _seed_progress(done, total, seed_val):
            if progress_cb is not None:
                progress_cb((seed_idx * total) + done, len(seeds) * total, seed_val, seed_idx + 1, len(seeds))
        runs.append(_calibrate_on_regimes(with_regimes, ticker, cfg, risk, trials, s, progress_cb=_seed_progress))
    aligned_cfg, aligned_risk = _aggregate_cfg_risk_from_runs(cfg, risk, runs)
    _, _, aligned_metrics = _run_backtest_from_regimes(with_regimes, ticker, aligned_cfg, aligned_risk)
    aligned_score = _score_metrics(aligned_metrics, cfg, aligned_cfg, risk, aligned_risk, len(with_regimes))
    return {"baseline_cfg": cfg, "baseline_risk": risk, "best_cfg": aligned_cfg, "best_risk": aligned_risk, "baseline_metrics": baseline_metrics, "best_metrics": aligned_metrics, "best_score": float(aligned_score), "trials": int(trials), "seeds": [int(s) for s in seeds], "runs": runs, "generated_at_utc": datetime.utcnow().isoformat()}


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Bull":    "rgba(0,200,83,0.12)",
    "Bear":    "rgba(255,82,82,0.12)",
    "Neutral": "rgba(100,100,100,0.06)",
}

CHART_BG = "#0d1117"
CHART_GRID = "rgba(139,148,158,0.18)"
CHART_FONT = "#e6edf3"
CHART_LEGEND = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)


def _regime_shapes(df: pd.DataFrame) -> list:
    shapes = []
    if df.empty:
        return shapes

    current_regime = df["regime"].iloc[0]
    start_time = df.index[0]

    for i in range(1, len(df)):
        regime = df["regime"].iloc[i]
        if regime != current_regime:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=start_time, x1=df.index[i], y0=0, y1=1,
                fillcolor=REGIME_COLORS.get(current_regime, "rgba(0,0,0,0)"),
                line=dict(width=0), layer="below",
            ))
            current_regime = regime
            start_time = df.index[i]

    # Append final segment explicitly
    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=start_time, x1=df.index[-1], y0=0, y1=1,
        fillcolor=REGIME_COLORS.get(current_regime, "rgba(0,0,0,0)"),
        line=dict(width=0), layer="below",
    ))

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
            "",
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
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        legend={**CHART_LEGEND, "font": dict(size=11, color=CHART_FONT)},
        font=dict(color=CHART_FONT, size=12),
        xaxis_rangeslider_visible=False,
        xaxis3=dict(rangeselector=dict(
            buttons=[
                dict(count=7,  label="7d",  step="day", stepmode="backward"),
                dict(count=30, label="30d", step="day", stepmode="backward"),
                dict(count=90, label="90d", step="day", stepmode="backward"),
                dict(step="all"),
            ],
            bgcolor="#161b22", activecolor="#238636", font=dict(color=CHART_FONT),
        ), type="date"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False)
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
        template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=title, font=dict(size=13)),
        legend={**CHART_LEGEND, "font": dict(size=11, color=CHART_FONT)},
        font=dict(color=CHART_FONT, size=12),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False)
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


def render_metric_color_legend():
    """Render a compact legend for dashboard metric colors."""
    render_legend()


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
    st.divider()

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
    render_metric_color_legend()

    if valid_folds:
        avg_ret    = np.mean([f["Total Return (%)"] for f in valid_folds])
        avg_sharpe = np.mean([f["Sharpe Ratio"]     for f in valid_folds])
        avg_dd     = np.mean([f["Max Drawdown (%)"] for f in valid_folds])
        avg_wr     = np.mean([f["Win Rate (%)"]     for f in valid_folds])
        wf_m1, wf_m2, wf_m3, wf_m4 = st.columns(4)
        for col, label, value, color in [
            (wf_m1, "Avg Return / Fold",   f"{avg_ret:+.2f}%", "#69f0ae" if avg_ret >= 0 else "#ff5252"),
            (wf_m2, "Avg Win Rate / Fold", f"{avg_wr:.1f}%", "#69f0ae"),
            (wf_m3, "Avg Sharpe / Fold",   f"{avg_sharpe:.3f}", "#58a6ff"),
            (wf_m4, "Avg Max DD / Fold",   f"{avg_dd:.2f}%", "#ff8a80"),
        ]:
            with col:
                st.markdown(
                    metric_card(
                        label=label,
                        value=value,
                        value_color=color,
                        label_class="wf-metric-label",
                        value_class="wf-metric-value",
                        card_class="wf-metric-card",
                    ),
                    unsafe_allow_html=True,
                )

    st.divider()

    # Lockbox OOS (moved near top summary)
    snapshot_data = {
        "ticker":          wf["ticker"],
        "train_window":    wf["train_window"],
        "test_window":     wf["test_window"],
        "lockbox_pct":     wf["lockbox_pct"],
        "fold_metrics":    fold_metrics,
        "lockbox_metrics": lb_metrics,
    }
    lb_title_col, lb_btn_col = st.columns([3, 1])
    with lb_title_col:
        st.markdown("#### Lockbox OOS (Gold-Standard — View Only, Never Tune To This)")
    with lb_btn_col:
        st.download_button(
            "Download JSON Snapshot",
            data=json.dumps(snapshot_data, indent=2, default=str),
            file_name=f"wf_{wf['ticker']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    if "Error" in lb_metrics:
        st.warning(f"Lockbox evaluation failed: {lb_metrics['Error']}")
    elif lb_metrics:
        lb_period = f"{lb_metrics.get('Period Start', '')} → {lb_metrics.get('Period End', '')}"
        st.caption(f"Period: {lb_period}")
        lb1, lb2, lb3, lb4, lb5, lb6 = st.columns(6)
        for col, label, value, color in [
            (lb1, "Lockbox Return",   f"{lb_metrics.get('Total Return (%)', 0):+.2f}%", "#69f0ae" if lb_metrics.get('Total Return (%)', 0) >= 0 else "#ff5252"),
            (lb2, "Win Rate",         f"{lb_metrics.get('Win Rate (%)', 0):.1f}%", "#69f0ae"),
            (lb3, "Sharpe",           f"{lb_metrics.get('Sharpe Ratio', 0):.3f}", "#58a6ff"),
            (lb4, "Max Drawdown",     f"{lb_metrics.get('Max Drawdown (%)', 0):.2f}%", "#ff8a80"),
            (lb5, "Alpha vs B&H",     f"{lb_metrics.get('Alpha (%)', 0):+.2f}%", "#58a6ff"),
            (lb6, "Total Fees",       f"${lb_metrics.get('Total Fees ($)', 0):,.2f}", "#d2a8ff"),
        ]:
            with col:
                st.markdown(
                    metric_card(
                        label=label,
                        value=value,
                        value_color=color,
                        label_class="wf-metric-label",
                        value_class="wf-metric-value",
                        card_class="wf-metric-card",
                    ),
                    unsafe_allow_html=True,
                )
    else:
        st.info("Lockbox metrics unavailable.")

    st.divider()

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

    st.divider()

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
                    f'padding:10px 12px;text-align:center;">'
                    f'<div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">{label}</div>'
                    f'<div style="color:{color};font-size:1.55rem;font-weight:800;">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

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

    st.divider()

    if not oos_df.empty and "equity" in oos_df.columns:
        st.markdown("#### OOS Equity Curve (All Test Windows Concatenated)")
        st.plotly_chart(build_equity_chart(oos_df, "OOS Equity vs Buy & Hold"), use_container_width=True)

    st.divider()

def render_calibration_tab(ticker: str, cfg: dict, risk: dict):
    st.markdown("### Calibration Lab")
    st.caption("Optimize a constrained parameter set on recent data, then inspect exactly what changed.")

    cal = st.session_state.get("last_calibration")
    cal_runs = cal.get("runs", []) if cal else []
    overall_agreement_top, _ = _seed_agreement_stats(cal_runs) if cal_runs else (0.0, [])
    low_agreement = bool(cal_runs) and overall_agreement_top < 0.70

    apply_anyway = False
    if low_agreement:
        st.warning(f"Low seed agreement ({overall_agreement_top*100:.1f}%). Applying calibrated params may be unstable.")
        apply_anyway = st.checkbox("Apply anyway (danger)", value=False, key="cal_apply_anyway")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("Trials")
        trials = st.number_input("Trials Value", min_value=20, max_value=200, value=40, step=10, key="cal_trials", label_visibility="collapsed")
    with c2:
        s1a, s1b = st.columns([5, 1])
        with s1a:
            st.markdown("Seed 1")
        with s1b:
            use_seed_1 = st.checkbox("Use Seed 1", value=True, key="cal_use_seed_1", label_visibility="collapsed")
        seed_1 = st.number_input("Seed 1 Value", min_value=1, max_value=99999, value=42, step=1, key="cal_seed_1", label_visibility="collapsed")
    with c3:
        s2a, s2b = st.columns([5, 1])
        with s2a:
            st.markdown("Seed 2")
        with s2b:
            use_seed_2 = st.checkbox("Use Seed 2", value=True, key="cal_use_seed_2", label_visibility="collapsed")
        seed_2 = st.number_input("Seed 2 Value", min_value=1, max_value=99999, value=123, step=1, key="cal_seed_2", label_visibility="collapsed")
    with c4:
        s3a, s3b = st.columns([5, 1])
        with s3a:
            st.markdown("Seed 3")
        with s3b:
            use_seed_3 = st.checkbox("Use Seed 3", value=True, key="cal_use_seed_3", label_visibility="collapsed")
        seed_3 = st.number_input("Seed 3 Value", min_value=1, max_value=99999, value=777, step=1, key="cal_seed_3", label_visibility="collapsed")
    with c5:
        st.markdown("Apply")
        apply_clicked = st.button(
            "Apply Calibrated Parameters",
            use_container_width=True,
            disabled=(cal is None) or (low_agreement and not apply_anyway),
        )

    if apply_clicked and cal is not None:
        st.session_state["applied_calibration_cfg"] = copy.deepcopy(cal["best_cfg"])
        st.session_state["applied_calibration_risk"] = copy.deepcopy(cal.get("best_risk", {}))
        st.session_state["last_calibration_apply_ts"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        st.success("Applied calibrated parameters to the active run.")
        st.rerun()

    run_clicked = st.button("Run Calibration", type="primary", use_container_width=True)

    if run_clicked:
        selected_seeds = []
        if use_seed_1:
            selected_seeds.append(int(seed_1))
        if use_seed_2:
            selected_seeds.append(int(seed_2))
        if use_seed_3:
            selected_seeds.append(int(seed_3))

        if not selected_seeds:
            st.warning("Select at least one seed to run calibration.")
            return

        progress_bar = st.progress(0)
        progress_text = st.empty()
        seed_count = len(selected_seeds)

        def _progress(done, total, seed_val, seed_num, seed_total):
            pct = int((done / max(total, 1)) * 100)
            progress_bar.progress(min(max(pct, 0), 100))
            progress_text.caption(
                f"Calibrating with {seed_count} seed{'s' if seed_count != 1 else ''} (this can take several minutes) — Seed {seed_num}/{seed_total} [{seed_val}] · Overall progress: {done}/{total}"
            )

        with st.spinner(f"Calibrating with {seed_count} seed{'s' if seed_count != 1 else ''} (this can take several minutes) and aligning parameters..."):
            cal = run_calibration_multi_seed(
                ticker,
                cfg,
                risk,
                trials=int(trials),
                seeds=selected_seeds,
                progress_cb=_progress,
            )
            st.session_state["last_calibration"] = cal
            st.session_state["last_calibration_run_ts"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            progress_bar.progress(100)
            progress_text.caption("Calibration complete.")
            progress_bar.empty()
            progress_text.empty()
            st.rerun()

    cal = st.session_state.get("last_calibration")
    if not cal:
        st.info("No calibration run yet. Click **Run Calibration**.")
        return

    base_m = cal["baseline_metrics"]
    best_m = cal["best_metrics"]
    base_cfg = cal["baseline_cfg"]
    best_cfg = cal["best_cfg"]
    base_risk = cal.get("baseline_risk", risk)
    best_risk = cal.get("best_risk", risk)

    runs = cal.get("runs", [])
    if runs:
        run_rows = []
        for r in runs:
            rm = r.get("best_metrics", {})
            run_rows.append({
                "Seed": r.get("seed"),
                "Sharpe": round(float(rm.get("Sharpe Ratio", 0.0)), 3),
                "Return (%)": round(float(rm.get("Total Return (%)", 0.0)), 2),
                "Max DD (%)": round(float(rm.get("Max Drawdown (%)", 0.0)), 2),
                "Score": round(float(r.get("best_score", 0.0)), 3),
            })
        st.markdown("#### Multi-Seed Run Summary")
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)

        # Seed agreement score (parameter stability across runs)
        overall_agreement, per_param_agreement = _seed_agreement_stats(runs)
        if overall_agreement >= 0.85:
            band = "High"
            band_color = "#69f0ae"
        elif overall_agreement >= 0.70:
            band = "Medium"
            band_color = "#ffa657"
        else:
            band = "Low"
            band_color = "#ff5252"

        st.markdown("#### Seed Agreement")
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;'>"
            f"<div style='color:#8b949e;font-size:0.82rem;text-transform:uppercase;letter-spacing:1px;'>Overall Agreement</div>"
            f"<div style='color:{band_color};font-size:1.4rem;font-weight:800;'>{overall_agreement*100:.1f}% ({band})</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(per_param_agreement), use_container_width=True, hide_index=True)

    st.markdown("#### Before vs After (Key Metrics)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", f"{best_m['Total Return (%)']:+.2f}%", delta=f"{best_m['Total Return (%)'] - base_m['Total Return (%)']:+.2f}%")
    m2.metric("Sharpe", f"{best_m['Sharpe Ratio']:.3f}", delta=f"{best_m['Sharpe Ratio'] - base_m['Sharpe Ratio']:+.3f}")
    m3.metric("Max DD", f"{best_m['Max Drawdown (%)']:.2f}%", delta=f"{best_m['Max Drawdown (%)'] - base_m['Max Drawdown (%)']:+.2f}%")
    m4.metric("Trades", int(best_m.get("Total Trades", 0)), delta=int(best_m.get("Total Trades", 0) - base_m.get("Total Trades", 0)))

    rows = []
    for k in CALIBRATION_SPACE.keys():
        before = base_cfg.get(k)
        after = best_cfg.get(k)
        delta = (after - before) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
        rows.append({"Category": "Signal", "Parameter": k, "Before": before, "After": after, "Delta": round(delta, 4) if delta is not None else "-", "Changed": "Yes" if before != after else "No"})
    for k in CALIBRATION_RISK_SPACE.keys():
        before = base_risk.get(k)
        after = best_risk.get(k)
        delta = (after - before) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
        rows.append({"Category": "Risk", "Parameter": k, "Before": before, "After": after, "Delta": round(delta, 4) if delta is not None else "-", "Changed": "Yes" if before != after else "No"})

    improved = (best_m.get("Sharpe Ratio", 0) > base_m.get("Sharpe Ratio", 0)) and (
        best_m.get("Max Drawdown (%)", 0) >= base_m.get("Max Drawdown (%)", 0) - 1.0
    )
    if improved:
        st.success("Calibration verdict: ✅ Improved profile (higher Sharpe without materially worse drawdown).")
    else:
        st.warning("Calibration verdict: ⚠ Mixed result. Review changes before adopting.")

    # Executive-style calibration commentary (3–5 bullets)
    sharpe_delta = float(best_m.get("Sharpe Ratio", 0) - base_m.get("Sharpe Ratio", 0))
    dd_delta = float(best_m.get("Max Drawdown (%)", 0) - base_m.get("Max Drawdown (%)", 0))
    ret_delta = float(best_m.get("Total Return (%)", 0) - base_m.get("Total Return (%)", 0))
    trades_delta = int(best_m.get("Total Trades", 0) - base_m.get("Total Trades", 0))

    commentary = []
    commentary.append(
        f"Performance profile: Return {ret_delta:+.2f}pp, Sharpe {sharpe_delta:+.3f}, Max DD {dd_delta:+.2f}pp versus baseline."
    )

    if best_cfg.get("p_bull_min", cfg.get("p_bull_min", 0)) > base_cfg.get("p_bull_min", cfg.get("p_bull_min", 0)):
        commentary.append("Signal quality tightened: higher confidence threshold favors cleaner entries over frequency.")
    elif best_cfg.get("p_bull_min", cfg.get("p_bull_min", 0)) < base_cfg.get("p_bull_min", cfg.get("p_bull_min", 0)):
        commentary.append("Signal quality loosened: lower confidence threshold increases participation and may raise churn.")

    if best_cfg.get("trend_min", cfg.get("trend_min", 0)) > base_cfg.get("trend_min", cfg.get("trend_min", 0)):
        commentary.append("Trend confirmation increased: setup now requires broader trend agreement before entry.")
    elif best_cfg.get("trend_min", cfg.get("trend_min", 0)) < base_cfg.get("trend_min", cfg.get("trend_min", 0)):
        commentary.append("Trend confirmation relaxed: faster entries, but with higher whipsaw risk in noisy regimes.")

    if best_cfg.get("volatility_max_pct", cfg.get("volatility_max_pct", 0)) < base_cfg.get("volatility_max_pct", cfg.get("volatility_max_pct", 0)):
        commentary.append("Risk posture is tighter: lower volatility cap filters unstable market periods.")
    elif best_cfg.get("volatility_max_pct", cfg.get("volatility_max_pct", 0)) > base_cfg.get("volatility_max_pct", cfg.get("volatility_max_pct", 0)):
        commentary.append("Risk posture is more permissive: wider volatility cap allows more opportunities in fast markets.")

    commentary.append(
        f"Execution intensity: trade count changed by {trades_delta:+d}; validate this aligns with desired turnover and costs."
    )

    st.markdown("#### Executive Commentary")
    for line in commentary[:5]:
        st.markdown(f"- {line}")

    st.markdown("#### Summary of parameter changes")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Calibration Snapshot (JSON)",
        data=json.dumps(cal, indent=2, default=str),
        file_name=f"calibration_{ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard layout
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ticker, cfg, risk = build_sidebar()

    force_auto_setup = bool(st.session_state.get("force_auto_setup", False))
    should_auto_setup = force_auto_setup

    if should_auto_setup:
        run_quick_cal = bool(st.session_state.get("run_quick_cal_after_refit", False))
        splash = st.empty()
        progress = st.progress(0)
        status = st.empty()

        def _render_splash(step_title: str, step_detail: str):
            splash.markdown(
                f"""
                <style>
                  @keyframes opiGlow {{ 0% {{ box-shadow: 0 0 0 rgba(105,240,174,0.0); }} 50% {{ box-shadow: 0 0 24px rgba(105,240,174,0.25); }} 100% {{ box-shadow: 0 0 0 rgba(105,240,174,0.0); }} }}
                  @keyframes opiScan {{ 0% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(220%); }} }}
                  @keyframes opiTicker {{ 0% {{ transform: translateX(0%); }} 100% {{ transform: translateX(-50%); }} }}
                  @keyframes btcRain {{ 0% {{ transform: translateY(-18px); opacity:0; }} 15% {{ opacity:0.75; }} 100% {{ transform: translateY(140px); opacity:0; }} }}
                  .opi-boot-wrap {{
                    position: relative;
                    overflow: hidden;
                    background: linear-gradient(120deg,#0f172a 0%, #111827 50%, #1f2937 100%);
                    border: 1px solid #30363d;
                    border-radius: 14px;
                    padding: 16px 18px;
                    margin: 6px 0 8px 0;
                    animation: opiGlow 2.2s ease-in-out infinite;
                  }}
                  .opi-boot-wrap:before {{
                    content: "";
                    position: absolute;
                    top:0; left:0;
                    width: 40%; height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(88,166,255,0.18), transparent);
                    animation: opiScan 2.4s linear infinite;
                  }}
                  .opi-topline {{ color:#8b949e;font-size:0.74rem;text-transform:uppercase;letter-spacing:1.2px; }}
                  .opi-rain {{ position:absolute; inset:0; pointer-events:none; z-index:0; }}
                  .opi-rain span {{ position:absolute; top:-20px; color:rgba(240,165,0,0.55); font-size:14px; animation: btcRain linear infinite; }}
                  .opi-rain span:nth-child(1) {{ left:6%;  animation-duration:2.6s; animation-delay:0.1s; }}
                  .opi-rain span:nth-child(2) {{ left:16%; animation-duration:3.1s; animation-delay:0.7s; }}
                  .opi-rain span:nth-child(3) {{ left:27%; animation-duration:2.4s; animation-delay:1.2s; }}
                  .opi-rain span:nth-child(4) {{ left:39%; animation-duration:3.4s; animation-delay:0.4s; }}
                  .opi-rain span:nth-child(5) {{ left:52%; animation-duration:2.9s; animation-delay:1.4s; }}
                  .opi-rain span:nth-child(6) {{ left:64%; animation-duration:3.2s; animation-delay:0.2s; }}
                  .opi-rain span:nth-child(7) {{ left:76%; animation-duration:2.7s; animation-delay:1.0s; }}
                  .opi-rain span:nth-child(8) {{ left:88%; animation-duration:3.0s; animation-delay:0.6s; }}
                  .opi-title {{ color:#e6edf3;font-size:1.15rem;font-weight:800;margin-top:4px; position:relative; z-index:1; }}
                  .opi-step {{ color:#69f0ae;font-size:0.95rem;font-weight:700;margin-top:8px; position:relative; z-index:1; }}
                  .opi-detail {{ color:#c9d1d9;font-size:0.88rem;margin-top:3px; position:relative; z-index:1; }}
                  .opi-meme-row {{ margin-top:10px;white-space:nowrap;overflow:hidden;border-top:1px solid rgba(139,148,158,0.2);padding-top:8px;line-height:1.35; position:relative; z-index:1; }}
                  .opi-meme-track {{ display:inline-block; min-width:200%; animation: opiTicker 6s linear infinite; color:#f0a500; font-size:0.82rem; }}
                </style>
                <div class='opi-boot-wrap'>
                  <div class='opi-rain'><span>₿</span><span>₿</span><span>₿</span><span>₿</span><span>₿</span><span>₿</span><span>₿</span><span>₿</span></div>
                  <div class='opi-topline'>Boot Sequence · Quant Degens Online 🚀</div>
                  <div class='opi-title'>⚡ Opi Startup Workflow</div>
                  <div class='opi-step'>{step_title}</div>
                  <div class='opi-detail'>{step_detail}</div>
                  <div class='opi-meme-row'>
                    <div class='opi-meme-track'>🐂 BULL MODE LOADING · 💎🙌 HOLD THE EDGE · 📈 GREEN CANDLES ONLY (IDEALLY) · 🧠 CALIBRATING ALPHA · 🚀 TO THE MOON? (RISK-MANAGED) · 🐂 BULL MODE LOADING · 💎🙌 HOLD THE EDGE · 📈 GREEN CANDLES ONLY (IDEALLY) · 🧠 CALIBRATING ALPHA · 🚀 TO THE MOON? (RISK-MANAGED) · 🐂 BULL MODE LOADING · 💎🙌 HOLD THE EDGE · 📈 GREEN CANDLES ONLY (IDEALLY) · 🧠 CALIBRATING ALPHA · 🚀 TO THE MOON? (RISK-MANAGED) · 🐂 BULL MODE LOADING · 💎🙌 HOLD THE EDGE · 📈 GREEN CANDLES ONLY (IDEALLY) · 🧠 CALIBRATING ALPHA · 🚀 TO THE MOON? (RISK-MANAGED) ·</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 1) Refresh data + refit model
        _render_splash("Refreshing market data", f"Refetching and refitting model for {ticker}...")
        progress.progress(15)
        status.caption(f"Step 1/{3 if run_quick_cal else 2} · Data refresh + model refit")
        fetch_raw_data(ticker)
        fit_hmm_cached(ticker)
        ts_now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        st.session_state["last_data_refresh_ts"] = ts_now
        st.session_state["last_model_refit_ts"] = ts_now

        # 2) Pull live BTC price (cached)
        _render_splash("Pulling BTC live price", "Getting latest BTC snapshot for header card...")
        progress.progress(50 if not run_quick_cal else 40)
        status.caption(f"Step 2/{3 if run_quick_cal else 2} · Pull BTC live price")
        fetch_live_btc_price()

        # 3) Optional lightweight calibration and apply
        if run_quick_cal:
            random_seed = int(np.random.randint(1, 100000))
            _render_splash("Running quick calibration", f"20 trials · 1 random seed ({random_seed}) · then auto-apply")
            progress.progress(70)
            status.caption("Step 3/3 · Calibration + apply")
            cal = run_calibration_multi_seed(
                ticker,
                copy.deepcopy(cfg),
                copy.deepcopy(risk),
                trials=20,
                seeds=[random_seed],
            )
            st.session_state["last_calibration"] = cal
            st.session_state["last_calibration_run_ts"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            st.session_state["applied_calibration_cfg"] = copy.deepcopy(cal["best_cfg"])
            st.session_state["applied_calibration_risk"] = copy.deepcopy(cal.get("best_risk", {}))
            st.session_state["last_calibration_apply_ts"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        st.session_state["force_auto_setup"] = False

        progress.progress(100)
        status.caption("Startup workflow complete.")
        splash.empty()
        progress.empty()
        status.empty()

    applied_cfg = st.session_state.get("applied_calibration_cfg")
    if applied_cfg:
        cfg.update(applied_cfg)
    applied_risk = st.session_state.get("applied_calibration_risk")
    if applied_risk:
        risk.update(applied_risk)

    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown(
            f'<h2 style="margin:0 0 0.35rem 0; padding:0; font-size:2.0rem; text-align:left;">HMM Regime-Based Trading System — {ticker}</h2>',
            unsafe_allow_html=True,
        )
        last_refresh = st.session_state.get("last_data_refresh_ts", "Never")
        last_refit = st.session_state.get("last_model_refit_ts", "Never")
        last_cal_run = st.session_state.get("last_calibration_run_ts", "Never")
        last_cal_apply = st.session_state.get("last_calibration_apply_ts", "Never")
        st.caption(
            f"Data refresh: {last_refresh}  •  Model refit: {last_refit}  •  Calibration run: {last_cal_run}  •  Calibration applied: {last_cal_apply}"
        )
        if last_cal_apply != "Never":
            st.markdown(
                "<span style='display:inline-block;padding:4px 10px;border:1px solid #2ea043;border-radius:999px;color:#69f0ae;font-size:0.8rem;'>Startup workflow completed</span>",
                unsafe_allow_html=True,
            )
    with header_right:
        btc_live = fetch_live_btc_price()
        btc_pull_ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        if btc_live is not None:
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                f'padding:8px 10px;text-align:center;">'
                f'<div style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;">'
                f'BTC-USD Live Price</div>'
                f'<div style="color:#69f0ae;font-size:1.45rem;font-weight:800;">${btc_live:,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                'padding:8px 10px;">'
                '<div style="color:#8b949e;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;">'
                'BTC-USD Live Price</div>'
                '<div style="color:#8b949e;font-size:0.95rem;font-weight:700;">Unavailable</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Live price pulled: {btc_pull_ts}")

    tab_main, tab_wf, tab_cal = st.tabs(["Decision", "Validation", "Calibration"])

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
        # Current regime & signal
        current_regime = result_df["regime"].iloc[-1]
        regime_detail  = result_df.get("regime_detail", result_df["regime"]).iloc[-1]
        signal_ok      = bool(result_df["signal_ok"].iloc[-1])
        is_long        = current_regime == "Bull" and signal_ok

        signal_label = "LONG" if is_long else "CASH"
        signal_cls   = "signal-long" if is_long else "signal-cash"
        regime_cls   = _regime_css_class(regime_detail)

        p_bull_val  = current_signals.get("p_bull")
        signals_map = current_signals.get("signals", {})
        buckets_map = current_signals.get("buckets", {})
        pass_rates  = current_signals.get("pass_rates", {})

        if p_bull_val is not None:
            conf_color = "#69f0ae" if p_bull_val >= cfg.get("p_bull_min", 0.55) else "#ff5252"
            p_bull_pct = p_bull_val * 100
            threshold_pct = cfg.get("p_bull_min", 0.55) * 100
            confidence_html = (
                f'<span style="color:{conf_color};font-size:1.35rem;font-weight:800;">{p_bull_pct:.2f}%</span>'
                f'<span style="color:#8b949e;font-size:0.78rem;margin-left:8px;">'
                f'(threshold {threshold_pct:.2f}%)</span>'
            )
        else:
            confidence_html = '<span style="color:#8b949e;font-size:1.0rem;font-weight:700;">Unavailable</span>'

        # First-principles decision summary (above signal/regime/confidence cards)
        blockers = [name for name, (ok, _) in signals_map.items() if not ok]

        def _blocker_bucket(name: str) -> int:
            trend_keys = ("Price > EMA", "MACD > Signal", "ROC(", "p_bull Slope")
            risk_keys = ("RSI <", "Volatility <", "Momentum >", "HMM Confidence")
            if any(k in name for k in trend_keys):
                return 0
            if any(k in name for k in risk_keys):
                return 1
            return 2

        blockers_sorted = sorted(
            blockers,
            key=lambda n: (_blocker_bucket(n), float(pass_rates.get(n, 1000.0)), n),
        )

        if is_long:
            st.success("Decision: LONG — Entry conditions are satisfied.")
        else:
            top_blockers = blockers_sorted[:3] if blockers_sorted else ["Regime / confidence gate"]
            st.warning("Decision: CASH — Top blockers: " + ", ".join(top_blockers))

        signal_card = card(
            f'<div class="main-takeaway-title">Current Signal</div><div>{pill(signal_label, signal_cls)}</div>',
            class_name="main-takeaway-card",
        )
        regime_card = card(
            f'<div class="main-takeaway-title">Detected Regime</div><div>{pill(regime_detail, regime_cls)}</div>',
            class_name="main-takeaway-card",
        )
        confidence_card = card(
            f'<div class="main-takeaway-title">Bull Confidence</div><div>{confidence_html}</div>',
            class_name="main-takeaway-card",
        )
        st.markdown(
            f'<div style="display:flex;justify-content:center;width:100%;"><div class="main-takeaway-grid">{signal_card}{regime_card}{confidence_card}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        gate_ok = signal_ok
        gate_text = "OPEN" if gate_ok else "CLOSED"

        total_pnl = float(metrics.get("Final Equity ($)", STARTING_CAPITAL) - STARTING_CAPITAL)
        bars_24h = 24 if len(result_df) > 24 else max(len(result_df) - 1, 1)
        pnl_24h = float(result_df["equity"].iloc[-1] - result_df["equity"].iloc[-bars_24h]) if "equity" in result_df.columns else 0.0
        open_trades = 1 if is_long else 0

        if len(result_df):
            ts = pd.to_datetime(result_df.index[-1], utc=True).tz_convert("America/New_York")
            last_ts = ts.strftime("%b %d, %Y • %I:%M %p ET")
        else:
            last_ts = "—"

        total_pnl_str = f"{'+' if total_pnl >= 0 else '-'}${abs(total_pnl):,.2f}"
        total_ret_str = f"{metrics.get('Total Return (%)', 0):+.2f}%"
        pnl_24h_str = f"{'+' if pnl_24h >= 0 else '-'}${abs(pnl_24h):,.2f}"
        gate_color = "#69f0ae" if gate_ok else "#ff8a80"

        st.markdown("### Live Trading Overview")
        o1, o2, o3, o4, o5 = st.columns(5)

        with o1:
            st.markdown(metric_card("TOTAL PNL", total_pnl_str, "#69f0ae" if total_pnl >= 0 else "#ff5252", subtext=total_ret_str), unsafe_allow_html=True)
        with o2:
            st.markdown(metric_card("PNL (24H)", pnl_24h_str, "#69f0ae" if pnl_24h >= 0 else "#ff5252"), unsafe_allow_html=True)
        with o3:
            st.markdown(metric_card("OPEN TRADES", open_trades, "#58a6ff"), unsafe_allow_html=True)
        with o4:
            st.markdown(metric_card("ENTRY GATE", gate_text, gate_color), unsafe_allow_html=True)
        with o5:
            st.markdown(metric_card("LAST UPDATE", last_ts, "#8b949e", value_class="summary-metric-sub", subtext="Local time"), unsafe_allow_html=True)

        st.markdown("### Equity")
        st.plotly_chart(build_equity_chart(result_df), use_container_width=True)

        st.markdown("### Recent Trade History")
        if trades_df.empty:
            st.info("No completed trades yet.")
        else:
            recent_cols = [c for c in ["Entry Time", "Exit Time", "Entry Price", "Exit Price", "Return (%)", "PnL ($)", "Exit Reason"] if c in trades_df.columns]
            recent = trades_df[recent_cols].copy().tail(10)
            if "Entry Time" in recent.columns:
                recent["Entry Time"] = pd.to_datetime(recent["Entry Time"]).dt.strftime("%Y-%m-%d %H:%M")
            if "Exit Time" in recent.columns:
                recent["Exit Time"] = pd.to_datetime(recent["Exit Time"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(recent, use_container_width=True, hide_index=True)

        with st.expander("Advanced diagnostics", expanded=False):
            st.markdown(f"### {ticker} Price Chart — Regime-Shaded Background")
            st.plotly_chart(build_chart(result_df, trades_df, cfg, ticker), use_container_width=True)
            st.markdown("### HMM State Summary")
            cleaned_state_stats = state_stats.dropna(how="all").copy()
            st.dataframe(cleaned_state_stats, use_container_width=True, height=260)

        with st.expander("Validation snapshot (rigor checks)", expanded=False):
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("Sharpe", f"{metrics.get('Sharpe Ratio', 0):.3f}")
            b2.metric("Max DD", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
            b3.metric("Win Rate", f"{metrics.get('Win Rate (%)', 0):.1f}%")
            b4.metric("Trades", int(metrics.get("Total Trades", 0)))
            b5.metric("Alpha", f"{metrics.get('Alpha (%)', 0):+.2f}%")
            b6.metric("Sortino", f"{metrics.get('Sortino Ratio', 0):.3f}")

            trend_score = buckets_map.get("Trend", (0, 0, 0))
            risk_score = buckets_map.get("Risk/Cond.", (0, 0, 0))
            st.markdown("**Validation Notes**")
            st.markdown(f"- Trend bucket: {trend_score}")
            st.markdown(f"- Risk/Condition bucket: {risk_score}")
            issues = sanity_report.get("issues", [])
            if issues:
                st.markdown(f"- Data issues: {len(issues)} flagged")
                for issue in issues[:3]:
                    st.markdown(f"  - {issue}")
            else:
                st.markdown("- Data issues: none")

        st.divider()
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

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — Calibration
    # ─────────────────────────────────────────────────────────────────────────
    with tab_cal:
        render_calibration_tab(ticker, cfg, risk)


if __name__ == "__main__":
    main()
