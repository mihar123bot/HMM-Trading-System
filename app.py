"""
app.py
Streamlit dashboard for the HMM Regime-Based BTC Trading System.

Layout
------
Sidebar : Live CONFIG sliders (all thresholds & periods)
Header  : title + last-updated timestamp
Row 1   : Current Signal pill + Current Regime badge + Vote scorecard
Row 2   : Interactive Plotly candlestick (background shaded by regime)
Row 3   : Performance metrics (Total Return, Alpha, Win Rate, Max DD, Sharpe)
Row 4   : Equity curve vs Buy & Hold
Row 5   : Trade log table + HMM State statistics table
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
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
from data_loader import fetch_asset_data, ASSETS, TICKER_TO_NAME
from hmm_engine import fit_hmm, predict_regimes, get_state_stats
from indicators import compute_indicators, get_current_signals, CONFIG as DEFAULT_CONFIG
from backtester import run_backtest, STARTING_CAPITAL

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
    .section-title {
        color:#8b949e; font-size:0.75rem;
        text-transform:uppercase; letter-spacing:2px;
        margin-bottom:4px;
    }
    hr { border-color:#21262d; }
    [data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — live CONFIG editor
# ──────────────────────────────────────────────────────────────────────────────

def build_sidebar() -> tuple[str, dict]:
    """Render sidebar controls and return (ticker, cfg)."""
    st.sidebar.markdown("## Asset Selection")
    asset_name = st.sidebar.selectbox(
        "Select Asset",
        options=list(ASSETS.keys()),
        index=0,
    )
    ticker = ASSETS[asset_name]
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Strategy Parameters")
    st.sidebar.markdown("All changes trigger an instant backtest rerun.")
    st.sidebar.markdown("---")

    cfg = {}

    st.sidebar.markdown("### Voting Gate")
    cfg["votes_required"] = st.sidebar.slider("Min Votes Required (out of 8)", 1, 8, DEFAULT_CONFIG["votes_required"], 1)

    st.sidebar.markdown("### Entry Thresholds")
    cfg["rsi_max"]            = st.sidebar.slider("RSI Max (< threshold)",  50, 100, DEFAULT_CONFIG["rsi_max"],                          1)
    cfg["momentum_min_pct"]   = st.sidebar.slider("Min Momentum (%)",      0.0, 10.0, float(DEFAULT_CONFIG["momentum_min_pct"]),        0.1)
    cfg["volatility_max_pct"] = st.sidebar.slider("Max Volatility (%)",    1.0, 20.0, float(DEFAULT_CONFIG["volatility_max_pct"]),      0.5)
    cfg["adx_min"]            = st.sidebar.slider("Min ADX",                10,  50, DEFAULT_CONFIG["adx_min"],                          1)

    st.sidebar.markdown("### Indicator Periods")
    cfg["rsi_period"]        = st.sidebar.slider("RSI Period",             5,  50, DEFAULT_CONFIG["rsi_period"],        1)
    cfg["momentum_period"]   = st.sidebar.slider("Momentum Period (bars)", 2,  50, DEFAULT_CONFIG["momentum_period"],   1)
    cfg["volume_sma_period"] = st.sidebar.slider("Volume SMA Period",      5, 100, DEFAULT_CONFIG["volume_sma_period"], 1)
    cfg["volatility_period"] = st.sidebar.slider("Volatility Window (bars)", 6, 168, DEFAULT_CONFIG["volatility_period"], 1)
    cfg["adx_period"]        = st.sidebar.slider("ADX Period",             5,  50, DEFAULT_CONFIG["adx_period"],        1)
    cfg["ema_fast"]          = st.sidebar.slider("EMA Fast Period",        5, 100, DEFAULT_CONFIG["ema_fast"],          1)
    cfg["ema_slow"]          = st.sidebar.slider("EMA Slow Period",       20, 500, DEFAULT_CONFIG["ema_slow"],          5)

    st.sidebar.markdown("### MACD Settings")
    cfg["macd_fast"]   = st.sidebar.slider("MACD Fast",   2,  50, DEFAULT_CONFIG["macd_fast"],   1)
    cfg["macd_slow"]   = st.sidebar.slider("MACD Slow",   5, 100, DEFAULT_CONFIG["macd_slow"],   1)
    cfg["macd_signal"] = st.sidebar.slider("MACD Signal", 2,  30, DEFAULT_CONFIG["macd_signal"], 1)

    st.sidebar.markdown("---")
    st.sidebar.caption("Default config is defined in `indicators.py → CONFIG`.")
    return ticker, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Cached data fetch — keyed by ticker
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_raw_data(ticker: str):
    return fetch_asset_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def fit_hmm_cached(ticker: str):
    """HMM fit is expensive — cache by ticker (TTL matches fetch cache)."""
    raw = fetch_raw_data(ticker)
    model, scaler, feat_df, state_map, bull_state, bear_state = fit_hmm(raw)
    with_regimes = predict_regimes(model, scaler, feat_df, state_map, raw)
    state_stats = get_state_stats(model, scaler, state_map, feat_df, raw)
    return with_regimes, state_stats


def run_pipeline(ticker: str, cfg: dict):
    """Run indicators + backtest with the given ticker and config."""
    with_regimes, state_stats = fit_hmm_cached(ticker)
    full = compute_indicators(with_regimes, cfg=cfg)
    result_df, trades_df, metrics = run_backtest(full)
    current_signals = get_current_signals(full)
    return result_df, trades_df, metrics, state_stats, current_signals


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
            shapes.append(
                dict(
                    type="rect", xref="x", yref="paper",
                    x0=start_time, x1=df.index[i],
                    y0=0, y1=1,
                    fillcolor=REGIME_COLORS.get(current_regime, "rgba(0,0,0,0)"),
                    line=dict(width=0), layer="below",
                )
            )
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

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"],   close=plot_df["Close"],
            name=ticker,
            increasing_line_color="#26a641", decreasing_line_color="#f85149",
            increasing_fillcolor="#26a641",  decreasing_fillcolor="#f85149",
        ),
        row=1, col=1,
    )

    if ema_fast_col in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df[ema_fast_col],
                       name=f"EMA {cfg['ema_fast']}",
                       line=dict(color="#f0a500", width=1), opacity=0.8),
            row=1, col=1,
        )
    if ema_slow_col in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df[ema_slow_col],
                       name=f"EMA {cfg['ema_slow']}",
                       line=dict(color="#58a6ff", width=1), opacity=0.8),
            row=1, col=1,
        )

    if not trades_df.empty:
        entries = trades_df[pd.to_datetime(trades_df["Entry Time"]) >= cutoff]
        exits   = trades_df[pd.to_datetime(trades_df["Exit Time"]) >= cutoff]
        if not entries.empty:
            fig.add_trace(go.Scatter(
                x=entries["Entry Time"], y=entries["Entry Price"], mode="markers",
                name="Entry",
                marker=dict(symbol="triangle-up", size=12, color="#69f0ae",
                            line=dict(color="#fff", width=1))),
                row=1, col=1)
        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=exits["Exit Time"], y=exits["Exit Price"], mode="markers",
                name="Exit",
                marker=dict(symbol="triangle-down", size=12, color="#ff5252",
                            line=dict(color="#fff", width=1))),
                row=1, col=1)

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["rsi"], name="RSI",
                   line=dict(color="#d2a8ff", width=1)),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,82,82,0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,200,83,0.5)",  row=2, col=1)
    fig.add_hline(y=cfg["rsi_max"], line_dash="dot",
                  line_color="rgba(255,255,255,0.35)", row=2, col=1)

    bar_colors = ["#26a641" if c >= o else "#f85149"
                  for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume",
               marker_color=bar_colors, opacity=0.6),
        row=3, col=1,
    )
    if vol_sma_col in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df[vol_sma_col],
                       name=f"Vol {cfg['volume_sma_period']}-SMA",
                       line=dict(color="#ffa657", width=1)),
            row=3, col=1,
        )

    fig.update_layout(
        shapes=_regime_shapes(plot_df),
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        xaxis3=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7,  label="7d",  step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(count=90, label="90d", step="day", stepmode="backward"),
                    dict(step="all"),
                ],
                bgcolor="#161b22", activecolor="#238636",
                font=dict(color="#e6edf3"),
            ),
            type="date",
        ),
    )
    return fig


def build_equity_chart(result_df: pd.DataFrame) -> go.Figure:
    close = result_df["Close"]
    bah   = STARTING_CAPITAL * (close / close.iloc[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df.index, y=result_df["equity"], name="Strategy",
        line=dict(color="#69f0ae", width=2),
        fill="tozeroy", fillcolor="rgba(105,240,174,0.07)"))
    fig.add_trace(go.Scatter(
        x=result_df.index, y=bah, name="Buy & Hold",
        line=dict(color="#58a6ff", width=1.5, dash="dash")))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Equity Curve vs Buy & Hold", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard layout
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    ticker, cfg = build_sidebar()
    asset_name = TICKER_TO_NAME.get(ticker, ticker)

    # ── Header ───────────────────────────────────────────────────────────────
    col_title, col_time = st.columns([3, 1])
    with col_title:
        st.markdown(f"## HMM Regime-Based Trading System — {asset_name}")
        st.markdown(
            '<p class="section-title">Hidden Markov Model · 7 States · '
            f'2.5× Leverage · 48h Cooldown · {cfg["votes_required"]}/8 votes required</p>',
            unsafe_allow_html=True,
        )
    with col_time:
        st.markdown(
            f'<p style="text-align:right;color:#8b949e;margin-top:18px;">'
            f'Last updated<br><b style="color:#e6edf3">'
            f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC</b></p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker} data · Fitting HMM · Running backtest…"):
        try:
            result_df, trades_df, metrics, state_stats, current_signals = run_pipeline(ticker, cfg)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.exception(e)
            st.stop()

    # ── Current regime & signal ──────────────────────────────────────────────
    current_regime = result_df["regime"].iloc[-1]
    vote_count     = int(result_df["vote_count"].iloc[-1])
    signal_ok      = bool(result_df["signal_ok"].iloc[-1])
    is_long        = current_regime == "Bull" and signal_ok

    signal_label = "LONG" if is_long else "CASH"
    signal_cls   = "signal-long" if is_long else "signal-cash"
    regime_cls   = (
        "regime-bull" if current_regime == "Bull"
        else "regime-bear" if current_regime == "Bear"
        else "regime-neutral"
    )

    col_sig, col_reg, col_votes = st.columns([1.5, 1.5, 4])

    with col_sig:
        st.markdown('<p class="section-title">Current Signal</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="{signal_cls}">{signal_label}</span>', unsafe_allow_html=True)

    with col_reg:
        st.markdown('<p class="section-title">Detected Regime</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="{regime_cls}">{current_regime}</span>', unsafe_allow_html=True)

    with col_votes:
        votes_ok   = vote_count >= cfg["votes_required"]
        vote_color = "#69f0ae" if votes_ok else "#ff5252"
        vote_label = "  ✅ ENTRY ALLOWED" if votes_ok else "  ❌ ENTRY BLOCKED"
        st.markdown(
            f'<p class="section-title">Confirmation Votes: '
            f'<b style="color:{vote_color}">{vote_count} / 8</b>{vote_label}</p>',
            unsafe_allow_html=True,
        )
        vote_cols = st.columns(4)
        for idx, (name, (passed, value)) in enumerate(current_signals.items()):
            icon  = "✅" if passed else "❌"
            color = "#69f0ae" if passed else "#ff5252"
            with vote_cols[idx % 4]:
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;'
                    f'border-radius:8px;padding:8px 10px;margin-bottom:6px;">'
                    f'<span style="color:{color};font-weight:700;">{icon} {name}</span><br>'
                    f'<span style="color:#8b949e;font-size:0.8rem;">{value}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Candlestick chart ─────────────────────────────────────────────────────
    st.markdown(f"### {ticker} Price Chart (Last 90 Days) — Regime-Shaded Background")
    st.markdown(
        '<p class="section-title">'
        '🟢 Green = Bull Run &nbsp;|&nbsp; 🔴 Red = Bear/Crash &nbsp;|&nbsp; ⬛ Grey = Neutral</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_chart(result_df, trades_df, cfg, ticker), use_container_width=True)

    st.markdown("---")

    # ── Performance metrics ───────────────────────────────────────────────────
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, label, value, help_text in [
        (m1, "Total Return",  f"{metrics['Total Return (%)']:+.2f}%",    f"Start: ${STARTING_CAPITAL:,.0f}"),
        (m2, "Alpha vs B&H",  f"{metrics['Alpha (%)']:+.2f}%",           f"B&H: {metrics['Buy & Hold Return (%)']:+.2f}%"),
        (m3, "Win Rate",      f"{metrics['Win Rate (%)']:.1f}%",          f"{metrics['Total Trades']} trades"),
        (m4, "Max Drawdown",  f"{metrics['Max Drawdown (%)']:.2f}%",      "Peak-to-trough"),
        (m5, "Sharpe Ratio",  f"{metrics['Sharpe Ratio']:.3f}",           "Annualised (hourly)"),
        (m6, "Final Equity",  f"${metrics['Final Equity ($)']:,.0f}",      "2.5× leveraged PnL"),
    ]:
        with col:
            st.metric(label=label, value=value, help=help_text)

    st.markdown("---")

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.plotly_chart(build_equity_chart(result_df), use_container_width=True)

    st.markdown("---")

    # ── Trade log + HMM state table ──────────────────────────────────────────
    col_trades, col_states = st.columns([3, 2])

    with col_trades:
        st.markdown("### Trade Log")
        if trades_df.empty:
            st.info("No trades executed in the backtest window.")
        else:
            disp = trades_df.copy()
            disp["Entry Time"] = pd.to_datetime(disp["Entry Time"]).dt.strftime("%Y-%m-%d %H:%M")
            disp["Exit Time"]  = pd.to_datetime(disp["Exit Time"]).dt.strftime("%Y-%m-%d %H:%M")

            def _color_ret(val):
                return f"color: {'#69f0ae' if val > 0 else '#ff5252'}; font-weight: bold"

            styled = (
                disp.style
                .applymap(_color_ret, subset=["Return (%)", "PnL ($)"])
                .format({
                    "Entry Price":      "${:,.2f}",
                    "Exit Price":       "${:,.2f}",
                    "PnL ($)":          "${:,.2f}",
                    "Equity After ($)": "${:,.2f}",
                    "Return (%)":       "{:+.2f}%",
                })
            )
            st.dataframe(styled, use_container_width=True, height=340)

    with col_states:
        st.markdown("### HMM State Summary")
        st.dataframe(
            state_stats.style.format({
                "Mean Return (%)": "{:+.4f}%",
                "Std Return (%)":  "{:.4f}%",
            }),
            use_container_width=True,
            height=340,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="color:#484f58;font-size:0.75rem;text-align:center;">'
        "HMM Regime Trading System · Research & educational purposes only · Not financial advice."
        "</p>",
        unsafe_allow_html=True,
    )

    if st.button("Refresh Data & Refit Model"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
