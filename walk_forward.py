"""
walk_forward.py
Rolling walk-forward validation engine.

Methodology
-----------
1. Reserve the last `lockbox_pct` of bars as a never-touched Lockbox OOS set.
2. On the remaining data, run rolling train/test folds:
     - Train  : fit a fresh HMM on the last `train_window_days` of data
     - Test   : trade the next `test_window_days` using the freshly fitted model
     - Repeat, stepping forward by test_window_days each iteration
3. Concatenate all test-window equity curves → composite OOS equity curve.
4. Evaluate the Lockbox using the model from the final training fold.
5. Save a JSON snapshot of the config + fold metrics to wf_results/.

Boundary guarantee (PDR-D)
--------------------------
  fold_train.index[-1] < fold_test.index[0] is asserted at runtime for every fold.
  No bar appears in both train and test windows.

This prevents in-sample overfitting: the Lockbox is only evaluated once,
at the very end. Sidebar tuning should be done on the WF OOS results only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from data_loader import fetch_asset_data
from hmm_engine import fit_hmm, predict_regimes
from indicators import compute_indicators
from backtester import run_backtest, STARTING_CAPITAL


def run_walk_forward(
    ticker: str,
    train_window_days: int = 180,
    test_window_days: int  = 14,
    lockbox_pct: float     = 0.20,
    cfg: dict              = None,
    risk: dict             = None,
    progress_callback: Callable[[float, str], None] = None,
) -> tuple[pd.DataFrame, list[dict], dict, pd.DataFrame]:
    """
    Run a rolling walk-forward analysis for *ticker*.

    Parameters
    ----------
    ticker             : asset ticker (e.g. "BTC-USD")
    train_window_days  : HMM training window in calendar days
    test_window_days   : trading test window in calendar days
    lockbox_pct        : fraction of total bars to reserve as lockbox OOS
                         (reserved from the end of history; never used for tuning)
    cfg                : indicator config overrides (passed to compute_indicators)
    risk               : risk / backtest config overrides
                         keys: stop_loss_pct, take_profit_pct, min_regime_bars,
                               fee_bps, slippage_bps, use_atr_stops, k_stop, k_tp
    progress_callback  : optional fn(progress: float, message: str) for UI updates

    Returns
    -------
    oos_df          : concatenated OOS equity DataFrame (all test windows)
    fold_metrics    : list of per-fold metric dicts (includes attribution + waterfall)
    lockbox_metrics : metrics dict from the final Lockbox evaluation
    state_stats     : HMM state summary from the last fitted model
    """
    risk = risk or {}

    def _progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    # ── 1. Fetch full history ─────────────────────────────────────────────────
    _progress(0.0, f"Fetching data for {ticker} …")
    raw = fetch_asset_data(ticker)

    n_total = len(raw)
    n_lockbox = max(int(n_total * lockbox_pct), test_window_days * 24)
    n_wf      = n_total - n_lockbox

    if n_wf < (train_window_days + test_window_days) * 24:
        raise ValueError(
            f"Not enough data for walk-forward with these parameters. "
            f"Have {n_wf} bars for WF (need ≥ {(train_window_days + test_window_days) * 24})."
        )

    data_wf      = raw.iloc[:n_wf]
    data_lockbox = raw.iloc[n_wf - train_window_days * 24:]  # includes train context for lockbox

    # ── 2. Build fold boundaries ──────────────────────────────────────────────
    train_bars = train_window_days * 24
    test_bars  = test_window_days  * 24

    fold_starts = list(range(0, n_wf - train_bars - test_bars + 1, test_bars))

    n_folds = len(fold_starts)
    if n_folds == 0:
        raise ValueError("No complete folds possible with current window settings.")

    # ── 3. Rolling walk-forward ───────────────────────────────────────────────
    fold_metrics_list: list[dict] = []
    oos_frames: list[pd.DataFrame] = []

    last_model = last_scaler = last_feat_df = last_state_map = last_bull_state = None
    last_config_hash = ""
    last_exec_rule   = ""

    for fold_idx, start in enumerate(fold_starts):
        train_end  = start + train_bars
        test_end   = min(train_end + test_bars, n_wf)

        fold_train = data_wf.iloc[start:train_end]
        fold_test  = data_wf.iloc[train_end:test_end]

        if len(fold_test) < 24:  # skip tiny tail folds
            continue

        # ── PDR-D: Assert no train/test index overlap ─────────────────────────
        assert fold_train.index[-1] < fold_test.index[0], (
            f"Fold {fold_idx + 1}: train end {fold_train.index[-1]} "
            f">= test start {fold_test.index[0]} — index overlap detected!"
        )

        _progress(
            (fold_idx + 1) / (n_folds + 1),
            f"Fold {fold_idx + 1}/{n_folds}  "
            f"[{fold_test.index[0].date()} → {fold_test.index[-1].date()}]",
        )

        # Fit HMM on training window
        try:
            model, scaler, feat_df, state_map, bull_state, bear_state = fit_hmm(fold_train)
        except Exception as e:
            # Skip fold if HMM fails (rare with small windows)
            fold_metrics_list.append({
                "Fold": fold_idx + 1,
                "Train Start": str(fold_train.index[0].date()),
                "Train End":   str(fold_train.index[-1].date()),
                "Test Start":  str(fold_test.index[0].date()),
                "Test End":    str(fold_test.index[-1].date()),
                "Error": str(e),
            })
            continue

        last_model, last_scaler = model, scaler
        last_state_map, last_bull_state = state_map, bull_state

        # Predict regimes on test window (need to build features from full fold context)
        try:
            # Combine enough train context for feature warm-up, then predict on test
            context = data_wf.iloc[max(0, train_end - 50):test_end]
            from hmm_engine import _build_features
            ctx_features = _build_features(context)
            ctx_features_aligned = ctx_features.loc[ctx_features.index.isin(fold_test.index)]

            if ctx_features_aligned.empty:
                ctx_features_aligned = ctx_features.iloc[-len(fold_test):]

            with_regimes = predict_regimes(model, scaler, ctx_features_aligned, state_map, fold_test, bull_state)
            last_feat_df = ctx_features_aligned
        except Exception:
            # Fallback: predict directly on test features
            test_features = _build_features(fold_test)
            with_regimes = predict_regimes(model, scaler, test_features, state_map, fold_test, bull_state)
            last_feat_df = test_features

        # Compute indicators
        with_indicators = compute_indicators(with_regimes, cfg=cfg)

        # Run backtest
        result_df, trades_df, metrics = run_backtest(
            with_indicators,
            ticker                     = ticker,
            stop_loss_pct              = risk.get("stop_loss_pct",               -3.0),
            take_profit_pct            = risk.get("take_profit_pct",              4.0),
            min_regime_bars            = risk.get("min_regime_bars",              3),
            fee_bps                    = risk.get("fee_bps",                     None),
            slippage_bps               = risk.get("slippage_bps",                None),
            use_atr_stops              = risk.get("use_atr_stops",               False),
            k_stop                     = risk.get("k_stop",                       2.0),
            k_tp                       = risk.get("k_tp",                         3.0),
            # Phase 3 risk gates
            use_vol_targeting          = risk.get("use_vol_targeting",           False),
            vol_target_pct             = risk.get("vol_target_pct",              30.0),
            vol_target_min_mult        = risk.get("vol_target_min_mult",          0.25),
            vol_target_max_mult        = risk.get("vol_target_max_mult",          1.0),
            kill_switch_enabled        = risk.get("kill_switch_enabled",         False),
            kill_switch_dd_pct         = risk.get("kill_switch_dd_pct",          10.0),
            kill_switch_cooldown_h     = risk.get("kill_switch_cooldown_h",       48),
            use_market_quality_filter  = risk.get("use_market_quality_filter",   False),
            stress_range_threshold     = risk.get("stress_range_threshold",       0.03),
            stress_force_flat          = risk.get("stress_force_flat",           False),
            stress_cooldown_hours      = risk.get("stress_cooldown_hours",        12),
        )

        last_config_hash = metrics.get("config_hash", "")
        last_exec_rule   = metrics.get("execution_rule_version", "")

        # Re-normalise equity to $10k starting for comparability across folds
        oos_frames.append(result_df[["equity", "position", "Close"]])

        fold_metrics_list.append({
            "Fold":                   fold_idx + 1,
            "Train Start":            str(fold_train.index[0].date()),
            "Train End":              str(fold_train.index[-1].date()),
            "Test Start":             str(fold_test.index[0].date()),
            "Test End":               str(fold_test.index[-1].date()),
            "Trades":                 metrics["Total Trades"],
            "Total Return (%)":       metrics["Total Return (%)"],
            "Win Rate (%)":           metrics["Win Rate (%)"],
            "Max Drawdown (%)":       metrics["Max Drawdown (%)"],
            "Sharpe Ratio":           metrics["Sharpe Ratio"],
            "Total Fees ($)":         metrics["Total Fees ($)"],
            "attribution":            metrics.get("attribution", {}),
            "waterfall":              metrics.get("waterfall", {}),
            "execution_rule_version": metrics.get("execution_rule_version", ""),
            "config_hash":            metrics.get("config_hash", ""),
        })

    # ── 4. Composite OOS equity ───────────────────────────────────────────────
    if oos_frames:
        oos_df = pd.concat(oos_frames)
        oos_df = oos_df[~oos_df.index.duplicated(keep="first")]
        oos_df.sort_index(inplace=True)
    else:
        oos_df = pd.DataFrame(columns=["equity", "position", "Close"])

    # ── 5. Lockbox evaluation ─────────────────────────────────────────────────
    lockbox_metrics: dict = {}

    _progress(0.95, "Evaluating Lockbox OOS …")

    if last_model is not None:
        try:
            from hmm_engine import _build_features as _bfeat, get_state_stats
            lb_features = _bfeat(data_lockbox)
            lb_test     = data_lockbox.iloc[len(data_lockbox) - n_lockbox:]

            lb_feat_test = lb_features.loc[lb_features.index.isin(lb_test.index)]
            if lb_feat_test.empty:
                lb_feat_test = lb_features.iloc[-n_lockbox:]

            lb_regimes = predict_regimes(
                last_model, last_scaler, lb_feat_test, last_state_map, lb_test, last_bull_state
            )
            lb_indicators = compute_indicators(lb_regimes, cfg=cfg)
            _, _, lockbox_metrics = run_backtest(
                lb_indicators,
                ticker                     = ticker,
                stop_loss_pct              = risk.get("stop_loss_pct",               -3.0),
                take_profit_pct            = risk.get("take_profit_pct",              4.0),
                min_regime_bars            = risk.get("min_regime_bars",              3),
                fee_bps                    = risk.get("fee_bps",                     None),
                slippage_bps               = risk.get("slippage_bps",                None),
                use_atr_stops              = risk.get("use_atr_stops",               False),
                k_stop                     = risk.get("k_stop",                       2.0),
                k_tp                       = risk.get("k_tp",                         3.0),
                use_vol_targeting          = risk.get("use_vol_targeting",           False),
                vol_target_pct             = risk.get("vol_target_pct",              30.0),
                vol_target_min_mult        = risk.get("vol_target_min_mult",          0.25),
                vol_target_max_mult        = risk.get("vol_target_max_mult",          1.0),
                kill_switch_enabled        = risk.get("kill_switch_enabled",         False),
                kill_switch_dd_pct         = risk.get("kill_switch_dd_pct",          10.0),
                kill_switch_cooldown_h     = risk.get("kill_switch_cooldown_h",       48),
                use_market_quality_filter  = risk.get("use_market_quality_filter",   False),
                stress_range_threshold     = risk.get("stress_range_threshold",       0.03),
                stress_force_flat          = risk.get("stress_force_flat",           False),
                stress_cooldown_hours      = risk.get("stress_cooldown_hours",        12),
            )
            lockbox_metrics["Period Start"] = str(lb_test.index[0].date())
            lockbox_metrics["Period End"]   = str(lb_test.index[-1].date())

            state_stats = get_state_stats(
                last_model, last_scaler, last_state_map, lb_feat_test, lb_test
            )
        except Exception as e:
            lockbox_metrics = {"Error": str(e)}
            state_stats = pd.DataFrame()
    else:
        state_stats = pd.DataFrame()

    # ── 6. Save JSON snapshot ─────────────────────────────────────────────────
    _save_snapshot(
        ticker, train_window_days, test_window_days, lockbox_pct,
        cfg, risk, fold_metrics_list, lockbox_metrics,
        config_hash=last_config_hash,
        execution_rule_version=last_exec_rule,
    )

    _progress(1.0, "Walk-forward complete.")

    return oos_df, fold_metrics_list, lockbox_metrics, state_stats


def _save_snapshot(
    ticker: str,
    train_window_days: int,
    test_window_days: int,
    lockbox_pct: float,
    cfg: dict,
    risk: dict,
    fold_metrics: list[dict],
    lockbox_metrics: dict,
    config_hash: str = "",
    execution_rule_version: str = "",
):
    """Save walk-forward results as a JSON snapshot for later review."""
    os.makedirs("wf_results", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("wf_results", f"snapshot_{ticker}_{ts}.json")

    snapshot = {
        "generated_utc":          datetime.utcnow().isoformat(),
        "ticker":                 ticker,
        "train_window_days":      train_window_days,
        "test_window_days":       test_window_days,
        "lockbox_pct":            lockbox_pct,
        "indicator_config":       cfg or {},
        "risk_config":            risk or {},
        "config_hash":            config_hash,
        "execution_rule_version": execution_rule_version,
        "n_folds":                len(fold_metrics),
        "fold_metrics":           fold_metrics,
        "lockbox_metrics":        lockbox_metrics,
    }

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    return path


def list_snapshots() -> list[str]:
    """Return sorted list of snapshot JSON file paths."""
    if not os.path.isdir("wf_results"):
        return []
    files = [
        os.path.join("wf_results", f)
        for f in os.listdir("wf_results")
        if f.endswith(".json")
    ]
    return sorted(files, reverse=True)


def load_snapshot(path: str) -> dict:
    """Load a saved walk-forward snapshot."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"

    def _print_progress(pct, msg):
        bar = "#" * int(pct * 30)
        print(f"\r[{bar:<30}] {pct*100:.0f}%  {msg}", end="", flush=True)

    print(f"Running walk-forward for {ticker} …\n")
    oos_df, folds, lb_metrics, _ = run_walk_forward(
        ticker,
        train_window_days=180,
        test_window_days=14,
        lockbox_pct=0.20,
        progress_callback=_print_progress,
    )

    print("\n\n=== Fold Summary ===")
    fold_df = pd.DataFrame([
        {k: v for k, v in f.items() if k not in ("attribution", "waterfall")}
        for f in folds
    ])
    if not fold_df.empty:
        print(fold_df.to_string(index=False))

    print("\n=== Attribution (last fold) ===")
    if folds and "attribution" in folds[-1]:
        for k, v in folds[-1]["attribution"].items():
            print(f"  {k}: {v}")

    print("\n=== Waterfall (last fold) ===")
    if folds and "waterfall" in folds[-1]:
        for k, v in folds[-1]["waterfall"].items():
            print(f"  {k}: {v}")

    print("\n=== Lockbox OOS ===")
    for k, v in lb_metrics.items():
        if k not in ("attribution", "waterfall"):
            print(f"  {k}: {v}")
