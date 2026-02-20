"""
walk_forward.py
Rolling walk-forward validation engine.

Methodology
-----------
1. Reserve the last `lockbox_pct` of bars as a never-touched Lockbox OOS set.
2. On the remaining data, run rolling train/test folds:
     - Train  : fit a fresh HMM on the last `train_window_days` of data
                (all bars with ts <= train_end_ts)
     - Test   : trade the next `test_window_days` using the freshly fitted model
                (all bars with ts > train_end_ts, i.e. strictly the next bar)
     - Repeat, stepping forward by test_window_days each iteration
3. Concatenate all test-window equity curves → composite OOS equity curve.
4. Evaluate the Lockbox using the model from the final training fold.
5. Save a JSON snapshot of the config + fold metrics to wf_results/.

Boundary guarantee (PDR-D)
--------------------------
  Train split  : ts <= train_end_ts  (iloc [start:train_end])
  Test split   : ts >  train_end_ts  (iloc [train_end:train_end+test_bars])
  Assertion    : fold_train.index.max() < fold_test.index.min()

  No bar can appear in both windows because the split is on a strict timestamp
  inequality. The assertion is runtime-enforced on every fold.

Regime QA
---------
  Each fold logs a 'regime_qa' dict with:
  - state_diagnostics  : per-HMM-state count/%, mean log-return (train)
  - bull_label_reason  : which state was mapped to Bull and why
  - pct_time_bull_train: % of train bars in Bull state
  - pct_time_bull_test : % of test bars in Bull state

  Print with: python walk_forward.py BTC-USD
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


def _build_regime_qa(
    model,
    scaler,
    state_map: dict,
    bull_states: list,   # list of ints — Bull is now a SET of persistent states
    bear_state: int,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> dict:
    """
    Compute per-fold Regime QA diagnostics.

    Returns a dict with:
      bull_states         : list of HMM state ints labelled Bull
      bear_state          : int HMM state index labelled Bear
      bull_label_reason   : human-readable explanation of Bull mapping
      state_diagnostics   : per-state dicts: count, %, mean_ret, std_ret, A_ss
      pct_time_bull_train : % of train bars in any Bull state
      pct_time_bull_test  : % of test bars in any Bull state (None if no test)
    """
    n_states = model.n_components

    # Decode training window
    X_train      = scaler.transform(train_features.values)
    states_train = model.predict(X_train)
    log_ret_train = train_features["log_return"].values
    n_train       = len(states_train)

    state_diag = []
    for s in range(n_states):
        mask = states_train == s
        rets = log_ret_train[mask]
        state_diag.append({
            "state":              s,
            "label":              state_map.get(s, "Neutral"),
            "count_train":        int(mask.sum()),
            "pct_train":          round(float(mask.mean() * 100), 1),
            "mean_ret_train_pct": round(float(rets.mean() * 100), 4) if len(rets) > 0 else 0.0,
            "std_ret_train_pct":  round(float(rets.std()  * 100), 4) if len(rets) > 1 else 0.0,
            "model_mean_scaled":  round(float(model.means_[s, 0]), 4),
            "A_ss":               round(float(model.transmat_[s, s]), 3),
        })

    bull_mask_train = np.isin(states_train, bull_states)
    pct_bull_train  = round(float(bull_mask_train.mean() * 100), 1)

    # Decode test window
    pct_bull_test   = None
    count_bull_test = 0
    total_test      = 0
    if test_features is not None and not test_features.empty:
        X_test      = scaler.transform(test_features.values)
        states_test = model.predict(X_test)
        total_test  = len(states_test)
        count_bull_test = int(np.isin(states_test, bull_states).sum())
        pct_bull_test = round(float(count_bull_test / total_test * 100), 1) if total_test > 0 else 0.0

    # Human-readable reason line
    bull_info_strs = []
    for s in bull_states:
        sd = state_diag[s]
        bull_info_strs.append(
            f"state {s} (occ={sd['pct_train']:.1f}%  "
            f"mean={sd['mean_ret_train_pct']:+.4f}%  "
            f"std={sd['std_ret_train_pct']:.4f}%  "
            f"A_ss={sd['A_ss']:.3f})"
        )
    bull_reason = "Bull states: " + " | ".join(bull_info_strs) if bull_info_strs else "Bull states: (none)"

    return {
        "bull_states":          bull_states,
        "bear_state":           bear_state,
        "bull_label_reason":    bull_reason,
        "state_diagnostics":    state_diag,
        "pct_time_bull_train":  pct_bull_train,
        "pct_time_bull_test":   pct_bull_test,
        "count_bull_test":      count_bull_test,
        "total_test_bars":      total_test,
    }


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
                               fee_bps, slippage_bps, use_atr_stops, k_stop, k_tp,
                               use_trailing_stop, trailing_stop_pct,
                               trail_atr_mult, trail_activation_pct,
                               regime_flip_grace_bars, use_pbull_sizing
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

    last_model = last_scaler = last_feat_df = last_state_map = None
    last_bull_states: list = []
    last_bear_state: int   = 0
    last_config_hash = ""
    last_exec_rule   = ""

    for fold_idx, start in enumerate(fold_starts):
        train_end  = start + train_bars

        # ── A: Timestamp-based split — train: ts <= train_end_ts ─────────────
        fold_train    = data_wf.iloc[start:train_end]          # [start, train_end)
        train_end_ts  = fold_train.index[-1]                   # last train timestamp

        # Test: strictly the next bar onwards (ts > train_end_ts), up to test_bars
        remaining  = data_wf[data_wf.index > train_end_ts]
        fold_test  = remaining.iloc[:test_bars]

        if len(fold_test) < 24:  # skip tiny tail folds
            continue

        # ── PDR-D: Strict timestamp assertion ────────────────────────────────
        train_idx = fold_train.index
        test_idx  = fold_test.index
        assert train_idx.max() < test_idx.min(), (
            f"Fold {fold_idx + 1}: train end {train_idx.max()} "
            f">= test start {test_idx.min()} — timestamp boundary violated!"
        )

        _progress(
            (fold_idx + 1) / (n_folds + 1),
            f"Fold {fold_idx + 1}/{n_folds}  "
            f"[{fold_test.index[0].date()} → {fold_test.index[-1].date()}]",
        )

        # Fit HMM on training window
        try:
            model, scaler, feat_df, state_map, bull_states, bear_state = fit_hmm(fold_train)
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
        last_state_map  = state_map
        last_bull_states = bull_states
        last_bear_state  = bear_state

        _train_features_for_qa = feat_df  # captured for QA after test features are built

        # Predict regimes on test window (need to build features from full fold context)
        try:
            # Combine 50 bars of train tail + all test bars for feature warm-up
            from hmm_engine import _build_features
            context = pd.concat([fold_train.iloc[-50:], fold_test])
            ctx_features = _build_features(context)
            ctx_features_aligned = ctx_features.loc[ctx_features.index.isin(fold_test.index)]

            if ctx_features_aligned.empty:
                ctx_features_aligned = ctx_features.iloc[-len(fold_test):]

            with_regimes = predict_regimes(model, scaler, ctx_features_aligned, state_map, fold_test, bull_states)
            last_feat_df = ctx_features_aligned
        except Exception:
            # Fallback: predict directly on test features
            test_features = _build_features(fold_test)
            with_regimes = predict_regimes(model, scaler, test_features, state_map, fold_test, bull_states)
            last_feat_df = test_features

        # ── B: Regime QA — now we have both train and test features ──────────
        regime_qa = _build_regime_qa(
            model, scaler, state_map, bull_states, bear_state,
            _train_features_for_qa, last_feat_df,
        )

        # Compute indicators
        with_indicators = compute_indicators(with_regimes, cfg=cfg)

        # Run backtest
        result_df, trades_df, metrics = run_backtest(
            with_indicators,
            ticker                     = ticker,
            stop_loss_pct              = risk.get("stop_loss_pct",               -2.5),
            take_profit_pct            = risk.get("take_profit_pct",              4.0),
            min_regime_bars            = risk.get("min_regime_bars",              2),
            fee_bps                    = risk.get("fee_bps",                      5),
            slippage_bps               = risk.get("slippage_bps",                 3),
            use_atr_stops              = risk.get("use_atr_stops",               False),
            k_stop                     = risk.get("k_stop",                       2.0),
            k_tp                       = risk.get("k_tp",                         3.0),
            # Phase 3 risk gates
            use_vol_targeting          = risk.get("use_vol_targeting",           False),
            vol_target_pct             = risk.get("vol_target_pct",              30.0),
            vol_target_min_mult        = risk.get("vol_target_min_mult",          0.25),
            vol_target_max_mult        = risk.get("vol_target_max_mult",          1.0),
            kill_switch_enabled        = risk.get("kill_switch_enabled",         True),
            kill_switch_dd_pct         = risk.get("kill_switch_dd_pct",          9.0),
            kill_switch_cooldown_h     = risk.get("kill_switch_cooldown_h",       24),
            use_market_quality_filter  = risk.get("use_market_quality_filter",   True),
            stress_range_threshold     = risk.get("stress_range_threshold",       0.04),
            stress_force_flat          = risk.get("stress_force_flat",           True),
            stress_cooldown_hours      = risk.get("stress_cooldown_hours",        24),
            use_trailing_stop          = risk.get("use_trailing_stop",           True),
            trailing_stop_pct          = risk.get("trailing_stop_pct",            2.0),
            trail_atr_mult             = risk.get("trail_atr_mult",               2.5),
            trail_activation_pct       = risk.get("trail_activation_pct",         1.5),
            regime_flip_grace_bars     = risk.get("regime_flip_grace_bars",       2),
            use_pbull_sizing           = risk.get("use_pbull_sizing",             True),
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
            # Full timestamps for strict boundary verification (dates above can collide same day)
            "_train_end_ts":          str(fold_train.index[-1]),
            "_test_start_ts":         str(fold_test.index[0]),
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
            "regime_qa":              regime_qa,
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
                last_model, last_scaler, lb_feat_test, last_state_map, lb_test, last_bull_states
            )
            lb_indicators = compute_indicators(lb_regimes, cfg=cfg)
            _, _, lockbox_metrics = run_backtest(
                lb_indicators,
                ticker                     = ticker,
                stop_loss_pct              = risk.get("stop_loss_pct",               -2.5),
                take_profit_pct            = risk.get("take_profit_pct",              4.0),
                min_regime_bars            = risk.get("min_regime_bars",              2),
                fee_bps                    = risk.get("fee_bps",                      5),
                slippage_bps               = risk.get("slippage_bps",                 3),
                use_atr_stops              = risk.get("use_atr_stops",               False),
                k_stop                     = risk.get("k_stop",                       2.0),
                k_tp                       = risk.get("k_tp",                         3.0),
                use_vol_targeting          = risk.get("use_vol_targeting",           False),
                vol_target_pct             = risk.get("vol_target_pct",              30.0),
                vol_target_min_mult        = risk.get("vol_target_min_mult",          0.25),
                vol_target_max_mult        = risk.get("vol_target_max_mult",          1.0),
                kill_switch_enabled        = risk.get("kill_switch_enabled",         True),
                kill_switch_dd_pct         = risk.get("kill_switch_dd_pct",          9.0),
                kill_switch_cooldown_h     = risk.get("kill_switch_cooldown_h",       24),
                use_market_quality_filter  = risk.get("use_market_quality_filter",   True),
                stress_range_threshold     = risk.get("stress_range_threshold",       0.04),
                stress_force_flat          = risk.get("stress_force_flat",           True),
                stress_cooldown_hours      = risk.get("stress_cooldown_hours",        24),
                use_trailing_stop          = risk.get("use_trailing_stop",           True),
                trailing_stop_pct          = risk.get("trailing_stop_pct",            2.0),
                trail_atr_mult             = risk.get("trail_atr_mult",               2.5),
                trail_activation_pct       = risk.get("trail_activation_pct",         1.5),
                regime_flip_grace_bars     = risk.get("regime_flip_grace_bars",       2),
                use_pbull_sizing           = risk.get("use_pbull_sizing",             True),
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
    SKIP = {"attribution", "waterfall", "regime_qa"}
    fold_df = pd.DataFrame([
        {k: v for k, v in f.items() if k not in SKIP}
        for f in folds
    ])
    if not fold_df.empty:
        print(fold_df.to_string(index=False))

    # ── B: Regime QA — print per fold ─────────────────────────────────────────
    print("\n=== Regime QA (all folds) ===")
    for fold in folds:
        qa = fold.get("regime_qa")
        if not qa:
            continue
        f_num = fold["Fold"]
        print(
            f"\n  ── Fold {f_num}  "
            f"[train {fold['Train Start']} → {fold['Train End']}  |  "
            f"test {fold['Test Start']} → {fold['Test End']}]"
        )
        print(f"  {qa['bull_label_reason']}")
        print(f"  State distribution (training window):")
        for sd in qa["state_diagnostics"]:
            is_bull = sd["state"] in qa.get("bull_states", [])
            is_bear = sd["state"] == qa["bear_state"]
            flag    = "  ← BULL" if is_bull else ("  ← BEAR" if is_bear else "")
            print(
                f"    State {sd['state']} ({sd['label']:8s}): "
                f"{sd['count_train']:4d} bars  "
                f"({sd['pct_train']:5.1f}%)  "
                f"mean={sd['mean_ret_train_pct']:+.4f}%  "
                f"std={sd['std_ret_train_pct']:.4f}%  "
                f"A_ss={sd['A_ss']:.3f}"
                f"{flag}"
            )
        print(f"  % time Bull — train: {qa['pct_time_bull_train']:.1f}%", end="")
        if qa.get("pct_time_bull_test") is not None:
            print(
                f"  |  test: {qa['pct_time_bull_test']:.1f}%"
                f"  ({qa['count_bull_test']}/{qa['total_test_bars']} bars)"
            )
        else:
            print()

    # ── Boundary verification: confirm Train End < Test Start ─────────────────
    print("\n=== Boundary Check (Train End vs Test Start — full timestamps) ===")
    boundary_ok = True
    for fold in folds:
        te_ts = fold.get("_train_end_ts", fold.get("Train End", ""))
        ts_ts = fold.get("_test_start_ts", fold.get("Test Start", ""))
        # Use pd.Timestamp for strict comparison (handles same-day same-date cases)
        te_t = pd.Timestamp(te_ts)
        ts_t = pd.Timestamp(ts_ts)
        ok   = te_t < ts_t
        flag = "OK" if ok else "VIOLATION!"
        if not ok:
            boundary_ok = False
        print(
            f"  Fold {fold['Fold']:2d}:  "
            f"train_end={te_ts}  "
            f"test_start={ts_ts}  "
            f"[{flag}]"
        )
    if boundary_ok:
        print("  All folds: train_end_ts < test_start_ts ✓")
    else:
        print("  *** BOUNDARY VIOLATIONS DETECTED ***")

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
