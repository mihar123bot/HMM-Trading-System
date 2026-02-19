"""
hmm_engine.py
Hidden Markov Model engine for market regime detection.

- GaussianHMM with configurable N_STATES (default 4)
- Features: log-returns, range (High-Low)/Close, volume volatility
- State labelling (persistent-regime algorithm):
    1. Viterbi-decode the training window.
    2. Compute per-state: occupancy (%), mean log-return, std log-return, A_ss.
    3. STRESS  : states with occupancy < MIN_OCC_STRESS — spike/outlier clusters
                 that are too rare to represent a sustainable regime.
    4. Bear    : non-STRESS state with the lowest mean return.
    5. Bull    : non-STRESS, non-Bear states with mean_return > 0.
                 If none qualify, promote the best non-STRESS, non-Bear state.
    6. Neutral : everything else.
  Bull is therefore a *set* of states, not a single argmax.  This prevents
  the K=7 "spike cluster" problem where Bull captures 1-5 extreme hourly bars
  and is never seen again in a 14-day OOS window.
- p_bull = SUM of posteriors across ALL Bull states (forward-backward)
- predict_proba() gives per-bar Bull confidence
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

N_STATES       = 4      # K=4: 1 Bear + 1-2 Bull + 1-2 Neutral; small enough to avoid spike clusters
RANDOM_STATE   = 42
MIN_OCC_STRESS = 0.05   # States occupying <5% of train bars → labelled STRESS (spike cluster)

REGIME_BULL    = "Bull"
REGIME_BEAR    = "Bear"
REGIME_NEUTRAL = "Neutral"
REGIME_STRESS  = "STRESS"     # sparse spike/outlier cluster — not a tradeable regime


def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the 3-feature DataFrame used by the HMM.

    Features
    --------
    1. log_return  : log(Close_t / Close_{t-1})
    2. range_ratio : (High - Low) / Close  — proxy for intrabar volatility
    3. vol_vol     : rolling 5-bar std of log(Volume)  — volume volatility
    """
    log_ret     = np.log(data["Close"] / data["Close"].shift(1))
    range_ratio = (data["High"] - data["Low"]) / data["Close"]
    log_vol     = np.log(data["Volume"].replace(0, np.nan))
    vol_vol     = log_vol.rolling(5).std()

    features = pd.DataFrame(
        {"log_return": log_ret, "range_ratio": range_ratio, "vol_vol": vol_vol}
    )
    features.dropna(inplace=True)
    return features


def _label_states(
    model,
    features_df: pd.DataFrame,
    X_scaled: np.ndarray,
    min_occ_stress: float = MIN_OCC_STRESS,
) -> tuple[dict, list, int]:
    """
    Label HMM states as Bull (set), Bear, Neutral, or STRESS.

    Algorithm
    ---------
    1. Viterbi-decode training features.
    2. Per-state stats: occupancy, mean log-return, std log-return, A_ss.
    3. STRESS: occupancy < min_occ_stress.
    4. Bear  : lowest mean return among non-STRESS states.
    5. Bull  : non-STRESS, non-Bear states with mean_return > 0.
               Fallback: best non-STRESS, non-Bear state (even if mean ≤ 0).
    6. Neutral: remainder.

    Returns
    -------
    state_map   : dict  {state_int → regime_label_str}
    bull_states : list  of state ints labelled Bull
    bear_state  : int   state labelled Bear
    """
    states      = model.predict(X_scaled)
    n           = len(states)
    log_ret     = features_df["log_return"].values
    n_states    = model.n_components

    # ── Per-state stats ───────────────────────────────────────────────────────
    info = []
    for s in range(n_states):
        mask  = states == s
        count = int(mask.sum())
        occ   = count / n
        rets  = log_ret[mask]
        info.append({
            "state":    s,
            "occ":      occ,
            "count":    count,
            "mean_ret": float(rets.mean()) if len(rets) > 0 else 0.0,
            "std_ret":  float(rets.std())  if len(rets) > 1 else 0.0,
            "A_ss":     float(model.transmat_[s, s]),
        })

    # ── Step 3: STRESS ────────────────────────────────────────────────────────
    for si in info:
        si["is_stress"] = si["occ"] < min_occ_stress

    non_stress = [si for si in info if not si["is_stress"]]

    # ── Step 4: Bear ──────────────────────────────────────────────────────────
    if non_stress:
        bear_info  = min(non_stress, key=lambda x: x["mean_ret"])
        bear_state = bear_info["state"]
    else:
        # All states are STRESS — fall back to global argmin
        bear_state = min(info, key=lambda x: x["mean_ret"])["state"]
        for si in info:
            si["is_stress"] = False   # reset so at least one non-STRESS exists
        non_stress = info

    # ── Step 5: Bull ──────────────────────────────────────────────────────────
    bull_candidates = [
        si for si in non_stress
        if si["mean_ret"] > 0 and si["state"] != bear_state
    ]

    if not bull_candidates:
        # Fallback: best remaining non-STRESS, non-Bear state
        rest = [si for si in non_stress if si["state"] != bear_state]
        if rest:
            bull_candidates = [max(rest, key=lambda x: x["mean_ret"])]
        else:
            # Only one state left — promote it to Bull anyway
            bull_candidates = [max(info, key=lambda x: x["mean_ret"])]

    bull_states = [si["state"] for si in bull_candidates]

    # ── Step 6: state_map ─────────────────────────────────────────────────────
    state_map: dict[int, str] = {}
    for si in info:
        s = si["state"]
        if si["is_stress"]:
            state_map[s] = REGIME_STRESS
        elif s in bull_states:
            state_map[s] = REGIME_BULL
        elif s == bear_state:
            state_map[s] = REGIME_BEAR
        else:
            state_map[s] = REGIME_NEUTRAL

    return state_map, bull_states, bear_state


def fit_hmm(data: pd.DataFrame, n_states: int = N_STATES, min_occ_stress: float = MIN_OCC_STRESS):
    """
    Fit a GaussianHMM on price data and label states using the persistent-regime algorithm.

    Parameters
    ----------
    data           : OHLCV DataFrame
    n_states       : number of HMM hidden states (default 4)
    min_occ_stress : occupancy threshold below which a state is labelled STRESS (default 0.05)

    Returns
    -------
    model       : fitted hmmlearn.hmm.GaussianHMM
    scaler      : fitted StandardScaler
    features_df : the feature DataFrame used for fitting (aligned index)
    state_map   : dict {state_int → regime_label_str}
    bull_states : list of ints — HMM state indices labelled Bull
    bear_state  : int          — HMM state index labelled Bear
    """
    features_df = _build_features(data)
    X           = features_df.values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = hmm.GaussianHMM(
        n_components    = n_states,
        covariance_type = "full",
        n_iter          = 200,
        random_state    = RANDOM_STATE,
        tol             = 1e-4,
    )
    model.fit(X_scaled)

    state_map, bull_states, bear_state = _label_states(
        model, features_df, X_scaled, min_occ_stress=min_occ_stress
    )

    return model, scaler, features_df, state_map, bull_states, bear_state


def predict_regimes(
    model,
    scaler,
    features_df: pd.DataFrame,
    state_map: dict,
    data: pd.DataFrame,
    bull_states: list = None,
) -> pd.DataFrame:
    """
    Run Viterbi decoding on features_df and attach regime labels to data.

    Also computes:
    - p_bull       : sum of posterior probabilities across all Bull states
    - regime_detail: Neutral is split into Neutral-LowVol / Neutral-HighVol
                     based on range_ratio relative to the Neutral median

    Parameters
    ----------
    bull_states : list of HMM state ints labelled Bull (from fit_hmm).
                  p_bull = sum of posteriors[:, s] for s in bull_states.

    Returns the original data DataFrame with added columns:
        'hmm_state'     : integer HMM state (0..n_states-1)
        'regime'        : string label (Bull / Bear / Neutral / STRESS)
        'regime_detail' : string label (Bull / Bear / Neutral-LowVol / Neutral-HighVol / STRESS)
        'p_bull'        : float [0, 1] — combined Bull state posterior probability
    """
    X_scaled = scaler.transform(features_df.values)

    # Viterbi hard-state path
    states = model.predict(X_scaled)

    # Forward-backward posteriors — shape (n, n_states)
    posteriors = model.predict_proba(X_scaled)

    # p_bull = sum of posteriors for all Bull states
    if bull_states:
        valid_bull = [s for s in bull_states if s < posteriors.shape[1]]
        if valid_bull:
            p_bull_arr = posteriors[:, valid_bull].sum(axis=1)
        else:
            p_bull_arr = posteriors.max(axis=1)
    else:
        p_bull_arr = posteriors.max(axis=1)

    regime_series = pd.Series(
        [state_map.get(s, REGIME_NEUTRAL) for s in states],
        index=features_df.index,
        name="regime",
    )
    state_series  = pd.Series(states,     index=features_df.index, name="hmm_state")
    p_bull_series = pd.Series(p_bull_arr, index=features_df.index, name="p_bull")

    # ── Neutral subtype: split by range_ratio relative to Neutral median ──────
    neutral_mask = regime_series == REGIME_NEUTRAL
    if neutral_mask.any():
        neutral_rr = features_df.loc[neutral_mask, "range_ratio"]
        rr_median  = float(neutral_rr.median())
    else:
        rr_median = float(features_df["range_ratio"].median())

    detail_vals = []
    for s, rr, reg in zip(states, features_df["range_ratio"].values, regime_series.values):
        if reg == REGIME_BULL:
            detail_vals.append("Bull")
        elif reg == REGIME_BEAR:
            detail_vals.append("Bear")
        elif reg == REGIME_STRESS:
            detail_vals.append("STRESS")
        else:
            detail_vals.append("Neutral-LowVol" if rr < rr_median else "Neutral-HighVol")
    regime_detail_series = pd.Series(detail_vals, index=features_df.index, name="regime_detail")

    result = data.copy()
    result = result.join(state_series,          how="left")
    result = result.join(regime_series,         how="left")
    result = result.join(regime_detail_series,  how="left")
    result = result.join(p_bull_series,         how="left")

    # Forward-fill the very first few rows that have no HMM state (feature warmup)
    result["hmm_state"]     = result["hmm_state"].ffill()
    result["regime"]        = result["regime"].ffill().fillna(REGIME_NEUTRAL)
    result["regime_detail"] = result["regime_detail"].ffill().fillna("Neutral-LowVol")
    result["p_bull"]        = result["p_bull"].ffill().fillna(0.0)

    return result


def get_state_stats(
    model, scaler, state_map: dict, features_df: pd.DataFrame, data: pd.DataFrame
) -> pd.DataFrame:
    """
    Return a summary DataFrame of each HMM state's characteristics.
    Includes: Label, Count, Occ (%), Mean Return (%), Std Return (%), A_ss.
    """
    X_scaled = scaler.transform(features_df.values)
    states   = model.predict(X_scaled)
    n        = len(states)

    log_ret = np.log(data["Close"] / data["Close"].shift(1)).reindex(features_df.index)

    rows = []
    for s in range(model.n_components):
        mask = states == s
        rets = log_ret.values[mask]
        rows.append({
            "State":            s,
            "Label":            state_map.get(s, REGIME_NEUTRAL),
            "Count":            int(mask.sum()),
            "Occ (%)":          round(float(mask.mean() * 100), 1),
            "Mean Return (%)":  round(float(np.mean(rets) * 100), 4) if len(rets) > 0 else 0.0,
            "Std Return (%)":   round(float(np.std(rets)  * 100), 4) if len(rets) > 0 else 0.0,
            "A_ss":             round(float(model.transmat_[s, s]), 3),
        })

    return pd.DataFrame(rows).set_index("State")
