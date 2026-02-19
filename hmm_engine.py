"""
hmm_engine.py
Hidden Markov Model engine for market regime detection.

- GaussianHMM with 7 hidden states
- Features: log-returns, range (High-Low)/Close, volume volatility
- Auto-labels states: Bull (highest mean return), Bear (lowest mean return)
- All other states are tagged as Neutral (split into LowVol / HighVol subtypes)
- predict_proba() provides per-bar Bull confidence (p_bull column)
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

N_STATES = 7
RANDOM_STATE = 42

REGIME_BULL = "Bull"
REGIME_BEAR = "Bear"
REGIME_NEUTRAL = "Neutral"


def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the 3-feature DataFrame used by the HMM.

    Features
    --------
    1. log_return  : log(Close_t / Close_{t-1})
    2. range_ratio : (High - Low) / Close  — proxy for intrabar volatility
    3. vol_vol     : rolling 5-bar std of log(Volume)  — volume volatility
    """
    log_ret = np.log(data["Close"] / data["Close"].shift(1))
    range_ratio = (data["High"] - data["Low"]) / data["Close"]
    log_vol = np.log(data["Volume"].replace(0, np.nan))
    vol_vol = log_vol.rolling(5).std()

    features = pd.DataFrame(
        {"log_return": log_ret, "range_ratio": range_ratio, "vol_vol": vol_vol}
    )
    features.dropna(inplace=True)
    return features


def fit_hmm(data: pd.DataFrame):
    """
    Fit a GaussianHMM on price data.

    Returns
    -------
    model       : fitted hmmlearn.hmm.GaussianHMM
    scaler      : fitted StandardScaler
    features_df : the feature DataFrame used for fitting (aligned index)
    state_map   : dict mapping HMM state int -> regime label string
    bull_state  : int  (HMM state index for Bull)
    bear_state  : int  (HMM state index for Bear)
    """
    features_df = _build_features(data)
    X = features_df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_STATE,
        tol=1e-4,
    )
    model.fit(X_scaled)

    # ---- Auto-label states by mean log-return ----
    # model.means_ shape: (n_states, n_features)
    # Feature 0 is log_return in scaled space; ordering is preserved
    mean_returns = model.means_[:, 0]

    bull_state = int(np.argmax(mean_returns))
    bear_state = int(np.argmin(mean_returns))

    state_map = {}
    for s in range(N_STATES):
        if s == bull_state:
            state_map[s] = REGIME_BULL
        elif s == bear_state:
            state_map[s] = REGIME_BEAR
        else:
            state_map[s] = REGIME_NEUTRAL

    return model, scaler, features_df, state_map, bull_state, bear_state


def predict_regimes(
    model,
    scaler,
    features_df: pd.DataFrame,
    state_map: dict,
    data: pd.DataFrame,
    bull_state: int = None,
) -> pd.DataFrame:
    """
    Run Viterbi decoding on features_df and attach regime labels to data.

    Also computes:
    - p_bull       : posterior probability of the Bull state (forward-backward)
    - regime_detail: Neutral is split into Neutral-LowVol / Neutral-HighVol
                     based on range_ratio relative to the Neutral median

    Returns the original data DataFrame with added columns:
        'hmm_state'     : integer HMM state (0–6)
        'regime'        : string label (Bull / Bear / Neutral)
        'regime_detail' : string label (Bull / Bear / Neutral-LowVol / Neutral-HighVol)
        'p_bull'        : float [0, 1] — Bull state posterior probability
    """
    X_scaled = scaler.transform(features_df.values)

    # Viterbi hard-state path
    states = model.predict(X_scaled)

    # Forward-backward posteriors — shape (n, n_states)
    posteriors = model.predict_proba(X_scaled)

    # Bull confidence: probability assigned to the bull state
    if bull_state is not None and bull_state < posteriors.shape[1]:
        p_bull_arr = posteriors[:, bull_state]
    else:
        # Fall back to max posterior if bull_state unknown
        p_bull_arr = posteriors.max(axis=1)

    regime_series = pd.Series(
        [state_map[s] for s in states], index=features_df.index, name="regime"
    )
    state_series = pd.Series(states, index=features_df.index, name="hmm_state")
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
        else:
            detail_vals.append("Neutral-LowVol" if rr < rr_median else "Neutral-HighVol")
    regime_detail_series = pd.Series(detail_vals, index=features_df.index, name="regime_detail")

    result = data.copy()
    result = result.join(state_series,        how="left")
    result = result.join(regime_series,       how="left")
    result = result.join(regime_detail_series, how="left")
    result = result.join(p_bull_series,       how="left")

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
    Return a summary DataFrame of each HMM state's characteristics
    (useful for the dashboard state-overview table).
    """
    X_scaled = scaler.transform(features_df.values)
    states = model.predict(X_scaled)

    log_ret = np.log(data["Close"] / data["Close"].shift(1)).reindex(features_df.index)

    rows = []
    for s in range(N_STATES):
        mask = states == s
        rets = log_ret.values[mask]
        rows.append(
            {
                "State": s,
                "Label": state_map[s],
                "Count": mask.sum(),
                "Mean Return (%)": float(np.mean(rets) * 100) if len(rets) else 0.0,
                "Std Return (%)": float(np.std(rets) * 100) if len(rets) else 0.0,
            }
        )

    return pd.DataFrame(rows).set_index("State")
