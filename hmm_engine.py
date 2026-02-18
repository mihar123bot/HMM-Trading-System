"""
hmm_engine.py
Hidden Markov Model engine for market regime detection.

- GaussianHMM with 7 hidden states
- Features: log-returns, range (High-Low)/Close, volume volatility
- Auto-labels states: Bull Run (highest mean return) and Bear/Crash (lowest mean return)
- All other states are tagged as Neutral
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


def _build_features(data: pd.DataFrame) -> np.ndarray:
    """
    Construct the 3-feature matrix used by the HMM.

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
    bull_state  : int  (HMM state index for Bull Run)
    bear_state  : int  (HMM state index for Bear/Crash)
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
    # Feature 0 is log_return in scaled space; use unscaled means for clarity
    mean_returns = model.means_[:, 0]  # scaled, but ordering is preserved

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
) -> pd.DataFrame:
    """
    Run Viterbi decoding on features_df and attach regime labels to data.

    Returns the original data DataFrame with two added columns:
        'hmm_state'  : integer HMM state
        'regime'     : string label (Bull / Bear / Neutral)
    """
    X_scaled = scaler.transform(features_df.values)
    states = model.predict(X_scaled)

    regime_series = pd.Series(
        [state_map[s] for s in states], index=features_df.index, name="regime"
    )
    state_series = pd.Series(states, index=features_df.index, name="hmm_state")

    result = data.copy()
    result = result.join(state_series, how="left")
    result = result.join(regime_series, how="left")

    # Forward-fill the very first few rows that have no HMM state
    result["hmm_state"].fillna(method="ffill", inplace=True)
    result["regime"].fillna(method="ffill", inplace=True)
    result["regime"].fillna(REGIME_NEUTRAL, inplace=True)

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
