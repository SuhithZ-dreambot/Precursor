"""
feature_engineering.py
Turns raw per-window flow stats into a temporal feature panel:
  - rolling mean/std over the last N windows (captures trend, not just
    current-state snapshot -- this is the "preserve temporal changes
    before aggregation collapses to a single label" idea from the deck)
  - rate-of-change (delta) features
  - EWMA + CUSUM drift statistics (the "statistical signal" half of the
    hybrid model)
"""
import numpy as np
import pandas as pd

RAW_COLS = [
    "flow_rate", "unique_ips", "unique_ports",
    "port_entropy", "syn_ack_ratio", "byte_rate", "inter_arrival_std",
]


def add_rolling_features(df, window=6):
    out = df.copy()
    for col in RAW_COLS:
        out[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        out[f"{col}_roll_std"] = df[col].rolling(window, min_periods=1).std().fillna(0)
        out[f"{col}_delta"] = df[col].diff().fillna(0)
    return out


def ewma_cusum(series, ewma_alpha=0.2, cusum_k=0.5):
    """
    EWMA baseline + two-sided CUSUM drift statistic on a standardized series.
    Returns the CUSUM magnitude (S+ and S- combined) at each point -- a
    classic, cheap, interpretable drift detector used as the "statistical
    signal" half of the hybrid fusion.
    """
    x = series.values.astype(float)
    mu0 = np.mean(x[: max(10, len(x) // 20)])   # baseline mean from early data
    sigma0 = np.std(x[: max(10, len(x) // 20)]) + 1e-6
    z = (x - mu0) / sigma0

    ewma = np.zeros_like(z)
    ewma[0] = z[0]
    for i in range(1, len(z)):
        ewma[i] = ewma_alpha * z[i] + (1 - ewma_alpha) * ewma[i - 1]

    s_pos = np.zeros_like(z)
    s_neg = np.zeros_like(z)
    for i in range(1, len(z)):
        s_pos[i] = max(0, s_pos[i - 1] + z[i] - cusum_k)
        s_neg[i] = max(0, s_neg[i - 1] - z[i] - cusum_k)
    cusum = s_pos + s_neg
    return ewma, cusum


def add_statistical_signal(df):
    out = df.copy()
    # Attacks in this synthetic set show up as entropy drop + SYN/ACK rise;
    # combine both into one drift-sensitive composite before CUSUM.
    composite = (
        -1.0 * zscore(df["port_entropy"])
        + 1.0 * zscore(df["syn_ack_ratio"])
        + 0.5 * zscore(df["unique_ports"])
    )
    out["stat_composite"] = composite
    ewma, cusum = ewma_cusum(pd.Series(composite))
    out["stat_ewma"] = ewma
    out["stat_cusum"] = cusum
    # squash CUSUM into a [0,1] "statistical risk" score
    out["statistical_risk"] = 1 / (1 + np.exp(-(cusum - np.percentile(cusum, 85)) / (np.std(cusum) + 1e-6)))
    return out


def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-6)


def make_future_labels(df, lead_windows=3):
    """
    Label each window with whether an attack ONSET occurs within the next
    `lead_windows` windows (but hasn't started yet). This is the
    "attack-in-future-window != attack-in-current-window" label from the deck.
    """
    is_attack = df["is_attack_now"].values
    n = len(is_attack)
    future_label = np.zeros(n, dtype=int)
    for i in range(n):
        window_end = min(n, i + 1 + lead_windows)
        # positive only if attack starts in the lookahead window AND is not
        # already active right now (this is a forecast, not a detector)
        if is_attack[i] == 0 and is_attack[i + 1:window_end].any():
            future_label[i] = 1
    return future_label
