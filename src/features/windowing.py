"""
Supervised windowing for multi-step time series forecasting.

Converts a 2D array of scaled features into (X, y) pairs using a
sliding window approach. X contains all features over the lookback
window; y contains only the target variable over the forecast horizon.
"""

import numpy as np


def make_windows(data, target_idx, lookback=120, horizon=24):
    """Create supervised windows from a scaled feature array.

    Args:
        data: 2D array of shape (n_timesteps, n_features).
        target_idx: column index of the target variable in data.
        lookback: number of past time steps used as input.
        horizon: number of future time steps to predict.

    Returns:
        X: 3D array (n_samples, lookback, n_features).
        y: 2D array (n_samples, horizon) — target variable only.
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, target_idx])
    return np.array(X), np.array(y)