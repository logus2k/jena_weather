"""
Preprocessing module for Jena Climate dataset.

Takes raw ingested data and produces:
- Hourly resampled, cleaned DataFrame
- Feature-engineered DataFrame (cyclical time features)
- Standardized train/val/test arrays with sliding-window sequences
- Fitted StandardScaler (for inverse-transforming predictions)
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

import xpto


def _fix_sensor_errors(df):
    """Replace -9999.0 equipment failures in wind columns with NaN."""
    for col in ["wv (m/s)", "max. wv (m/s)"]:
        if col in df.columns:
            df[col] = df[col].replace(-9999.0, np.nan)
    return df


def _resample_hourly(df, features, freq="1h"):
    """Resample to hourly means, drop incomplete hours."""
    df_h = (
        df.set_index("Date Time")[features]
        .resample(freq)
        .mean()
        .reset_index()
    )
    df_h = df_h.dropna().reset_index(drop=True)
    return df_h


def _add_time_features(df):
    """Add cyclical time features and encode wind direction."""
    out = df.copy()
    hour = out.index.hour
    doy = out.index.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if "wd (deg)" in out.columns:
        wd = out["wd (deg)"].astype(float)
        out["wd_sin"] = np.sin(2 * np.pi * wd / 360.0)
        out["wd_cos"] = np.cos(2 * np.pi * wd / 360.0)
        out.drop(columns=["wd (deg)"], inplace=True)

    return out


def _make_windows(data, target_idx, seq_len, horizon):
    """Create sliding-window input/output pairs.

    Parameters
    ----------
    data : np.ndarray, shape (T, F)
    target_idx : int
    seq_len : int  (L - input window)
    horizon : int  (H - forecast steps)

    Returns
    -------
    X : np.ndarray, shape (N, L, F)
    y : np.ndarray, shape (N, H)
    """
    n_samples = len(data) - seq_len - horizon
    X = sliding_window_view(data, window_shape=seq_len, axis=0)
    X = X.transpose(0, 2, 1)[:n_samples].copy()
    y = sliding_window_view(data[seq_len:, target_idx], window_shape=horizon)
    y = y[:n_samples].copy()
  
    return X, y


def preprocess(df_raw, cfg):
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw ingested dataframe with ``Date Time`` column.
    cfg : OmegaConf DictConfig or dict
        Configuration with ``data.*`` keys.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        scaler : StandardScaler (fitted on train)
        feature_cols : list[str] (final feature column names)
        target_col : str
    """
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg["data"]
    features = list(data_cfg.features)
    target_col = data_cfg.target
    train_frac = data_cfg.split.train
    val_frac = data_cfg.split.val
    seq_len = int(data_cfg.sequence_length)
    horizon = int(data_cfg.forecast_horizon)
    freq = data_cfg.resample_freq

    # All 14 original numeric columns for resampling
    all_numeric = [
        "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
        "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
        "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)",
    ]

    # Step 1: Fix sensor errors
    df = _fix_sensor_errors(df_raw.copy())

    # Step 2: Resample to hourly
    df_h = _resample_hourly(df, all_numeric, freq=freq)
    print(f"[Preprocessing] Hourly resampled: {len(df_h):,} records")

    # Step 3: Select features and set datetime index
    df_work = df_h[["Date Time"] + features].copy()
    df_work = df_work.set_index("Date Time")

    # Step 4: Temporal split
    n = len(df_work)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    df_train = df_work.iloc[:n_train]
    df_val = df_work.iloc[n_train:n_train + n_val]
    df_test = df_work.iloc[n_train + n_val:]
    print(f"[Preprocessing] Split: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")

    # Step 5: Time feature engineering
    df_train = _add_time_features(df_train)
    df_val = _add_time_features(df_val)
    df_test = _add_time_features(df_test)
    feat_cols = list(df_train.columns)
    print(f"[Preprocessing] Features after engineering: {len(feat_cols)} {feat_cols}")

    # Step 6: Standardize (fit on train only)
    scaler = StandardScaler()
    scaler.fit(df_train[feat_cols])
    train_scaled = scaler.transform(df_train[feat_cols]).astype(np.float32)
    val_scaled = scaler.transform(df_val[feat_cols]).astype(np.float32)
    test_scaled = scaler.transform(df_test[feat_cols]).astype(np.float32)

    # Step 7: Sliding windows
    target_idx = feat_cols.index(target_col)
    X_train, y_train = _make_windows(train_scaled, target_idx, seq_len, horizon)
    X_val, y_val = _make_windows(val_scaled, target_idx, seq_len, horizon)
    X_test, y_test = _make_windows(test_scaled, target_idx, seq_len, horizon)

    print(f"[Preprocessing] Windows: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[Preprocessing] Windows: X_val={X_val.shape}, X_test={X_test.shape}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feat_cols,
        "target_col": target_col,
    }
