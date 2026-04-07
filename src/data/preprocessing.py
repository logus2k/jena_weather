"""Data preprocessing - resample, clean, select features, split."""

import numpy as np
import pandas as pd


def resample_hourly(df, time_col="Date Time", freq="1h"):
    """Resample to hourly means, drop NaN rows."""
    df_hourly = (
        df.set_index(time_col)
          .resample(freq)
          .mean()
          .reset_index()
    )
    n_before = len(df_hourly)
    df_hourly = df_hourly.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df_hourly)
    return df_hourly, n_dropped


def select_features(df, features, time_col="Date Time"):
    """Select target features plus time column."""
    columns = [time_col] + [f for f in features if f != time_col]
    return df[columns].copy()


def temporal_split(df, train_ratio=0.70, val_ratio=0.15):
    """Split chronologically into train/val/test DataFrames."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    return df_train, df_val, df_test


def preprocess(df, features, time_col="Date Time", freq="1h",
               train_ratio=0.70, val_ratio=0.15):
    """Full preprocessing pipeline: resample -> select features -> split.

    Returns (df_train, df_val, df_test, summary).
    """
    df_hourly, n_nan_dropped = resample_hourly(df, time_col, freq)
    df_model = select_features(df_hourly, features, time_col)
    df_train, df_val, df_test = temporal_split(df_model, train_ratio, val_ratio)

    summary = {
        "original_rows": len(df),
        "hourly_rows": len(df_hourly),
        "nan_dropped": n_nan_dropped,
        "selected_features": features,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
    }

    return df_train, df_val, df_test, summary
