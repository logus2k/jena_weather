"""
Feature engineering for the Jena Climate forecasting pipeline.

Derives cyclical temporal encodings (hour-of-day, day-of-year) and
wind vector decompositions (Cartesian components, gust metrics) from
the raw meteorological variables. Expands the 6 original features
to a 16-dimensional input vector used by all models.
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str = "Date Time") -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df[time_col].dt.hour
    df["dayofyear"] = df[time_col].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    return df


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    wd_rad = np.deg2rad(df["wd (deg)"])

    df["wd_sin"] = np.sin(wd_rad)
    df["wd_cos"] = np.cos(wd_rad)

    df["wx"] = df["wv (m/s)"] * np.cos(wd_rad)
    df["wy"] = df["wv (m/s)"] * np.sin(wd_rad)

    df["wind_gap"] = df["max. wv (m/s)"] - df["wv (m/s)"]
    df["gust_ratio"] = df["max. wv (m/s)"] / (df["wv (m/s)"] + 1e-6)

    return df


def get_final_feature_columns() -> list:
    return [
        "T (degC)",
        "p (mbar)",
        "rh (%)",
        "wv (m/s)",
        "max. wv (m/s)",
        "wd (deg)",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "wd_sin",
        "wd_cos",
        "wx",
        "wy",
        "wind_gap",
        "gust_ratio",
    ]
