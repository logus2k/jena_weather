"""Data preparation - scaling and windowing for model training."""

from src.features.scaling import get_scaler
from src.features.windowing import make_windows


def prepare_data(cfg, df_train, df_val, df_test, feature_cols, target_idx,
                 lookback, horizon):
    """Scale features and create sliding windows for train/val/test.

    Args:
        cfg: dict with at least 'scaler_name' key
        df_train, df_val, df_test: DataFrames with feature columns
        feature_cols: list of column names to use as features
        target_idx: index of the target column in feature_cols
        lookback: number of input time steps
        horizon: number of forecast time steps

    Returns:
        (scaler, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    scaler_name = cfg.get("scaler_name", "standard")
    scaler = get_scaler(scaler_name)

    X_train_scaled = scaler.fit_transform(df_train[feature_cols])
    X_val_scaled = scaler.transform(df_val[feature_cols])
    X_test_scaled = scaler.transform(df_test[feature_cols])

    X_train, y_train = make_windows(X_train_scaled, target_idx, lookback, horizon)
    X_val, y_val = make_windows(X_val_scaled, target_idx, lookback, horizon)
    X_test, y_test = make_windows(X_test_scaled, target_idx, lookback, horizon)

    return scaler, X_train, y_train, X_val, y_val, X_test, y_test
