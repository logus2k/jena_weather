"""
Fitness evaluation for evolutionary individuals.

Implements the complete pipeline for scoring a candidate configuration:
scale data → create windows → build GRU → train → predict → compute
validation MAE in °C. The fitness (lower is better) is the original-
scale MAE, ensuring invariance to the scaler choice in the genotype.
"""

from src.features.scaling import get_scaler
from src.features.windowing import make_windows
from src.models.gru import build_gru_model
from src.models.train_eval import (
    train_model,
    evaluate_scaled_forecasts,
    evaluate_original_scale_forecasts,
)
import numpy as np


def inverse_target_with_scaler(y_scaled, scaler, target_idx, n_features):
    """Inverse-transform target predictions using any sklearn scaler.

    Works with StandardScaler, RobustScaler, and MinMaxScaler by
    constructing a dummy array and applying inverse_transform.
    """
    original_shape = y_scaled.shape
    y_flat = y_scaled.reshape(-1)

    dummy = np.zeros((len(y_flat), n_features), dtype=float)
    dummy[:, target_idx] = y_flat

    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(original_shape)


def evaluate_individual(
    cfg,
    df_train,
    df_val,
    final_feature_cols,
    target_idx,
    lookback=120,
    horizon=24,
    epochs=20,
    verbose=0,
):
    scaler = get_scaler(cfg["scaler_name"])

    X_train_df = df_train[final_feature_cols].copy()
    X_val_df = df_val[final_feature_cols].copy()

    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)

    effective_lookback = cfg.get("lookback", lookback)

    X_train, y_train = make_windows(
        X_train_scaled,
        target_idx,
        effective_lookback,
        horizon,
    )
    X_val, y_val = make_windows(
        X_val_scaled,
        target_idx,
        effective_lookback,
        horizon,
    )

    model = build_gru_model(
        L=effective_lookback,
        n_features=X_train.shape[2],
        H=horizon,
        units1=cfg["units1"],
        units2=cfg["units2"],
        units3=cfg["units3"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        l2=cfg["l2"],
        dense_units=cfg["dense_units"],
        dense_activation=cfg["dense_activation"],
        learning_rate=cfg["learning_rate"],
        clipnorm=cfg["clipnorm"],
        optimizer_name=cfg["optimizer_name"],
        weight_decay=cfg["weight_decay"],
        loss_name=cfg["loss_name"],
        gaussian_noise_std=cfg["gaussian_noise_std"],
    )

    train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=cfg["batch_size"],
        epochs=epochs,
        verbose=verbose,
    )

    y_pred_val = model.predict(X_val, verbose=0)

    scaled_metrics = evaluate_scaled_forecasts(y_val, y_pred_val)

    n_features = len(final_feature_cols)
    y_val_inv = inverse_target_with_scaler(y_val, scaler, target_idx, n_features)
    y_pred_val_inv = inverse_target_with_scaler(y_pred_val, scaler, target_idx, n_features)

    original_metrics = evaluate_original_scale_forecasts(y_val_inv, y_pred_val_inv)

    return {
        "cfg": cfg,
        "fitness": original_metrics["mae"],
        "metrics": {
            "mae_scaled": scaled_metrics["mae_scaled"],
            "rmse_scaled": scaled_metrics["rmse_scaled"],
            "mae_degC": original_metrics["mae"],
            "rmse_degC": original_metrics["rmse"],
        },
    }