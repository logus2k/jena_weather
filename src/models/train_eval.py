"""
Training and evaluation utilities for Keras forecasting models.

Provides a standard training wrapper with early stopping and learning
rate reduction, plus evaluation functions that compute MAE/RMSE on
both scaled (normalized) and original-scale (°C) predictions.
"""

import numpy as np
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_default_callbacks(es_cfg=None, lr_cfg=None):
    """Return EarlyStopping + ReduceLROnPlateau callbacks.

    Parameters are driven by Hydra config sub-blocks when provided. All
    values fall back to the project defaults if a sub-block is missing,
    so older code paths that call `get_default_callbacks()` with no
    arguments continue to work unchanged.

    Args:
        es_cfg: `cfg.training.early_stopping` (OmegaConf DictConfig or plain dict), expected keys:
            - patience (int, default 6)
            - restore_best_weights (bool, default True)
        lr_cfg: `cfg.training.lr_reduction` (OmegaConf DictConfig or plain dict), expected keys:
            - factor (float, default 0.5)
            - patience (int, default 3)
            - min_lr (float, default 1e-5)
    """
    es_patience = int(es_cfg.get("patience", 6)) if es_cfg is not None else 6
    es_restore = bool(es_cfg.get("restore_best_weights", True)) if es_cfg is not None else True
    lr_factor = float(lr_cfg.get("factor", 0.5)) if lr_cfg is not None else 0.5
    lr_patience = int(lr_cfg.get("patience", 3)) if lr_cfg is not None else 3
    lr_min = float(lr_cfg.get("min_lr", 1e-5)) if lr_cfg is not None else 1e-5

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=es_patience,
        restore_best_weights=es_restore,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr_min,
    )

    return [early_stopping, reduce_lr]


def train_model(model, X_train, y_train, X_val, y_val,
                batch_size=128, epochs=60, verbose=1,
                es_cfg=None, lr_cfg=None):
    """Standard training wrapper. `es_cfg` and `lr_cfg` are passed through
    to `get_default_callbacks` so users can drive early stopping and LR
    reduction from Hydra config."""
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_default_callbacks(es_cfg=es_cfg, lr_cfg=lr_cfg),
        verbose=verbose
    )
    return history


def evaluate_scaled_forecasts(y_true, y_pred):
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    return {"mae_scaled": mae, "rmse_scaled": rmse}


def inverse_scale_target(y_scaled, mean, std):
    return y_scaled * std + mean


def evaluate_original_scale_forecasts(y_true_inv, y_pred_inv):
    mae = mean_absolute_error(y_true_inv.flatten(), y_pred_inv.flatten())
    rmse = np.sqrt(mean_squared_error(y_true_inv.flatten(), y_pred_inv.flatten()))
    return {"mae": mae, "rmse": rmse}