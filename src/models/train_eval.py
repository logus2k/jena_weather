"""
Training and evaluation utilities for Keras forecasting models.

Provides a standard training wrapper with early stopping and learning
rate reduction, plus evaluation functions that compute MAE/RMSE on
both scaled (normalized) and original-scale (°C) predictions.
"""

import numpy as np
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_default_callbacks():
    """Return EarlyStopping (patience=6) and ReduceLROnPlateau (patience=3) callbacks."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )

    return [early_stopping, reduce_lr]


def train_model(model, X_train, y_train, X_val, y_val, batch_size=128, epochs=60, verbose=1):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_default_callbacks(),
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