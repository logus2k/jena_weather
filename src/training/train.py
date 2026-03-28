"""
Training module for Jena Climate forecasting.

Supports GRU (Keras) and Linear (sklearn) models, with MLflow tracking.
"""

import numpy as np
import mlflow


def _build_gru_model(seq_len, n_features, horizon, cfg):
    """Build a stacked GRU model with Keras."""
    import tensorflow as tf
    from tensorflow import keras

    units1 = int(cfg.model.units1)
    units2 = int(cfg.model.units2)
    dropout = float(cfg.model.dropout)

    inputs = keras.Input(shape=(seq_len, n_features))
    x = keras.layers.GRU(units1, return_sequences=True,
                         dropout=dropout, recurrent_dropout=0.0)(inputs)
    x = keras.layers.GRU(units2, return_sequences=False,
                         dropout=dropout, recurrent_dropout=0.0)(x)
    outputs = keras.layers.Dense(horizon)(x)

    model = keras.Model(inputs, outputs)

    lr = float(cfg.training.learning_rate)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _build_linear_model():
    """Build a simple linear regression baseline (sklearn)."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


class _MLflowEpochLogger:
    """Keras callback that logs per-epoch metrics to MLflow."""

    def __init__(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        from tensorflow import keras
        self._base = keras.callbacks.Callback
        # Build dynamically so the class inherits at runtime
        pass

    @staticmethod
    def create():
        from tensorflow import keras

        class EpochLogger(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    return
                mlflow.log_metric("train_loss", logs.get("loss", 0), step=epoch)
                mlflow.log_metric("val_loss", logs.get("val_loss", 0), step=epoch)
                mlflow.log_metric("train_mae", logs.get("mae", 0), step=epoch)
                mlflow.log_metric("val_mae", logs.get("val_mae", 0), step=epoch)

        return EpochLogger()


def train(prep_result, cfg):
    """Train a model and log everything to MLflow.

    Parameters
    ----------
    prep_result : dict
        Output from ``preprocess()``. Must contain X_train, y_train,
        X_val, y_val, X_test, y_test, scaler, feature_cols, target_col.
    cfg : OmegaConf DictConfig
        Full resolved configuration.

    Returns
    -------
    dict with keys:
        model : trained model object
        history : training history (Keras) or None (sklearn)
        test_metrics : dict of {metric_name: value} in original scale (degC)
    """
    X_train = prep_result["X_train"]
    y_train = prep_result["y_train"]
    X_val = prep_result["X_val"]
    y_val = prep_result["y_val"]
    X_test = prep_result["X_test"]
    y_test = prep_result["y_test"]
    scaler = prep_result["scaler"]
    feat_cols = prep_result["feature_cols"]
    target_col = prep_result["target_col"]

    model_type = cfg.model.type
    epochs = int(cfg.training.epochs)
    batch_size = int(cfg.training.batch_size)
    seq_len = int(cfg.data.sequence_length)
    horizon = int(cfg.data.forecast_horizon)
    n_features = X_train.shape[2]

    # Log all config params to MLflow
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", float(cfg.training.learning_rate))
    mlflow.log_param("sequence_length", seq_len)
    mlflow.log_param("forecast_horizon", horizon)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("train_samples", X_train.shape[0])

    if model_type == "GRU":
        mlflow.log_param("units1", int(cfg.model.units1))
        mlflow.log_param("units2", int(cfg.model.units2))
        mlflow.log_param("dropout", float(cfg.model.dropout))

    print(f"[Training] Model: {model_type}, Epochs: {epochs}, Batch: {batch_size}")

    if model_type == "GRU":
        model = _build_gru_model(seq_len, n_features, horizon, cfg)
        model.summary()

        from tensorflow import keras
        callbacks = [
            _MLflowEpochLogger.create(),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
            ),
        ]

        # Log total_epochs for noted's epoch progress bar
        mlflow.log_metric("total_epochs", epochs, step=0)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Predict on test
        y_pred_scaled = model.predict(X_test, verbose=0)

    elif model_type == "Linear":
        # Flatten windows for sklearn: (N, L*F) -> predict (N, H)
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)

        model = _build_linear_model()
        model.fit(X_tr_flat, y_train)
        history = None

        y_pred_scaled = model.predict(X_te_flat)

        # Log train/val metrics manually for Linear
        from sklearn.metrics import mean_absolute_error
        train_mae = mean_absolute_error(y_train.reshape(-1), model.predict(X_tr_flat).reshape(-1))
        val_mae = mean_absolute_error(y_val.reshape(-1), model.predict(X_val_flat).reshape(-1))
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("val_mae", val_mae)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Inverse-scale predictions to degrees Celsius
    t_idx = feat_cols.index(target_col)
    t_mean = scaler.mean_[t_idx]
    t_std = scaler.scale_[t_idx]

    y_test_c = y_test * t_std + t_mean
    y_pred_c = y_pred_scaled * t_std + t_mean

    # Compute metrics in original scale
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    yt = y_test_c.reshape(-1)
    yp = y_pred_c.reshape(-1)

    test_mae = float(mean_absolute_error(yt, yp))
    test_rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    test_r2 = float(r2_score(yt, yp))

    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_r2", test_r2)

    print(f"[Training] Test MAE:  {test_mae:.4f} degC")
    print(f"[Training] Test RMSE: {test_rmse:.4f} degC")
    print(f"[Training] Test R2:   {test_r2:.4f}")

    return {
        "model": model,
        "history": history,
        "test_metrics": {"test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2},
        "y_test_c": y_test_c,
        "y_pred_c": y_pred_c,
    }
