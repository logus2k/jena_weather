"""Evaluation metrics - compute and compare model performance."""

import numpy as np
from src.models.train_eval import (
    evaluate_scaled_forecasts,
    evaluate_original_scale_forecasts,
    inverse_scale_target,
)
from src.evolution.phenotype import inverse_target_with_scaler
from src.utils.env import set_global_seed
from src.data.preparation import prepare_data
from src.training.pipeline import train_pipeline


def evaluate_model(model, X_test, y_test, scaler, target_idx, n_features):
    """Evaluate a model on test data in both scaled and original space.

    Returns dict with scaled and original-scale metrics.
    """
    y_pred = model.predict(X_test, verbose=0)

    scaled = evaluate_scaled_forecasts(y_test, y_pred)

    y_test_inv = inverse_target_with_scaler(y_test, scaler, target_idx, n_features)
    y_pred_inv = inverse_target_with_scaler(y_pred, scaler, target_idx, n_features)

    original = evaluate_original_scale_forecasts(y_test_inv, y_pred_inv)

    return {
        "mae_scaled": scaled["mae_scaled"],
        "rmse_scaled": scaled["rmse_scaled"],
        "mae": original["mae"],
        "rmse": original["rmse"],
        "y_pred": y_pred,
        "y_pred_inv": y_pred_inv,
        "y_test_inv": y_test_inv,
    }


def evaluate_with_seed(seed, cfg, feature_cols, df_train, df_val, df_test,
                       target_col="T (degC)", lookback=120, horizon=24,
                       epochs=30):
    """Train and evaluate a model with a specific random seed.

    Used for robustness assessment across multiple seeds.
    """
    set_global_seed(seed)

    target_idx = feature_cols.index(target_col)
    effective_lookback = cfg.get("lookback", lookback)

    scaler, X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        cfg, df_train, df_val, df_test, feature_cols, target_idx,
        effective_lookback, horizon,
    )

    model, history = train_pipeline(
        cfg, X_train, y_train, X_val, y_val,
        effective_lookback, X_train.shape[2], horizon,
        epochs=epochs, verbose=0,
    )

    metrics = evaluate_model(model, X_test, y_test, scaler, target_idx,
                             len(feature_cols))

    return {
        "seed": seed,
        "mae_scaled": metrics["mae_scaled"],
        "rmse_scaled": metrics["rmse_scaled"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }


def compare_models(results_dict):
    """Compare multiple model results.

    Args:
        results_dict: dict of {model_name: metrics_dict}

    Returns:
        sorted list of (model_name, metrics) by MAE ascending.
    """
    ranked = sorted(results_dict.items(), key=lambda x: x[1]["mae"])
    return ranked
