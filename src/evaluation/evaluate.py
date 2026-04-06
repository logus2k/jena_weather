"""
Evaluation module for Jena Climate forecasting.

Generates evaluation plots and optionally registers the model
in MLflow Model Registry.
"""

import os
import tempfile
import numpy as np
import matplotlib
if 'inline' not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature


def _infer_model_signature(model, y_pred_c):
    """Infer MLflow model signature from the model's input/output shapes.

    Works for Keras, sklearn, and other models by generating a dummy input
    matching the model's expected shape and using the actual prediction output.
    """
    try:
        sample_input = None
        sample_output = y_pred_c[:1] if y_pred_c is not None else None

        # Keras/TF model
        if hasattr(model, 'input_shape'):
            shape = model.input_shape
            # Replace None (batch dim) with 1
            concrete_shape = tuple(s if s is not None else 1 for s in shape)
            sample_input = np.random.randn(*concrete_shape).astype(np.float32)
            if sample_output is None:
                sample_output = model.predict(sample_input, verbose=0)

        # sklearn model
        elif hasattr(model, 'n_features_in_'):
            sample_input = np.random.randn(1, model.n_features_in_).astype(np.float32)
            if sample_output is None:
                sample_output = model.predict(sample_input)

        if sample_input is not None and sample_output is not None:
            return infer_signature(sample_input, sample_output)
    except Exception:
        pass
    return None


def _plot_predictions(y_true, y_pred, n_points=500):
    """Plot predicted vs actual temperature for the first n_points."""
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(min(n_points, len(y_true)))
    ax.plot(t, y_true[:n_points], label="Actual", alpha=0.8, linewidth=1)
    ax.plot(t, y_pred[:n_points], label="Predicted", alpha=0.8, linewidth=1)
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Temperature (degC)")
    ax.set_title("GRU Forecast vs Actual Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_horizon_mae(y_true_2d, y_pred_2d):
    """Bar chart of MAE per forecast horizon hour."""
    mae_by_h = np.mean(np.abs(y_true_2d - y_pred_2d), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    hours = np.arange(1, len(mae_by_h) + 1)
    ax.bar(hours, mae_by_h, color="#4a90d9", alpha=0.8)
    ax.set_xlabel("Forecast Horizon (hours ahead)")
    ax.set_ylabel("MAE (degC)")
    ax.set_title("Prediction Error by Forecast Horizon")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def evaluate(train_result, cfg, register_model=False, model_name="JenaWeatherGRU"):
    """Generate evaluation artifacts and optionally register the model.

    Parameters
    ----------
    train_result : dict
        Output from ``train()``. Must contain model, test_metrics,
        y_test_c, y_pred_c.
    cfg : OmegaConf DictConfig
        Full resolved configuration.
    register_model : bool
        If True, log the model to MLflow and register it.
    model_name : str
        Name for the registered model in MLflow Registry.

    Returns
    -------
    dict with keys:
        test_metrics : dict
        registered : bool (whether model was registered)
        model_version : int or None
    """
    model = train_result["model"]
    metrics = train_result["test_metrics"]
    y_test_c = train_result["y_test_c"]
    y_pred_c = train_result["y_pred_c"]
    model_type = cfg.model.type

    print(f"[Evaluation] Model: {model_type}")
    print(f"[Evaluation] MAE: {metrics['test_mae']:.4f} degC")
    print(f"[Evaluation] RMSE: {metrics['test_rmse']:.4f} degC")
    print(f"[Evaluation] R2: {metrics['test_r2']:.4f}")

    # Generate and log plots
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prediction vs actual
        fig1 = _plot_predictions(y_test_c.reshape(-1), y_pred_c.reshape(-1))
        path1 = os.path.join(tmpdir, "predictions_vs_actual.png")
        fig1.savefig(path1, dpi=150)
        plt.show()
        plt.close(fig1)
        if mlflow.active_run():
            mlflow.log_artifact(path1)
            print(f"[Evaluation] Logged artifact: predictions_vs_actual.png")

        # Per-horizon MAE
        fig2 = _plot_horizon_mae(y_test_c, y_pred_c)
        path2 = os.path.join(tmpdir, "horizon_mae.png")
        fig2.savefig(path2, dpi=150)
        plt.show()
        plt.close(fig2)
        if mlflow.active_run():
            mlflow.log_artifact(path2)
            print(f"[Evaluation] Logged artifact: horizon_mae.png")

    # Only log model and register if an MLflow run is active
    model_version = None
    if mlflow.active_run():
        signature = _infer_model_signature(model, y_pred_c)
        log_kwargs = {"name": "model", "signature": signature}

        if register_model:
            print(f"[Evaluation] Registering model as '{model_name}'...")
            log_kwargs["registered_model_name"] = model_name

        model_info = None
        if model_type == "GRU":
            import tensorflow as tf
            model_info = mlflow.tensorflow.log_model(model, **log_kwargs)
        elif model_type == "Linear":
            model_info = mlflow.sklearn.log_model(model, **log_kwargs)

        # Store the direct model URI as a tag for reliable loading (MLflow 3.x)
        if model_info and hasattr(model_info, 'model_uri'):
            mlflow.set_tag("noted.model_uri", model_info.model_uri)

        if register_model:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                model_version = max(int(v.version) for v in versions)
                print(f"[Evaluation] Registered: {model_name} v{model_version}")
        else:
            print(f"[Evaluation] Model logged as artifact (not registered)")

    return {
        "test_metrics": metrics,
        "registered": register_model,
        "model_version": model_version,
    }
