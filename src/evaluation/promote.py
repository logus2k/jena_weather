"""Model promotion - compare against champion and auto-promote if better."""

import logging
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def get_champion_metrics(model_name, metric_name="mae"):
    """Get the MAE of the current champion model.

    Returns (version, mae) or (None, None) if no champion exists.
    """
    client = MlflowClient()

    try:
        # Find version with @champion alias
        versions = client.get_model_version_by_alias(model_name, "champion")
        run_id = versions.run_id
        run = client.get_run(run_id)
        mae = run.data.metrics.get(metric_name)
        return versions.version, mae
    except Exception:
        # No champion alias set or model doesn't exist
        return None, None


def register_and_promote(model, model_name, run_id, new_mae,
                         metric_name="mae"):
    """Register a model and promote to champion if it beats the current one.

    Args:
        model: trained Keras model
        model_name: MLflow registry name
        run_id: MLflow run ID for this training run
        new_mae: MAE of the new model
        metric_name: metric to compare

    Returns:
        dict with promotion result.
    """
    client = MlflowClient()

    champion_version, champion_mae = get_champion_metrics(model_name,
                                                          metric_name)

    # Register the model
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    new_version = mv.version

    result = {
        "model_name": model_name,
        "new_version": new_version,
        "new_mae": new_mae,
        "champion_version": champion_version,
        "champion_mae": champion_mae,
        "promoted": False,
    }

    # Compare and promote if better
    if champion_mae is None or new_mae < champion_mae:
        client.set_registered_model_alias(model_name, "champion", new_version)
        result["promoted"] = True
        improvement = ((champion_mae - new_mae) / champion_mae * 100
                       if champion_mae else None)
        result["improvement_pct"] = improvement
        logger.info(
            "Promoted %s v%s as champion (MAE: %.4f -> %.4f, %.1f%% improvement)",
            model_name, new_version, champion_mae or 0, new_mae,
            improvement or 0,
        )
    else:
        logger.info(
            "Kept %s v%s as champion (MAE: %.4f <= %.4f)",
            model_name, champion_version, champion_mae, new_mae,
        )

    return result
