"""
Jena Weather training pipeline DAG.

4-stage pipeline: Ingest -> Preprocess -> Train -> Evaluate.
Calls the modular scripts in src/ with config built from DAG params.
"""

import os
import sys
from airflow.sdk import dag, task, Param
from airflow.models import Variable
from datetime import datetime

# Dynamic schedule from Airflow Variable (editable from noted UI)
_schedule = Variable.get("jena_training_pipeline_schedule", default_var=None)

# Project root inside the Airflow worker container
PROJECT_ROOT = "/opt/airflow/dags/jena_weather"


def _build_cfg(params):
    """Build an OmegaConf DictConfig from DAG params, mirroring config.yaml."""
    from omegaconf import OmegaConf

    cfg_dict = {
        "model": {
            "type": params["model_type"],
            "units1": params.get("units1", 128),
            "units2": params.get("units2", 64),
            "dropout": params.get("dropout", 0.2),
        },
        "data": {
            "file": "data/jena_climate_2009_2016.csv",
            "resample_freq": "1h",
            "features": [
                "T (degC)", "p (mbar)", "rh (%)",
                "wv (m/s)", "max. wv (m/s)", "wd (deg)",
            ],
            "target": "T (degC)",
            "split": {"train": 0.70, "val": 0.15, "test": 0.15},
            "sequence_length": 120,
            "forecast_horizon": 24,
        },
        "training": {
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
        },
    }
    return OmegaConf.create(cfg_dict)


def _add_project_to_path():
    """Add project root to sys.path so src/ modules are importable."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


@dag(
    dag_id="jena_training_pipeline",
    schedule=_schedule,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["noted", "jena_weather", "training"],
    params={
        "model_type": Param("GRU", type="string", description="Model architecture"),
        "epochs": Param(10, type="integer", description="Training epochs"),
        "batch_size": Param(256, type="integer", description="Batch size"),
        "learning_rate": Param(0.0005, type="number", description="Learning rate"),
        "units1": Param(128, type="integer", description="GRU first layer units"),
        "units2": Param(64, type="integer", description="GRU second layer units"),
        "dropout": Param(0.2, type="number", description="Dropout rate"),
        "register_model": Param(False, type="boolean", description="Register model in MLflow Registry"),
        "hydra_config_hash": Param("", type="string", description="Hydra config hash for lineage"),
    },
)
def jena_training_pipeline():

    @task
    def ingest_data(**context):
        """Load and validate the Jena Climate CSV."""
        _add_project_to_path()
        from src.ingestion.ingest import ingest

        params = context["params"]
        cfg = _build_cfg(params)
        df = ingest(cfg, project_root=PROJECT_ROOT)

        # Save to temp parquet for the next task
        out_path = os.path.join("/opt/airflow/data", "jena_pipeline_ingested.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[DAG] Saved ingested data to {out_path} ({len(df)} rows)")
        return out_path

    @task
    def preprocess_data(ingest_path, **context):
        """Resample, engineer features, standardize, create sliding windows."""
        import pandas as pd
        import joblib
        import numpy as np
        _add_project_to_path()
        from src.preprocessing.preprocess import preprocess

        params = context["params"]
        cfg = _build_cfg(params)

        df_raw = pd.read_parquet(ingest_path)
        # Restore Date Time as datetime (parquet preserves the type)
        result = preprocess(df_raw, cfg)

        # Save arrays and scaler for the next task
        out_dir = os.path.join("/opt/airflow/data", "jena_pipeline_prep")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "arrays.npz"),
            X_train=result["X_train"], y_train=result["y_train"],
            X_val=result["X_val"], y_val=result["y_val"],
            X_test=result["X_test"], y_test=result["y_test"],
        )
        joblib.dump(result["scaler"], os.path.join(out_dir, "scaler.joblib"))
        joblib.dump(result["feature_cols"], os.path.join(out_dir, "feature_cols.joblib"))
        joblib.dump(result["target_col"], os.path.join(out_dir, "target_col.joblib"))
        print(f"[DAG] Saved preprocessed data to {out_dir}")
        return out_dir

    @task
    def train_model(prep_dir, **context):
        """Train the model with MLflow tracking."""
        import numpy as np
        import joblib
        import mlflow
        _add_project_to_path()
        from src.training.train import train

        params = context["params"]
        cfg = _build_cfg(params)

        # Load preprocessed data
        arrays = np.load(os.path.join(prep_dir, "arrays.npz"))
        prep_result = {
            "X_train": arrays["X_train"], "y_train": arrays["y_train"],
            "X_val": arrays["X_val"], "y_val": arrays["y_val"],
            "X_test": arrays["X_test"], "y_test": arrays["y_test"],
            "scaler": joblib.load(os.path.join(prep_dir, "scaler.joblib")),
            "feature_cols": joblib.load(os.path.join(prep_dir, "feature_cols.joblib")),
            "target_col": joblib.load(os.path.join(prep_dir, "target_col.joblib")),
        }

        # MLflow setup
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("jena_weather")

        with mlflow.start_run(run_name=f"pipeline_{cfg.model.type}"):
            # Log lineage tags
            if params.get("hydra_config_hash"):
                mlflow.set_tag("hydra.config_hash", params["hydra_config_hash"])
                mlflow.log_param("hydra_config_hash", params["hydra_config_hash"])

            result = train(prep_result, cfg)

            # Save train result for evaluation task
            out_dir = os.path.join("/opt/airflow/data", "jena_pipeline_train")
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(out_dir, "predictions.npz"),
                y_test_c=result["y_test_c"],
                y_pred_c=result["y_pred_c"],
            )
            joblib.dump(result["model"], os.path.join(out_dir, "model.joblib"))
            joblib.dump(result["test_metrics"], os.path.join(out_dir, "metrics.joblib"))

            # Store the active run ID so evaluate can resume it
            run_id = mlflow.active_run().info.run_id
            joblib.dump(run_id, os.path.join(out_dir, "run_id.joblib"))

            print(f"[DAG] Training complete. MLflow run: {run_id}")
            return out_dir

    @task
    def evaluate_model(train_dir, **context):
        """Generate evaluation plots and optionally register the model."""
        import numpy as np
        import joblib
        import mlflow
        _add_project_to_path()
        from src.evaluation.evaluate import evaluate

        params = context["params"]
        cfg = _build_cfg(params)
        register = params.get("register_model", False)

        # Load training results
        preds = np.load(os.path.join(train_dir, "predictions.npz"))
        model = joblib.load(os.path.join(train_dir, "model.joblib"))
        metrics = joblib.load(os.path.join(train_dir, "metrics.joblib"))
        run_id = joblib.load(os.path.join(train_dir, "run_id.joblib"))

        train_result = {
            "model": model,
            "test_metrics": metrics,
            "y_test_c": preds["y_test_c"],
            "y_pred_c": preds["y_pred_c"],
        }

        # Resume the same MLflow run from training
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("jena_weather")

        with mlflow.start_run(run_id=run_id):
            result = evaluate(train_result, cfg,
                              register_model=register,
                              model_name="JenaWeatherGRU")

        status = "registered" if result["registered"] else "evaluated"
        print(f"[DAG] Evaluation complete ({status})")
        print(f"[DAG] MAE: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}, R2: {metrics['test_r2']:.4f}")
        return result["test_metrics"]

    # Wire the 4-stage pipeline
    ingested = ingest_data()
    preprocessed = preprocess_data(ingested)
    trained = train_model(preprocessed)
    evaluate_model(trained)


jena_training_pipeline()
