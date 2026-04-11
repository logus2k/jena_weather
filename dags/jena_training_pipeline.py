"""Jena Weather Training Pipeline - Airflow DAG.

Automates the full MLOps lifecycle:
  Ingest -> Preprocess -> Feature Engineering -> Evidently Quality ->
  Train -> Evaluate -> Register & Auto-Promote -> Evidently Drift

Uses the same src/ modules as the notebook. Configuration via Hydra.
"""

import sys
import os
from datetime import datetime

from airflow.decorators import dag, task

# Add project root to path so src/ modules are importable
PROJECT_ROOT = "/opt/airflow/dags/jena_weather"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Evidently
EVIDENTLY_PROJECT_NAME = "Jena Weather"
EVIDENTLY_URL = "http://noted-evidently:8000"


def _get_or_create_evidently_project():
    """Get the Evidently project by name, creating it if it doesn't exist."""
    from evidently.ui.workspace import RemoteWorkspace
    ws = RemoteWorkspace(EVIDENTLY_URL)
    matches = [p for p in ws.list_projects() if p.name == EVIDENTLY_PROJECT_NAME]
    if matches:
        return ws, matches[0].id
    project = ws.create_project(EVIDENTLY_PROJECT_NAME)
    print(f"Created Evidently project: {project.id}")
    return ws, project.id
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "Jena Weather Forecaster"


@dag(
    dag_id="jena_training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["jena_weather", "training", "mlops"],
    params={
        "model_config": "gru_baseline",
        "scaler_config": "standard",
        "epochs": 50,
        "batch_size": 128,
        "seed": 42,
        "hydra_config_hash": "",
    },
)
def jena_training_pipeline():

    @task
    def ingest_data(**context):
        """Load, validate, and clean the raw dataset."""
        from src.data.ingestion import ingest

        dataset_path = os.path.join(PROJECT_ROOT, "data", "jena_climate_2009_2016.csv")
        df, summary = ingest(dataset_path)

        print(f"Ingested {summary['rows']} rows, {summary['columns']} columns")
        print(f"Duplicates removed: {summary['duplicates_removed']}")

        # Save intermediate result as parquet for next task
        output_path = "/tmp/jena_ingested.parquet"
        df.to_parquet(output_path, index=False)

        return {"path": output_path, "summary": summary}

    @task
    def preprocess_data(ingest_result, **context):
        """Resample, feature engineer, and split the data."""
        import pandas as pd
        from src.data.preprocessing import resample_hourly, select_features, temporal_split
        from src.features.engineering import add_time_features, add_wind_features, get_final_feature_columns

        params = context["params"]

        df = pd.read_parquet(ingest_result["path"])

        # Resample to hourly
        df_hourly, n_dropped = resample_hourly(df)
        print(f"Hourly resampled: {len(df_hourly)} rows ({n_dropped} NaN dropped)")

        # Select features (from Hydra config)
        features = ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"]
        df_model = select_features(df_hourly, features)

        # Feature engineering
        df_feat = add_time_features(df_model, time_col="Date Time")
        df_feat = add_wind_features(df_feat)
        final_cols = get_final_feature_columns()
        print(f"Features: {len(final_cols)} -> {final_cols}")

        # Temporal split
        df_train, df_val, df_test = temporal_split(df_feat, 0.70, 0.15)
        print(f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

        # Save splits
        train_path = "/tmp/jena_train.parquet"
        val_path = "/tmp/jena_val.parquet"
        test_path = "/tmp/jena_test.parquet"
        df_feat_path = "/tmp/jena_feat.parquet"
        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)
        df_test.to_parquet(test_path, index=False)
        df_feat.to_parquet(df_feat_path, index=False)

        return {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "feat_path": df_feat_path,
            "feature_cols": final_cols,
            "target": "T (degC)",
        }

    @task
    def evidently_quality(preprocess_result, **context):
        """Generate data quality report and save to Evidently workspace."""
        import pandas as pd
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataSummaryPreset

        df_feat = pd.read_parquet(preprocess_result["feat_path"])
        feature_cols = preprocess_result["feature_cols"]

        data_def = DataDefinition(numerical_columns=feature_cols, timestamp="Date Time")
        dataset = Dataset.from_pandas(df_feat, data_definition=data_def)

        report = Report([DataSummaryPreset()], tags=["data-quality", "jena-weather", "pipeline"])
        snapshot = report.run(dataset)

        ws, project_id = _get_or_create_evidently_project()
        ws.add_run(project_id, snapshot, include_data=False)

        print("Data quality report saved to Evidently")
        return {"status": "ok"}

    @task
    def train_model_task(preprocess_result, **context):
        """Train the GRU model using the configured architecture."""
        import pandas as pd
        import numpy as np
        import mlflow
        from src.data.preparation import prepare_data
        from src.training.pipeline import train_pipeline
        from src.utils.env import set_global_seed

        params = context["params"]
        seed = params.get("seed", 42)
        set_global_seed(seed)

        # Load data
        df_train = pd.read_parquet(preprocess_result["train_path"])
        df_val = pd.read_parquet(preprocess_result["val_path"])
        df_test = pd.read_parquet(preprocess_result["test_path"])
        feature_cols = preprocess_result["feature_cols"]
        target = preprocess_result["target"]
        target_idx = feature_cols.index(target)

        # Load model config
        import yaml
        model_config_name = params.get("model_config", "gru_baseline")
        scaler_config_name = params.get("scaler_config", "standard")
        config_dir = os.path.join(PROJECT_ROOT, "config")

        with open(os.path.join(config_dir, "model", f"{model_config_name}.yaml")) as f:
            model_cfg = yaml.safe_load(f)
        with open(os.path.join(config_dir, "data", "default.yaml")) as f:
            data_cfg = yaml.safe_load(f)

        lookback = data_cfg.get("lookback", 120)
        horizon = data_cfg.get("horizon", 24)
        epochs = params.get("epochs", 50)
        batch_size = params.get("batch_size", 128)

        # Prepare config dict for prepare_data
        cfg = dict(model_cfg)
        cfg["scaler_name"] = scaler_config_name
        cfg["batch_size"] = batch_size

        # Scale and window
        scaler, X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
            cfg, df_train, df_val, df_test, feature_cols, target_idx, lookback, horizon,
        )

        # Train
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Jena Weather Forecasting")

        with mlflow.start_run(run_name=f"Pipeline - {model_config_name}") as run:
            mlflow.log_params({
                "model_type": model_cfg.get("type", "GRU"),
                "model_config": model_config_name,
                "scaler": scaler_config_name,
                "lookback": lookback,
                "horizon": horizon,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
            })

            model, history = train_pipeline(
                cfg, X_train, y_train, X_val, y_val,
                lookback, X_train.shape[2], horizon,
                epochs=epochs, verbose=1,
            )

            # Log training metrics
            hist = history.history
            for key in hist:
                for step, val in enumerate(hist[key]):
                    mlflow.log_metric(key, float(val), step=step + 1)

            # Evaluate
            from src.models.train_eval import evaluate_scaled_forecasts, evaluate_original_scale_forecasts
            from src.evolution.phenotype import inverse_target_with_scaler

            y_pred = model.predict(X_test, verbose=2)
            scaled_metrics = evaluate_scaled_forecasts(y_test, y_pred)

            y_test_inv = inverse_target_with_scaler(y_test, scaler, target_idx, len(feature_cols))
            y_pred_inv = inverse_target_with_scaler(y_pred, scaler, target_idx, len(feature_cols))
            original_metrics = evaluate_original_scale_forecasts(y_test_inv, y_pred_inv)

            mlflow.log_metrics({
                "test_mae_scaled": scaled_metrics["mae_scaled"],
                "test_rmse_scaled": scaled_metrics["rmse_scaled"],
                "test_mae_degC": original_metrics["mae"],
                "test_rmse_degC": original_metrics["rmse"],
            })

            mlflow.tensorflow.log_model(model, "model")

            print(f"Run ID: {run.info.run_id}")
            print(f"MAE: {original_metrics['mae']:.4f} degC")
            print(f"RMSE: {original_metrics['rmse']:.4f} degC")

            # Save test arrays for drift detection
            np.savez("/tmp/jena_test_arrays.npz",
                      y_test=y_test, y_pred=y_pred,
                      y_test_inv=y_test_inv, y_pred_inv=y_pred_inv)

            return {
                "run_id": run.info.run_id,
                "mae": original_metrics["mae"],
                "rmse": original_metrics["rmse"],
                "train_path": preprocess_result["train_path"],
                "test_path": preprocess_result["test_path"],
                "feature_cols": feature_cols,
            }

    @task
    def promote_model(train_result, **context):
        """Register model and auto-promote if better than current champion."""
        from src.evaluation.promote import register_and_promote

        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        result = register_and_promote(
            model=None,
            model_name=MODEL_NAME,
            run_id=train_result["run_id"],
            new_mae=train_result["mae"],
        )

        print(f"Registered: v{result['new_version']}")
        print(f"Promoted: {result['promoted']}")
        if result["promoted"] and result.get("improvement_pct") is not None:
            print(f"Improvement: {result['improvement_pct']:.1f}%")

        return result

    @task
    def evidently_drift(train_result, **context):
        """Compare training vs test data distributions."""
        import pandas as pd
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataDriftPreset

        df_train = pd.read_parquet(train_result["train_path"])
        df_test = pd.read_parquet(train_result["test_path"])
        feature_cols = train_result["feature_cols"]

        ref_def = DataDefinition(numerical_columns=feature_cols)
        ref_dataset = Dataset.from_pandas(df_train[feature_cols], data_definition=ref_def)
        cur_dataset = Dataset.from_pandas(df_test[feature_cols], data_definition=ref_def)

        report = Report(
            [DataDriftPreset()],
            tags=["drift", "jena-weather", "pipeline"],
            metadata={"run_id": train_result["run_id"]},
        )
        snapshot = report.run(cur_dataset, ref_dataset)

        ws, project_id = _get_or_create_evidently_project()
        ws.add_run(project_id, snapshot, include_data=False)

        print("Drift report saved to Evidently")

        drift_dict = snapshot.dict()
        for metric in drift_dict.get("metrics", []):
            name = metric.get("metric_name", "")
            if "Drift" in name:
                print(f"  {name}: {metric.get('value', '')}")

        return {"status": "ok"}

    # DAG flow
    ingested = ingest_data()
    preprocessed = preprocess_data(ingested)
    quality = evidently_quality(preprocessed)
    trained = train_model_task(preprocessed)
    promoted = promote_model(trained)
    drift = evidently_drift(trained)

    # Dependencies: quality runs in parallel with training,
    # promotion after training, drift after training
    quality
    trained >> promoted
    trained >> drift


jena_training_pipeline()
