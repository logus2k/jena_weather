"""Jena Weather Training Pipeline - Airflow DAG.

Automates the full MLOps lifecycle:
  Ingest -> Preprocess -> Feature Engineering -> Evidently Quality ->
  Train -> Evaluate -> Register & Auto-Promote -> Evidently Drift

Uses the same src/ modules as the notebook. Configuration comes from the
project's Hydra config folder (config/config.yaml + config/data/<opt>.yaml
+ config/model/<opt>.yaml + config/scaler/<opt>.yaml). DAG params can be
used to override individual keys on top of the composed config.

This matches the Hydra-driven flow in emi_tutorial3_jena_weather.ipynb so
the DAG and the notebook behave identically given the same configuration.
"""

import sys
import os
import yaml
from datetime import datetime

from airflow.decorators import dag, task

# Add project root to path so src/ modules are importable
PROJECT_ROOT = "/opt/airflow/dags/jena_weather"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

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


def _load_yaml(path):
    """Load a YAML file, returning an empty dict if the file is missing."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _compose_config(params):
    """Compose the effective Hydra config from the project's config folder.

    Resolution order (later values overwrite earlier ones for the same key):
      1. config/config.yaml top-level keys (seed, training, ...)
      2. config/data/<data_config>.yaml    -> merged under cfg['data']
      3. config/model/<model_config>.yaml  -> merged under cfg['model']
      4. config/scaler/<scaler_config>.yaml -> merged under cfg['scaler']
      5. DAG params that shadow specific keys (epochs, batch_size, seed)

    The `defaults:` list in config.yaml is used only as a fallback when a
    DAG param doesn't specify a particular group. This matches Hydra's
    composition semantics while keeping the DAG dependency-free.
    """
    # 1. Main config (inline training block, seed, etc.)
    main_cfg = _load_yaml(os.path.join(CONFIG_DIR, "config.yaml"))
    defaults_list = main_cfg.pop("defaults", [])

    # Extract default group selections from the defaults list
    default_selections = {}
    for entry in defaults_list:
        if isinstance(entry, dict):
            for group, selection in entry.items():
                default_selections[group] = selection

    # 2. Resolve group selections: DAG params win over defaults list
    data_name = params.get("data_config") or default_selections.get("data")
    model_name = params.get("model_config") or default_selections.get("model")
    scaler_name = params.get("scaler_config") or default_selections.get("scaler")

    cfg = dict(main_cfg)

    if data_name:
        cfg["data"] = _load_yaml(os.path.join(CONFIG_DIR, "data", f"{data_name}.yaml"))
    if model_name:
        cfg["model"] = _load_yaml(os.path.join(CONFIG_DIR, "model", f"{model_name}.yaml"))
    if scaler_name:
        cfg["scaler"] = _load_yaml(os.path.join(CONFIG_DIR, "scaler", f"{scaler_name}.yaml"))

    # 3. Apply param-level overrides on top of the composed config
    if params.get("seed") is not None:
        cfg["seed"] = params["seed"]
    if params.get("epochs") is not None:
        cfg.setdefault("training", {})["epochs"] = params["epochs"]
    if params.get("batch_size") is not None:
        cfg.setdefault("training", {})["batch_size"] = params["batch_size"]

    # Stash the resolved selection names so downstream tasks can log them
    cfg["_selections"] = {
        "data": data_name,
        "model": model_name,
        "scaler": scaler_name,
    }

    return cfg


def _assemble_hydra_bundle(cfg, project_root):
    """Assemble the Hydra bundle (config/ tree + selections.json +
    resolved.yaml) as a {relative_path: bytes} dict.

    The format matches noted's `HydraManager.assemble_bundle_from_source`
    so runs produced by this DAG appear in the Configuration Composer's
    Experiment Run mode alongside runs produced by Run Manager.
    """
    import json as _json
    import hashlib as _hashlib

    bundle = {}
    config_dir = os.path.join(project_root, "config")
    config_top = "config"  # We always use `config/` as the top folder name

    # 1. Verbatim copy of every file under config/
    for dirpath, _dirnames, filenames in os.walk(config_dir):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, config_dir)
            try:
                with open(full, "rb") as f:
                    bundle[f"{config_top}/{rel}"] = f.read()
            except OSError:
                continue

    # 2. selections.json - what options this run used
    selections = cfg.get("_selections", {}) or {}
    group_selections = {
        g: s for g, s in selections.items() if s
    }
    # Overrides that went on top of the composed config (training.epochs,
    # training.batch_size, seed, etc.) would live in here, but the DAG
    # currently collapses them into the composed cfg rather than tracking
    # a separate override list. Leave empty - the composed resolved.yaml
    # already reflects the effective values.
    overrides = {}
    selections_doc = {
        "group_selections": group_selections,
        "overrides": overrides,
    }
    bundle["selections.json"] = _json.dumps(
        selections_doc, indent=2, sort_keys=True
    ).encode("utf-8")

    # 3. resolved.yaml - the full composed config (minus internal helper keys)
    resolved = {k: v for k, v in cfg.items() if not k.startswith("_")}
    resolved_yaml = yaml.dump(resolved, default_flow_style=False, sort_keys=False)
    bundle["resolved.yaml"] = resolved_yaml.encode("utf-8")

    # Compute the hash (same formula noted uses) for tagging
    config_hash = _hashlib.sha256(resolved_yaml.encode()).hexdigest()

    return bundle, f"sha256:{config_hash}"


def _write_bundle_to_dir(bundle, target_dir):
    """Write a bundle dict (from _assemble_hydra_bundle) to a target
    directory, creating subdirectories as needed."""
    for rel_path, content in bundle.items():
        full = os.path.join(target_dir, rel_path)
        parent = os.path.dirname(full) or target_dir
        os.makedirs(parent, exist_ok=True)
        with open(full, "wb") as f:
            f.write(content)


MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "Jena Weather Forecaster"


@dag(
    dag_id="jena_training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["jena_weather", "training", "mlops"],
    params={
        "data_config": "jena_full_dataset",
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
        """Load, validate, and clean the raw dataset. Uses the data file
        specified by the selected data config (cfg['data']['file'])."""
        from src.data.ingestion import ingest

        cfg = _compose_config(context["params"])
        data_cfg = cfg.get("data", {})
        rel_file = data_cfg.get("file", "data/jena_climate_2009_2016.csv")
        dataset_path = os.path.join(PROJECT_ROOT, rel_file)
        print(f"Data config: {cfg.get('_selections', {}).get('data')}")
        print(f"Loading dataset: {dataset_path}")

        df, summary = ingest(dataset_path)

        print(f"Ingested {summary['rows']} rows, {summary['columns']} columns")
        print(f"Duplicates removed: {summary['duplicates_removed']}")

        # Save intermediate result as parquet for next task
        output_path = "/tmp/jena_ingested.parquet"
        df.to_parquet(output_path, index=False)

        return {"path": output_path, "summary": summary}

    @task
    def preprocess_data(ingest_result, **context):
        """Resample, feature engineer, and split the data according to the
        composed Hydra config (features, target, split ratios all come
        from cfg['data'])."""
        import pandas as pd
        from src.data.preprocessing import resample_hourly, select_features, temporal_split
        from src.features.engineering import add_time_features, add_wind_features, get_final_feature_columns

        cfg = _compose_config(context["params"])
        data_cfg = cfg.get("data", {})

        df = pd.read_parquet(ingest_result["path"])

        # Resample to hourly
        df_hourly, n_dropped = resample_hourly(df)
        print(f"Hourly resampled: {len(df_hourly)} rows ({n_dropped} NaN dropped)")

        # Select features from Hydra config
        features = list(data_cfg.get("features", [
            "T (degC)", "p (mbar)", "rh (%)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"
        ]))
        target = data_cfg.get("target", "T (degC)")
        print(f"Features: {features}")
        print(f"Target: {target}")
        df_model = select_features(df_hourly, features)

        # Feature engineering
        df_feat = add_time_features(df_model, time_col="Date Time")
        df_feat = add_wind_features(df_feat)
        final_cols = get_final_feature_columns()
        print(f"Engineered features: {len(final_cols)} -> {final_cols}")

        # Temporal split from Hydra config
        split_cfg = data_cfg.get("split", {})
        train_ratio = float(split_cfg.get("train", 0.70))
        val_ratio = float(split_cfg.get("val", 0.15))
        df_train, df_val, df_test = temporal_split(df_feat, train_ratio, val_ratio)
        print(f"Split ({train_ratio}/{val_ratio}): train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

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
            "target": target,
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
        """Train the GRU model using the full composed Hydra config.

        Reads model, training, data (lookback/horizon) and scaler sections
        from the composed cfg. Passes early stopping and LR reduction
        sub-blocks through to train_pipeline so they drive the Keras
        callbacks the same way the notebook does.
        """
        import pandas as pd
        import numpy as np
        import mlflow
        from src.data.preparation import prepare_data
        from src.training.pipeline import train_pipeline
        from src.utils.env import set_global_seed

        cfg = _compose_config(context["params"])
        selections = cfg.get("_selections", {})
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        scaler_cfg = cfg.get("scaler", {})
        training_cfg = cfg.get("training", {})

        seed = int(cfg.get("seed", 42))
        set_global_seed(seed)

        # Load data splits from preprocess result
        df_train = pd.read_parquet(preprocess_result["train_path"])
        df_val = pd.read_parquet(preprocess_result["val_path"])
        df_test = pd.read_parquet(preprocess_result["test_path"])
        feature_cols = preprocess_result["feature_cols"]
        target = preprocess_result["target"]
        target_idx = feature_cols.index(target)

        lookback = int(data_cfg.get("lookback", 120))
        horizon = int(data_cfg.get("horizon", 24))
        epochs = int(training_cfg.get("epochs", 50))
        batch_size = int(training_cfg.get("batch_size", 128))

        # Build the model-shaped cfg dict that build_model_from_cfg expects.
        # We start from the model config and overlay scaler name and training
        # knobs that build_model_from_cfg reads (learning_rate, clipnorm,
        # batch_size). Early stopping and LR reduction go through their own
        # kwargs below.
        model_training_cfg = dict(model_cfg)
        model_training_cfg["scaler_name"] = scaler_cfg.get("name", selections.get("scaler") or "standard")
        model_training_cfg["batch_size"] = batch_size
        if "learning_rate" in training_cfg:
            model_training_cfg["learning_rate"] = training_cfg["learning_rate"]
        if "clipnorm" in training_cfg:
            model_training_cfg["clipnorm"] = training_cfg["clipnorm"]

        es_cfg = training_cfg.get("early_stopping") or None
        lr_cfg = training_cfg.get("lr_reduction") or None

        # Scale and window
        scaler, X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
            model_training_cfg, df_train, df_val, df_test,
            feature_cols, target_idx, lookback, horizon,
        )

        # Train. Experiment name matches the project_id so notebook runs,
        # Run Manager runs, and DAG runs all land in the same MLflow
        # experiment (noted's run-execute handler uses project_id as the
        # experiment name too).
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("jena_weather")

        run_name = f"Pipeline - {selections.get('model', 'model')} / {selections.get('data', 'data')}"
        with mlflow.start_run(run_name=run_name) as run:
            logged_params = {
                "data_config": selections.get("data", ""),
                "model_config": selections.get("model", ""),
                "scaler_config": selections.get("scaler", ""),
                "model_type": model_cfg.get("type", "GRU"),
                "scaler": model_training_cfg["scaler_name"],
                "lookback": lookback,
                "horizon": horizon,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": training_cfg.get("learning_rate", ""),
                "clipnorm": training_cfg.get("clipnorm", ""),
                "seed": seed,
            }
            if es_cfg:
                logged_params["es_patience"] = es_cfg.get("patience", "")
                logged_params["es_restore_best"] = es_cfg.get("restore_best_weights", "")
            if lr_cfg:
                logged_params["lr_factor"] = lr_cfg.get("factor", "")
                logged_params["lr_patience"] = lr_cfg.get("patience", "")
                logged_params["lr_min"] = lr_cfg.get("min_lr", "")
            mlflow.log_params(logged_params)

            model, history = train_pipeline(
                model_training_cfg, X_train, y_train, X_val, y_val,
                lookback, X_train.shape[2], horizon,
                epochs=epochs, verbose=1,
                es_cfg=es_cfg, lr_cfg=lr_cfg,
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

            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, y_pred)
            mlflow.tensorflow.log_model(model, name="model", signature=signature)

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
    def log_hydra_lineage(train_result, **context):
        """Upload the full Hydra bundle (config/ tree + selections.json +
        resolved.yaml) to the MLflow run produced by train_model_task.

        This is what makes the run discoverable from noted's Configuration
        Composer > Experiment Run mode: the Composer filters runs by
        presence of a `hydra/` artifact folder, and this task is the one
        that uploads it.

        Fails loud: if the bundle upload fails (MLflow unreachable, disk
        full, permission denied), this task turns red in the Airflow UI
        and the overall DAG run is marked failed. The upstream training
        task still holds its own params/metrics/model on the MLflow run
        side - only the Hydra lineage is missing. Retrying this task
        alone is safe and idempotent.
        """
        import tempfile
        import mlflow
        from mlflow.tracking import MlflowClient

        cfg = _compose_config(context["params"])
        run_id = train_result["run_id"]

        bundle, config_hash = _assemble_hydra_bundle(cfg, PROJECT_ROOT)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bundle_to_dir(bundle, tmpdir)
            client.log_artifacts(run_id, tmpdir, "hydra")

        # Tag the run with the config hash for easy lookup from the UI
        client.set_tag(run_id, "noted.hydra_config_hash", config_hash)

        selections = cfg.get("_selections", {}) or {}
        print(f"Logged Hydra bundle to run {run_id}")
        print(f"  config_hash: {config_hash}")
        print(f"  selections: {selections}")
        print(f"  artifacts: {sorted(bundle.keys())}")

        return {"run_id": run_id, "config_hash": config_hash}

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
    lineage = log_hydra_lineage(trained)
    promoted = promote_model(trained)
    drift = evidently_drift(trained)

    # Dependencies: quality runs in parallel with training,
    # lineage/promotion/drift all fan out from training.
    quality
    trained >> lineage
    trained >> promoted
    trained >> drift


jena_training_pipeline()
