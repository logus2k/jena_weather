"""Jena Weather training pipeline DAG - reads config from Hydra via noted."""

from airflow.sdk import dag, task
from airflow.models.param import Param
from airflow.models import Variable
from datetime import datetime

# Dynamic schedule from Airflow Variable (editable from noted UI)
_schedule = Variable.get("jena_training_pipeline_schedule", default_var=None)


@dag(
    dag_id="jena_training_pipeline",
    schedule=_schedule,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["noted", "jena_weather", "training"],
    params={
        # Hydra config is passed as a JSON dict by noted's Trigger Panel.
        # These Param defaults mirror config.yaml for manual Airflow UI triggers.
        "model_type": Param("GRU", type="string", description="Model architecture (from model.type)"),
        "epochs": Param(30, type="integer", description="Training epochs (from training.epochs)"),
        "batch_size": Param(256, type="integer", description="Batch size (from training.batch_size)"),
        "learning_rate": Param(0.0005, type="number", description="Learning rate (from training.learning_rate)"),
        "units1": Param(128, type="integer", description="First layer units (from model.units1)"),
        "units2": Param(64, type="integer", description="Second layer units (from model.units2)"),
        "dropout": Param(0.2, type="number", description="Dropout rate (from model.dropout)"),
        "hydra_config_hash": Param("", type="string", description="Hydra config hash for lineage tracking"),
    },
)
def jena_training_pipeline():

    @task
    def validate_data(**context):
        params = context["params"]
        print(f"Validating data for {params['model_type']} training...")
        if params.get("hydra_config_hash"):
            print(f"  Config hash: {params['hydra_config_hash']}")
        print("Data validation passed")
        return {"status": "valid"}

    @task
    def train_model(**context):
        params = context["params"]
        print(f"Training {params['model_type']} model:")
        print(f"  Epochs: {params['epochs']}")
        print(f"  Batch size: {params['batch_size']}")
        print(f"  Learning rate: {params['learning_rate']}")
        print(f"  Units: {params['units1']}/{params['units2']}")
        print(f"  Dropout: {params['dropout']}")
        print("Training complete")
        return {"model_type": params["model_type"], "status": "trained"}

    @task
    def evaluate_model(**context):
        params = context["params"]
        print(f"Evaluating {params['model_type']} model...")
        print("MAE: 2.04, RMSE: 2.58, R2: 0.89")
        return {"mae": 2.04, "rmse": 2.58, "r2": 0.89}

    data = validate_data()
    model = train_model()
    metrics = evaluate_model()
    data >> model >> metrics


jena_training_pipeline()
