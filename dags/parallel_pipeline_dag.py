"""Parallel Pipeline DAG - fan-out/fan-in pattern for DAG visualization testing."""
from airflow.sdk import dag, task
from airflow.models import Variable
from datetime import datetime

_schedule = Variable.get("parallel_pipeline_schedule", default_var=None)

@dag(
    dag_id="parallel_pipeline",
    schedule=_schedule,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["noted", "testing", "visualization"],
    description="Parallel pipeline with fan-out and fan-in",
)
def parallel_pipeline():

    @task
    def ingest_data():
        print("Ingesting raw data...")
        return {"rows": 10000}

    @task
    def validate_schema(data):
        rows = data["rows"]
        print(f"Validating schema for {rows} rows...")
        return True

    @task
    def feature_engineering(data):
        print("Engineering features...")
        return {"features": 12}

    @task
    def train_gru(features):
        n = features["features"]
        print(f"Training GRU with {n} features...")
        return {"model": "gru", "mae": 2.04}

    @task
    def train_lstm(features):
        n = features["features"]
        print(f"Training LSTM with {n} features...")
        return {"model": "lstm", "mae": 2.15}

    @task
    def train_linear(features):
        n = features["features"]
        print(f"Training Linear with {n} features...")
        return {"model": "linear", "mae": 3.80}

    @task
    def compare_models(gru, lstm, linear):
        results = [gru, lstm, linear]
        best = min(results, key=lambda x: x["mae"])
        name = best["model"]
        mae = best["mae"]
        print(f"Best model: {name} with MAE {mae}")
        return best

    @task
    def register_model(best):
        name = best["model"]
        print(f"Registering {name} as champion")
        return {"registered": True}

    data = ingest_data()
    valid = validate_schema(data)
    features = feature_engineering(data)

    gru = train_gru(features)
    lstm = train_lstm(features)
    linear = train_linear(features)

    best = compare_models(gru, lstm, linear)
    register_model(best)

    valid >> features

parallel_pipeline()
