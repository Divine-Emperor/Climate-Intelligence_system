"""
models/model_registry.py
─────────────────────────
MLflow-based model registry wrapper.
Tracks experiments, logs metrics, and registers model versions.
"""
import os
import sys
import mlflow
import mlflow.keras
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MLFLOW_TRACKING


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    os.makedirs(MLFLOW_TRACKING, exist_ok=True)
    logger.info(f"MLflow tracking at: {MLFLOW_TRACKING}")


def log_training_run(
    model_name: str,
    city: str,
    params: dict,
    metrics: dict,
    model,
    artifact_paths: list = None,
):
    """Log a complete training run to MLflow."""
    setup_mlflow()
    experiment_name = f"climate_ai_{model_name}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{city}_{model_name}"):
        mlflow.log_params({"city": city, **params})
        mlflow.log_metrics(metrics)

        if model is not None:
            mlflow.keras.log_model(model, artifact_path=model_name)

        if artifact_paths:
            for path in artifact_paths:
                if os.path.exists(path):
                    mlflow.log_artifact(path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Logged run {run_id} to experiment '{experiment_name}'.")
        return run_id


def list_experiments():
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    return [e.name for e in client.search_experiments()]


def get_best_model(experiment_name: str, metric: str = "val_loss"):
    """Retrieve URI of the best model from an MLflow experiment."""
    setup_mlflow()
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} ASC"],
    )
    if runs.empty:
        return None
    best_run = runs.iloc[0]
    return mlflow.get_artifact_uri(run_id=best_run["run_id"])
