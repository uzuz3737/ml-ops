import os
import mlflow
from mlflow.tracking import MlflowClient
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_mlflow_client(config: dict = None) -> MlflowClient:
    """Returns configured MlflowClient."""
    if config is None:
        config = load_config()
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        config["mlflow"].get("tracking_uri", "http://localhost:5000"),
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as e:
        logger.warning(f"Could not connect to tracking URI {tracking_uri}: {e}. Falling back to default.")
    return MlflowClient()


def register_and_promote_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "xgboost_model",
    target_stage: str = "Production",
) -> str:
    """Registers model from run and transitions it to target stage (e.g. Production)."""
    client = get_mlflow_client()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info(f"Registering model from {model_uri} as '{model_name}'...")
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    logger.info(f"Transitioning model '{model_name}' version {model_version.version} to {target_stage}...")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=target_stage,
        archive_existing_versions=True,
    )

    logger.info(f"Model {model_name} version {model_version.version} is now in {target_stage}!")
    return model_version.version


def get_latest_production_model_uri(model_name: str, config: dict = None) -> str:
    """Fetches URI for current production model."""
    client = get_mlflow_client(config)
    try:
        latest = client.get_latest_versions(name=model_name, stages=["Production"])
        if latest:
            return f"models:/{model_name}/Production"
    except Exception as e:
        logger.warning(f"Failed to query MLflow registry for {model_name}: {e}")
    return ""
