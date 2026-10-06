import json
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger
from src.features.engineering import prepare_features_and_target
from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.models.registry import register_and_promote_model

logger = get_logger(__name__)


def setup_mlflow(config: dict):
    """Initializes MLflow tracking URI and experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"].get("tracking_uri", "http://localhost:5000"))
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured with URI: {tracking_uri}, Experiment: {experiment_name}")


def build_models(config: dict) -> dict:
    model_config = config["model"]
    return {
        "xgboost": xgb.XGBClassifier(**model_config["xgboost"]),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(**model_config["logistic_regression"]),
        ),
        "random_forest": RandomForestClassifier(**model_config["random_forest"]),
        "lightgbm": LGBMClassifier(**model_config["lightgbm"], verbosity=-1),
    }


def train_model(config: dict = None):
    """Train all candidates, compare metrics, and persist the best model."""
    if config is None:
        config = load_config()

    root = get_project_root()
    # Rebuild splits so datasets created before column normalization are repaired.
    from src.data.ingestion import ingest_data
    train_df, val_df, _ = ingest_data(config)
    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)
    setup_mlflow(config)
    target_col = config["data"]["target_col"]
    raw_source = str(root / config["data"]["raw_data_file"])
    train_dataset = mlflow.data.from_pandas(
        train_df,
        source=raw_source,
        name="CreditCard_train",
        targets=target_col,
    )
    validation_dataset = mlflow.data.from_pandas(
        val_df,
        source=raw_source,
        name="CreditCard_validation",
        targets=target_col,
    )

    saved_dir = root / "models" / "saved"
    temp_dir = root / "models" / "temp"
    saved_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    results, trained_models, run_ids = {}, {}, {}

    for name, model in build_models(config).items():
        logger.info("Training candidate model: %s", name)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val.to_numpy(), probabilities)
        metrics["training_seconds"] = training_seconds

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(validation_dataset, context="validation")
            mlflow.log_params({
                "model_type": name,
                "features_count": X_train.shape[1],
                "train_samples": X_train.shape[0],
                "val_samples": X_val.shape[0],
            })
            mlflow.log_metrics(metrics)
            cm_path = temp_dir / f"{name}_confusion_matrix.png"
            roc_path = temp_dir / f"{name}_roc_curve.png"
            plot_confusion_matrix(y_val.to_numpy(), (probabilities >= 0.5).astype(int), str(cm_path))
            plot_roc_curve(y_val.to_numpy(), probabilities, str(roc_path))
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
            mlflow.log_artifact(str(roc_path), artifact_path="evaluation_plots")
            signature = infer_signature(X_train.head(10), model.predict(X_train.head(10)))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(2),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            run_ids[name] = run.info.run_id

        results[name] = metrics
        trained_models[name] = model
        joblib.dump(model, saved_dir / f"{name}.joblib")

    metric_name = config["model"].get("selection_metric", "roc_auc")
    best_name = max(results, key=lambda candidate: results[candidate][metric_name])
    best_model = trained_models[best_name]
    joblib.dump(best_model, saved_dir / "best_model.joblib")
    summary = {"selection_metric": metric_name, "best_model": best_name, "models": results}
    with open(saved_dir / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    try:
        register_and_promote_model(
            run_id=run_ids[best_name],
            model_name=config["mlflow"]["registered_model_name"],
            artifact_path="model",
        )
    except Exception as exc:
        logger.warning("Registry promotion failed; local best model remains available: %s", exc)

    logger.info("Best model: %s (%s=%.4f)", best_name, metric_name, results[best_name][metric_name])
    return best_model, summary


if __name__ == "__main__":
    train_model()
