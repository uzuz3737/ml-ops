import os
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
import mlflow

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger
from src.features.engineering import prepare_features_and_target
from serving.schemas import (
    CustomerFeatures,
    SinglePredictionResponse,
    BatchCustomerFeatures,
    BatchPredictionResponse,
)
from serving.metrics import (
    PrometheusMiddleware,
    get_metrics_response,
    MODEL_PREDICTIONS_TOTAL,
    PREDICTION_PROBABILITY_HISTOGRAM,
    ACTIVE_MODEL_VERSION,
)

logger = get_logger(__name__)

# Global model state
state = {
    "model": None,
    "model_version": "unknown",
    "model_source": "none",
    "config": None,
}


def load_model(config: dict):
    """Loads model from MLflow Model Registry or local fallback artifact."""
    root = get_project_root()
    reg_name = config["mlflow"].get("registered_model_name", "CreditCardDefaultBestModel")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"].get("tracking_uri", "http://localhost:5000"))

    # 1. Try MLflow Registry Production Model
    try:
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{reg_name}/Production"
        logger.info(f"Attempting to load model from MLflow Registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        state["model"] = model
        state["model_version"] = "Production-Registry"
        state["model_source"] = model_uri
        ACTIVE_MODEL_VERSION.labels(model_name=reg_name, version="Production", source="mlflow_registry").set(1)
        logger.info("Successfully loaded model from MLflow Registry.")
        return
    except Exception as e:
        logger.warning(f"Could not load model from MLflow Registry: {e}. Checking local fallback...")

    # 2. Try Local Saved Model
    fallback_path = root / config["serving"].get("fallback_model_path", "models/saved/xgb_model.joblib")
    if fallback_path.exists():
        logger.info(f"Loading fallback model from: {fallback_path}")
        model = joblib.load(fallback_path)
        state["model"] = model
        comparison_path = root / "models" / "saved" / "comparison.json"
        selected_name = "best-local-model"
        if comparison_path.exists():
            import json
            selected_name = json.loads(comparison_path.read_text(encoding="utf-8"))["best_model"]
        state["model_version"] = selected_name
        state["model_source"] = str(fallback_path)
        ACTIVE_MODEL_VERSION.labels(model_name=reg_name, version="local-joblib", source="filesystem").set(1)
        logger.info("Successfully loaded fallback model.")
        return

    logger.warning("No pre-trained model found! Starting service in cold-start mode.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config()
    state["config"] = config
    load_model(config)
    yield
    # Shutdown
    logger.info("Shutting down serving application...")


app = FastAPI(
    title="Credit Card Default Scoring API",
    description="MLOps model comparison inference service with Prometheus monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Attach Prometheus Middleware
app.add_middleware(PrometheusMiddleware)


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics_response()


@app.get("/health", tags=["Health"])
def health_check():
    """Health and readiness probe."""
    is_ready = state["model"] is not None
    return {
        "status": "healthy",
        "model_loaded": is_ready,
        "model_version": state["model_version"],
        "model_source": state["model_source"],
    }


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict(customer: CustomerFeatures):
    """Predicts credit card default probability for a single customer."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        # Convert Pydantic model to DataFrame
        input_dict = customer.model_dump()
        df = pd.DataFrame([input_dict])

        # Feature engineering
        X, _ = prepare_features_and_target(df, state["config"])

        # Predict
        prob = float(state["model"].predict_proba(X)[:, 1][0])
        pred = int(prob >= 0.5)

        # Update Prometheus Metrics
        MODEL_PREDICTIONS_TOTAL.labels(prediction_class=str(pred)).inc()
        PREDICTION_PROBABILITY_HISTOGRAM.observe(prob)

        return SinglePredictionResponse(
            default_prediction=pred,
            default_probability=round(prob, 4),
            model_version=state["model_version"],
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchCustomerFeatures):
    """Batch prediction endpoint."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        raw_list = [c.model_dump() for c in payload.customers]
        df = pd.DataFrame(raw_list)

        X, _ = prepare_features_and_target(df, state["config"])
        probs = state["model"].predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results = []
        for p, prob in zip(preds, probs):
            pred_int = int(p)
            prob_float = float(prob)
            MODEL_PREDICTIONS_TOTAL.labels(prediction_class=str(pred_int)).inc()
            PREDICTION_PROBABILITY_HISTOGRAM.observe(prob_float)
            results.append(
                SinglePredictionResponse(
                    default_prediction=pred_int,
                    default_probability=round(prob_float, 4),
                    model_version=state["model_version"],
                )
            )

        return BatchPredictionResponse(
            predictions=results,
            count=len(results),
            model_version=state["model_version"],
        )
    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.app:app", host="0.0.0.0", port=8000, reload=True)
