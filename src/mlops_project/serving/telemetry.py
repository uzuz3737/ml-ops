"""Prometheus metrics and structured logs for the API.

No request IDs, model versions or raw features as labels — they are unbounded.
"""

from __future__ import annotations

import json
import logging
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "api_requests_total",
    "HTTP requests by endpoint and status",
    ["endpoint", "status"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request handling time",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
BATCH_SIZE = Histogram(
    "api_batch_size",
    "Instances per accepted /predict request",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    registry=REGISTRY,
)
VALIDATION_FAILURES = Counter(
    "api_validation_failures_total", "Rejected inputs by rule", ["rule"], registry=REGISTRY
)
PREDICTIONS = Counter(
    "model_predictions_total", "Predictions by label", ["label"], registry=REGISTRY
)
PROBABILITY = Histogram(
    "model_default_probability",
    "Predicted default probability",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)
DEPLOYMENTS = Counter(
    "model_deployments_total",
    "Watcher actions by result",
    ["action", "result"],
    registry=REGISTRY,
)
MODEL_READY = Gauge("model_ready", "1 when a verified model is loaded", registry=REGISTRY)
MODEL_LOADED_AT = Gauge(
    "model_loaded_timestamp_seconds", "When the current model was activated", registry=REGISTRY
)
FEEDBACK = Counter(
    "feedback_labels_total", "Feedback labels by result", ["result"], registry=REGISTRY
)

logger = logging.getLogger("mlops_project.serving")


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    record = {"ts": time.time(), "component": "api", "event": event, **fields}
    logger.log(level, json.dumps(record, default=str, sort_keys=True))
