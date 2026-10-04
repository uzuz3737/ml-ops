"""Credit-default prediction API (P3).

Run locally: uvicorn mlops_project.serving.app:app --port 8000
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..monitoring.exporters import MonitoringCollector
from ..pipelines.contracts import PipelineError
from . import telemetry
from .bundle import ModelSlot, load_model
from .events import EventStore, FeedbackError
from .schema import check_predict_body
from .settings import Settings, load_settings
from .watcher import DeploymentWatcher

_ENDPOINTS = {"/predict", "/health", "/ready", "/metrics", "/feedback"}


def _error(status: int, request_id: str, code: str, message: str, details=None) -> JSONResponse:
    body = {"request_id": request_id, "error": {"code": code, "message": message}}
    body["error"]["details"] = details or []
    return JSONResponse(body, status_code=status, headers={"X-Request-ID": request_id})


async def _read_json(request: Request, limit: int, request_id: str):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return None, _error(413, request_id, "payload_too_large", f"Body exceeds {limit} bytes.")
    raw = await request.body()
    if len(raw) > limit:
        return None, _error(413, request_id, "payload_too_large", f"Body exceeds {limit} bytes.")
    try:
        return json.loads(raw), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, _error(400, request_id, "malformed_json", "Body is not valid JSON.")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    slot = ModelSlot()
    events = EventStore(settings.events_dir)
    watcher = DeploymentWatcher(settings, slot)
    collector = MonitoringCollector(settings.monitoring_dir, settings.features)
    # Concurrent predict_proba calls fight over the GIL and all slow down;
    # scoring one batch at a time keeps each call at full speed.
    inference_lock = threading.Lock()

    def score(model, instances):
        with inference_lock:
            return model.predict(instances)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()
        thread = None
        telemetry.MODEL_READY.set(0)
        if settings.candidate:
            # benchmark mode: serve one candidate, ignore the shared deployment channel
            if settings.candidate.get("schema_version") != settings.schema_version:
                raise PipelineError("incompatible_schema", "Candidate schema differs.")
            slot.swap(load_model(settings.candidate, settings.project_root, settings.features))
            telemetry.MODEL_READY.set(1)
        else:
            telemetry.REGISTRY.register(collector)
            watcher.check_once()
            thread = threading.Thread(target=watcher.run, args=(stop,), daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread:
                thread.join(timeout=5)
                telemetry.REGISTRY.unregister(collector)

    app = FastAPI(title="Credit default API", version="1", lifespan=lifespan)
    app.state.settings = settings
    app.state.slot = slot
    app.state.events = events
    app.state.watcher = watcher

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        endpoint = request.url.path if request.url.path in _ENDPOINTS else "other"
        started = time.perf_counter()
        response = await call_next(request)
        telemetry.LATENCY.labels(endpoint).observe(time.perf_counter() - started)
        telemetry.REQUESTS.labels(endpoint, str(response.status_code)).inc()
        return response

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        model = slot.get()
        if model is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {
            "status": "ready",
            "model_version": model.model_version,
            "schema_version": model.schema_version,
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(telemetry.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict")
    async def predict(request: Request):
        request_id = uuid4().hex
        body, failure = await _read_json(request, settings.max_body_bytes, request_id)
        if failure:
            return failure
        problems = check_predict_body(body, settings.features, settings.max_batch_size)
        if problems:
            for problem in problems:
                telemetry.VALIDATION_FAILURES.labels(problem["rule"]).inc()
            return _error(422, request_id, "schema_violation", "Request rejected.", problems)

        model = slot.get()
        if model is None:
            return _error(503, request_id, "model_not_ready", "No verified model is loaded.")
        instances = body["instances"]
        try:
            predictions = await run_in_threadpool(score, model, instances)
        except PipelineError as error:
            telemetry.log_event("predict_failed", request_id=request_id, error=error.code)
            return _error(500, request_id, "prediction_failed", "Model could not score the batch.")
        except Exception:
            telemetry.logger.exception("predict crashed request_id=%s", request_id)
            return _error(500, request_id, "internal_error", "Unexpected server error.")

        await run_in_threadpool(
            events.record_predictions, request_id, model, instances, predictions
        )
        telemetry.BATCH_SIZE.observe(len(instances))
        for item in predictions:
            telemetry.PREDICTIONS.labels(str(item["label"])).inc()
            telemetry.PROBABILITY.observe(item["default_probability"])
        return JSONResponse(
            {
                "request_id": request_id,
                "model_version": model.model_version,
                "predictions": predictions,
            },
            headers={"X-Request-ID": request_id},
        )

    @app.post("/feedback")
    async def feedback(request: Request):
        request_id = uuid4().hex
        body, failure = await _read_json(request, settings.max_body_bytes, request_id)
        if failure:
            return failure
        try:
            stored = await run_in_threadpool(events.record_feedback, body)
        except FeedbackError as error:
            telemetry.FEEDBACK.labels(error.code).inc()
            return _error(error.status, request_id, error.code, error.message)
        telemetry.FEEDBACK.labels("accepted").inc(stored)
        return JSONResponse({"request_id": body["request_id"], "stored": stored}, status_code=202)

    return app


def __getattr__(name: str):
    # uvicorn imports `app`; build it lazily so importing this module stays cheap in tests.
    if name == "app":
        value = create_app()
        globals()["app"] = value
        return value
    raise AttributeError(name)
