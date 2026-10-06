"""Dependency-light Airflow client for the internal pipeline worker."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request

from .contracts import (
    CONTRACT_VERSION,
    REQUIRED_PASS,
    PipelineError,
    json_bytes,
    validate_run_id,
    validate_step,
)

MAX_RESPONSE_BYTES = 1024 * 1024


def execute_remote(
    step: str,
    run_id: str,
    config_path: str,
    *,
    worker_url: str | None = None,
    timeout_seconds: float | None = None,
    data_file: str | None = None,
) -> dict:
    validate_step(step)
    if data_file is not None and (not isinstance(data_file, str) or not data_file):
        raise PipelineError("invalid_data_file", "data_file must be a nonempty project path.")
    validate_run_id(run_id)
    if not isinstance(config_path, str) or not config_path:
        raise PipelineError(
            "invalid_config", "The remote configuration path must be a nonempty string."
        )
    url = worker_url or os.environ.get("MLOPS_WORKER_URL", "http://pipeline-worker:8100")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise PipelineError(
            "invalid_worker_url",
            "Worker URL must be an HTTP(S) endpoint without credentials, query or fragment.",
        )
    try:
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get("MLOPS_STEP_TIMEOUT_SECONDS", "1910")
        )
    except (TypeError, ValueError):
        raise PipelineError(
            "invalid_timeout", "Worker timeout must be a finite positive number."
        ) from None
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise PipelineError("invalid_timeout", "Worker timeout must be a finite positive number.")
    body = {"step": step, "run_id": run_id, "config_path": config_path}
    if data_file is not None:
        body["data_file"] = data_file
    request = urllib.request.Request(
        url.rstrip("/") + "/steps",
        data=json_bytes(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = _response_json(response)
    except urllib.error.HTTPError as error:
        try:
            payload = _response_json(error)
        finally:
            error.close()
        details = payload.get("error", {})
        if (
            isinstance(details, dict)
            and isinstance(details.get("code"), str)
            and isinstance(details.get("message"), str)
        ):
            raise PipelineError(details["code"], details["message"]) from None
        raise PipelineError("worker_rejected", "The worker rejected this pipeline stage.") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise PipelineError(
            "worker_unavailable", "The pipeline worker could not be reached before its timeout."
        ) from None
    if (
        type(payload.get("contract_version")) is not int
        or payload["contract_version"] != CONTRACT_VERSION
        or payload.get("run_id") != run_id
        or payload.get("step") != step
        or payload.get("state") != "succeeded"
        or not isinstance(payload.get("result"), dict)
        or type(payload["result"].get("contract_version")) is not int
        or payload["result"]["contract_version"] != CONTRACT_VERSION
        or payload["result"].get("run_id") != run_id
        or ("passed" in payload["result"] and type(payload["result"]["passed"]) is not bool)
        or payload["result"].get("passed") is False
        or payload["result"].get("status") in {"failed", "error", "rejected"}
        or (step in REQUIRED_PASS and payload["result"].get("passed") is not True)
    ):
        raise PipelineError(
            "invalid_worker_response",
            "The worker returned mismatched or unsuccessful stage evidence.",
        )
    json_bytes(payload)
    return payload


def _response_json(response) -> dict:
    data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise PipelineError(
            "invalid_worker_response", "The worker response exceeded the evidence size limit."
        )
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise PipelineError(
            "invalid_worker_response", "The worker returned invalid JSON evidence."
        ) from None
    if not isinstance(payload, dict):
        raise PipelineError("invalid_worker_response", "The worker response must be a JSON object.")
    return payload
