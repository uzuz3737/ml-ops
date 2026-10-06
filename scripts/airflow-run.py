"""Trigger the pinned Airflow REST API and verify the exact deployed API version.

Invoked inside pipeline-worker by both bootstrap wrappers; only Python stdlib is
needed. A successful trigger alone never produces a successful exit status.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from mlops_project.pipelines.contracts import (
    STAGES,
    PipelineError,
    confined_path,
    data_file_reference,
    load_config,
    validate_run_id,
)

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
DAG_ID = "credit_default_pipeline"


class LaunchError(RuntimeError):
    """A safe operational error that does not include credentials or raw data."""


class StatusError(LaunchError):
    def __init__(self, method: str, status: int):
        super().__init__(f"{method} request failed with HTTP {status}")
        self.status = status


class UnavailableError(LaunchError):
    """A service may still be starting."""


def request_json(url: str, *, method: str = "GET", payload=None, authorization=None):
    headers = {"Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    data = None
    if payload is not None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(
            Request(url, data=data, headers=headers, method=method), timeout=15
        ) as response:
            body = response.read(2_000_001)
    except HTTPError as error:
        status = error.code
        error.close()
        raise StatusError(method, status) from None
    except (URLError, TimeoutError, OSError):
        raise UnavailableError(f"{method} request could not reach the required service") from None
    if len(body) > 2_000_000:
        raise LaunchError("Service response exceeds the permitted size")
    try:
        result = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise LaunchError("Service returned invalid JSON") from None
    if not isinstance(result, dict):
        raise LaunchError("Service response must be a JSON object")
    return result


def read_record(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise LaunchError(f"Required evidence is missing or invalid: {path.name}") from None
    if not isinstance(value, dict):
        raise LaunchError(f"Evidence must be an object: {path.name}")
    return value


def verify_deployment(project_root: Path, run_id: str, api_url: str, config: dict | None = None):
    """Check persisted acknowledgement plus live readiness and prediction shape."""
    config = config or {}
    try:
        artifacts = confined_path(
            config.get("pipeline", {}).get("artifact_root", "artifacts"), project_root
        )
        serving = config.get("serving", {})
        deployments = confined_path(
            serving.get("deployments_dir", "artifacts/deployments"), project_root
        )
        fixture_path = confined_path(
            serving.get("smoke_fixture", "examples/predict.json"), project_root
        )
    except (PipelineError, AttributeError, TypeError):
        raise LaunchError(
            "Deployment evidence and smoke fixture must use valid project-local paths"
        ) from None
    run_directory = artifacts / "runs" / run_id
    deploy_record = read_record(run_directory / "deploy.json")
    verify_record = read_record(run_directory / "verify.json")
    for step, record in (("deploy", deploy_record), ("verify", verify_record)):
        if (
            type(record.get("contract_version")) is not int
            or record["contract_version"] != 1
            or record.get("run_id") != run_id
            or record.get("step") != step
            or record.get("state") != "succeeded"
        ):
            raise LaunchError(
                "Deployment and verification evidence must match the successful pipeline run"
            )
    deployment = deploy_record.get("result", {})
    if not isinstance(deployment, dict):
        raise LaunchError("Deploy result is invalid")
    deployment_id = deployment.get("deployment_id")
    version = deployment.get("model_version")
    if not isinstance(deployment_id, str) or not IDENTIFIER.fullmatch(deployment_id):
        raise LaunchError("Deploy result must identify a safe deployment_id")
    if not isinstance(version, str) or not version:
        raise LaunchError("Deploy result must identify an exact string model_version")
    verification = verify_record.get("result", {})
    if (
        not isinstance(verification, dict)
        or verification.get("passed") is not True
        or verification.get("model_version") != version
        or verification.get("deployment_id") != deployment_id
    ):
        raise LaunchError("Verification evidence does not match the deployment's exact identity")
    acknowledgement = read_record(deployments / "acknowledgements" / f"{deployment_id}.json")
    if (
        type(acknowledgement.get("contract_version")) is not int
        or acknowledgement["contract_version"] != 1
        or acknowledgement.get("status") != "loaded"
        or acknowledgement.get("deployment_id") != deployment_id
        or acknowledgement.get("model_version") != version
    ):
        raise LaunchError("Runtime acknowledgement does not match the deployed version")
    ready = request_json(api_url.rstrip("/") + "/ready")
    if ready.get("status") != "ready" or ready.get("model_version") != version:
        raise LaunchError("Live readiness does not confirm the exact deployed version")
    fixture = read_record(fixture_path)
    instances = fixture.get("instances")
    if not isinstance(instances, list) or not instances:
        raise LaunchError("Prediction smoke fixture needs nonempty instances")
    prediction = request_json(api_url.rstrip("/") + "/predict", method="POST", payload=fixture)
    predictions = prediction.get("predictions")
    if (
        prediction.get("model_version") != version
        or not isinstance(predictions, list)
        or len(predictions) != len(instances)
    ):
        raise LaunchError("Prediction smoke response has the wrong version or count")
    for item in predictions:
        if (
            not isinstance(item, dict)
            or type(item.get("label")) is not int
            or item["label"] not in (0, 1)
        ):
            raise LaunchError("Prediction smoke labels violate the API contract")
        probability = item.get("default_probability")
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise LaunchError("Prediction smoke probabilities violate the API contract")
    return {
        "deployment_id": deployment_id,
        "model_version": version,
        "prediction_count": len(predictions),
    }


def explain_failure(project_root: Path, run_id: str, config: dict) -> None:
    """Print the stage that stopped the run and, for bad data, what was wrong."""
    try:
        artifacts = confined_path(
            config["pipeline"].get("artifact_root", "artifacts"), project_root
        )
    except PipelineError:
        return
    run_directory = artifacts / "runs" / run_id
    for step in STAGES:
        try:
            record = read_record(run_directory / f"{step}.json")
        except LaunchError:
            continue
        if record.get("state") == "failed":
            error = record.get("error", {})
            print(f"Stopped at stage '{step}': {error.get('code')} — {error.get('message')}")
            break
    try:
        alert = read_record(run_directory / "data-validation-alert.json")
    except LaunchError:
        return
    print(f"Data validation alert: {len(alert.get('failures', []))} failed checks")
    for key in ("missing_columns", "unexpected_columns"):
        if alert.get(key):
            print(f"  {key}: {', '.join(alert[key])}")
    for item in alert.get("failure_summary", [])[:10]:
        print(f"  {item['field']}: {item['rule']} in {item['rows']} rows")
    print(f"  details: artifacts/runs/{run_id}/validation-report.json")


def launch(args):
    project_root = Path(args.project_root).resolve()
    config = (project_root / args.config).resolve()
    try:
        relative_config = config.relative_to(project_root / "configs")
    except ValueError:
        raise LaunchError("Config must stay inside the project's configs directory") from None
    if not config.is_file() or relative_config.suffix not in (".yaml", ".yml"):
        raise LaunchError("Config must identify an existing YAML file")
    try:
        project_config, _ = load_config(config, project_root)
    except PipelineError:
        raise LaunchError("Project configuration does not satisfy the pipeline contract") from None
    if project_config["pipeline"].get("dag_id", DAG_ID) != DAG_ID:
        raise LaunchError("pipeline.dag_id must match the installed credit_default_pipeline DAG")
    run_id = args.run_id or datetime.now(UTC).strftime("p0-%Y%m%dT%H%M%S-") + uuid4().hex[:8]
    try:
        validate_run_id(run_id)
    except PipelineError:
        raise LaunchError(
            "Run ID must contain 1–128 safe alphanumeric, dot, underscore or hyphen characters"
        ) from None
    conf = {"pipeline_run_id": run_id, "config_path": "configs/" + relative_config.as_posix()}
    data_file = getattr(args, "data_file", "")
    if data_file:
        try:
            conf["data_file"] = data_file_reference(data_file, project_root)["uri"]
        except PipelineError as error:
            raise LaunchError(f"--data-file: {error.message}") from None
    username = os.environ.get("AIRFLOW_ADMIN_USERNAME", "")
    password = os.environ.get("AIRFLOW_ADMIN_PASSWORD", "")
    if not username or not password:
        raise LaunchError("Airflow administrator credentials are required in .env")
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    authorization = "Basic " + token
    airflow_url = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080").rstrip("/")
    dag_url = f"{airflow_url}/api/v1/dags/{DAG_ID}"
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        try:
            request_json(dag_url, authorization=authorization)
            break
        except (StatusError, UnavailableError) as error:
            if isinstance(error, StatusError) and error.status != 404:
                raise
            if time.monotonic() >= deadline:
                raise LaunchError(
                    "Timed out waiting for the DAG; inspect Airflow import errors and authentication"
                ) from None
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    request_json(dag_url, method="PATCH", payload={"is_paused": False}, authorization=authorization)
    run_url = dag_url + "/dagRuns/" + quote(run_id, safe="")
    request_json(
        dag_url + "/dagRuns",
        method="POST",
        payload={"dag_run_id": run_id, "conf": conf},
        authorization=authorization,
    )
    print(f"Triggered Airflow run {run_id}; waiting for its terminal result.", flush=True)
    last_state = None
    while time.monotonic() < deadline:
        record = request_json(run_url, authorization=authorization)
        state = record.get("state")
        if state != last_state:
            print(f"Airflow run state: {state}", flush=True)
            last_state = state
        if state == "failed":
            explain_failure(project_root, run_id, project_config)
            raise LaunchError(
                f"Airflow run {run_id} failed; inspect task logs and artifacts/runs/{run_id}"
            )
        if state == "success":
            verification = verify_deployment(
                project_root, run_id, os.environ.get("API_URL", "http://api:8000"), project_config
            )
            evidence = {
                "run_id": run_id,
                "dag_id": DAG_ID,
                "state": "success",
                "verified_at": datetime.now(UTC).isoformat(),
                **verification,
            }
            artifacts = confined_path(
                project_config["pipeline"].get("artifact_root", "artifacts"), project_root
            )
            evidence_path = artifacts / "runs" / run_id / "launch.json"
            temporary = evidence_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            temporary.replace(evidence_path)
            print(
                f"Verified deployment {verification['deployment_id']} running model {verification['model_version']}.",
                flush=True,
            )
            return 0
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise LaunchError(
        f"Airflow run {run_id} exceeded the timeout; it may still be running, inspect it before retrying"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--project-root", default="/workspace")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--data-file", default="", help="Train from this CSV/JSON/XLS(X) under data/ instead of UCI"
    )
    args = parser.parse_args(argv)
    if not 30 <= args.timeout_seconds <= 7200:
        parser.error("--timeout-seconds must be from 30 through 7200")
    try:
        return launch(args)
    except LaunchError as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
