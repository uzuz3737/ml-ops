"""Submit validated labeled alerts to Airflow with deduplication and cooldown."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .contracts import (
    CONTRACT_VERSION,
    PipelineError,
    configuration_snapshot,
    confined_path,
    json_bytes,
    load_config,
    project_directory,
    validate_run_id,
)
from .jsonio import atomic_json, file_lock, read_record


class AirflowClient:
    """Airflow 2 stable REST API client; credentials never enter persisted evidence."""

    def __init__(self):
        self.url = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080").rstrip("/")
        parts = urllib.parse.urlsplit(self.url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise PipelineError(
                "invalid_airflow_url",
                "AIRFLOW_URL must be a valid HTTP(S) endpoint without credentials.",
            )
        username = os.environ.get("AIRFLOW_USERNAME") or os.environ.get("AIRFLOW_ADMIN_USERNAME")
        password = os.environ.get("AIRFLOW_PASSWORD") or os.environ.get("AIRFLOW_ADMIN_PASSWORD")
        if not username or not password:
            raise PipelineError(
                "airflow_credentials_missing",
                "Set AIRFLOW_ADMIN_USERNAME and AIRFLOW_ADMIN_PASSWORD (or AIRFLOW_USERNAME and AIRFLOW_PASSWORD) for the Airflow API.",
            )
        token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        self.headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

    def __call__(self, method: str, path: str, payload: dict | None = None) -> dict:
        request = urllib.request.Request(
            self.url + path,
            method=method,
            data=json_bytes(payload) if payload is not None else None,
            headers=self.headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            code = {404: "airflow_not_found", 409: "airflow_conflict"}.get(
                status, "airflow_rejected"
            )
            raise PipelineError(code, "The Airflow API rejected the retraining request.") from None
        except (OSError, TimeoutError, urllib.error.URLError):
            raise PipelineError(
                "airflow_unavailable",
                "The Airflow API could not be reached; reconcile the recorded pending trigger.",
            ) from None
        try:
            result = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise PipelineError(
                "invalid_airflow_response", "Airflow returned invalid JSON evidence."
            ) from None
        if len(data) > 1024 * 1024 or not isinstance(result, dict):
            raise PipelineError(
                "invalid_airflow_response", "Airflow returned unsupported response evidence."
            )
        return result


def _count(value, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        code = "policy_unset" if name.startswith("minimum_") else "invalid_alert"
        raise PipelineError(code, f"{name} must be an integer of at least {minimum}.")
    return value


def _number(value, name: str, *, minimum: float = 0, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise PipelineError(
            "policy_unset", f"{name} must contain a calibrated finite numeric value."
        )
    return float(value)


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError("invalid_alert", f"{name} must contain a nonempty identifier.")
    return value


def _timestamp(value, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise PipelineError(
            "invalid_alert", f"{name} must be an ISO-8601 timestamp with a timezone."
        ) from None
    if parsed.tzinfo is None:
        raise PipelineError("invalid_alert", f"{name} must include a timezone.")
    return parsed.astimezone(UTC)


def _validate_alert(alert: dict, monitoring: dict, now: datetime) -> str:
    if (
        not isinstance(alert, dict)
        or type(alert.get("contract_version")) is not int
        or alert["contract_version"] != 1
    ):
        raise PipelineError("invalid_alert", "Alert evidence requires contract_version: 1.")
    trigger_id = validate_run_id(alert.get("trigger_id"))
    # Keep the generated pipeline ID within the worker's 128-character limit.
    validate_run_id("retrain-" + trigger_id)
    for field in ("alert_id", "model_version", "dataset_version", "candidate_dataset_version"):
        _text(alert.get(field), field)
    if alert["candidate_dataset_version"] == alert["dataset_version"]:
        raise PipelineError(
            "invalid_alert", "Retraining requires a newly approved candidate dataset version."
        )
    for field in (
        "candidate_dataset_approved",
        "labels_validated",
        "separate_training_validation",
        "final_test_excluded",
        "threshold_crossed",
    ):
        if alert.get(field) is not True:
            raise PipelineError(
                "invalid_alert", f"{field} must explicitly be true before retraining."
            )
    if alert.get("reason") not in {"data_drift", "concept_drift", "quality"}:
        raise PipelineError(
            "invalid_alert", "Alert reason must be data_drift, concept_drift or quality."
        )
    observed = _number(alert.get("observed"), "alert.observed")
    threshold = _number(alert.get("threshold"), "alert.threshold")
    threshold_key = (
        "data_drift_threshold"
        if alert["reason"] == "data_drift"
        else "quality_degradation_threshold"
    )
    configured = _number(monitoring.get(threshold_key), threshold_key)
    if threshold != configured:
        raise PipelineError(
            "stale_evidence", "The alert threshold does not match the calibrated monitoring policy."
        )
    # All reasons use a higher-is-worse drift or quality-degradation score.
    # A caller's threshold_crossed flag alone cannot authorize retraining.
    if observed <= configured:
        raise PipelineError(
            "threshold_not_crossed",
            "The observed drift or quality-degradation score must exceed its calibrated threshold.",
        )
    if _timestamp(alert.get("created_at"), "created_at") > now:
        raise PipelineError("invalid_alert", "Alert creation time must not be in the future.")
    sample_count = _count(alert.get("sample_count"), "sample_count", minimum=1)
    labeled = _count(alert.get("labeled_sample_count"), "labeled_sample_count")
    positives = _count(alert.get("positive_label_count"), "positive_label_count")
    negatives = _count(alert.get("negative_label_count"), "negative_label_count")
    if labeled > sample_count or positives + negatives != labeled:
        raise PipelineError(
            "invalid_alert",
            "Labeled class counts must sum to the labeled count within the observed sample count.",
        )
    coverage = _number(alert.get("label_coverage"), "label_coverage", maximum=1)
    if not math.isclose(coverage, labeled / sample_count, rel_tol=0, abs_tol=1e-12):
        raise PipelineError(
            "invalid_alert", "Label coverage must equal labeled samples divided by all samples."
        )
    minimum_samples = _count(monitoring.get("minimum_samples"), "minimum_samples", minimum=1)
    minimum_labels = _count(
        monitoring.get("minimum_labeled_samples"), "minimum_labeled_samples", minimum=1
    )
    minimum_classes = _count(
        monitoring.get("minimum_class_samples"), "minimum_class_samples", minimum=1
    )
    minimum_coverage = _number(
        monitoring.get("minimum_label_coverage"), "minimum_label_coverage", maximum=1
    )
    if (
        sample_count < minimum_samples
        or labeled < minimum_labels
        or positives < minimum_classes
        or negatives < minimum_classes
        or coverage < minimum_coverage
    ):
        raise PipelineError(
            "insufficient_labels",
            "The alert lacks the calibrated sample count, class counts or label coverage for controlled retraining.",
        )
    return trigger_id


def _matching_run(response: dict, run_id: str, conf: dict) -> bool:
    return (
        isinstance(response, dict)
        and response.get("dag_run_id") == run_id
        and isinstance(response.get("conf"), dict)
        and all(response["conf"].get(key) == value for key, value in conf.items())
    )


def _ensure_no_active_runs(client, path: str) -> None:
    offset = 0
    while True:
        response = client("GET", f"{path}?limit=100&offset={offset}", None)
        runs = response.get("dag_runs")
        total = response.get("total_entries")
        if not isinstance(runs, list) or type(total) is not int or total < 0:
            raise PipelineError(
                "invalid_airflow_response", "Airflow must return paginated DAG-run evidence."
            )
        if any(
            not isinstance(run, dict)
            or run.get("state") not in {"queued", "running", "success", "failed"}
            for run in runs
        ):
            raise PipelineError(
                "invalid_airflow_response",
                "Airflow returned a missing or unsupported DAG-run state.",
            )
        if any(run["state"] in {"queued", "running"} for run in runs):
            raise PipelineError(
                "active_run",
                "A pipeline run is already queued or running; retraining submission is blocked.",
            )
        offset += len(runs)
        if offset >= total:
            return
        if not runs:
            raise PipelineError(
                "invalid_airflow_response", "Airflow pagination omitted remaining DAG runs."
            )


def submit_retraining(
    alert: dict,
    config_path: str | Path,
    state_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    now: datetime | None = None,
    client=None,
) -> dict:
    root = Path(project_root).resolve() if project_root else project_directory()
    config, config_file = load_config(config_path, root)
    monitor_file = confined_path(
        config.get("monitoring_config", "configs/monitoring.yaml"), root, must_exist=True
    )
    try:
        monitoring = yaml.safe_load(monitor_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise PipelineError(
            "invalid_policy", "Monitoring policy must be readable UTF-8 YAML."
        ) from None
    if (
        not isinstance(monitoring, dict)
        or type(monitoring.get("contract_version")) is not int
        or monitoring["contract_version"] != 1
        or not isinstance(monitoring.get("retraining"), dict)
    ):
        raise PipelineError(
            "invalid_policy",
            "Monitoring policy requires contract_version: 1 and a retraining mapping.",
        )
    policy = monitoring["retraining"]
    if policy.get("enabled") is not True:
        raise PipelineError(
            "retraining_disabled",
            "Retraining is disabled until the owners calibrate monitoring thresholds and supply validated labeled data.",
        )
    for flag in ("require_valid_labels", "require_separate_training_validation"):
        if policy.get(flag) is not True:
            raise PipelineError("invalid_policy", f"retraining.{flag} must be enabled.")
    current = now or datetime.now(UTC)
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise PipelineError(
            "invalid_time", "The controller time must be a timezone-aware datetime."
        )
    current = current.astimezone(UTC)
    cooldown = _number(policy.get("cooldown_seconds"), "retraining.cooldown_seconds")
    trigger_id = _validate_alert(alert, monitoring, current)
    dag_id = validate_run_id(policy.get("dag_id"))
    expected_dag = config["pipeline"].get("dag_id", "credit_default_pipeline")
    if dag_id != expected_dag:
        raise PipelineError(
            "invalid_policy", "The monitoring policy must target the configured training DAG."
        )
    policy_hash, _ = configuration_snapshot(config, config_file, root)
    alert_hash = hashlib.sha256(json_bytes(alert)).hexdigest()
    artifacts = confined_path(config["pipeline"].get("artifact_root", "artifacts"), root)
    configured_state = config["pipeline"].get("retraining_state_dir", artifacts / "retraining")
    state_root = confined_path(state_dir or configured_state, root)
    if state_root != confined_path(configured_state, root):
        raise PipelineError(
            "invalid_config",
            "A custom state_dir must match pipeline.retraining_state_dir so stage workers can verify receipts.",
        )
    state_root.mkdir(parents=True, exist_ok=True)
    run_id = "retrain-" + trigger_id
    conf = {
        "mode": "retrain",
        "trigger_id": trigger_id,
        "pipeline_run_id": run_id,
        "config_path": config_file.relative_to(root).as_posix(),
        "candidate_dataset_version": alert["candidate_dataset_version"],
    }
    run_path = "/api/v1/dags/" + urllib.parse.quote(dag_id, safe="") + "/dagRuns"
    exact_path = run_path + "/" + urllib.parse.quote(run_id, safe="")
    with file_lock(state_root / ".controller.lock"):
        receipt_path = state_root / f"{trigger_id}.json"
        if receipt_path.exists():
            receipt = read_record(receipt_path)
            if (
                receipt.get("alert_sha256") != alert_hash
                or receipt.get("policy_sha256") != policy_hash
            ):
                raise PipelineError(
                    "trigger_conflict",
                    "This trigger ID already belongs to different alert or policy evidence.",
                )
            if receipt.get("state") == "submitted":
                return {**receipt, "state": "already_submitted"}
        else:
            receipt = {
                "contract_version": CONTRACT_VERSION,
                "trigger_id": trigger_id,
                "pipeline_run_id": run_id,
                "state": "pending",
                "alert_sha256": alert_hash,
                "policy_sha256": policy_hash,
                "created_at": current.isoformat().replace("+00:00", "Z"),
                "retraining_request": {
                    field: alert[field]
                    for field in (
                        "trigger_id",
                        "alert_id",
                        "model_version",
                        "dataset_version",
                        "candidate_dataset_version",
                        "candidate_dataset_approved",
                        "labels_validated",
                        "separate_training_validation",
                        "final_test_excluded",
                        "reason",
                    )
                },
            }
        requester = client if client is not None else AirflowClient()
        # A crashed or ambiguous prior POST must be reconciled before another POST.
        if receipt_path.exists():
            try:
                existing = requester("GET", exact_path, None)
            except PipelineError as error:
                if error.code != "airflow_not_found":
                    raise
            else:
                if not _matching_run(existing, run_id, conf):
                    raise PipelineError(
                        "trigger_conflict",
                        "Airflow already has this run ID with different retraining evidence.",
                    )
                return _submitted(receipt, receipt_path, state_root, current)
        last_path = state_root / "last-submission.json"
        if last_path.exists():
            last = read_record(last_path)
            last_time = _timestamp(last.get("submitted_at"), "last submitted_at")
            if (current - last_time).total_seconds() < cooldown:
                raise PipelineError(
                    "cooldown",
                    "Retraining cooldown has not elapsed since the last accepted submission.",
                )
        _ensure_no_active_runs(requester, run_path)
        current_config, _ = load_config(config_file, root)
        current_hash, _ = configuration_snapshot(current_config, config_file, root)
        if current_hash != policy_hash:
            raise PipelineError(
                "stale_evidence", "Retraining configuration changed before submission."
            )
        atomic_json(receipt_path, receipt)
        try:
            response = requester("POST", run_path, {"dag_run_id": run_id, "conf": conf})
        except PipelineError as error:
            if error.code not in {
                "airflow_conflict",
                "airflow_unavailable",
                "invalid_airflow_response",
            }:
                raise
            try:
                response = requester("GET", exact_path, None)
            except PipelineError:
                raise PipelineError(
                    "submission_pending",
                    "Airflow submission could not be reconciled; retry the same trigger ID after checking Airflow.",
                ) from None
        if not _matching_run(response, run_id, conf):
            raise PipelineError(
                "trigger_conflict", "Airflow run evidence does not match this retraining trigger."
            )
        return _submitted(receipt, receipt_path, state_root, current)


def _submitted(receipt: dict, receipt_path: Path, state_root: Path, now: datetime) -> dict:
    timestamp = now.isoformat().replace("+00:00", "Z")
    result = {**receipt, "state": "submitted", "submitted_at": timestamp}
    atomic_json(
        state_root / "last-submission.json",
        {"trigger_id": receipt["trigger_id"], "submitted_at": timestamp},
    )
    atomic_json(receipt_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alert", required=True)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--project-root")
    args = parser.parse_args()
    root = Path(args.project_root).resolve() if args.project_root else project_directory()
    try:
        alert_file = confined_path(args.alert, root, must_exist=True)
        alert = read_record(alert_file)
        print(json.dumps(submit_retraining(alert, args.config, project_root=root), indent=2))
        return 0
    except PipelineError as error:
        print(
            json.dumps({"contract_version": CONTRACT_VERSION, "error": error.as_dict()}, indent=2)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
