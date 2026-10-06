"""Serialize approved model delivery and verify the actual serving bundle.

P3 implements the shared-volume watcher and API; P2 implements registry
finalization. No alias change or successful manifest write alone is delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import yaml

from mlops_project.artifacts import atomic_json, file_lock, read_record, utc_now

from .contracts import (
    PipelineError,
    configuration_snapshot,
    confined_path,
    json_bytes,
    load_config,
    project_directory,
    resolve_adapter,
    sha256_file,
    validate_run_id,
)
from .runner import _run_adapter

_IDENTITY = (
    "model_name",
    "model_version",
    "schema_version",
    "artifact_uri",
    "artifact_sha256",
    "code_commit",
    "dataset_version",
    "policy_version",
)
_VERSION = re.compile(r"[1-9][0-9]*\Z")


def _number(value, name, maximum=600):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise PipelineError("invalid_config", f"{name} must be finite, positive and <= {maximum}.")
    return float(value)


def _settings(context):
    root = Path(context["project_root"]).resolve()
    run_dir = confined_path(context["run_dir"], root)
    run_dir.mkdir(parents=True, exist_ok=True)
    serving = context["config"].get("serving", {})
    pipeline = context["config"]["pipeline"]
    directory = confined_path(serving.get("deployments_dir", "artifacts/deployments"), root)
    api_url = serving.get("api_url", "")
    parsed = urlsplit(api_url) if isinstance(api_url, str) else None
    if (
        not parsed
        or parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise PipelineError(
            "invalid_config", "serving.api_url requires an HTTP URL without credentials."
        )
    timeout = _number(pipeline.get("deployment_timeout_seconds", 300), "deployment_timeout_seconds")
    poll = _number(pipeline.get("deployment_poll_seconds", 1), "deployment_poll_seconds", timeout)
    stage_budget = pipeline.get("stage_timeout_seconds")
    if (
        type(stage_budget) not in (int, float)
        or stage_budget < 2 * timeout + 2 * min(30, timeout) + 10
    ):
        raise PipelineError(
            "invalid_config",
            "Stage timeout must leave time for delivery, registry finalization and recovery.",
        )
    fixture = read_record(
        confined_path(serving.get("smoke_fixture", "examples/predict.json"), root, must_exist=True)
    )
    if not isinstance(fixture.get("instances"), list) or not fixture["instances"]:
        raise PipelineError("invalid_fixture", "Prediction smoke fixture needs nonempty instances.")
    reference = pipeline.get("registry_finalize")
    resolve_adapter(reference, "register")  # Fail before any serving mutation.
    return root, run_dir, directory, api_url.rstrip("/"), timeout, poll, fixture, reference


def _frozen(context):
    root = Path(context["project_root"]).resolve()
    path = confined_path(context["config_path"], root, must_exist=True)
    digest, _ = configuration_snapshot(context["config"], path, root)
    if digest != context["config_sha256"]:
        raise PipelineError("stale_evidence", "Deployment configuration changed during this run.")


def _approved_manifest(context, root, run_dir):
    _frozen(context)
    approval = context.get("inputs", {}).get("approve", {})
    if (
        type(approval.get("contract_version")) is not int
        or approval.get("contract_version") != 1
        or approval.get("passed") is not True
        or approval.get("status") != "approved"
        or approval.get("run_id") != context["run_id"]
    ):
        raise PipelineError(
            "unapproved_deployment", "Delivery requires this run's passed approval."
        )
    report_path = run_dir / "gate-report.json"
    report = read_record(report_path)
    expected_sha = approval.get("gate_report_sha256")
    if expected_sha != sha256_file(report_path):
        raise PipelineError("artifact_mismatch", "Approval gate report checksum changed.")
    if (
        report.get("passed") is not True
        or report.get("run_id") != context["run_id"]
        or report.get("config_sha256") != context["config_sha256"]
        or report.get("candidate_model_version") != approval.get("model_version")
        or report.get("contract_version") != 1
        or not isinstance(report.get("checks"), list)
        or len(report["checks"]) != 5
        or any(
            check.get("passed") is not True for check in report["checks"] if isinstance(check, dict)
        )
        or any(not isinstance(check, dict) for check in report["checks"])
    ):
        raise PipelineError(
            "stale_evidence", "Approval does not match the passed run/config/model report."
        )
    policy = confined_path(
        context["config"].get("quality_gates", "configs/quality_gates.yaml"), root, must_exist=True
    )
    if report.get("policy_sha256") != sha256_file(policy):
        raise PipelineError("stale_evidence", "The approval policy changed after the gate report.")
    schema_path = confined_path(
        context["config"].get("schema_config", "configs/data_schema.yaml"), root, must_exist=True
    )
    try:
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise PipelineError(
            "invalid_config", "Serving schema configuration is unreadable."
        ) from None
    if not isinstance(schema, dict) or schema.get("schema_version") != approval.get(
        "schema_version"
    ):
        raise PipelineError(
            "stale_evidence", "Candidate schema differs from the configured serving schema."
        )
    confined_path(schema.get("schema_path", "schemas/credit_default.pbtxt"), root, must_exist=True)
    for field in _IDENTITY:
        if (
            not isinstance(approval.get(field), str)
            or not approval[field]
            or (field != "model_version" and report.get(field) != approval[field])
        ):
            raise PipelineError(
                "stale_evidence", f"Approval/report {field} identity is missing or mismatched."
            )
    if approval["model_name"] != context["config"]["serving"].get("model_name"):
        raise PipelineError("stale_evidence", "Approval identifies another serving model.")
    if not _VERSION.fullmatch(approval["model_version"]):
        raise PipelineError(
            "invalid_model_version", "Delivery requires an exact positive decimal MLflow version."
        )
    if not isinstance(approval.get("approval_id"), str) or not approval["approval_id"]:
        raise PipelineError("unapproved_deployment", "Approval must identify its policy decision.")
    return {
        "contract_version": 1,
        "action": "deploy",
        "deployment_id": "deploy-" + hashlib.sha256(context["run_id"].encode()).hexdigest()[:24],
        "run_id": context["run_id"],
        **{field: approval[field] for field in _IDENTITY},
        "approval_id": approval["approval_id"],
        "gate_report_uri": report_path.relative_to(root).as_posix(),
        "gate_report_sha256": expected_sha,
        "config_sha256": context["config_sha256"],
        "policy_sha256": report["policy_sha256"],
    }


def _confirmed(path, root, config):
    if not path.exists():
        return None
    value = read_record(path)
    if value.get("confirmed") is not True or value.get("contract_version") != 1:
        raise PipelineError(
            "invalid_deployment_state", "Rollback targets must be confirmed approved bundles."
        )
    validate_run_id(value.get("deployment_id"))
    for field in (*_IDENTITY, "approval_id", "gate_report_uri", "gate_report_sha256"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise PipelineError(
                "invalid_deployment_state", "Confirmed bundle identity is incomplete."
            )
    if not _VERSION.fullmatch(value["model_version"]):
        raise PipelineError(
            "invalid_model_version", "Rollback requires an exact registered model version."
        )
    schema_path = confined_path(
        config.get("schema_config", "configs/data_schema.yaml"), root, must_exist=True
    )
    try:
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise PipelineError(
            "invalid_config", "Serving schema configuration is unreadable."
        ) from None
    if (
        value["model_name"] != config["serving"].get("model_name")
        or not isinstance(schema, dict)
        or value["schema_version"] != schema.get("schema_version")
    ):
        raise PipelineError(
            "incompatible_rollback",
            "Confirmed bundles must match the configured model name and serving schema.",
        )
    report_path = confined_path(value["gate_report_uri"], root, must_exist=True)
    report = read_record(report_path)
    if value["gate_report_sha256"] != sha256_file(report_path) or report.get("passed") is not True:
        raise PipelineError(
            "artifact_mismatch", "The confirmed rollback bundle lost its approval evidence."
        )
    for field in _IDENTITY:
        expected = report.get("candidate_model_version" if field == "model_version" else field)
        if expected != value[field]:
            raise PipelineError(
                "stale_evidence", "Rollback bundle and approval report identities differ."
            )
    return value


def _http(api_url, endpoint, timeout, payload=None):
    data = json_bytes(payload) if payload is not None else None
    request = Request(
        api_url + endpoint,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(2_000_001)
            status = response.status
    except HTTPError as error:
        try:
            if error.code == 503:
                return 503, {}
            raise PipelineError(
                "serving_probe_failed", "API returned an unexpected HTTP error."
            ) from None
        finally:
            error.close()
    except (OSError, TimeoutError, URLError):
        raise PipelineError(
            "serving_unavailable", "The API could not be reached for verification."
        ) from None
    try:
        result = json.loads(body)
    except (UnicodeError, ValueError):
        raise PipelineError("invalid_response", "The API returned invalid JSON.") from None
    if len(body) > 2_000_000 or not isinstance(result, dict):
        raise PipelineError("invalid_response", "The API response must be a bounded JSON object.")
    return status, result


def _wait(directory, manifest, api_url, timeout, poll, fixture, *, unload=False):
    deadline = time.monotonic() + timeout
    last_error = PipelineError(
        "deployment_timeout", "Runtime did not acknowledge the requested bundle."
    )
    while time.monotonic() < deadline:
        try:
            ack = read_record(directory / "acknowledgements" / f"{manifest['deployment_id']}.json")
            if (
                type(ack.get("contract_version")) is not int
                or ack.get("contract_version") != 1
                or ack.get("deployment_id") != manifest["deployment_id"]
            ):
                raise PipelineError(
                    "stale_acknowledgement", "Runtime ACK belongs to another deployment."
                )
            if ack.get("status") in ("failed", "error", "rejected"):
                raise PipelineError(
                    "model_load_failed", "Runtime rejected the requested model bundle."
                )
            request_timeout = max(0.001, min(5, deadline - time.monotonic()))
            if unload:
                if ack.get("status") != "unloaded":
                    raise PipelineError(
                        "stale_acknowledgement", "Runtime has not acknowledged unloading."
                    )
                ready_status, _ = _http(api_url, "/ready", request_timeout)
                health_status, _ = _http(
                    api_url, "/health", max(0.001, min(5, deadline - time.monotonic()))
                )
                if ready_status != 503 or health_status != 200:
                    raise PipelineError(
                        "unload_unverified", "Unloaded API must be unready and live."
                    )
                return {"status": "unloaded", "deployment_id": manifest["deployment_id"]}
            if (
                ack.get("status") != "loaded"
                or ack.get("model_version") != manifest["model_version"]
            ):
                raise PipelineError(
                    "stale_acknowledgement", "Runtime ACK does not identify the requested version."
                )
            status, ready = _http(api_url, "/ready", request_timeout)
            if (
                status != 200
                or ready.get("status") != "ready"
                or ready.get("model_version") != manifest["model_version"]
                or ready.get("schema_version") != manifest["schema_version"]
            ):
                raise PipelineError(
                    "stale_runtime", "Readiness does not identify the requested model/schema."
                )
            status, prediction = _http(
                api_url, "/predict", max(0.001, min(5, deadline - time.monotonic())), fixture
            )
            predictions = prediction.get("predictions")
            if (
                status != 200
                or prediction.get("model_version") != manifest["model_version"]
                or not isinstance(predictions, list)
                or len(predictions) != len(fixture["instances"])
            ):
                raise PipelineError(
                    "invalid_prediction", "Prediction smoke response has the wrong version/count."
                )
            for item in predictions:
                probability = item.get("default_probability") if isinstance(item, dict) else None
                if (
                    not isinstance(item, dict)
                    or type(item.get("label")) is not int
                    or item["label"] not in (0, 1)
                    or type(probability) not in (int, float)
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                ):
                    raise PipelineError(
                        "invalid_prediction",
                        "Prediction smoke violates binary label/probability contract.",
                    )
            return {
                "status": "loaded",
                "deployment_id": manifest["deployment_id"],
                "model_version": manifest["model_version"],
                "prediction_count": len(predictions),
                "verified_at": utc_now(),
            }
        except PipelineError as error:
            if error.code == "model_load_failed":
                raise
            last_error = error
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    raise PipelineError(
        "deployment_timeout", f"Runtime verification timed out ({last_error.code})."
    )


def _finalize(context, reference, manifest, previous, timeout, run_dir, reason):
    callback_context = {
        **context,
        "step": "register",
        "action": manifest["action"],
        "model_name": context["config"]["serving"]["model_name"],
        "model_version": manifest.get("model_version"),
        "previous_model_version": previous.get("model_version") if previous else None,
        "deployment_id": manifest["deployment_id"],
        "reason": reason,
    }
    result = _run_adapter(reference, callback_context, min(30, timeout), run_dir)
    if (
        not isinstance(result, dict)
        or type(result.get("contract_version")) is not int
        or result.get("contract_version") != 1
        or result.get("passed") is not True
        or "model_version" not in result
        or result["model_version"] != manifest.get("model_version")
    ):
        raise PipelineError(
            "registry_finalize_failed",
            "Registry finalization did not confirm the exact runtime version.",
        )
    json_bytes(result)
    return result


def _audit(context, directory, deployment_id, value, label):
    record = {
        "contract_version": 1,
        "run_id": context["run_id"],
        "deployment_id": deployment_id,
        "recorded_at": utc_now(),
        **value,
    }
    atomic_json(directory / "history" / deployment_id / f"{label}-{uuid4().hex}.json", record)
    path = Path(context["run_dir"]) / f"deployment-{label}.json"
    atomic_json(path, record)
    return [{"uri": path.name, "sha256": sha256_file(path)}]


def _recover(context, settings, failed, previous, reason):
    _, run_dir, directory, api_url, timeout, poll, fixture, reference = settings
    if previous:
        manifest = {
            **previous,
            "action": "rollback",
            "deployment_id": "rollback-" + uuid4().hex,
            "reason": reason,
            "failed_deployment_id": failed["deployment_id"],
        }
        for key in ("confirmed", "confirmed_at", "runtime", "registry"):
            manifest.pop(key, None)
    else:
        manifest = {
            "contract_version": 1,
            "action": "unload",
            "deployment_id": "cancel-" + uuid4().hex,
            "status": "unavailable",
            "reason": "first_deployment_failed",
            "failed_deployment_id": failed["deployment_id"],
        }
    atomic_json(directory / "desired-model.json", manifest)
    proof = _wait(directory, manifest, api_url, timeout, poll, fixture, unload=not previous)
    registry = _finalize(context, reference, manifest, failed, timeout, run_dir, reason)
    if previous:
        atomic_json(
            directory / "active-model.json",
            {
                **manifest,
                "confirmed": True,
                "confirmed_at": utc_now(),
                "runtime": proof,
                "registry": registry,
            },
        )
    else:
        (directory / "active-model.json").unlink(missing_ok=True)
        (directory / "previous-model.json").unlink(missing_ok=True)
    _audit(
        context,
        directory,
        manifest["deployment_id"],
        {"state": "recovered", "manifest": manifest, "runtime": proof, "registry": registry},
        "recovery",
    )
    return {
        "state": "rolled_back" if previous else "unloaded",
        "deployment_id": manifest["deployment_id"],
        "model_version": manifest.get("model_version"),
    }


def _deliver(context, settings, manifest, previous, *, label):
    _, run_dir, directory, api_url, timeout, poll, fixture, reference = settings
    _audit(
        context,
        directory,
        manifest["deployment_id"],
        {"state": "pending", "manifest": manifest, "previous_manifest": previous},
        label,
    )
    atomic_json(directory / "desired-model.json", manifest)
    try:
        proof = _wait(directory, manifest, api_url, timeout, poll, fixture)
        _frozen(context)
        registry = _finalize(
            context,
            reference,
            manifest,
            previous,
            timeout,
            run_dir,
            manifest.get("reason", "approved_delivery"),
        )
        _frozen(context)
        confirmed = {
            **manifest,
            "confirmed": True,
            "confirmed_at": utc_now(),
            "runtime": proof,
            "registry": registry,
        }
        if previous:
            atomic_json(directory / "previous-model.json", previous)
        else:
            (directory / "previous-model.json").unlink(missing_ok=True)
        atomic_json(directory / "active-model.json", confirmed)
        artifacts = _audit(
            context,
            directory,
            manifest["deployment_id"],
            {
                "state": "deployed",
                "manifest": manifest,
                "previous_manifest": previous,
                "runtime": proof,
                "registry": registry,
            },
            label,
        )
    except PipelineError as error:
        recovery = {"state": "recovery_failed"}
        try:
            recovery = _recover(context, settings, manifest, previous, error.code)
        except PipelineError as recovery_error:
            recovery["error"] = recovery_error.as_dict()
        _audit(
            context,
            directory,
            manifest["deployment_id"],
            {
                "state": "failed",
                "manifest": manifest,
                "previous_manifest": previous,
                "error": error.as_dict(),
                "recovery": recovery,
            },
            label,
        )
        if recovery["state"] == "recovery_failed":
            raise PipelineError(
                "recovery_failed",
                "Delivery failed and runtime/registry recovery could not be verified; inspect deployment evidence.",
            ) from None
        raise PipelineError(
            "deployment_failed",
            "Delivery failed; prior runtime/registry state was restored and verified.",
        ) from None
    return {
        "contract_version": 1,
        "run_id": context["run_id"],
        "passed": True,
        "status": "deployed",
        "deployment_id": manifest["deployment_id"],
        "model_version": manifest["model_version"],
        "model_name": manifest["model_name"],
        "schema_version": manifest["schema_version"],
        "artifact_sha256": manifest["artifact_sha256"],
        "artifacts": artifacts,
    }


def deploy(context: dict) -> dict:
    settings = _settings(context)
    root, run_dir, directory, *_ = settings
    manifest = _approved_manifest(context, root, run_dir)
    with file_lock(directory / ".deployment.lock", timeout_seconds=5):
        active = _confirmed(directory / "active-model.json", root, context["config"])
        if active and active["deployment_id"] == manifest["deployment_id"]:
            if any(
                active[field] != manifest[field] for field in (*_IDENTITY, "gate_report_sha256")
            ):
                raise PipelineError(
                    "stale_evidence", "Repeated deployment ID changed its approved identity."
                )
            active = _confirmed(directory / "previous-model.json", root, context["config"])
        return _deliver(context, settings, manifest, active, label="deploy")


def verify(context: dict) -> dict:
    settings = _settings(context)
    root, _, directory, api_url, timeout, poll, fixture, _ = settings
    result = context.get("inputs", {}).get("deploy", {})
    with file_lock(directory / ".deployment.lock", timeout_seconds=5):
        active = _confirmed(directory / "active-model.json", root, context["config"])
        if (
            not active
            or active["deployment_id"] != result.get("deployment_id")
            or active["model_version"] != result.get("model_version")
        ):
            raise PipelineError(
                "stale_evidence", "Verification requires the currently confirmed deployment."
            )
        try:
            proof = _wait(directory, active, api_url, timeout, poll, fixture)
        except PipelineError as error:
            previous = _confirmed(directory / "previous-model.json", root, context["config"])
            try:
                recovery = _recover(context, settings, active, previous, error.code)
            except PipelineError as recovery_error:
                _audit(
                    context,
                    directory,
                    active["deployment_id"],
                    {
                        "state": "verification_failed",
                        "error": error.as_dict(),
                        "recovery": {"state": "recovery_failed", "error": recovery_error.as_dict()},
                    },
                    "verify",
                )
                raise PipelineError(
                    "recovery_failed",
                    "Verification failed and prior serving state could not be restored.",
                ) from None
            _audit(
                context,
                directory,
                active["deployment_id"],
                {"state": "verification_failed", "error": error.as_dict(), "recovery": recovery},
                "verify",
            )
            raise PipelineError(
                "deployment_failed",
                "Verification failed; prior runtime/registry state was restored.",
            ) from None
        artifacts = _audit(
            context,
            directory,
            active["deployment_id"],
            {"state": "verified", "manifest": active, "runtime": proof},
            "verify",
        )
        return {
            "contract_version": 1,
            "run_id": context["run_id"],
            "passed": True,
            "deployment_id": active["deployment_id"],
            "model_version": active["model_version"],
            "artifacts": artifacts,
        }


def rollback(config_path="configs/project.yaml", *, reason, project_root=None):
    """Restore the previous confirmed approved bundle without retraining."""
    if not isinstance(reason, str) or not reason.strip():
        raise PipelineError("invalid_reason", "Rollback requires a recorded reason.")
    root = Path(project_root).resolve() if project_root else project_directory()
    config, path = load_config(config_path, root)
    config_hash, _ = configuration_snapshot(config, path, root)
    run_id = "rollback-" + uuid4().hex
    run_dir = (
        confined_path(config["pipeline"].get("artifact_root", "artifacts"), root) / "runs" / run_id
    )
    context = {
        "contract_version": 1,
        "run_id": run_id,
        "step": "deploy",
        "project_root": str(root),
        "run_dir": str(run_dir),
        "config": config,
        "config_path": str(path),
        "config_sha256": config_hash,
        "inputs": {},
    }
    settings = _settings(context)
    directory = settings[2]
    with file_lock(directory / ".deployment.lock", timeout_seconds=5):
        active = _confirmed(directory / "active-model.json", root, config)
        previous = _confirmed(directory / "previous-model.json", root, config)
        if not active or not previous:
            raise PipelineError(
                "rollback_unavailable",
                "Rollback requires current and previous confirmed approved bundles.",
            )
        manifest = {
            **previous,
            "action": "rollback",
            "deployment_id": run_id,
            "reason": reason,
            "failed_deployment_id": active["deployment_id"],
        }
        for key in ("confirmed", "confirmed_at", "runtime", "registry"):
            manifest.pop(key, None)
        return _deliver(context, settings, manifest, active, label="rollback")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true", required=True)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--project-root")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(rollback(args.config, reason=args.reason, project_root=args.project_root)))
        return 0
    except PipelineError as error:
        print(json.dumps({"error": error.as_dict()}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
