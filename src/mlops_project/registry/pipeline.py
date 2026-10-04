"""Registry stages; P0 remains responsible for approval and runtime verification."""

import hashlib
import json
import re
import tempfile
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlops_project.pipelines.contracts import (
    confined_path,
    json_bytes,
    sha256_file,
    validate_run_id,
)
from mlops_project.pipelines.runner import atomic_json, file_lock
from mlops_project.training.bundle import load_bundle
from mlops_project.training.inputs import verified_path
from mlops_project.training.pipeline import tracking_uri


def _client(context):
    uri = tracking_uri(context)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _missing(exc):
    return exc.error_code == "RESOURCE_DOES_NOT_EXIST"


def register(context):
    result = context["inputs"]["evaluate"]
    if result.get("passed") is not True or result.get("run_id") != context["run_id"]:
        raise ValueError("Registration requires passed evaluation from this run")
    root = Path(context["project_root"])
    bundle_path = verified_path(
        {"uri": result["artifact_uri"], "sha256": result["artifact_sha256"]}, root
    )
    bundle = load_bundle(bundle_path, result["artifact_sha256"], result["schema_version"])
    for key in ("run_id", "code_commit", "dataset_version", "schema_version", "config_sha256"):
        if bundle["manifest"][key] != result[key]:
            raise ValueError("Registration bundle provenance differs from evaluation")
    if result["config_sha256"] != context["config_sha256"]:
        raise ValueError("Registration configuration changed")
    client = _client(context)
    name = context["config"]["serving"]["model_name"]
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
        raise ValueError("Use a portable alphanumeric registered model name")
    run = client.get_run(result["mlflow_run_id"])
    if (
        run.info.status != "FINISHED"
        or result["mlflow_model_uri"] != f"runs:/{run.info.run_id}/model"
    ):
        raise ValueError("Only a finished exact tracked experiment can be registered")
    for key in (
        "artifact_sha256",
        "code_commit",
        "dataset_version",
        "schema_version",
        "config_sha256",
        "run_id",
    ):
        if run.data.tags.get(key) != result[key]:
            raise ValueError(f"Tracking/evaluation provenance differs: {key}")
    try:
        client.get_registered_model(name)
    except MlflowException as exc:
        if not _missing(exc):
            raise
        client.create_registered_model(name)
    directory = confined_path(context["run_dir"], root)
    # P0 serializes DAG runs; this additionally serializes independent direct calls.
    with file_lock(directory / ".p2-register.lock"):
        matching = [
            v
            for v in client.search_model_versions(f"name = '{name}'")
            if v.tags.get("pipeline_run_id") == context["run_id"]
        ]
        if matching:
            if (
                len(matching) != 1
                or matching[0].tags.get("artifact_sha256") != result["artifact_sha256"]
                or matching[0].run_id != result["mlflow_run_id"]
            ):
                raise ValueError("Conflicting existing registry identity for run")
            version = matching[0]
        else:
            if client.get_run(result["mlflow_run_id"]).info.status != "FINISHED":
                raise ValueError("Only a finished tracked experiment can be registered")
            tags = {
                "pipeline_run_id": context["run_id"],
                "artifact_sha256": result["artifact_sha256"],
                "code_commit": result["code_commit"],
                "dataset_version": result["dataset_version"],
                "schema_version": result["schema_version"],
                "config_sha256": context["config_sha256"],
                "bundle_uri": result["artifact_uri"],
                "lifecycle_state": "validated",
            }
            version = client.create_model_version(
                name, result["mlflow_model_uri"], run_id=result["mlflow_run_id"], tags=tags
            )
        output = {
            **result,
            "status": "validated",
            "model_name": name,
            "model_version": str(version.version),
            "registry_uri": f"models:/{name}/{version.version}",
        }
        path = directory / "registration.json"
        atomic_json(path, output)
        output["artifacts"] = [
            *result["artifacts"],
            {"uri": "registration.json", "sha256": sha256_file(path)},
        ]
        return output


def record_gate_decision(context, manifest):
    """Persist an actual P0 gate decision; never compute/invent service measurements."""
    root = Path(context["project_root"])
    report_path = confined_path(manifest["gate_report_uri"], root, must_exist=True)
    if sha256_file(report_path) != manifest["gate_report_sha256"]:
        raise ValueError("Gate report checksum mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("contract_version") != 1 or report.get("model_name") != manifest["model_name"]:
        raise ValueError("Invalid gate contract/model identity")
    client = _client(context)
    name, version = manifest["model_name"], manifest["model_version"]
    record = client.get_model_version(name, version)
    for key in (
        "artifact_sha256",
        "schema_version",
        "dataset_version",
        "code_commit",
        "config_sha256",
    ):
        if record.tags.get(key) != report.get(key) or report.get(key) != manifest.get(key):
            raise ValueError(f"Gate/registered-model provenance mismatch: {key}")
    if (
        report.get("candidate_model_version") != version
        or report.get("run_id") != record.tags["pipeline_run_id"]
    ):
        raise ValueError("Gate decision belongs to another model/run")
    if type(report.get("passed")) is not bool or not report.get("checks"):
        raise ValueError("Malformed gate decision")
    if any(type(c.get("passed")) is not bool for c in report["checks"]):
        raise ValueError("Invalid gate checks")
    if report["passed"] != all(c["passed"] for c in report["checks"]):
        raise ValueError("Gate outcome contradicts checks")
    state = "approved" if report["passed"] else "rejected"
    if record.tags.get("lifecycle_state") == "deployed":
        if (
            state != "approved"
            or record.tags.get("gate_report_sha256") != manifest["gate_report_sha256"]
        ):
            raise ValueError("Cannot alter the gate decision of a deployed version")
    else:
        client.set_model_version_tag(name, version, "lifecycle_state", state)
    client.set_model_version_tag(
        name, version, "gate_report_sha256", manifest["gate_report_sha256"]
    )
    client.set_model_version_tag(name, version, "approver", report["approver"])
    client.set_model_version_tag(name, version, "approval_time", report["created_at"])
    return state


def _alias(client, name, alias):
    # MLflow 2.19 reports an absent alias as INVALID_PARAMETER_VALUE; use the
    # model's mapping so missing aliases do not hide actual connection failures.
    version = client.get_registered_model(name).aliases.get(alias)
    return str(version) if version is not None else None


def _set_alias(client, name, alias, version):
    if version is None:
        if _alias(client, name, alias) is not None:
            client.delete_registered_model_alias(name, alias)
    else:
        client.set_registered_model_alias(name, alias, version)


def finalize_deployment(context):
    """Called ONLY by P0 after ACK/readiness/exact-version prediction verification."""
    action, version, name = context["action"], context["model_version"], context["model_name"]
    if action not in {"deploy", "rollback", "unload"} or (action == "unload") != (version is None):
        raise ValueError("Invalid deployment action/version")
    if version is not None and (
        not isinstance(version, str) or not version.isdecimal() or int(version) < 1
    ):
        raise ValueError("Exact positive decimal model version required")
    deployment_id = validate_run_id(context["deployment_id"])
    root = Path(context["project_root"])
    directory = confined_path(
        context["config"]["serving"].get("deployments_dir", "artifacts/deployments"), root
    )
    desired = json.loads((directory / "desired-model.json").read_text(encoding="utf-8"))
    if type(desired.get("contract_version")) is not int or desired["contract_version"] != 1:
        raise ValueError("Invalid desired-manifest contract")
    if (
        desired.get("action") != action
        or desired.get("deployment_id") != deployment_id
        or desired.get("model_version") != version
    ):
        raise ValueError("Callback does not match the controller's desired manifest")
    if name != context["config"]["serving"]["model_name"] or (
        version and desired.get("model_name") != name
    ):
        raise ValueError("Unexpected registered model")
    client = _client(context)
    identity = {
        "action": action,
        "model_name": name,
        "model_version": version,
        "previous_model_version": context.get("previous_model_version"),
        "deployment_id": deployment_id,
        "reason": context["reason"],
        "desired_sha256": hashlib.sha256(json_bytes(desired)).hexdigest(),
    }
    journal = directory / "registry" / f"{deployment_id}.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(directory / "registry" / ".finalize.lock"):
        old = {a: _alias(client, name, a) for a in ("champion", "previous")}
        if journal.exists():
            record = json.loads(journal.read_text(encoding="utf-8"))
            if record["identity"] != identity:
                raise ValueError("Deployment ID reused with changed registry evidence")
            if record["state"] == "completed":
                if old != record["after"]:
                    raise ValueError("Completed callback is stale after a newer deployment")
                return {"contract_version": 1, "passed": True, "model_version": version}
        else:
            if action == "deploy":
                state = record_gate_decision(context, desired)
                if state != "approved":
                    raise ValueError("Rejected candidate cannot be deployed")
            elif action == "rollback":
                target = client.get_model_version(name, version)
                if not target.tags.get("gate_report_sha256") or target.tags.get(
                    "lifecycle_state"
                ) not in {"approved", "deployed", "retired", "rolled_back"}:
                    raise ValueError("Rollback requires a previously approved version")
                if target.tags["artifact_sha256"] != desired.get("artifact_sha256"):
                    raise ValueError("Rollback artifact differs from approved version")
            previous = context.get("previous_model_version")
            if action == "rollback":
                # Restore the target's last healthy mapping, not the failed
                # candidate passed by P0 as previous_model_version.
                previous = target.tags.get("previous_model_version") or None
            if previous == version:
                previous = None
            after = {"champion": version, "previous": previous if action != "unload" else None}
            record = {"identity": identity, "before": old, "after": after, "state": "pending"}
            atomic_json(journal, record)
        try:
            for alias, target in record["after"].items():
                _set_alias(client, name, alias, target)
            previous = record["before"]["champion"]
            if previous and previous != version:
                client.set_model_version_tag(
                    name,
                    previous,
                    "lifecycle_state",
                    "rolled_back" if action in {"rollback", "unload"} else "retired",
                )
            if version:
                if action == "deploy":
                    client.set_model_version_tag(
                        name, version, "previous_model_version", record["after"]["previous"] or ""
                    )
                client.set_model_version_tag(name, version, "lifecycle_state", "deployed")
                client.set_model_version_tag(name, version, "deployment_id", deployment_id)
                client.set_model_version_tag(name, version, "deployment_action", action)
                client.set_model_version_tag(name, version, "deployment_reason", context["reason"])
            if {a: _alias(client, name, a) for a in record["after"]} != record["after"]:
                raise RuntimeError("Registry aliases did not converge")
        except Exception:
            # Preserve pending journal for retry; P0 handles verified runtime recovery.
            for alias, target in record["before"].items():
                _set_alias(client, name, alias, target)
            raise
        atomic_json(journal, {**record, "state": "completed"})
    return {"contract_version": 1, "passed": True, "model_version": version}


def retrieve_bundle(context, model_name, model_version, destination):
    """Retrieve a checksum-verified immutable tracked bundle for P3/recovery."""
    if not isinstance(model_version, str) or not re.fullmatch(r"[1-9][0-9]*", model_version):
        raise ValueError("Retrieve by an exact version, never an alias")
    client = _client(context)
    version = client.get_model_version(model_name, model_version)
    root = Path(context["project_root"])
    path = confined_path(destination, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        downloaded = Path(
            client.download_artifacts(version.run_id, "bundle/bundle.joblib", temporary)
        )
        bundle = load_bundle(
            downloaded, version.tags["artifact_sha256"], version.tags["schema_version"]
        )
        for key in ("code_commit", "dataset_version", "schema_version", "config_sha256"):
            if bundle["manifest"][key] != version.tags[key]:
                raise ValueError("Retrieved artifact/registry provenance differs")
        temporary_target = Path(temporary) / "verified.joblib"
        temporary_target.write_bytes(downloaded.read_bytes())
        temporary_target.replace(path)
    return {
        "model_name": model_name,
        "model_version": model_version,
        "artifact_uri": path.relative_to(root).as_posix(),
        "artifact_sha256": version.tags["artifact_sha256"],
    }
