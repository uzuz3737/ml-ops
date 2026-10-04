import json
from pathlib import Path

import pytest
from mlflow import MlflowClient

from mlops_project.pipelines.contracts import sha256_file
from mlops_project.pipelines.runner import atomic_json
from mlops_project.registry.pipeline import finalize_deployment, record_gate_decision


@pytest.fixture
def registry_context(tmp_path):
    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_registered_model("credit-default")
    experiment = client.create_experiment(
        "registry-tests", artifact_location=(tmp_path / "tracked").as_uri()
    )
    for i in range(1, 4):
        run = client.create_run(experiment)
        client.set_terminated(run.info.run_id)
        client.create_model_version(
            "credit-default",
            (tmp_path / f"bundle{i}").as_uri(),
            run_id=run.info.run_id,
            tags={
                "pipeline_run_id": f"p2-{i}",
                "artifact_sha256": str(i) * 64,
                "schema_version": "credit-v1",
                "dataset_version": "data-v1",
                "code_commit": "a" * 40,
                "config_sha256": "b" * 64,
                "lifecycle_state": "validated",
            },
        )
    directory = tmp_path / "artifacts/deployments"
    directory.mkdir(parents=True)
    return {
        "project_root": str(tmp_path),
        "run_id": "p2-1",
        "run_dir": str(tmp_path / "artifacts/runs/p2-1"),
        "config_sha256": "b" * 64,
        "config": {
            "training": {"tracking_uri": uri},
            "serving": {"model_name": "credit-default", "deployments_dir": "artifacts/deployments"},
        },
    }, client


def desired(context, action, version, previous=None, passed=True, deployment_id=None):
    root = Path(context["project_root"])
    directory = root / "artifacts/deployments"
    identity = {
        "contract_version": 1,
        "model_name": "credit-default",
        "model_version": version,
        "artifact_sha256": (version or "0") * 64,
        "schema_version": "credit-v1",
        "dataset_version": "data-v1",
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "action": action,
        "deployment_id": deployment_id or f"{action}-{version or 'none'}",
    }
    report = {
        **identity,
        "candidate_model_version": version,
        "run_id": f"p2-{version}",
        "passed": passed,
        "checks": [{"passed": passed}],
        "approver": "policy:test",
        "created_at": "2026-10-04T00:00:00Z",
    }
    report_path = directory / f"gate-{version}.json"
    atomic_json(report_path, report)
    manifest = {
        **identity,
        "gate_report_uri": report_path.relative_to(root).as_posix(),
        "gate_report_sha256": sha256_file(report_path),
    }
    atomic_json(directory / "desired-model.json", manifest)
    return {
        **context,
        **identity,
        "previous_model_version": previous,
        "reason": "test live verification",
    }, manifest


def test_real_sqlite_registry_deploy_rollback_unload_and_retries(registry_context):
    context, client = registry_context
    first, _ = desired(context, "deploy", "1")
    assert finalize_deployment(first)["model_version"] == "1"
    assert finalize_deployment(first)["passed"] is True
    second, _ = desired(context, "deploy", "2", "1")
    finalize_deployment(second)
    assert client.get_model_version_by_alias("credit-default", "champion").version == 2
    assert client.get_model_version_by_alias("credit-default", "previous").version == 1
    rollback, _ = desired(context, "rollback", "1", "1")
    finalize_deployment(rollback)
    assert client.get_model_version_by_alias("credit-default", "champion").version == 1
    assert client.get_model_version("credit-default", "2").tags["lifecycle_state"] == "rolled_back"
    unload, _ = desired(context, "unload", None)
    assert finalize_deployment(unload)["model_version"] is None
    assert client.get_registered_model("credit-default").aliases == {}


def test_rejected_gate_retained_without_alias_mutation(registry_context):
    context, client = registry_context
    callback, manifest = desired(context, "deploy", "1", passed=False)
    assert record_gate_decision(context, manifest) == "rejected"
    with pytest.raises(ValueError, match="Rejected"):
        finalize_deployment(callback)
    assert client.get_registered_model("credit-default").aliases == {}
    assert client.get_model_version("credit-default", "1").tags["lifecycle_state"] == "rejected"


def test_bad_gate_identity_cannot_approve(registry_context):
    context, client = registry_context
    callback, manifest = desired(context, "deploy", "1")
    path = Path(context["project_root"]) / manifest["gate_report_uri"]
    report = json.loads(path.read_text())
    report["artifact_sha256"] = "f" * 64
    atomic_json(path, report)
    manifest["gate_report_sha256"] = sha256_file(path)
    atomic_json(
        Path(context["project_root"]) / "artifacts/deployments/desired-model.json", manifest
    )
    with pytest.raises(ValueError, match="provenance"):
        finalize_deployment(callback)
    assert not client.get_registered_model("credit-default").aliases


def test_unapproved_rollback_and_changed_retry_rejected(registry_context):
    context, _ = registry_context
    callback, _ = desired(context, "rollback", "3")
    with pytest.raises(ValueError, match="previously approved"):
        finalize_deployment(callback)
    callback, _ = desired(context, "deploy", "1")
    finalize_deployment(callback)
    with pytest.raises(ValueError, match="reused"):
        finalize_deployment({**callback, "reason": "different evidence"})


def test_alias_failure_restores_mapping_and_pending_retry(registry_context, monkeypatch):
    from mlops_project.registry import pipeline

    context, client = registry_context
    first, _ = desired(context, "deploy", "1")
    finalize_deployment(first)
    second, _ = desired(context, "deploy", "2", "1")
    original = pipeline._set_alias
    failed = False

    def fail_once(client, name, alias, version):
        nonlocal failed
        if alias == "previous" and version == "1" and not failed:
            failed = True
            raise RuntimeError("Injected registry outage")
        return original(client, name, alias, version)

    monkeypatch.setattr(pipeline, "_set_alias", fail_once)
    with pytest.raises(RuntimeError, match="outage"):
        finalize_deployment(second)
    assert client.get_model_version_by_alias("credit-default", "champion").version == 1
    finalize_deployment(second)
    assert client.get_model_version_by_alias("credit-default", "champion").version == 2
