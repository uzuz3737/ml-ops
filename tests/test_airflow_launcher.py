"""Exercise the common launcher's HTTP completion and deployment boundary."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mlops_project.pipelines.contracts import STAGES


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "airflow-run.py"
    spec = importlib.util.spec_from_file_location("airflow_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def deployment(tmp_path):
    run_directory = tmp_path / "artifacts" / "runs" / "run-1"
    run_directory.mkdir(parents=True)
    identity = {"deployment_id": "deployment-1", "model_version": "7"}
    (run_directory / "deploy.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "run_id": "run-1",
                "step": "deploy",
                "state": "succeeded",
                "result": identity,
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "verify.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "run_id": "run-1",
                "step": "verify",
                "state": "succeeded",
                "result": {"passed": True, **identity},
            }
        ),
        encoding="utf-8",
    )
    acknowledgements = tmp_path / "artifacts" / "deployments" / "acknowledgements"
    acknowledgements.mkdir(parents=True)
    (acknowledgements / "deployment-1.json").write_text(
        json.dumps({"contract_version": 1, "status": "loaded", **identity}), encoding="utf-8"
    )
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "predict.json").write_text(
        json.dumps({"instances": [{"LIMIT_BAL": 200000}]}), encoding="utf-8"
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "contract_version": 1,
                "pipeline": {
                    "dag_id": "credit_default_pipeline",
                    "stage_timeout_seconds": 5,
                    "adapters": {step: "mlops_project.fixture:run" for step in STAGES},
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def responses(url, **kwargs):
    if url.endswith("/ready"):
        return {"status": "ready", "model_version": "7"}
    if url.endswith("/predict"):
        assert kwargs["method"] == "POST"
        assert kwargs["payload"]["instances"]
        return {"model_version": "7", "predictions": [{"label": 0, "default_probability": 0.1}]}
    raise AssertionError(f"Unexpected request {url}")


def test_verifies_persisted_acknowledgement_and_live_version(launcher, deployment, monkeypatch):
    monkeypatch.setattr(launcher, "request_json", responses)
    assert launcher.verify_deployment(deployment, "run-1", "http://api:8000") == {
        "deployment_id": "deployment-1",
        "model_version": "7",
        "prediction_count": 1,
    }


def test_rejects_alias_change_without_matching_runtime_version(launcher, deployment, monkeypatch):
    def stale_version(url, **kwargs):
        return (
            {"status": "ready", "model_version": "6"}
            if url.endswith("/ready")
            else responses(url, **kwargs)
        )

    monkeypatch.setattr(launcher, "request_json", stale_version)
    with pytest.raises(launcher.LaunchError, match="exact deployed version"):
        launcher.verify_deployment(deployment, "run-1", "http://api:8000")


def test_missing_acknowledgement_is_a_failure(launcher, deployment):
    (deployment / "artifacts" / "deployments" / "acknowledgements" / "deployment-1.json").unlink()
    with pytest.raises(launcher.LaunchError, match="evidence is missing"):
        launcher.verify_deployment(deployment, "run-1", "http://api:8000")


def test_unversioned_acknowledgement_is_a_failure(launcher, deployment):
    path = deployment / "artifacts" / "deployments" / "acknowledgements" / "deployment-1.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("contract_version", None)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="Runtime acknowledgement does not match"):
        launcher.verify_deployment(deployment, "run-1", "http://api:8000")


def test_dag_failure_cannot_be_reported_as_success(launcher, deployment, monkeypatch):
    monkeypatch.setenv("AIRFLOW_ADMIN_USERNAME", "test-user")
    monkeypatch.setenv("AIRFLOW_ADMIN_PASSWORD", "test-secret")
    requests = []

    def failed_dag(url, **kwargs):
        requests.append((url, kwargs))
        if url.endswith("/dagRuns/run-1"):
            return {"state": "failed"}
        return {"dag_id": "credit_default_pipeline"}

    monkeypatch.setattr(launcher, "request_json", failed_dag)
    args = SimpleNamespace(
        project_root=str(deployment),
        config="configs/project.yaml",
        run_id="run-1",
        timeout_seconds=30,
    )
    with pytest.raises(launcher.LaunchError, match="run-1 failed"):
        launcher.launch(args)
    assert not (deployment / "artifacts" / "runs" / "run-1" / "launch.json").exists()
    assert any(item[1].get("method") == "POST" for item in requests)


def test_authentication_failure_is_immediate_and_safe(launcher, deployment, monkeypatch):
    monkeypatch.setenv("AIRFLOW_ADMIN_USERNAME", "test-user")
    monkeypatch.setenv("AIRFLOW_ADMIN_PASSWORD", "test-secret")
    monkeypatch.setattr(
        launcher,
        "request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(launcher.StatusError("GET", 401)),
    )
    args = SimpleNamespace(
        project_root=str(deployment),
        config="configs/project.yaml",
        run_id="run-1",
        timeout_seconds=30,
    )
    with pytest.raises(launcher.StatusError, match="HTTP 401") as error:
        launcher.launch(args)
    assert "test-secret" not in str(error.value)


def test_running_dag_timeout_fails_without_completion_evidence(launcher, deployment, monkeypatch):
    monkeypatch.setenv("AIRFLOW_ADMIN_USERNAME", "test-user")
    monkeypatch.setenv("AIRFLOW_ADMIN_PASSWORD", "test-secret")
    elapsed = [0.0]
    monkeypatch.setattr(launcher.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        launcher.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
    )
    monkeypatch.setattr(launcher, "request_json", lambda *_args, **_kwargs: {"state": "running"})
    args = SimpleNamespace(
        project_root=str(deployment),
        config="configs/project.yaml",
        run_id="run-1",
        timeout_seconds=30,
    )
    with pytest.raises(launcher.LaunchError, match="may still be running"):
        launcher.launch(args)
    assert elapsed[0] == 30
    assert not (deployment / "artifacts" / "runs" / "run-1" / "launch.json").exists()


def test_unsafe_run_id_is_rejected_before_any_http_call(launcher, deployment, monkeypatch):
    monkeypatch.setattr(
        launcher, "request_json", lambda *_args, **_kwargs: pytest.fail("Unexpected HTTP request")
    )
    args = SimpleNamespace(
        project_root=str(deployment),
        config="configs/project.yaml",
        run_id="../outside",
        timeout_seconds=30,
    )
    with pytest.raises(launcher.LaunchError, match="Run ID"):
        launcher.launch(args)


def test_conflicting_dag_config_is_rejected_before_post(launcher, deployment, monkeypatch):
    config_path = deployment / "configs/project.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["pipeline"]["dag_id"] = "different_dag"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        launcher, "request_json", lambda *_args, **_kwargs: pytest.fail("Unexpected HTTP request")
    )
    args = SimpleNamespace(
        project_root=str(deployment),
        config="configs/project.yaml",
        run_id="run-1",
        timeout_seconds=30,
    )
    with pytest.raises(launcher.LaunchError, match="dag_id"):
        launcher.launch(args)


def test_configured_evidence_and_fixture_paths_are_used(launcher, deployment, monkeypatch):
    (deployment / "artifacts").rename(deployment / "custom-artifacts")
    (deployment / "examples/predict.json").rename(deployment / "examples/custom.json")
    monkeypatch.setattr(launcher, "request_json", responses)
    config = {
        "pipeline": {"artifact_root": "custom-artifacts"},
        "serving": {
            "deployments_dir": "custom-artifacts/deployments",
            "smoke_fixture": "examples/custom.json",
        },
    }
    assert (
        launcher.verify_deployment(deployment, "run-1", "http://api:8000", config)["model_version"]
        == "7"
    )


def test_stage_records_from_another_run_are_rejected(launcher, deployment):
    path = deployment / "artifacts/runs/run-1/verify.json"
    record = json.loads(path.read_text())
    record["run_id"] = "stale-run"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="successful pipeline run"):
        launcher.verify_deployment(deployment, "run-1", "http://api:8000")
