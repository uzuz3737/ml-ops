"""Delivery checks use a synthetic watcher/API, never actual model measurements."""

import copy
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from mlops_project.pipelines import deployment
from mlops_project.pipelines.contracts import STAGES, PipelineError, configuration_snapshot
from mlops_project.pipelines.gates import approve
from mlops_project.pipelines.runner import atomic_json, read_record


@pytest.fixture
def system(tmp_path, monkeypatch):
    policy = {
        "contract_version": 1,
        "policy_version": "synthetic-policy",
        "gates": {
            "metric": {
                "name": "average_precision",
                "direction": "maximize",
                "positive_class": 1,
                "minimum": 0.4,
            },
            "comparison": {"max_regression": 0.02},
            "service": {
                "p95_latency_ms_max": 300,
                "throughput_rps_min": 20,
                "error_rate_max": 0.01,
                "batch_size": 1,
                "concurrency": 10,
                "measurement_seconds": 300,
            },
        },
    }
    config = {
        "contract_version": 1,
        "quality_gates": "policy.yaml",
        "schema_config": "schema.yaml",
        "serving": {
            "api_url": "http://synthetic-api:8000",
            "model_name": "test-model",
            "smoke_fixture": "predict.json",
            "deployments_dir": "deployments",
        },
        "pipeline": {
            "stage_timeout_seconds": 20,
            "deployment_timeout_seconds": 0.2,
            "deployment_poll_seconds": 0.1,
            "registry_finalize": "mlops_project.registry.pipeline:finalize_deployment",
            "adapters": {step: f"mlops_project.synthetic:{step}" for step in STAGES},
        },
    }
    config_path = tmp_path / "project.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
    (tmp_path / "schema.yaml").write_text(
        yaml.safe_dump({"schema_version": "test-schema", "schema_path": "schema.pbtxt"}),
        encoding="utf-8",
    )
    (tmp_path / "schema.pbtxt").write_text("# synthetic schema fixture\n", encoding="utf-8")
    atomic_json(tmp_path / "predict.json", {"instances": [{"LIMIT_BAL": 100}]})
    directory = tmp_path / "deployments"
    state = {
        "version": None,
        "schema": None,
        "fail_version": None,
        "fail_unload": False,
        "bad_prediction": False,
        "bad_ack": False,
        "unversioned_ack": False,
        "registry_calls": [],
        "registry_fail_action": None,
        "clock": 0.0,
        "desired": [],
    }

    def write(path, value):
        atomic_json(path, value)
        if path.name != "desired-model.json":
            return
        state["desired"].append(copy.deepcopy(value))
        if value["action"] == "unload":
            if not state["fail_unload"]:
                state["version"] = None
            ack = {
                "contract_version": 1,
                "deployment_id": value["deployment_id"],
                "status": "loaded" if state["fail_unload"] else "unloaded",
            }
        else:
            state["version"] = value["model_version"]
            state["schema"] = value["schema_version"]
            ack = {
                "contract_version": 1,
                "deployment_id": value["deployment_id"],
                "model_version": "999" if state["bad_ack"] else state["version"],
                "status": "error" if state["fail_version"] == state["version"] else "loaded",
            }
            if state["unversioned_ack"]:
                ack.pop("contract_version", None)
        atomic_json(directory / "acknowledgements" / f"{value['deployment_id']}.json", ack)

    def http(api_url, endpoint, timeout, payload=None):
        assert timeout <= 5
        if endpoint == "/ready":
            return (
                (503, {})
                if state["version"] is None
                else (
                    200,
                    {
                        "status": "ready",
                        "model_version": state["version"],
                        "schema_version": state["schema"],
                    },
                )
            )
        if endpoint == "/health":
            return 200, {"status": "healthy"}
        assert endpoint == "/predict"
        return 200, {
            "model_version": state["version"],
            "predictions": [
                {"label": True if state["bad_prediction"] else 1, "default_probability": 0.6}
                for _ in payload["instances"]
            ],
        }

    def finalize(reference, context, timeout, run_dir):
        assert timeout <= config["pipeline"]["deployment_timeout_seconds"]
        state["registry_calls"].append(copy.deepcopy(context))
        if context["action"] == state["registry_fail_action"]:
            raise PipelineError("registry_unavailable", "Synthetic registry failure.")
        return {"contract_version": 1, "passed": True, "model_version": context["model_version"]}

    def context(version="2", run_id="run-2"):
        config_sha, _ = configuration_snapshot(config, config_path, tmp_path)
        common = {
            "contract_version": 1,
            "run_id": run_id,
            "artifact_sha256": version[-1] * 64,
            "code_commit": "a" * 40,
            "dataset_version": "synthetic-data",
            "schema_version": "test-schema",
        }
        evidence = {
            "evaluate": {
                **common,
                "passed": True,
                "metrics": {"average_precision": 0.5},
                "validation_id": "validation",
                "comparison": {"validation_id": "validation", "average_precision": 0.51},
            },
            "register": {
                **common,
                "model_name": "test-model",
                "model_version": version,
                "artifact_uri": "models:/test-model/" + version,
            },
            "benchmark": {
                **common,
                "passed": True,
                "model_version": version,
                "hardware": "synthetic",
                "metrics": {
                    "p50_latency_ms": 20,
                    "p95_latency_ms": 40,
                    "throughput_rps": 40,
                    "error_rate": 0,
                },
                "workload": {"batch_size": 1, "concurrency": 10, "measurement_seconds": 300},
            },
        }
        result = {
            "contract_version": 1,
            "run_id": run_id,
            "step": "deploy",
            "config": config,
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "project_root": str(tmp_path),
            "run_dir": str(tmp_path / "runs" / run_id),
            "inputs": evidence,
        }
        result["inputs"] = {"approve": approve(result)}
        return result

    monkeypatch.setattr(deployment, "atomic_json", write)
    monkeypatch.setattr(deployment, "_http", http)
    monkeypatch.setattr(deployment, "resolve_adapter", lambda *_: lambda _: None)
    monkeypatch.setattr(deployment, "_run_adapter", finalize)
    monkeypatch.setattr(deployment.time, "monotonic", lambda: state["clock"])
    monkeypatch.setattr(
        deployment.time,
        "sleep",
        lambda seconds: state.__setitem__("clock", state["clock"] + seconds),
    )
    return tmp_path, directory, state, context


def test_delivery_confirms_runtime_then_finalizes_exact_registry_version(system):
    root, directory, state, context = system
    result = deployment.deploy(context())
    active = read_record(directory / "active-model.json")
    assert active["confirmed"] is True and active["model_version"] == "2"
    assert result["deployment_id"] == active["deployment_id"]
    assert state["registry_calls"][0]["action"] == "deploy"
    assert state["registry_calls"][0]["model_version"] == "2"
    assert not (directory / "previous-model.json").exists()
    assert all(
        (root / "runs/run-2" / artifact["uri"]).is_file() for artifact in result["artifacts"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("run_id", "another-run"),
        ("schema_version", "other-schema"),
        ("model_name", "another-model"),
        ("gate_report_sha256", "f" * 64),
    ],
)
def test_unapproved_or_changed_identity_never_changes_desired_model(system, field, value):
    _, directory, state, context = system
    current = context()
    current["inputs"]["approve"][field] = value
    with pytest.raises(PipelineError):
        deployment.deploy(current)
    assert not (directory / "desired-model.json").exists()
    assert not state["registry_calls"]


def test_missing_callback_is_detected_before_serving_mutation(system, monkeypatch):
    _, directory, _, context = system
    monkeypatch.setattr(
        deployment,
        "resolve_adapter",
        lambda *_: (_ for _ in ()).throw(
            PipelineError("adapter_unavailable", "P2 callback missing")
        ),
    )
    with pytest.raises(PipelineError, match="P2 callback missing"):
        deployment.deploy(context())
    assert not (directory / "desired-model.json").exists()


def test_changed_policy_or_schema_cannot_reuse_an_approval(system):
    root, directory, _, context = system
    current = context()
    (root / "schema.pbtxt").write_text("changed schema", encoding="utf-8")
    with pytest.raises(PipelineError, match="configuration changed"):
        deployment.deploy(current)
    assert not (directory / "desired-model.json").exists()


def test_failed_candidate_restores_actual_previous_version_and_registry(system):
    _, directory, state, context = system
    deployment.deploy(context("1", "run-1"))
    state["fail_version"] = "2"
    with pytest.raises(PipelineError) as error:
        deployment.deploy(context())
    assert error.value.code == "deployment_failed"
    assert state["version"] == "1"
    assert read_record(directory / "active-model.json")["model_version"] == "1"
    assert state["registry_calls"][-1]["action"] == "rollback"
    assert state["registry_calls"][-1]["previous_model_version"] == "2"
    assert state["desired"][-1]["action"] == "rollback"
    assert state["desired"][-1]["artifact_uri"] == "models:/test-model/1"


def test_registry_failure_restores_loaded_prior_bundle(system):
    _, directory, state, context = system
    deployment.deploy(context("1", "run-1"))
    state["registry_fail_action"] = "deploy"
    with pytest.raises(PipelineError) as error:
        deployment.deploy(context())
    assert error.value.code == "deployment_failed"
    assert state["version"] == "1"
    assert read_record(directory / "active-model.json")["model_version"] == "1"
    assert [call["action"] for call in state["registry_calls"]] == ["deploy", "deploy", "rollback"]


def test_first_failure_unloads_candidate_and_clears_registry(system):
    root, directory, state, context = system
    state["registry_fail_action"] = "deploy"
    with pytest.raises(PipelineError) as error:
        deployment.deploy(context())
    assert error.value.code == "deployment_failed"
    assert state["version"] is None
    assert not (directory / "active-model.json").exists()
    assert state["desired"][-1]["action"] == "unload"
    assert state["registry_calls"][-1]["model_version"] is None
    assert (
        read_record(root / "runs/run-2/deployment-deploy.json")["recovery"]["state"] == "unloaded"
    )


def test_unverified_unload_reports_recovery_failure(system):
    root, _, state, context = system
    state["fail_version"] = "2"
    state["fail_unload"] = True
    with pytest.raises(PipelineError) as error:
        deployment.deploy(context())
    assert error.value.code == "recovery_failed"
    evidence = read_record(root / "runs/run-2/deployment-deploy.json")
    assert evidence["recovery"]["state"] == "recovery_failed"
    assert not state["registry_calls"]


@pytest.mark.parametrize("failure", ["bad_ack", "unversioned_ack", "bad_prediction"])
def test_stale_ack_and_bool_prediction_fail_closed_and_cleanup(system, failure):
    _, directory, state, context = system
    state[failure] = True
    with pytest.raises(PipelineError):
        deployment.deploy(context())
    assert not (directory / "active-model.json").exists()
    assert state["version"] is None


def test_repeated_deployment_keeps_same_identity_and_previous_target(system):
    _, directory, _, context = system
    deployment.deploy(context("1", "run-1"))
    current = context()
    first = deployment.deploy(current)
    second = deployment.deploy(current)
    assert first["deployment_id"] == second["deployment_id"]
    assert read_record(directory / "previous-model.json")["model_version"] == "1"


def test_verify_rechecks_active_runtime_and_restores_prior_on_failure(system):
    _, directory, state, context = system
    deployment.deploy(context("1", "run-1"))
    current = context()
    result = deployment.deploy(current)
    current["inputs"] = {"deploy": result}
    assert deployment.verify(current)["passed"] is True
    state["version"] = "999"
    with pytest.raises(PipelineError) as error:
        deployment.verify(current)
    assert error.value.code == "deployment_failed"
    assert state["version"] == "1"
    assert read_record(directory / "active-model.json")["model_version"] == "1"


def test_verify_cannot_undo_an_unrelated_newer_deployment(system):
    _, _, state, context = system
    old = context("1", "run-1")
    old_result = deployment.deploy(old)
    deployment.deploy(context())
    old["inputs"] = {"deploy": old_result}
    with pytest.raises(PipelineError) as error:
        deployment.verify(old)
    assert error.value.code == "stale_evidence" and state["version"] == "2"


def test_manual_rollback_requires_prior_confirmation_and_restores_bundle(system):
    root, directory, state, context = system
    deployment.deploy(context("1", "run-1"))
    with pytest.raises(PipelineError) as error:
        deployment.rollback("project.yaml", reason="synthetic demo", project_root=root)
    assert error.value.code == "rollback_unavailable"
    deployment.deploy(context())
    result = deployment.rollback("project.yaml", reason="synthetic demo", project_root=root)
    assert result["model_version"] == "1" and state["version"] == "1"
    assert read_record(directory / "previous-model.json")["model_version"] == "2"
    assert state["registry_calls"][-1]["reason"] == "synthetic demo"


def test_invalid_timeout_leaves_time_for_recovery(system):
    _, directory, _, context = system
    current = context()
    current["config"]["pipeline"]["deployment_timeout_seconds"] = 20
    with pytest.raises(PipelineError, match="leave time"):
        deployment.deploy(current)
    assert not (directory / "desired-model.json").exists()


def test_callback_executes_in_real_bounded_child_with_exact_action(system, monkeypatch):
    root, _, _, context = system
    import mlops_project.pipelines.runner as runner

    source = Path(deployment.__file__).resolve().parents[1]
    shutil.copytree(
        source, root / "src/mlops_project", ignore=shutil.ignore_patterns("__pycache__")
    )
    registry = root / "src/mlops_project/registry"
    registry.mkdir(exist_ok=True)
    (registry / "__init__.py").write_text("", encoding="utf-8")
    (registry / "pipeline.py").write_text(
        "from pathlib import Path\nimport json\ndef finalize_deployment(context):\n    Path(context['run_dir'], 'callback.json').write_text(json.dumps(context))\n    return {'contract_version': 1, 'passed': True, 'model_version': context['model_version']}\n",
        encoding="utf-8",
    )
    callback_context = context()
    callback_context["config"]["pipeline"]["deployment_timeout_seconds"] = 2
    Path(callback_context["config_path"]).write_text(
        yaml.safe_dump(callback_context["config"]), encoding="utf-8"
    )
    monkeypatch.setattr(deployment, "_run_adapter", runner._run_adapter)
    result = deployment.deploy(context())
    callback = read_record(root / "runs/run-2/callback.json")
    assert callback["action"] == "deploy"
    assert callback["deployment_id"] == result["deployment_id"]
    assert callback["model_version"] == "2"


def test_real_localhost_http_preserves_version_and_rejects_bad_json():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ready":
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"model_version":"2"}' if self.path == "/valid" else b"invalid json")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert deployment._http(url, "/valid", 1) == (200, {"model_version": "2"})
        assert deployment._http(url, "/ready", 1) == (503, {})
        with pytest.raises(PipelineError, match="invalid JSON"):
            deployment._http(url, "/invalid", 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
