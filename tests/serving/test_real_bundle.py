"""End to end on real components: P1 split -> P2 trained bundle -> P3 deploy and predict."""
# ruff: noqa: E402

import json

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from serving_fixtures import EXAMPLE, make_settings, read_ack, request_deploy

from mlops_project.data import pipeline as data_pipeline
from mlops_project.data.policy import FEATURES, TARGET
from mlops_project.pipelines.contracts import sha256_file
from mlops_project.pipelines.runner import atomic_json
from mlops_project.serving.app import create_app
from mlops_project.training import pipeline as training_pipeline


def _rows():
    result = []
    for i in range(400):
        row = dict.fromkeys(FEATURES, 0)
        row.update(
            ID=i + 1,
            LIMIT_BAL=(i + 1) * 1000,
            AGE=21 + i % 50,
            SEX=1 + i % 2,
            EDUCATION=i % 7,
            MARRIAGE=i % 4,
            PAY_0=(i % 5) - 2,
        )
        row[TARGET] = int(i % 5 >= 3)
        result.append(row)
    return result


@pytest.fixture
def trained_bundle(tmp_path, monkeypatch):
    run_dir = tmp_path / "artifacts/runs/e2e"
    run_dir.mkdir(parents=True)
    data = run_dir / "data.json"
    data.write_text(json.dumps(_rows()))
    source = {
        "contract_version": 1,
        "dataset_version": "fixture",
        "schema_version": "credit-default-v1",
        "data": data_pipeline._ref(data),
        "artifacts": [data_pipeline._ref(data)],
        "passed": True,
    }
    (tmp_path / "requirements.lock").write_text("fixture lock\n")
    context = {
        "contract_version": 1,
        "run_id": "e2e",
        "run_dir": str(run_dir),
        "project_root": str(tmp_path),
        "config_sha256": "b" * 64,
        "inputs": {"validate": source},
        "config": {
            "project": {"seed": 42},
            "splits": {"train": 0.6, "validation": 0.2, "final_test": 0.1, "monitoring": 0.1},
            "serving": {"model_name": "credit-default"},
            "training": {
                "tracking_uri": f"sqlite:///{tmp_path.as_posix()}/mlflow.db",
                "dependency_lock": "requirements.lock",
                "experiment_name": "e2e",
            },
        },
    }
    monkeypatch.setattr(training_pipeline, "code_commit", lambda root: "a" * 40)
    split = data_pipeline.split(context)
    features = data_pipeline.features({**context, "inputs": {"split": split}})
    return training_pipeline.train_baseline({**context, "inputs": {"features": features}})


def _deploy_manifest(root, result, version):
    report = root / f"artifacts/runs/e2e/gate-report-{version}.json"
    atomic_json(
        report,
        {
            "passed": True,
            "candidate_model_version": version,
            "artifact_sha256": result["artifact_sha256"],
        },
    )
    return {
        "contract_version": 1,
        "action": "deploy",
        "deployment_id": f"deploy-real-{version}",
        "model_name": "credit-default",
        "model_version": version,
        "schema_version": "credit-default-v1",
        "artifact_uri": result["artifact_uri"],
        "artifact_sha256": result["artifact_sha256"],
        "gate_report_uri": report.relative_to(root).as_posix(),
        "gate_report_sha256": sha256_file(report),
    }


def test_real_p2_bundle_serves_through_watcher(trained_bundle, tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        request_deploy(settings, _deploy_manifest(tmp_path, trained_bundle, "1"))
        client.app.state.watcher.check_once()

        assert read_ack(settings, "deploy-real-1")["status"] == "loaded"
        assert client.get("/ready").json()["model_version"] == "1"

        response = client.post("/predict", json=EXAMPLE)
        assert response.status_code == 200
        prediction = response.json()["predictions"][0]
        assert prediction["label"] in (0, 1)
        assert 0 <= prediction["default_probability"] <= 1


def test_benchmark_measures_real_bundle_in_separate_process(trained_bundle, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    import yaml

    from mlops_project.serving.pipeline import benchmark

    repo = Path(__file__).resolve().parents[2]
    for name in ("configs/project.yaml", "configs/data_schema.yaml", "examples/predict.json"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, tmp_path / name)
    gates = yaml.safe_load((repo / "configs/quality_gates.yaml").read_text(encoding="utf-8"))
    gates["gates"]["service"].update(concurrency=2, measurement_seconds=2)
    (tmp_path / "configs/quality_gates.yaml").write_text(yaml.safe_dump(gates))
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))

    config = yaml.safe_load((tmp_path / "configs/project.yaml").read_text(encoding="utf-8"))
    registered = {**trained_bundle, "model_name": "credit-default", "model_version": "1"}
    result = benchmark(
        {
            "contract_version": 1,
            "run_id": "e2e",
            "project_root": str(tmp_path),
            "run_dir": str(tmp_path / "artifacts/runs/e2e"),
            "config": config,
            "config_path": "configs/project.yaml",
            "inputs": {"register": registered},
        }
    )

    assert result["passed"] is True
    assert result["model_version"] == "1"
    assert result["workload"] == {"batch_size": 1, "concurrency": 2, "measurement_seconds": 2}
    metrics = result["metrics"]
    assert metrics["error_rate"] == 0
    assert 0 < metrics["p50_latency_ms"] <= metrics["p95_latency_ms"]
    assert metrics["throughput_rps"] > 0
