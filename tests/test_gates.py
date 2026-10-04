"""Policy tests use explicit synthetic evidence, not observed model performance."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from mlops_project.pipelines.contracts import PipelineError
from mlops_project.pipelines.gates import approve


@pytest.fixture
def context(tmp_path):
    policy = {
        "contract_version": 1,
        "policy_version": "test-policy",
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
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
    common = {
        "contract_version": 1,
        "run_id": "test-run",
        "code_commit": "a" * 40,
        "artifact_sha256": "b" * 64,
        "dataset_version": "test-data",
        "schema_version": "test-schema",
    }
    return {
        "project_root": str(tmp_path),
        "run_dir": str(tmp_path / "run"),
        "run_id": "test-run",
        "config_sha256": "c" * 64,
        "config": {"quality_gates": "policy.yaml"},
        "inputs": {
            "evaluate": {
                **common,
                "passed": True,
                "metrics": {"average_precision": 0.5},
                "validation_id": "validation-1",
                "comparison": {"average_precision": 0.51, "validation_id": "validation-1"},
            },
            "register": {
                **common,
                "model_version": "2",
                "model_name": "test-model",
                "artifact_uri": "models:/test-model/2",
            },
            "benchmark": {
                **common,
                "passed": True,
                "model_version": "2",
                "hardware": "synthetic test fixture",
                "metrics": {
                    "p50_latency_ms": 50,
                    "p95_latency_ms": 100,
                    "throughput_rps": 30,
                    "error_rate": 0,
                },
                "workload": {"batch_size": 1, "concurrency": 10, "measurement_seconds": 300},
            },
        },
    }


def test_approved_report_is_tied_to_exact_model_and_policy(context):
    result = approve(context)
    assert result["passed"] is True
    assert result["status"] == "approved"
    report = json.loads((Path(context["run_dir"]) / "gate-report.json").read_text())
    assert report["candidate_model_version"] == "2"
    assert report["policy_version"] == "test-policy"
    assert report["artifact_uri"] == result["artifact_uri"] == "models:/test-model/2"
    assert report["config_sha256"] == context["config_sha256"]
    assert len(report["checks"]) == 5


@pytest.mark.parametrize(
    "metric,value", [("p95_latency_ms", 500), ("throughput_rps", 5), ("error_rate", 0.5)]
)
def test_failed_operational_gate_records_rejection(context, metric, value):
    context["inputs"]["benchmark"]["metrics"][metric] = value
    result = approve(context)
    assert result["passed"] is False
    assert result["status"] == "rejected"


def test_quality_regression_is_rejected(context):
    context["inputs"]["evaluate"]["metrics"]["average_precision"] = 0.3
    assert approve(context)["passed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "artifact_sha256",
        "code_commit",
        "schema_version",
        "dataset_version",
        "run_id",
        "model_version",
    ],
)
def test_stale_benchmark_cannot_approve_another_candidate(context, field):
    context["inputs"]["benchmark"][field] = "different"
    with pytest.raises(PipelineError, match="evidence|differs|version"):
        approve(context)


def test_comparison_requires_same_validation_population(context):
    context["inputs"]["evaluate"]["comparison"]["validation_id"] = "another-population"
    with pytest.raises(PipelineError, match="same validation"):
        approve(context)


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, 2])
def test_undefined_or_invalid_quality_gate_fails_closed(context, value):
    path = Path(context["project_root"]) / "policy.yaml"
    policy = yaml.safe_load(path.read_text())
    policy["gates"]["metric"]["minimum"] = value
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(PipelineError):
        approve(context)


def test_wrong_measurement_workload_cannot_satisfy_slo(context):
    context["inputs"]["benchmark"]["workload"]["concurrency"] = 1
    with pytest.raises(PipelineError, match="workload"):
        approve(context)


def test_approval_never_mutates_input_evidence(context):
    before = copy.deepcopy(context["inputs"])
    approve(context)
    assert context["inputs"] == before


@pytest.mark.parametrize("version", ["champion", "0", "-1", "1.0", "02"])
def test_mutable_alias_or_invalid_registry_version_cannot_be_approved(context, version):
    context["inputs"]["register"]["model_version"] = version
    context["inputs"]["benchmark"]["model_version"] = version
    with pytest.raises(PipelineError, match="exact positive MLflow version"):
        approve(context)


def test_missing_artifact_location_cannot_be_approved(context):
    del context["inputs"]["register"]["artifact_uri"]
    with pytest.raises(PipelineError, match="artifact_uri"):
        approve(context)


def test_malformed_metric_evidence_fails_with_typed_error(context):
    context["inputs"]["evaluate"]["metrics"] = []
    with pytest.raises(PipelineError, match="metrics must be a mapping"):
        approve(context)
