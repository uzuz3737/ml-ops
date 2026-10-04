"""Promotion policy evaluation; no model is approved without comparable evidence."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mlops_project.pipelines.contracts import PipelineError, confined_path, json_bytes

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_MODEL_VERSION = re.compile(r"[1-9][0-9]*\Z")


def _number(value: Any, name: str, *, minimum: float = 0, maximum: float | None = None):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PipelineError(
            "missing_gate_evidence", f"{name} must be a configured/measured number."
        )
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise PipelineError("invalid_gate_evidence", f"{name} is outside its permitted range.")
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError("missing_gate_evidence", f"{name} must be a nonempty string.")
    return value


def _write_report(path: Path, report: dict) -> str:
    content = json_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def approve(context: dict) -> dict:
    """Compare the candidate's actual validation/service evidence with a frozen policy.

    Context comes from runner.run_step. Registry alias mutations and actual serving
    belong to the component owners; this function emits an auditable policy decision.
    """
    root = Path(context["project_root"]).resolve()
    run_dir = confined_path(context["run_dir"], root)
    policy_path = confined_path(
        context["config"].get("quality_gates", "configs/quality_gates.yaml"), root, must_exist=True
    )
    try:
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise PipelineError(
            "invalid_policy", "Quality policy must be readable UTF-8 YAML."
        ) from None
    if (
        not isinstance(policy, dict)
        or type(policy.get("contract_version")) is not int
        or policy["contract_version"] != 1
    ):
        raise PipelineError("invalid_policy", "Quality policy requires contract_version: 1.")
    policy_version = _text(policy.get("policy_version"), "policy_version")
    try:
        metric_policy = policy["gates"]["metric"]
        comparison_policy = policy["gates"]["comparison"]
        service_policy = policy["gates"]["service"]
        evidence = {
            stage: context["inputs"][stage] for stage in ("evaluate", "register", "benchmark")
        }
    except (KeyError, TypeError):
        raise PipelineError(
            "missing_gate_evidence",
            "Approval requires evaluate, register, benchmark and all policy sections.",
        ) from None
    if not all(
        isinstance(section, dict) for section in (metric_policy, comparison_policy, service_policy)
    ):
        raise PipelineError("invalid_policy", "Every quality-policy section must be a mapping.")
    if not all(isinstance(result, dict) for result in evidence.values()):
        raise PipelineError(
            "missing_gate_evidence", "Each upstream evidence result must be a mapping."
        )
    if (
        metric_policy.get("name") != "average_precision"
        or metric_policy.get("direction") != "maximize"
        or type(metric_policy.get("positive_class")) is not int
        or metric_policy["positive_class"] != 1
    ):
        raise PipelineError(
            "invalid_policy", "Policy must maximize average_precision for positive class 1."
        )
    minimum_ap = _number(metric_policy.get("minimum"), "minimum average_precision", maximum=1)
    regression_limit = _number(comparison_policy.get("max_regression"), "max_regression", maximum=1)
    p95_limit = _number(
        service_policy.get("p95_latency_ms_max"), "p95_latency_ms_max", minimum=0.001
    )
    throughput_min = _number(
        service_policy.get("throughput_rps_min"), "throughput_rps_min", minimum=0.001
    )
    error_limit = _number(service_policy.get("error_rate_max"), "error_rate_max", maximum=1)

    for stage, result in evidence.items():
        if (
            not isinstance(result, dict)
            or type(result.get("contract_version")) is not int
            or result["contract_version"] != 1
            or result.get("run_id") != context["run_id"]
        ):
            raise PipelineError(
                "stale_evidence", f"{stage} evidence must belong to this run and contract."
            )
        if stage != "register" and result.get("passed") is not True:
            raise PipelineError("gate_failed", f"{stage} did not pass its required checks.")
        for name in ("artifact_sha256", "code_commit", "dataset_version", "schema_version"):
            value = _text(result.get(name), f"{stage}.{name}")
            if value != evidence["register"].get(name):
                raise PipelineError(
                    "stale_evidence",
                    f"{name} differs across candidate/evaluation/benchmark evidence.",
                )
        if not _SHA.fullmatch(result["artifact_sha256"]) or not _COMMIT.fullmatch(
            result["code_commit"]
        ):
            raise PipelineError(
                "invalid_gate_evidence",
                "Model checksum and full code commit must be lowercase hexadecimal identifiers.",
            )

    evaluated, registered, benchmark = (
        evidence[name] for name in ("evaluate", "register", "benchmark")
    )
    model_version = _text(registered.get("model_version"), "model_version")
    if not _MODEL_VERSION.fullmatch(model_version):
        raise PipelineError(
            "invalid_gate_evidence",
            "model_version must identify an exact positive MLflow version, not an alias.",
        )
    model_name = _text(registered.get("model_name"), "model_name")
    artifact_uri = _text(registered.get("artifact_uri"), "artifact_uri")
    if benchmark.get("model_version") != model_version:
        raise PipelineError(
            "stale_evidence", "Benchmark measured another registered model version."
        )
    validation_id = _text(evaluated.get("validation_id"), "validation_id")
    comparison = evaluated.get("comparison", {})
    if not isinstance(comparison, dict) or comparison.get("validation_id") != validation_id:
        raise PipelineError(
            "stale_evidence", "Candidate and comparison must use the same validation population."
        )
    evaluated_metrics = evaluated.get("metrics", {})
    if not isinstance(evaluated_metrics, dict):
        raise PipelineError("missing_gate_evidence", "Evaluation metrics must be a mapping.")
    ap = _number(evaluated_metrics.get("average_precision"), "average_precision", maximum=1)
    reference_ap = _number(
        comparison.get("average_precision"), "comparison.average_precision", maximum=1
    )
    metrics = benchmark.get("metrics", {})
    if not isinstance(metrics, dict):
        raise PipelineError("missing_gate_evidence", "Benchmark metrics must be a mapping.")
    p50 = _number(metrics.get("p50_latency_ms"), "p50_latency_ms")
    p95 = _number(metrics.get("p95_latency_ms"), "p95_latency_ms")
    throughput = _number(metrics.get("throughput_rps"), "throughput_rps")
    error_rate = _number(metrics.get("error_rate"), "error_rate", maximum=1)
    if p50 > p95:
        raise PipelineError("invalid_gate_evidence", "p50 cannot exceed p95 latency.")
    _text(benchmark.get("hardware"), "benchmark.hardware")
    workload = benchmark.get("workload", {})
    if not isinstance(workload, dict):
        raise PipelineError("missing_gate_evidence", "Benchmark workload must be a mapping.")
    for name in ("batch_size", "concurrency", "measurement_seconds"):
        expected = _number(service_policy.get(name), f"service.{name}", minimum=1)
        observed = _number(workload.get(name), f"workload.{name}", minimum=1)
        if observed != expected:
            raise PipelineError(
                "stale_evidence",
                f"Benchmark workload {name} does not match the declared SLO workload.",
            )

    checks = [
        {
            "name": "minimum_average_precision",
            "threshold": minimum_ap,
            "observed": ap,
            "passed": ap >= minimum_ap,
        },
        {
            "name": "allowed_average_precision_regression",
            "threshold": regression_limit,
            "observed": reference_ap - ap,
            "passed": reference_ap - ap <= regression_limit,
        },
        {
            "name": "p95_latency_ms",
            "threshold": p95_limit,
            "observed": p95,
            "passed": p95 <= p95_limit,
        },
        {
            "name": "throughput_rps",
            "threshold": throughput_min,
            "observed": throughput,
            "passed": throughput >= throughput_min,
        },
        {
            "name": "error_rate",
            "threshold": error_limit,
            "observed": error_rate,
            "passed": error_rate <= error_limit,
        },
    ]
    passed = all(check["passed"] for check in checks)
    report = {
        "contract_version": 1,
        "run_id": context["run_id"],
        "candidate_model_version": model_version,
        "model_name": model_name,
        "artifact_sha256": registered["artifact_sha256"],
        "artifact_uri": artifact_uri,
        "code_commit": registered["code_commit"],
        "dataset_version": registered["dataset_version"],
        "schema_version": registered["schema_version"],
        "validation_id": validation_id,
        "policy_version": policy_version,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "config_sha256": context["config_sha256"],
        "created_at": datetime.now(UTC).isoformat(),
        "approver": f"policy:{policy_version}",
        "passed": passed,
        "checks": checks,
        "p50_latency_ms": p50,
        "hardware": benchmark["hardware"],
        "workload": workload,
    }
    report_path = run_dir / "gate-report.json"
    report_sha = _write_report(report_path, report)
    return {
        "contract_version": 1,
        "run_id": context["run_id"],
        "passed": passed,
        "status": "approved" if passed else "rejected",
        "approval_id": f"{context['run_id']}-{report_sha[:12]}",
        "model_version": model_version,
        "model_name": model_name,
        "artifact_sha256": registered["artifact_sha256"],
        "artifact_uri": artifact_uri,
        "code_commit": registered["code_commit"],
        "dataset_version": registered["dataset_version"],
        "schema_version": registered["schema_version"],
        "policy_version": policy_version,
        "gate_report_sha256": report_sha,
        "artifacts": [{"uri": "gate-report.json", "sha256": report_sha}],
    }
