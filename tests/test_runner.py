"""Integration plumbing checks use synthetic adapters, never trained-model claims."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mlops_project.pipelines.contracts import (
    STAGES,
    PipelineError,
    configuration_snapshot,
    sha256_file,
)
from mlops_project.pipelines.runner import _adapter_child, _run_adapter, run_step


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "project.yaml"
        self.config = {
            "contract_version": 1,
            "quality_gates": "quality.yaml",
            "pipeline": {
                "stage_timeout_seconds": 5,
                "adapters": {step: f"mlops_project.data.pipeline:{step}" for step in STAGES},
            },
        }
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        (self.root / "quality.yaml").write_text("policy: original\n", encoding="utf-8")

    def run_step(self, step, run_id="test-run"):
        return run_step(step, run_id, self.config_path, project_root=self.root)

    def successful_adapter(self, reference, context, timeout, run_dir):
        return {"contract_version": 1, "passed": True, "stage": context["step"]}

    def test_path_traversal_and_unknown_stage_are_rejected(self):
        for run_id in ("../escape", "..", "with/slash", "manual:bad", "", "CON", "name."):
            with self.subTest(run_id=run_id), self.assertRaises(PipelineError) as caught:
                self.run_step("ingest", run_id)
            self.assertEqual(caught.exception.code, "invalid_run_id")
        with self.assertRaises(PipelineError):
            self.run_step("unknown")
        self.assertFalse((self.root / "artifacts").exists())

    def test_dependencies_block_execution(self):
        with patch("mlops_project.pipelines.runner._run_adapter") as adapter:
            with self.assertRaises(PipelineError) as caught:
                self.run_step("train_baseline")
        self.assertEqual(caught.exception.code, "missing_evidence")
        adapter.assert_not_called()

    def test_failed_validation_is_persisted_and_stops_split(self):
        with patch(
            "mlops_project.pipelines.runner._run_adapter", side_effect=self.successful_adapter
        ):
            self.run_step("ingest")
        with patch(
            "mlops_project.pipelines.runner._run_adapter",
            return_value={"contract_version": 1, "passed": False},
        ):
            with self.assertRaises(PipelineError) as caught:
                self.run_step("validate")
        self.assertEqual(caught.exception.code, "gate_failed")
        record = json.loads((self.root / "artifacts/runs/test-run/validate.json").read_text())
        self.assertEqual(record["state"], "failed")
        with self.assertRaises(PipelineError) as caught:
            self.run_step("split")
        self.assertEqual(caught.exception.code, "upstream_failed")

    def test_missing_validation_flag_fails_closed(self):
        with patch(
            "mlops_project.pipelines.runner._run_adapter", side_effect=self.successful_adapter
        ):
            self.run_step("ingest")
        with patch(
            "mlops_project.pipelines.runner._run_adapter", return_value={"contract_version": 1}
        ):
            with self.assertRaises(PipelineError) as caught:
                self.run_step("validate")
        self.assertEqual(caught.exception.code, "missing_gate_evidence")

    def test_successful_retry_is_idempotent(self):
        with patch(
            "mlops_project.pipelines.runner._run_adapter", side_effect=self.successful_adapter
        ) as adapter:
            first = self.run_step("ingest")
            second = self.run_step("ingest")
        self.assertEqual(first, second)
        self.assertEqual(adapter.call_count, 1)
        self.assertEqual(first["result"]["run_id"], "test-run")

    def test_changed_referenced_policy_requires_new_run(self):
        with patch(
            "mlops_project.pipelines.runner._run_adapter", side_effect=self.successful_adapter
        ):
            self.run_step("ingest")
        (self.root / "quality.yaml").write_text("policy: changed\n", encoding="utf-8")
        with self.assertRaises(PipelineError) as caught:
            self.run_step("ingest")
        self.assertEqual(caught.exception.code, "stale_evidence")

    def test_configuration_changed_during_execution_records_failure(self):
        def change_policy(reference, context, timeout, run_dir):
            (self.root / "quality.yaml").write_text("policy: changed\n", encoding="utf-8")
            return {"contract_version": 1}

        with patch("mlops_project.pipelines.runner._run_adapter", side_effect=change_policy):
            with self.assertRaises(PipelineError) as caught:
                self.run_step("ingest")
        self.assertEqual(caught.exception.code, "stale_evidence")
        record = json.loads((self.root / "artifacts/runs/test-run/ingest.json").read_text())
        self.assertEqual(record["state"], "failed")

    def test_upstream_record_tampering_invalidates_successful_retry(self):
        with patch(
            "mlops_project.pipelines.runner._run_adapter", side_effect=self.successful_adapter
        ):
            self.run_step("ingest")
            self.run_step("validate")
        upstream = self.root / "artifacts/runs/test-run/ingest.json"
        record = json.loads(upstream.read_text())
        record["result"]["tampered"] = True
        upstream.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(PipelineError) as caught:
            self.run_step("validate")
        self.assertEqual(caught.exception.code, "stale_evidence")

    def test_artifact_checksums_are_verified_on_retry(self):
        def adapter(reference, context, timeout, run_dir):
            artifact = run_dir / "manifest.json"
            artifact.write_text("{}", encoding="utf-8")
            return {
                "contract_version": 1,
                "artifacts": [{"uri": "manifest.json", "sha256": sha256_file(artifact)}],
            }

        with patch("mlops_project.pipelines.runner._run_adapter", side_effect=adapter):
            self.run_step("ingest")
        (self.root / "artifacts/runs/test-run/manifest.json").write_text(
            "changed", encoding="utf-8"
        )
        with self.assertRaises(PipelineError) as caught:
            self.run_step("ingest")
        self.assertEqual(caught.exception.code, "artifact_mismatch")

    def test_missing_real_adapter_names_owner_and_records_failure(self):
        self.config["pipeline"]["adapters"]["ingest"] = (
            "mlops_project.unimplemented_test.pipeline:ingest"
        )
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        with self.assertRaises(PipelineError) as caught:
            self.run_step("ingest")
        self.assertEqual(caught.exception.code, "adapter_unavailable")
        self.assertIn("@OuanEng", caught.exception.message)
        record = json.loads((self.root / "artifacts/runs/test-run/ingest.json").read_text())
        self.assertEqual(record["state"], "failed")

    def test_child_timeout_stops_waiting_and_fails_explicitly(self):
        with patch(
            "mlops_project.pipelines.runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired("adapter", 0.01),
        ):
            with self.assertRaises(PipelineError) as caught:
                _run_adapter("mlops_project.data.pipeline:ingest", {}, 0.01, self.root)
        self.assertEqual(caught.exception.code, "stage_timeout")

    def test_adapter_error_messages_do_not_expose_internal_values(self):
        context = self.root / "context.json"
        result = self.root / "result.json"
        context.write_text('{"step":"ingest"}', encoding="utf-8")
        with patch(
            "mlops_project.pipelines.runner.resolve_adapter",
            return_value=lambda _: (_ for _ in ()).throw(RuntimeError("secret-token-value")),
        ):
            self.assertEqual(
                _adapter_child("mlops_project.data.pipeline:ingest", str(context), str(result)), 1
            )
        self.assertNotIn("secret-token-value", result.read_text())

    def test_retraining_without_approved_controller_receipt_is_blocked(self):
        with self.assertRaises(PipelineError) as caught:
            self.run_step("ingest", "retrain-trigger-1")
        self.assertEqual(caught.exception.code, "missing_evidence")

    def test_retraining_receipt_passes_new_dataset_to_adapter_and_is_immutable(self):
        config_hash, _ = configuration_snapshot(self.config, self.config_path, self.root)
        state = self.root / "artifacts/retraining"
        state.mkdir(parents=True)
        request = {
            "trigger_id": "trigger-1",
            "candidate_dataset_version": "new-regime-v2",
            "candidate_dataset_approved": True,
            "labels_validated": True,
            "separate_training_validation": True,
            "final_test_excluded": True,
        }
        receipt = {
            "contract_version": 1,
            "state": "submitted",
            "trigger_id": "trigger-1",
            "pipeline_run_id": "retrain-trigger-1",
            "policy_sha256": config_hash,
            "retraining_request": request,
        }
        path = state / "trigger-1.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        with patch(
            "mlops_project.pipelines.runner._run_adapter", side_effect=self.successful_adapter
        ) as adapter:
            self.run_step("ingest", "retrain-trigger-1")
        context = adapter.call_args.args[1]
        self.assertEqual(context["retraining"]["candidate_dataset_version"], "new-regime-v2")
        receipt["retraining_request"]["candidate_dataset_version"] = "tampered-regime"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaises(PipelineError) as caught:
            self.run_step("validate", "retrain-trigger-1")
        self.assertEqual(caught.exception.code, "stale_evidence")

    def test_all_stages_execute_real_child_protocol_and_real_approval_with_synthetic_evidence(self):
        # This tests subprocess transport and boundaries. These temporary adapters
        # produce synthetic metadata; they do not implement the teammates' ML work.
        source = Path(__file__).resolve().parents[1] / "src/mlops_project"
        fixture_package = self.root / "src/mlops_project"
        shutil.copytree(source, fixture_package, ignore=shutil.ignore_patterns("__pycache__"))
        (fixture_package / "fixture_adapter.py").write_text(
            """
import hashlib
from pathlib import Path
from mlops_project.pipelines.contracts import DEPENDENCIES

def run(context):
    stage = context["step"]
    assert set(context["inputs"]) == set(DEPENDENCIES[stage])
    path = Path(context["run_dir"]) / (stage + "-fixture.txt")
    path.write_text("synthetic boundary evidence " + stage, encoding="utf-8")
    common = {
        "contract_version": 1, "run_id": context["run_id"], "passed": True,
        "code_commit": "a" * 40, "artifact_sha256": "b" * 64,
        "dataset_version": "synthetic-data", "schema_version": "synthetic-schema",
        "artifacts": [{"uri": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
    }
    if stage == "evaluate":
        common.update(metrics={"average_precision": 0.5}, validation_id="fixture-validation",
                      comparison={"average_precision": 0.51, "validation_id": "fixture-validation"})
    if stage == "register":
        common.update(model_name="fixture-model", model_version="2", artifact_uri="models:/fixture-model/2")
    if stage == "benchmark":
        common.update(model_version="2", hardware="synthetic test fixture",
                      metrics={"p50_latency_ms": 50, "p95_latency_ms": 100, "throughput_rps": 30, "error_rate": 0},
                      workload={"batch_size": 1, "concurrency": 10, "measurement_seconds": 300})
    return common
""",
            encoding="utf-8",
        )
        self.config["pipeline"]["adapters"] = {
            step: "mlops_project.fixture_adapter:run" for step in STAGES
        }
        self.config["pipeline"]["adapters"]["approve"] = "mlops_project.pipelines.gates:approve"
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        policy = {
            "contract_version": 1,
            "policy_version": "synthetic-contract-test",
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
        (self.root / "quality.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
        records = {step: self.run_step(step) for step in STAGES}
        self.assertTrue(all(record["state"] == "succeeded" for record in records.values()))
        self.assertEqual(records["approve"]["result"]["status"], "approved")
        gate_report = self.root / "artifacts/runs/test-run/gate-report.json"
        self.assertEqual(
            records["approve"]["result"]["gate_report_sha256"], sha256_file(gate_report)
        )
        self.assertEqual(self.run_step("verify"), records["verify"])


if __name__ == "__main__":
    unittest.main()
