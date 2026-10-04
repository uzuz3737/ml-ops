"""Controller policy tests inject an Airflow requester and synthetic labeled alerts."""

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from mlops_project.pipelines.contracts import STAGES, PipelineError
from mlops_project.pipelines.retraining import submit_retraining


class FakeAirflow:
    def __init__(self):
        self.calls = []
        self.runs = []
        self.existing = None
        self.post_error = None
        self.exact_error = None

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if method == "GET" and "?" in path:
            return {"dag_runs": self.runs, "total_entries": len(self.runs)}
        if method == "GET":
            if self.exact_error:
                raise PipelineError(self.exact_error, "Injected network condition.")
            if self.existing is None:
                raise PipelineError("airflow_not_found", "Run not found.")
            return self.existing
        if self.post_error:
            if self.post_error in {"airflow_conflict", "airflow_unavailable"}:
                self.existing = payload
            raise PipelineError(self.post_error, "Injected POST condition.")
        self.existing = payload
        return payload


class RetrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "project.yaml"
        config = {
            "contract_version": 1,
            "monitoring_config": "monitoring.yaml",
            "pipeline": {
                "stage_timeout_seconds": 5,
                "dag_id": "credit_default_pipeline",
                "adapters": {step: f"mlops_project.data.pipeline:{step}" for step in STAGES},
            },
        }
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.policy = {
            "contract_version": 1,
            "minimum_samples": 100,
            "minimum_labeled_samples": 70,
            "minimum_class_samples": 10,
            "minimum_label_coverage": 0.7,
            "data_drift_threshold": 0.1,
            "quality_degradation_threshold": 0.2,
            "retraining": {
                "enabled": True,
                "cooldown_seconds": 3600,
                "require_valid_labels": True,
                "require_separate_training_validation": True,
                "dag_id": "credit_default_pipeline",
            },
        }
        self.write_policy()
        self.now = datetime(2026, 10, 4, tzinfo=UTC)
        self.alert = {
            "contract_version": 1,
            "trigger_id": "trigger-1",
            "alert_id": "alert-1",
            "model_version": "1",
            "dataset_version": "original-v1",
            "candidate_dataset_version": "regime-v2",
            "candidate_dataset_approved": True,
            "labels_validated": True,
            "separate_training_validation": True,
            "final_test_excluded": True,
            "reason": "data_drift",
            "observed": 0.25,
            "threshold": 0.1,
            "threshold_crossed": True,
            "created_at": "2026-10-03T23:55:00Z",
            "sample_count": 100,
            "labeled_sample_count": 80,
            "positive_label_count": 20,
            "negative_label_count": 60,
            "label_coverage": 0.8,
        }
        self.client = FakeAirflow()

    def write_policy(self):
        (self.root / "monitoring.yaml").write_text(yaml.safe_dump(self.policy), encoding="utf-8")

    def submit(self, **updates):
        return submit_retraining(
            {**self.alert, **updates},
            self.config_path,
            project_root=self.root,
            now=self.now,
            client=self.client,
        )

    def test_disabled_policy_never_contacts_airflow(self):
        self.policy["retraining"]["enabled"] = False
        self.write_policy()
        with self.assertRaises(PipelineError) as caught:
            self.submit()
        self.assertEqual(caught.exception.code, "retraining_disabled")
        self.assertEqual(self.client.calls, [])

    def test_unset_calibration_cannot_submit(self):
        self.policy["minimum_samples"] = None
        self.write_policy()
        with self.assertRaises(PipelineError) as caught:
            self.submit()
        self.assertEqual(caught.exception.code, "policy_unset")
        self.assertEqual(self.client.calls, [])

    def test_claimed_threshold_crossing_cannot_bypass_observed_score(self):
        for observed in (0, 0.05, 0.1):
            with self.subTest(observed=observed), self.assertRaises(PipelineError) as caught:
                self.submit(observed=observed)
            self.assertEqual(caught.exception.code, "threshold_not_crossed")
        self.assertEqual(self.client.calls, [])

    def test_quality_and_concept_drift_use_degradation_threshold(self):
        for reason in ("quality", "concept_drift"):
            with self.subTest(reason=reason), self.assertRaises(PipelineError) as caught:
                self.submit(reason=reason, threshold=0.2, observed=0.15)
            self.assertEqual(caught.exception.code, "threshold_not_crossed")
        self.assertEqual(self.client.calls, [])

    def test_unlabeled_or_small_sample_alerts_do_not_submit(self):
        with self.assertRaises(PipelineError) as caught:
            self.submit(
                labeled_sample_count=30,
                positive_label_count=15,
                negative_label_count=15,
                label_coverage=0.3,
            )
        self.assertEqual(caught.exception.code, "insufficient_labels")
        self.assertEqual(self.client.calls, [])

    def test_class_imbalance_blocks_even_when_coverage_passes(self):
        with self.assertRaises(PipelineError) as caught:
            self.submit(positive_label_count=2, negative_label_count=78)
        self.assertEqual(caught.exception.code, "insufficient_labels")

    def test_separate_training_and_test_exclusion_are_required(self):
        for flag in (
            "candidate_dataset_approved",
            "labels_validated",
            "separate_training_validation",
            "final_test_excluded",
            "threshold_crossed",
        ):
            with self.subTest(flag=flag), self.assertRaises(PipelineError):
                self.submit(**{flag: False})
        self.assertEqual(self.client.calls, [])

    def test_active_dag_run_blocks_submission(self):
        self.client.runs = [{"dag_run_id": "manual", "state": "running"}]
        with self.assertRaises(PipelineError) as caught:
            self.submit()
        self.assertEqual(caught.exception.code, "active_run")
        self.assertFalse(any(method == "POST" for method, _, _ in self.client.calls))

    def test_success_uses_deterministic_id_and_deduplicates_without_http(self):
        receipt = self.submit()
        self.assertEqual(receipt["state"], "submitted")
        posts = [payload for method, _, payload in self.client.calls if method == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["dag_run_id"], "retrain-trigger-1")
        self.assertEqual(posts[0]["conf"]["candidate_dataset_version"], "regime-v2")
        previous_count = len(self.client.calls)
        self.assertEqual(self.submit()["state"], "already_submitted")
        self.assertEqual(len(self.client.calls), previous_count)

    def test_same_trigger_cannot_change_evidence(self):
        self.submit()
        with self.assertRaises(PipelineError) as caught:
            self.submit(observed=0.5)
        self.assertEqual(caught.exception.code, "trigger_conflict")

    def test_cooldown_blocks_new_trigger(self):
        self.submit()
        with self.assertRaises(PipelineError) as caught:
            self.submit(trigger_id="trigger-2", alert_id="alert-2")
        self.assertEqual(caught.exception.code, "cooldown")
        self.now += timedelta(hours=2)
        self.assertEqual(
            self.submit(trigger_id="trigger-2", alert_id="alert-2")["state"], "submitted"
        )

    def test_post_conflict_requires_matching_existing_conf(self):
        self.client.post_error = "airflow_conflict"
        self.assertEqual(self.submit()["state"], "submitted")
        self.assertTrue(any("/retrain-trigger-1" in path for _, path, _ in self.client.calls))

    def test_ambiguous_post_failure_reconciles_matching_run(self):
        self.client.post_error = "airflow_unavailable"
        self.assertEqual(self.submit()["state"], "submitted")
        self.assertEqual(sum(method == "POST" for method, _, _ in self.client.calls), 1)

    def test_unresolved_post_stays_pending_then_reconciles_without_second_post(self):
        self.client.post_error = "airflow_unavailable"
        self.client.exact_error = "airflow_unavailable"
        with self.assertRaises(PipelineError) as caught:
            self.submit()
        self.assertEqual(caught.exception.code, "submission_pending")
        receipt_path = self.root / "artifacts/retraining/trigger-1.json"
        self.assertEqual(json.loads(receipt_path.read_text())["state"], "pending")
        self.client.exact_error = None
        self.assertEqual(self.submit()["state"], "submitted")
        self.assertEqual(sum(method == "POST" for method, _, _ in self.client.calls), 1)

    def test_conflicting_airflow_run_is_never_accepted(self):
        def conflict_client(method, path, payload):
            if method == "GET" and "?" in path:
                return {"dag_runs": [], "total_entries": 0}
            if method == "POST":
                raise PipelineError("airflow_conflict", "Existing run.")
            return {"dag_run_id": "retrain-trigger-1", "conf": {"mode": "manual"}}

        self.client = conflict_client
        with self.assertRaises(PipelineError) as caught:
            self.submit()
        self.assertEqual(caught.exception.code, "trigger_conflict")

    def test_active_run_on_later_page_is_detected(self):
        def paginated(method, path, payload):
            if "offset=0" in path:
                return {"dag_runs": [{"state": "success"}] * 100, "total_entries": 101}
            return {"dag_runs": [{"state": "queued"}], "total_entries": 101}

        self.client = paginated
        with self.assertRaises(PipelineError) as caught:
            self.submit()
        self.assertEqual(caught.exception.code, "active_run")


if __name__ == "__main__":
    unittest.main()
