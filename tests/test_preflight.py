import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mlops_project.pipelines.contracts import STAGES
from mlops_project.pipelines.preflight import preflight


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "project.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "contract_version": 1,
                    "quality_gates": "quality.yaml",
                    "pipeline": {
                        "stage_timeout_seconds": 5,
                        "registry_finalize": "mlops_project.registry.pipeline:finalize_deployment",
                        "adapters": {
                            step: f"mlops_project.data.pipeline:{step}" for step in STAGES
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self.quality = {
            "gates": {
                "metric": {"minimum": 0.3},
                "comparison": {"max_regression": 0.02},
                "service": {
                    "p95_latency_ms_max": 300,
                    "error_rate_max": 0.01,
                    "throughput_rps_min": 20,
                },
            }
        }
        self.write_quality()

    def write_quality(self):
        (self.root / "quality.yaml").write_text(yaml.safe_dump(self.quality), encoding="utf-8")

    def test_missing_modules_are_reported_and_block_pipeline(self):
        config = yaml.safe_load(self.config.read_text())
        config["pipeline"]["adapters"]["ingest"] = "mlops_project.unimplemented:ingest"
        self.config.write_text(yaml.safe_dump(config))
        report = preflight(self.config, project_root=self.root)
        self.assertFalse(report["passed"])
        errors = [check.get("error", {}) for check in report["checks"]]
        self.assertTrue(any("@OuanEng" in error.get("message", "") for error in errors))

    def test_unset_quality_threshold_blocks_even_with_adapters(self):
        self.quality["gates"]["metric"]["minimum"] = None
        self.write_quality()
        with patch(
            "mlops_project.pipelines.preflight.resolve_adapter", return_value=lambda context: {}
        ):
            report = preflight(self.config, project_root=self.root)
        self.assertFalse(report["passed"])
        self.assertEqual(report["checks"][-1]["error"]["code"], "unset_quality_gate")

    def test_valid_plumbing_and_numeric_policy_pass_preflight(self):
        with patch(
            "mlops_project.pipelines.preflight.resolve_adapter", return_value=lambda context: {}
        ):
            report = preflight(self.config, project_root=self.root)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["checks"]), len(STAGES) + 3)

    def test_bad_config_fails_without_importing_adapters(self):
        self.config.write_text("pipeline: []", encoding="utf-8")
        with patch("mlops_project.pipelines.preflight.resolve_adapter") as resolve:
            report = preflight(self.config, project_root=self.root)
        self.assertFalse(report["passed"])
        resolve.assert_not_called()

    def test_outside_policy_path_cannot_pass_preflight(self):
        config = yaml.safe_load(self.config.read_text())
        config["quality_gates"] = "../outside.yaml"
        self.config.write_text(yaml.safe_dump(config), encoding="utf-8")
        with patch(
            "mlops_project.pipelines.preflight.resolve_adapter", return_value=lambda context: {}
        ):
            report = preflight(self.config, project_root=self.root)
        self.assertEqual(report["checks"][-1]["error"]["code"], "path_outside_root")


if __name__ == "__main__":
    unittest.main()
