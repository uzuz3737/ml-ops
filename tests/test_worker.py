import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from mlops_project.pipelines.client import execute_remote
from mlops_project.pipelines.contracts import PipelineError
from mlops_project.pipelines.worker import make_server


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.server = make_server("127.0.0.1", 0, project_root=self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, payload):
        request = urllib.request.Request(
            self.url + "/steps",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_health_describes_worker_liveness(self):
        with urllib.request.urlopen(self.url + "/health") as response:
            self.assertEqual(json.load(response), {"status": "alive", "contract_version": 1})

    def test_disallowed_config_never_calls_runner(self):
        with patch("mlops_project.pipelines.worker.run_step") as runner:
            status, result = self.post(
                {"step": "ingest", "run_id": "test", "config_path": "other.yaml"}
            )
        self.assertEqual(status, 403)
        self.assertEqual(result["error"]["code"], "config_not_allowed")
        runner.assert_not_called()

    def test_escape_config_is_forbidden(self):
        status, result = self.post(
            {"step": "ingest", "run_id": "test", "config_path": "../outside.yaml"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(result["error"]["code"], "path_outside_root")

    def test_extra_keys_cannot_inject_adapter(self):
        status, result = self.post(
            {
                "step": "ingest",
                "run_id": "test",
                "config_path": "configs/project.yaml",
                "adapter": "os:system",
            }
        )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_missing_adapter_returns_failure_with_owner(self):
        with patch(
            "mlops_project.pipelines.worker.run_step",
            side_effect=PipelineError("adapter_unavailable", "@OuanEng must implement ingest."),
        ):
            status, result = self.post(
                {"step": "ingest", "run_id": "test", "config_path": "configs/project.yaml"}
            )
        self.assertEqual(status, 503)
        self.assertIn("@OuanEng", result["error"]["message"])

    def test_success_returns_stage_record(self):
        record = {
            "contract_version": 1,
            "run_id": "test",
            "step": "ingest",
            "state": "succeeded",
            "result": {"contract_version": 1, "run_id": "test"},
        }
        with patch("mlops_project.pipelines.worker.run_step", return_value=record):
            status, result = self.post(
                {"step": "ingest", "run_id": "test", "config_path": "configs/project.yaml"}
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, record)

    def test_airflow_client_and_real_worker_http_agree_on_evidence_contract(self):
        record = {
            "contract_version": 1,
            "run_id": "wire-test",
            "step": "validate",
            "state": "succeeded",
            "result": {"contract_version": 1, "run_id": "wire-test", "passed": True},
        }
        with patch("mlops_project.pipelines.worker.run_step", return_value=record):
            result = execute_remote(
                "validate",
                "wire-test",
                "configs/project.yaml",
                worker_url=self.url,
                timeout_seconds=5,
            )
        self.assertEqual(result, record)


if __name__ == "__main__":
    unittest.main()
