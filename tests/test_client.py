import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from mlops_project.pipelines.client import execute_remote
from mlops_project.pipelines.contracts import PipelineError


class ClientTests(unittest.TestCase):
    def payload(self, **updates):
        return {
            "contract_version": 1,
            "step": "ingest",
            "run_id": "test",
            "state": "succeeded",
            "result": {"contract_version": 1, "run_id": "test"},
            **updates,
        }

    def execute(self):
        return execute_remote(
            "ingest",
            "test",
            "configs/project.yaml",
            worker_url="http://worker:8100",
            timeout_seconds=5,
        )

    def test_success_validates_identity_and_sends_exact_fields(self):
        payload = self.payload()
        with patch(
            "mlops_project.pipelines.client.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(payload).encode()),
        ) as opener:
            self.assertEqual(self.execute(), payload)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://worker:8100/steps")
        self.assertEqual(
            json.loads(request.data),
            {"step": "ingest", "run_id": "test", "config_path": "configs/project.yaml"},
        )

    def test_mismatched_or_failed_response_cannot_succeed(self):
        for update in (
            {"run_id": "stale"},
            {"state": "failed"},
            {"contract_version": 2},
            {"step": "deploy"},
            {"result": {"contract_version": 1, "run_id": "stale"}},
            {"result": {"contract_version": 1, "run_id": "test", "passed": False}},
            {"result": {"contract_version": 1, "run_id": "test", "passed": "true"}},
        ):
            with (
                self.subTest(update=update),
                patch(
                    "mlops_project.pipelines.client.urllib.request.urlopen",
                    return_value=io.BytesIO(json.dumps(self.payload(**update)).encode()),
                ),
            ):
                with self.assertRaises(PipelineError) as caught:
                    self.execute()
            self.assertEqual(caught.exception.code, "invalid_worker_response")

    def test_non_json_is_safe_error(self):
        with patch(
            "mlops_project.pipelines.client.urllib.request.urlopen",
            return_value=io.BytesIO(b"<html>proxy error</html>"),
        ):
            with self.assertRaises(PipelineError) as caught:
                self.execute()
        self.assertEqual(caught.exception.code, "invalid_worker_response")

    def test_validation_requires_explicit_pass_evidence(self):
        payload = self.payload(step="validate")
        with patch(
            "mlops_project.pipelines.client.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(payload).encode()),
        ):
            with self.assertRaises(PipelineError) as caught:
                execute_remote("validate", "test", "configs/project.yaml")
        self.assertEqual(caught.exception.code, "invalid_worker_response")

    def test_http_failures_remain_failures(self):
        error = urllib.error.HTTPError(
            "http://worker/steps",
            503,
            "Service unavailable",
            {},
            io.BytesIO(
                b'{"error":{"code":"adapter_unavailable","message":"Owner must implement adapter."}}'
            ),
        )
        with patch("mlops_project.pipelines.client.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(PipelineError) as caught:
                self.execute()
        self.assertEqual(caught.exception.code, "adapter_unavailable")

    def test_network_failure_is_safe(self):
        with patch(
            "mlops_project.pipelines.client.urllib.request.urlopen",
            side_effect=urllib.error.URLError("internal-host-secret"),
        ):
            with self.assertRaises(PipelineError) as caught:
                self.execute()
        self.assertEqual(caught.exception.code, "worker_unavailable")
        self.assertNotIn("internal-host-secret", caught.exception.message)

    def test_invalid_timeout_or_url_is_rejected_before_http(self):
        for timeout in (0, -1, float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaises(PipelineError):
                execute_remote("ingest", "test", "project.yaml", timeout_seconds=timeout)
        with self.assertRaises(PipelineError):
            execute_remote("ingest", "test", "project.yaml", worker_url="http://name:secret@worker")


if __name__ == "__main__":
    unittest.main()
