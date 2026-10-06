"""Internal HTTP worker for the Airflow stage adapters, not the prediction API."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .contracts import CONTRACT_VERSION, PipelineError, confined_path, json_bytes, project_directory
from .runner import run_step

MAX_REQUEST_BYTES = 16 * 1024
ERROR_STATUS = {
    "invalid_run_id": 400,
    "invalid_step": 400,
    "invalid_request": 400,
    "path_outside_root": 403,
    "config_not_allowed": 403,
    "missing_file": 422,
    "invalid_data_file": 422,
    "data_validation_failed": 422,
    "invalid_config": 422,
    "invalid_adapter": 422,
    "stage_busy": 409,
    "stale_evidence": 409,
    "upstream_failed": 409,
    "missing_evidence": 409,
    "invalid_evidence": 409,
    "gate_failed": 422,
    "missing_gate_evidence": 422,
    "adapter_unavailable": 503,
    "missing_dependency": 503,
    "stage_timeout": 504,
}


def make_server(
    host: str = "127.0.0.1",
    port: int = 8100,
    *,
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    allowed_config_paths: list[str | Path] | None = None,
) -> ThreadingHTTPServer:
    root = Path(project_root).resolve() if project_root else project_directory()
    allowed = {
        confined_path(path, root) for path in (allowed_config_paths or ["configs/project.yaml"])
    }

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            # Persist safe execution evidence in run records; do not print body/errors.
            return

        def respond(self, status: int, value: dict):
            body = json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "alive", "contract_version": CONTRACT_VERSION})
            else:
                self.respond(
                    404, {"error": {"code": "not_found", "message": "Unknown worker endpoint."}}
                )

        def do_POST(self):
            if self.path != "/steps":
                self.respond(
                    404, {"error": {"code": "not_found", "message": "Unknown worker endpoint."}}
                )
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise PipelineError("invalid_request", "Chunked requests are not accepted.")
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    raise PipelineError(
                        "invalid_request", "A valid Content-Length is required."
                    ) from None
                if length < 0:
                    raise PipelineError("invalid_request", "Content-Length is required.")
                if length > MAX_REQUEST_BYTES:
                    self.respond(
                        413,
                        {
                            "error": {
                                "code": "payload_too_large",
                                "message": "Worker requests must be at most 16 KiB.",
                            }
                        },
                    )
                    return
                if (
                    self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    != "application/json"
                ):
                    self.respond(
                        415,
                        {
                            "error": {
                                "code": "invalid_content_type",
                                "message": "Use application/json.",
                            }
                        },
                    )
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    raise PipelineError("invalid_request", "Incomplete request body.")
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    raise PipelineError(
                        "invalid_request", "Request body must contain a valid JSON object."
                    ) from None
                required = {"step", "run_id", "config_path"}
                if (
                    not isinstance(payload, dict)
                    or not required <= set(payload)
                    or not set(payload) <= required | {"data_file"}
                ):
                    raise PipelineError(
                        "invalid_request",
                        "Request keys must be step, run_id, config_path and optional data_file.",
                    )
                if not isinstance(payload["config_path"], str):
                    raise PipelineError("invalid_request", "config_path must be a string.")
                if "data_file" in payload and not isinstance(payload["data_file"], str):
                    raise PipelineError("invalid_request", "data_file must be a string.")
                config_path = confined_path(payload["config_path"], root)
                if config_path not in allowed:
                    raise PipelineError(
                        "config_not_allowed", "The worker only accepts its configured project file."
                    )
                record = run_step(
                    payload["step"],
                    payload["run_id"],
                    config_path,
                    artifact_root=artifact_root,
                    project_root=root,
                    data_file=payload.get("data_file"),
                )
                self.respond(200, record)
            except PipelineError as error:
                self.respond(ERROR_STATUS.get(error.code, 500), {"error": error.as_dict()})
            except (TimeoutError, ConnectionError):
                self.close_connection = True
            except Exception:
                self.respond(
                    500,
                    {
                        "error": {
                            "code": "worker_failed",
                            "message": "The worker failed to execute this request.",
                        }
                    },
                )

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--project-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()
    server = make_server(
        args.host,
        args.port,
        project_root=args.project_root,
        artifact_root=args.artifact_root,
        allowed_config_paths=[args.config],
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
