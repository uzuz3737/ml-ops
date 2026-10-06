"""Follow P0's desired-model.json and acknowledge every deployment ID.

deploy/rollback: verify approval evidence and checksum, smoke-predict the new
bundle while the old one keeps serving, then swap. unload: clear the slot.
A failed load leaves the current model untouched and ACKs status "failed".
"""

from __future__ import annotations

import logging
import re
import threading
import time

from mlops_project.artifacts import atomic_json, read_record, utc_now

from ..pipelines.contracts import PipelineError, confined_path, sha256_file, validate_run_id
from . import telemetry
from .bundle import ModelSlot, load_model
from .settings import Settings

_VERSION = re.compile(r"[1-9][0-9]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED = (
    "model_name",
    "model_version",
    "schema_version",
    "artifact_uri",
    "artifact_sha256",
    "gate_report_uri",
    "gate_report_sha256",
)


class DeploymentWatcher:
    def __init__(self, settings: Settings, slot: ModelSlot):
        self.settings = settings
        self.slot = slot
        self.handled: str | None = None

    @property
    def desired_path(self):
        return self.settings.deployments_dir / "desired-model.json"

    def _ack(self, deployment_id: str, **fields) -> None:
        record = {"contract_version": 1, "deployment_id": deployment_id, **fields}
        record["acknowledged_at"] = utc_now()
        path = self.settings.deployments_dir / "acknowledgements" / f"{deployment_id}.json"
        atomic_json(path, record)

    def _check_approval(self, manifest: dict) -> None:
        for name in _REQUIRED:
            if not isinstance(manifest.get(name), str) or not manifest[name]:
                raise PipelineError("invalid_manifest", f"Manifest is missing {name}.")
        if manifest["model_name"] != self.settings.model_name:
            raise PipelineError("invalid_manifest", "Manifest names a different model.")
        if manifest["schema_version"] != self.settings.schema_version:
            raise PipelineError("incompatible_schema", "Manifest schema differs from serving.")
        if not _VERSION.fullmatch(manifest["model_version"]):
            raise PipelineError("invalid_model_version", "Model version must be exact.")
        if not _SHA.fullmatch(manifest["artifact_sha256"]):
            raise PipelineError("invalid_manifest", "artifact_sha256 must be lowercase SHA-256.")
        root = self.settings.project_root
        report_path = confined_path(manifest["gate_report_uri"], root, must_exist=True)
        if sha256_file(report_path) != manifest["gate_report_sha256"]:
            raise PipelineError("artifact_mismatch", "Gate report checksum changed.")
        report = read_record(report_path)
        if (
            report.get("passed") is not True
            or report.get("candidate_model_version") != manifest["model_version"]
            or report.get("artifact_sha256") != manifest["artifact_sha256"]
        ):
            raise PipelineError("unapproved_deployment", "Gate report does not approve this model.")

    def _activate(self, manifest: dict) -> None:
        self._check_approval(manifest)
        model = load_model(manifest, self.settings.project_root, self.settings.features)
        instances = self.settings.smoke_fixture.get("instances") or []
        if len(model.predict(instances)) != len(instances):
            raise PipelineError("invalid_model_output", "Smoke prediction failed.")
        self.slot.swap(model)
        telemetry.MODEL_READY.set(1)
        telemetry.MODEL_LOADED_AT.set(time.time())

    def handle(self, manifest: dict) -> None:
        deployment_id = validate_run_id(manifest.get("deployment_id"))
        action = manifest.get("action")
        try:
            if (
                type(manifest.get("contract_version")) is not int
                or manifest["contract_version"] != 1
            ):
                raise PipelineError("invalid_manifest", "Unsupported manifest contract_version.")
            if action == "unload":
                self.slot.swap(None)
                telemetry.MODEL_READY.set(0)
                self._ack(deployment_id, status="unloaded")
            elif action in ("deploy", "rollback"):
                self._activate(manifest)
                self._ack(
                    deployment_id,
                    status="loaded",
                    model_version=manifest["model_version"],
                    schema_version=manifest["schema_version"],
                )
            else:
                raise PipelineError("invalid_manifest", "Unknown manifest action.")
        except Exception as caught:
            error = caught
            if not isinstance(error, PipelineError):
                telemetry.logger.exception("unexpected load error")
                error = PipelineError("model_load_failed", "Bundle failed to load or predict.")
            telemetry.DEPLOYMENTS.labels(str(action), "failed").inc()
            telemetry.log_event(
                "deployment_failed",
                logging.WARNING,
                deployment_id=deployment_id,
                action=action,
                error=error.code,
            )
            self._ack(deployment_id, status="failed", error=error.as_dict())
            return
        telemetry.DEPLOYMENTS.labels(action, "ok").inc()
        current = self.slot.get()
        telemetry.log_event(
            "deployment_applied",
            deployment_id=deployment_id,
            action=action,
            model_version=current.model_version if current else None,
        )

    def check_once(self) -> None:
        if not self.desired_path.exists():
            return
        try:
            manifest = read_record(self.desired_path)
            deployment_id = validate_run_id(manifest.get("deployment_id"))
        except PipelineError as error:
            telemetry.log_event("bad_desired_manifest", logging.WARNING, error=error.code)
            return
        if deployment_id == self.handled:
            return
        self.handle(manifest)
        self.handled = deployment_id

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.check_once()
            except Exception:  # keep watching; the next manifest may be fine
                telemetry.logger.exception("watcher loop error")
            stop.wait(self.settings.poll_seconds)
