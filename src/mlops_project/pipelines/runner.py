"""Execute trusted adapters with deadlines, atomic evidence and failure barriers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .contracts import (
    CONTRACT_VERSION,
    DEPENDENCIES,
    PipelineError,
    configuration_snapshot,
    confined_path,
    json_bytes,
    load_config,
    project_directory,
    resolve_adapter,
    sha256_file,
    validate_result,
    validate_run_id,
    validate_step,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path, timeout_seconds: float = 5):
    """OS locks release after a process crash and work on Windows and Linux."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise PipelineError(
                        "stage_busy", "Another worker is already executing this pipeline stage."
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PipelineError(
            "missing_evidence", "A required upstream stage record is missing or unreadable."
        ) from None
    if not isinstance(record, dict):
        raise PipelineError("invalid_evidence", "A stage record must be a JSON object.")
    return record


def _run_adapter(reference: str, context: dict, timeout: float, run_dir: Path) -> dict:
    """A child process ensures timeouts stop adapter execution, not just waiting."""
    with tempfile.TemporaryDirectory(prefix=".adapter-", dir=run_dir) as temporary:
        context_path = Path(temporary) / "context.json"
        result_path = Path(temporary) / "result.json"
        atomic_json(context_path, context)
        environment = os.environ.copy()
        source_path = str(Path(__file__).resolve().parents[2])
        project_source = Path(context.get("project_root", source_path)) / "src"
        sources = [str(project_source)] if project_source.is_dir() else []
        sources.append(source_path)
        environment["PYTHONPATH"] = (
            os.pathsep.join(sources) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlops_project.pipelines.runner",
                    "--adapter-child",
                    reference,
                    str(context_path),
                    str(result_path),
                ],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise PipelineError(
                "stage_timeout", "Adapter execution exceeded pipeline.stage_timeout_seconds."
            ) from None
        except OSError:
            raise PipelineError(
                "worker_unavailable", "The adapter process could not start."
            ) from None
        try:
            payload = read_record(result_path)
        except PipelineError:
            raise PipelineError(
                "adapter_failed", "Adapter execution failed without a valid result."
            ) from None
        if completed.returncode != 0 or "error" in payload:
            error = payload.get("error", {})
            if (
                isinstance(error, dict)
                and isinstance(error.get("code"), str)
                and isinstance(error.get("message"), str)
            ):
                raise PipelineError(error["code"], error["message"])
            raise PipelineError("adapter_failed", "Adapter execution failed.")
        return payload.get("result")


def _retraining_context(run_id: str, config: dict, config_hash: str, artifacts: Path, root: Path):
    if not run_id.startswith("retrain-"):
        return None
    trigger_id = validate_run_id(run_id[len("retrain-") :])
    state_root = confined_path(
        config["pipeline"].get("retraining_state_dir", artifacts / "retraining"), root
    )
    # The controller keeps this lock through POST and receipt commit, eliminating
    # the race in which Airflow starts before its receipt is marked submitted.
    # Airflow may accept POST before a 30s response timeout, then require a
    # further 30s GET reconciliation. The stage must wait for that commit.
    with file_lock(state_root / ".controller.lock", timeout_seconds=75):
        receipt = read_record(state_root / f"{trigger_id}.json")
        request = receipt.get("retraining_request")
        if (
            type(receipt.get("contract_version")) is not int
            or receipt.get("contract_version") != CONTRACT_VERSION
            or receipt.get("state") != "submitted"
            or receipt.get("trigger_id") != trigger_id
            or receipt.get("pipeline_run_id") != run_id
            or receipt.get("policy_sha256") != config_hash
            or not isinstance(request, dict)
        ):
            raise PipelineError(
                "unapproved_retraining",
                "Retraining requires a matching submitted controller receipt and frozen configuration.",
            )
        if (
            request.get("trigger_id") != trigger_id
            or not isinstance(request.get("candidate_dataset_version"), str)
            or not request["candidate_dataset_version"]
            or any(
                request.get(flag) is not True
                for flag in (
                    "candidate_dataset_approved",
                    "labels_validated",
                    "separate_training_validation",
                    "final_test_excluded",
                )
            )
        ):
            raise PipelineError(
                "unapproved_retraining",
                "The controller receipt lacks an approved independent labeled retraining dataset.",
            )
        return {**request, "receipt_sha256": sha256_file(state_root / f"{trigger_id}.json")}


def run_step(
    step: str,
    run_id: str,
    config_path: str | Path,
    artifact_root: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict:
    validate_step(step)
    validate_run_id(run_id)
    root = Path(project_root).resolve() if project_root else project_directory()
    config, config_file = load_config(config_path, root)
    artifacts = confined_path(
        artifact_root or config["pipeline"].get("artifact_root", "artifacts"), root
    )
    run_dir = confined_path(Path("runs") / run_id, artifacts)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash, config_snapshot = configuration_snapshot(config, config_file, root)
    retraining = _retraining_context(run_id, config, config_hash, artifacts, root)
    with file_lock(run_dir / ".run.lock"):
        manifest_path = run_dir / "run.json"
        if manifest_path.exists():
            manifest = read_record(manifest_path)
            if (
                manifest.get("run_id") != run_id
                or manifest.get("config_sha256") != config_hash
                or manifest.get("retraining") != retraining
            ):
                raise PipelineError(
                    "stale_evidence",
                    "This run ID already belongs to a different configuration; use a new run ID.",
                )
        else:
            atomic_json(
                manifest_path,
                {
                    "contract_version": CONTRACT_VERSION,
                    "run_id": run_id,
                    "config_sha256": config_hash,
                    "configuration_snapshot": config_snapshot,
                    "retraining": retraining,
                    "created_at": utc_now(),
                },
            )
    with file_lock(run_dir / f".{step}.lock"):
        inputs, input_hashes = {}, {}
        for dependency in DEPENDENCIES[step]:
            dependency_path = run_dir / f"{dependency}.json"
            record = read_record(dependency_path)
            if (
                record.get("state") != "succeeded"
                or record.get("run_id") != run_id
                or record.get("step") != dependency
                or record.get("config_sha256") != config_hash
            ):
                raise PipelineError(
                    "upstream_failed",
                    f"Stage {dependency} has not succeeded with this run's configuration.",
                )
            inputs[dependency] = validate_result(record.get("result"), dependency, run_id, run_dir)
            input_hashes[dependency] = sha256_file(dependency_path)
        record_path = run_dir / f"{step}.json"
        if record_path.exists():
            previous = read_record(record_path)
            if previous.get("state") == "succeeded":
                if (
                    previous.get("run_id") != run_id
                    or previous.get("step") != step
                    or previous.get("config_sha256") != config_hash
                    or previous.get("input_sha256") != input_hashes
                ):
                    raise PipelineError(
                        "stale_evidence",
                        "A successful stage has changed provenance; start a new pipeline run.",
                    )
                validate_result(previous.get("result"), step, run_id, run_dir)
                return previous
        record = {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "step": step,
            "state": "running",
            "started_at": utc_now(),
            "config_sha256": config_hash,
            "input_sha256": input_hashes,
        }
        atomic_json(record_path, record)
        context = {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "step": step,
            "config": config,
            "config_path": str(config_file),
            "config_sha256": config_hash,
            "project_root": str(root),
            "run_dir": str(run_dir),
            "inputs": inputs,
            "retraining": retraining,
        }
        try:
            result = _run_adapter(
                config["pipeline"]["adapters"][step],
                context,
                config["pipeline"]["stage_timeout_seconds"],
                run_dir,
            )
            current_config, _ = load_config(config_file, root)
            current_hash, _ = configuration_snapshot(current_config, config_file, root)
            if current_hash != config_hash:
                raise PipelineError(
                    "stale_evidence",
                    "Configuration changed during adapter execution; start a new run.",
                )
            record.update(state="succeeded", result=validate_result(result, step, run_id, run_dir))
        except PipelineError as error:
            record.update(state="failed", error=error.as_dict(), finished_at=utc_now())
            atomic_json(record_path, record)
            raise
        except Exception:
            error = PipelineError("adapter_failed", "Adapter execution failed unexpectedly.")
            record.update(state="failed", error=error.as_dict(), finished_at=utc_now())
            atomic_json(record_path, record)
            raise error from None
        record["finished_at"] = utc_now()
        atomic_json(record_path, record)
        return record


def _adapter_child(reference: str, context_path: str, result_path: str) -> int:
    try:
        context = read_record(Path(context_path))
        adapter = resolve_adapter(reference, context["step"])
        result = adapter(context)
        atomic_json(Path(result_path), {"result": result})
        return 0
    except PipelineError as error:
        atomic_json(Path(result_path), {"error": error.as_dict()})
    except Exception:
        atomic_json(
            Path(result_path),
            {
                "error": {
                    "code": "adapter_failed",
                    "message": "The adapter raised an internal error; its owner must investigate.",
                }
            },
        )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-child", nargs=3, metavar=("ADAPTER", "CONTEXT", "RESULT"))
    parser.add_argument("--step")
    parser.add_argument("--run-id")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--project-root")
    args = parser.parse_args()
    if args.adapter_child:
        return _adapter_child(*args.adapter_child)
    if not args.step or not args.run_id:
        parser.error("--step and --run-id are required")
    try:
        print(
            json.dumps(
                run_step(args.step, args.run_id, args.config, project_root=args.project_root)
            )
        )
        return 0
    except PipelineError as error:
        print(json.dumps({"error": error.as_dict()}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
