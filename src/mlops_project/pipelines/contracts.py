"""Small, dependency-light contracts shared by the worker and Airflow DAG."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 1
STAGES = (
    "ingest",
    "validate",
    "split",
    "features",
    "train_baseline",
    "train_candidate_1",
    "train_candidate_2",
    "evaluate",
    "register",
    "benchmark",
    "approve",
    "deploy",
    "verify",
)
DEPENDENCIES = {
    "ingest": (),
    "validate": ("ingest",),
    "split": ("validate",),
    "features": ("split",),
    "train_baseline": ("features",),
    "train_candidate_1": ("features",),
    "train_candidate_2": ("features",),
    "evaluate": ("train_baseline", "train_candidate_1", "train_candidate_2"),
    "register": ("evaluate",),
    "benchmark": ("register",),
    "approve": ("evaluate", "register", "benchmark"),
    "deploy": ("approve",),
    "verify": ("deploy",),
}
STAGE_OWNERS = {
    **dict.fromkeys(("ingest", "validate", "split", "features"), "@OuanEng (P1)"),
    **dict.fromkeys(
        ("train_baseline", "train_candidate_1", "train_candidate_2", "evaluate", "register"),
        "@pairot230 (P2)",
    ),
    "benchmark": "@thanachaithongbai-hue (P3)",
    **dict.fromkeys(("deploy", "verify"), "P0 (integration owner)"),
    "approve": "P0 (integration owner)",
}
REQUIRED_PASS = {"validate", "evaluate", "benchmark", "approve", "verify"}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ADAPTER = re.compile(r"mlops_project(?:\.[A-Za-z_]\w*)+:[A-Za-z_]\w*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.IGNORECASE)


class PipelineError(RuntimeError):
    """An error safe to persist or return over the worker's HTTP interface."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def validate_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or not _RUN_ID.fullmatch(run_id)
        or run_id.endswith(".")
        or _WINDOWS_DEVICE.fullmatch(run_id)
    ):
        raise PipelineError(
            "invalid_run_id",
            "Use a 1–128 character run ID containing letters, digits, '.', '_' or '-'.",
        )
    return run_id


def validate_step(step: str) -> str:
    if not isinstance(step, str) or step not in STAGES:
        raise PipelineError("invalid_step", "Unknown pipeline stage.")
    return step


def project_directory() -> Path:
    return Path(__file__).resolve().parents[3]


def confined_path(value: str | Path, root: Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PipelineError(
            "path_outside_root", "The requested path is outside the allowed project directory."
        )
    if must_exist and not resolved.is_file():
        raise PipelineError("missing_file", "A required project file is missing.")
    return resolved


DATA_FILE_SUFFIXES = (".csv", ".json", ".xls", ".xlsx")


def data_file_reference(value, root: Path) -> dict:
    """Pin an operator-supplied dataset under data/ so a run can only ingest that content."""
    if not isinstance(value, str) or not value.strip():
        raise PipelineError("invalid_data_file", "data_file must be a nonempty project path.")
    path = confined_path(value, root)
    if not path.is_relative_to((root / "data").resolve()):
        raise PipelineError("path_outside_root", "data_file must be inside the data/ directory.")
    if path.suffix.lower() not in DATA_FILE_SUFFIXES:
        raise PipelineError(
            "invalid_data_file", "data_file must be a .csv, .json, .xls or .xlsx file."
        )
    if not path.is_file():
        raise PipelineError("missing_file", "data_file does not exist under data/.")
    return {"uri": path.relative_to(root.resolve()).as_posix(), "sha256": sha256_file(path)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError):
        raise PipelineError(
            "invalid_result", "Adapter results must contain finite, JSON-compatible values."
        ) from None


def load_config(
    config_path: str | Path, project_root: str | Path | None = None
) -> tuple[dict, Path]:
    root = Path(project_root).resolve() if project_root else project_directory()
    path = confined_path(config_path, root, must_exist=True)
    try:
        import yaml
    except ImportError:
        raise PipelineError(
            "missing_dependency", "Install the pinned worker dependencies (PyYAML is required)."
        ) from None
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise PipelineError(
            "invalid_config", "Project configuration is not valid UTF-8 YAML."
        ) from None
    if (
        not isinstance(config, dict)
        or type(config.get("contract_version")) is not int
        or config.get("contract_version") != CONTRACT_VERSION
        or not isinstance(config.get("pipeline"), dict)
    ):
        raise PipelineError(
            "invalid_config", "Project configuration must contain a pipeline mapping."
        )
    pipeline = config["pipeline"]
    adapters = pipeline.get("adapters")
    if not isinstance(adapters, dict):
        raise PipelineError(
            "invalid_config", "pipeline.adapters must map each stage to a module:function."
        )
    if set(adapters) != set(STAGES):
        raise PipelineError(
            "invalid_config",
            "pipeline.adapters must contain exactly the documented pipeline stages.",
        )
    for step in STAGES:
        validate_adapter_reference(adapters[step])
    timeout = pipeline.get("stage_timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise PipelineError(
            "invalid_config", "pipeline.stage_timeout_seconds must be a finite positive number."
        )
    return config, path


def configuration_snapshot(config: dict, config_path: Path, root: Path) -> tuple[str, dict]:
    """Bind a run to project, policy, monitoring and reviewed schema content."""
    import yaml

    paths = [config_path]
    schema_config = None
    for key in ("quality_gates", "schema_config", "monitoring_config"):
        if key in config:
            if not isinstance(config[key], str):
                raise PipelineError(
                    "invalid_config", f"{key} must reference a project-local configuration file."
                )
            path = confined_path(config[key], root)
            paths.append(path)
            if key == "schema_config":
                schema_config = path
    if schema_config is not None and schema_config.is_file():
        try:
            schema = yaml.safe_load(schema_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            raise PipelineError(
                "invalid_config", "Schema configuration is not valid YAML."
            ) from None
        if isinstance(schema, dict) and "schema_path" in schema:
            if not isinstance(schema["schema_path"], str):
                raise PipelineError(
                    "invalid_config",
                    "schema_path must reference the reviewed project-local TFDV schema.",
                )
            paths.append(confined_path(schema["schema_path"], root))
    snapshot = {
        path.relative_to(root).as_posix(): sha256_file(path) if path.is_file() else None
        for path in paths
    }
    return hashlib.sha256(json_bytes(snapshot)).hexdigest(), snapshot


def validate_adapter_reference(reference: str) -> str:
    if not isinstance(reference, str) or not _ADAPTER.fullmatch(reference):
        raise PipelineError(
            "invalid_adapter",
            "Adapters must use a trusted mlops_project.module:function reference.",
        )
    return reference


def resolve_adapter(reference: str, step: str):
    validate_adapter_reference(reference)
    module_name, function_name = reference.split(":")
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except (ImportError, AttributeError):
        raise PipelineError(
            "adapter_unavailable",
            f"{STAGE_OWNERS[step]} must implement {reference} and install its pinned dependencies.",
        ) from None
    if not callable(function):
        raise PipelineError(
            "adapter_unavailable", f"{STAGE_OWNERS[step]} must expose a callable {reference}."
        )
    return function


def validate_result(result: Any, step: str, run_id: str, run_dir: Path) -> dict:
    if (
        not isinstance(result, dict)
        or type(result.get("contract_version")) is not int
        or result["contract_version"] != CONTRACT_VERSION
    ):
        raise PipelineError("invalid_result", "Adapter results require contract_version: 1.")
    if "run_id" in result and result["run_id"] != run_id:
        raise PipelineError(
            "stale_evidence", "Adapter evidence belongs to a different pipeline run."
        )
    if "passed" in result and type(result["passed"]) is not bool:
        raise PipelineError("invalid_result", "The passed flag must be a boolean.")
    if result.get("passed") is False or result.get("status") in {"failed", "error", "rejected"}:
        raise PipelineError(
            "gate_failed",
            f"The {step} stage reported a failed check; downstream execution is blocked.",
        )
    if step in REQUIRED_PASS and result.get("passed") is not True:
        raise PipelineError(
            "missing_gate_evidence", f"The {step} stage must explicitly report passed: true."
        )
    normalized = {**result, "run_id": run_id}
    json_bytes(normalized)
    artifacts = normalized.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise PipelineError(
            "invalid_result", "artifacts must be a list of {uri, sha256} references."
        )
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("uri"), str)
            or not isinstance(artifact.get("sha256"), str)
            or not _SHA256.fullmatch(artifact["sha256"])
        ):
            raise PipelineError(
                "invalid_result",
                "Every artifact reference needs a local uri and lowercase SHA-256.",
            )
        path = confined_path(artifact["uri"], run_dir, must_exist=True)
        if sha256_file(path) != artifact["sha256"]:
            raise PipelineError(
                "artifact_mismatch", "An artifact's content does not match its recorded checksum."
            )
    return normalized
