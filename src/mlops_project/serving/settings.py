"""Runtime settings for the API process, read from the shared project config."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..pipelines.contracts import PipelineError, confined_path, load_config, project_directory


@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_name: str
    schema_version: str
    features: tuple[str, ...]
    deployments_dir: Path
    events_dir: Path | None
    monitoring_dir: Path
    smoke_fixture: dict
    max_batch_size: int = 1000
    max_body_bytes: int = 1_000_000
    poll_seconds: float = 1.0
    # Set only for the isolated benchmark server (no watcher, no event logging).
    candidate: dict | None = field(default=None)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value < 1:
        raise PipelineError("invalid_config", f"{name} must be a positive integer.")
    return value


def load_settings() -> Settings:
    root = Path(os.environ.get("MLOPS_PROJECT_ROOT") or project_directory()).resolve()
    config, _ = load_config(os.environ.get("MLOPS_CONFIG", "configs/project.yaml"), root)
    serving = config.get("serving", {})

    schema_path = confined_path(
        config.get("schema_config", "configs/data_schema.yaml"), root, must_exist=True
    )
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    features = schema.get("features") if isinstance(schema, dict) else None
    if not isinstance(features, list) or not features or not schema.get("schema_version"):
        raise PipelineError("invalid_config", "data_schema.yaml needs schema_version and features.")

    fixture_path = confined_path(
        serving.get("smoke_fixture", "examples/predict.json"), root, must_exist=True
    )
    artifact_root = confined_path(config["pipeline"].get("artifact_root", "artifacts"), root)

    candidate = None
    candidate_file = os.environ.get("MLOPS_CANDIDATE_MANIFEST")
    if candidate_file:
        candidate = json.loads(Path(candidate_file).read_text(encoding="utf-8"))

    return Settings(
        project_root=root,
        model_name=serving.get("model_name", "credit-default"),
        schema_version=schema["schema_version"],
        features=tuple(features),
        deployments_dir=confined_path(
            serving.get("deployments_dir", "artifacts/deployments"), root
        ),
        events_dir=None if candidate else artifact_root / "serving" / "events",
        monitoring_dir=artifact_root / "monitoring" / "exported",
        smoke_fixture=json.loads(fixture_path.read_text(encoding="utf-8")),
        max_batch_size=_int_env("MLOPS_MAX_BATCH_SIZE", 1000),
        max_body_bytes=_int_env("MLOPS_MAX_BODY_BYTES", 1_000_000),
        poll_seconds=float(os.environ.get("MLOPS_WATCH_SECONDS", "1")),
        candidate=candidate,
    )
