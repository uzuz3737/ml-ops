"""Load a trusted model bundle and hold the one currently serving.

Reads P2's bundle.joblib: {"pipeline": fitted preprocessor + estimator,
"manifest": {threshold, feature_columns, schema_version, environment, ...}}.
Same checks as training.bundle.load_bundle, minus the mlflow import, so the
API image does not need MLflow installed. Nothing is fitted here.
"""

from __future__ import annotations

import importlib.metadata
import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..pipelines.contracts import PipelineError, confined_path, sha256_file


@dataclass(frozen=True)
class LoadedModel:
    deployment_id: str
    model_name: str
    model_version: str
    schema_version: str
    artifact_sha256: str
    features: tuple[str, ...]
    predictor: Any
    threshold: float
    positive_index: int

    def predict(self, instances: list[dict]) -> list[dict]:
        rows = _frame(instances, self.features)
        probabilities = self.predictor.predict_proba(rows)
        results = []
        for row in probabilities:
            probability = float(row[self.positive_index])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise PipelineError("invalid_model_output", "Model returned a bad probability.")
            results.append(
                {"label": int(probability >= self.threshold), "default_probability": probability}
            )
        if len(results) != len(instances):
            raise PipelineError("invalid_model_output", "Model returned the wrong row count.")
        return results


def _frame(instances: list[dict], features: tuple[str, ...]):
    # Named columns keep ColumnTransformer happy; fall back to plain rows in tests.
    try:
        import pandas as pd
    except ImportError:
        return [[item[name] for name in features] for item in instances]
    return pd.DataFrame([[item[name] for name in features] for item in instances], columns=features)


def resolve_artifact(uri: str, root: Path) -> Path:
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    elif "://" in uri or uri.startswith(("models:/", "runs:/")):
        raise PipelineError(
            "unsupported_artifact_uri", "Only project-local bundle paths are supported for now."
        )
    return confined_path(uri, root, must_exist=True)


# Pickled sklearn objects only load reliably with the exact versions used to train.
PINNED_PACKAGES = ("scikit-learn", "numpy", "pandas", "joblib")


def runtime_environment() -> dict:
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": {name: version(name) for name in PINNED_PACKAGES},
    }


def _unpack(obj, schema_version: str, features: tuple[str, ...]) -> tuple[Any, dict]:
    if not isinstance(obj, dict) or "pipeline" not in obj or "manifest" not in obj:
        raise PipelineError("invalid_bundle", "Bundle must contain pipeline and manifest.")
    meta = obj["manifest"]
    if not isinstance(meta, dict) or type(meta.get("contract_version")) is not int:
        raise PipelineError("invalid_bundle", "Bundle manifest is missing contract_version.")
    if meta["contract_version"] != 1:
        raise PipelineError("invalid_bundle", "Unsupported bundle contract_version.")
    if meta.get("schema_version") != schema_version:
        raise PipelineError("incompatible_schema", "Bundle schema differs from the manifest.")
    columns = meta.get("feature_columns")
    if not isinstance(columns, list) or set(columns) != set(features):
        raise PipelineError("incompatible_schema", "Bundle features differ from serving schema.")
    threshold = meta.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise PipelineError("invalid_bundle", "Bundle threshold must be a number.")
    if not 0 < threshold < 1:
        raise PipelineError("invalid_bundle", "Bundle threshold must be inside (0, 1).")
    expected = meta.get("environment") or {}
    actual = runtime_environment()
    packages = expected.get("packages") or {}
    if expected.get("python") != actual["python"] or any(
        packages.get(name) != actual["packages"][name] for name in PINNED_PACKAGES
    ):
        raise PipelineError(
            "incompatible_environment", "Bundle was trained with other library versions."
        )
    return obj["pipeline"], meta


def load_model(manifest: dict, root: Path, features: tuple[str, ...]) -> LoadedModel:
    path = resolve_artifact(manifest["artifact_uri"], root)
    if sha256_file(path) != manifest["artifact_sha256"]:
        raise PipelineError("artifact_mismatch", "Bundle checksum differs from the manifest.")

    import joblib

    try:
        loaded = joblib.load(path)
    except Exception:
        raise PipelineError("model_load_failed", "Bundle could not be deserialized.") from None
    predictor, meta = _unpack(loaded, manifest["schema_version"], features)

    classes = [int(c) for c in getattr(predictor, "classes_", [])]
    if not hasattr(predictor, "predict_proba") or set(classes) != {0, 1}:
        raise PipelineError("invalid_bundle", "Model needs predict_proba over classes 0/1.")

    return LoadedModel(
        deployment_id=manifest["deployment_id"],
        model_name=manifest["model_name"],
        model_version=manifest["model_version"],
        schema_version=manifest["schema_version"],
        artifact_sha256=manifest["artifact_sha256"],
        features=tuple(meta["feature_columns"]),
        predictor=predictor,
        threshold=float(meta["threshold"]),
        positive_index=classes.index(1),
    )


class ModelSlot:
    """Requests grab the model once, so a swap never splits a batch."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model: LoadedModel | None = None

    def get(self) -> LoadedModel | None:
        with self._lock:
            return self._model

    def swap(self, model: LoadedModel | None) -> LoadedModel | None:
        with self._lock:
            old, self._model = self._model, model
            return old
