"""Load a trusted model bundle and hold the one currently serving.

Bundle format (agree with P2 before the first real model): a joblib file
containing {"model": fitted sklearn pipeline, "threshold": float}. The
pipeline already includes P1's fitted transforms; nothing is fitted here.
"""

from __future__ import annotations

import math
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


def _unpack(obj) -> tuple[Any, float]:
    if not isinstance(obj, dict) or "model" not in obj or "threshold" not in obj:
        raise PipelineError("invalid_bundle", "Bundle must contain model and threshold.")
    threshold = obj["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise PipelineError("invalid_bundle", "Bundle threshold must be a number.")
    if not 0 < threshold < 1:
        raise PipelineError("invalid_bundle", "Bundle threshold must be inside (0, 1).")
    return obj["model"], float(threshold)


def load_model(manifest: dict, root: Path, features: tuple[str, ...]) -> LoadedModel:
    path = resolve_artifact(manifest["artifact_uri"], root)
    if sha256_file(path) != manifest["artifact_sha256"]:
        raise PipelineError("artifact_mismatch", "Bundle checksum differs from the manifest.")

    import joblib

    try:
        predictor, threshold = _unpack(joblib.load(path))
    except PipelineError:
        raise
    except Exception:
        raise PipelineError("model_load_failed", "Bundle could not be deserialized.") from None

    classes = [int(c) for c in getattr(predictor, "classes_", [])]
    if not hasattr(predictor, "predict_proba") or 1 not in classes:
        raise PipelineError("invalid_bundle", "Model needs predict_proba and class 1.")

    return LoadedModel(
        deployment_id=manifest["deployment_id"],
        model_name=manifest["model_name"],
        model_version=manifest["model_version"],
        schema_version=manifest["schema_version"],
        artifact_sha256=manifest["artifact_sha256"],
        features=features,
        predictor=predictor,
        threshold=threshold,
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
