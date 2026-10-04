"""Load trusted checksum-bound P2 artifacts; never fit at prediction time."""

import importlib.metadata
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlops_project.pipelines.contracts import sha256_file

PACKAGES = ("scikit-learn", "numpy", "pandas", "joblib", "mlflow")


def environment():
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def load_bundle(path, checksum, schema_version=None):
    """Only call for a trusted pipeline artifact, after validating approval/provenance."""
    path = Path(path)
    if sha256_file(path) != checksum:
        raise ValueError("Model artifact checksum mismatch")
    bundle = joblib.load(path)
    metadata = bundle["manifest"]
    if type(metadata.get("contract_version")) is not int or metadata["contract_version"] != 1:
        raise ValueError("Unsupported bundle contract")
    if schema_version is not None and metadata["schema_version"] != schema_version:
        raise ValueError("Incompatible model schema")
    actual = environment()
    expected = metadata["environment"]
    if expected["python"] != actual["python"] or any(
        expected["packages"][p] != actual["packages"][p]
        for p in ("scikit-learn", "numpy", "pandas", "joblib")
    ):
        raise ValueError("Incompatible serialization environment")
    return bundle


def probabilities(pipeline, frame):
    classes = list(pipeline.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Expected binary classes 0 and 1")
    p = np.asarray(pipeline.predict_proba(frame))[:, classes.index(1)]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid estimator probabilities")
    return p


def predict_bundle(bundle, instances):
    if not isinstance(instances, list) or not instances:
        raise ValueError("Nonempty raw instances required")
    features = bundle["manifest"]["feature_columns"]
    for row in instances:
        if not isinstance(row, dict) or set(row) != set(features):
            raise ValueError("Feature keys must match the immutable bundle signature")
        if any(type(row[f]) not in (int, float) or not np.isfinite(row[f]) for f in features):
            raise ValueError("Features must be finite numeric values")
    frame = pd.DataFrame(instances, columns=features)
    p = probabilities(bundle["pipeline"], frame)
    return [
        {"label": int(v >= bundle["manifest"]["threshold"]), "default_probability": float(v)}
        for v in p
    ]
