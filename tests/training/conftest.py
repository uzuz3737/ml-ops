"""P1-contract fixtures only; this is not a production data/feature implementation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_project.pipelines.contracts import sha256_file
from mlops_project.pipelines.runner import atomic_json


@pytest.fixture
def training_context(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    from mlops_project.training import inputs, pipeline

    monkeypatch.setattr(inputs, "shared_preprocessor", lambda source, manifest: StandardScaler())
    monkeypatch.setattr(pipeline, "shared_preprocessor", lambda source, manifest: StandardScaler())
    monkeypatch.setattr(pipeline, "code_commit", lambda root: "a" * 40)
    root, directory = tmp_path, tmp_path / "artifacts/runs/test-p2"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(42)
    frames = {}
    for name, offset, size in (("train", 0, 180), ("validation", 180, 80), ("final_test", 900, 20)):
        x = rng.normal(size=(size, 3))
        y = (x[:, 0] + 0.4 * x[:, 1] + rng.normal(scale=0.3, size=size) > 0.7).astype(int)
        frame = pd.DataFrame(x, columns=["a", "b", "c"])
        frame["ID"], frame["target"] = np.arange(offset, offset + size), y
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False)
        frames[name] = {"uri": path.name, "sha256": sha256_file(path), "format": "csv"}
    manifest = {
        "contract_version": 1,
        "dataset_version": "synthetic-v1",
        "schema_version": "fixture-v1",
        "validation_id": "fixture-validation-v1",
        "feature_columns": ["a", "b", "c"],
        "target_column": "target",
        "identifier": "ID",
        "partitions": frames,
        "protected_ids": list(range(900, 920)),
    }
    atomic_json(root / "split.json", manifest)
    lock = Path(__file__).resolve().parents[2] / "src/mlops_project/training/requirements.lock"
    (root / "requirements.lock").write_bytes(lock.read_bytes())
    return {
        "contract_version": 1,
        "run_id": "test-p2",
        "project_root": str(root),
        "run_dir": str(directory),
        "config_sha256": "b" * 64,
        "config": {
            "project": {"seed": 42},
            "serving": {"model_name": "credit-default"},
            "training": {
                "tracking_uri": f"sqlite:///{root.as_posix()}/mlflow.db",
                "dependency_lock": "requirements.lock",
                "experiment_name": "p2-tests",
            },
        },
        "inputs": {
            "features": {
                "split_manifest": {"uri": "split.json", "sha256": sha256_file(root / "split.json")}
            }
        },
    }


@pytest.fixture
def trained(training_context):
    from mlops_project.training.pipeline import (
        evaluate,
        train_baseline,
        train_candidate_1,
        train_candidate_2,
    )

    context = training_context
    results = {
        function.__name__: function(context)
        for function in (train_baseline, train_candidate_1, train_candidate_2)
    }
    evaluation = evaluate({**context, "inputs": results})
    return context, results, evaluation


def rewrite_manifest(context, change):
    path = Path(context["project_root"]) / "split.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    atomic_json(path, manifest)
    context["inputs"]["features"]["split_manifest"]["sha256"] = sha256_file(path)
