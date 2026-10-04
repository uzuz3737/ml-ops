"""P1 split/features output feeds P2 training without any glue code."""
# ruff: noqa: E402

import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

from mlops_project.data import pipeline as data_pipeline
from mlops_project.data.policy import FEATURES, TARGET
from mlops_project.pipelines.contracts import sha256_file
from mlops_project.training import inputs
from mlops_project.training import pipeline as training_pipeline
from mlops_project.training.bundle import load_bundle, predict_bundle


def _rows():
    result = []
    for i in range(400):
        row = dict.fromkeys(FEATURES, 0)
        row.update(
            ID=i + 1,
            LIMIT_BAL=(i + 1) * 1000,
            AGE=21 + i % 50,
            SEX=1 + i % 2,
            EDUCATION=i % 7,
            MARRIAGE=i % 4,
            PAY_0=(i % 5) - 2,
        )
        row[TARGET] = int(i % 5 >= 3)
        result.append(row)
    return result


@pytest.fixture
def p1_features(tmp_path):
    run_dir = tmp_path / "artifacts/runs/contract-test"
    run_dir.mkdir(parents=True)
    data = run_dir / "data.json"
    data.write_text(json.dumps(_rows()))
    source = {
        "contract_version": 1,
        "dataset_version": "fixture",
        "schema_version": "credit-default-v1",
        "data": data_pipeline._ref(data),
        "artifacts": [data_pipeline._ref(data)],
        "passed": True,
    }
    context = {
        "contract_version": 1,
        "run_id": "contract-test",
        "run_dir": str(run_dir),
        "project_root": str(tmp_path),
        "config_sha256": "b" * 64,
        "inputs": {"validate": source},
        "config": {
            "project": {"seed": 42},
            "splits": {"train": 0.6, "validation": 0.2, "final_test": 0.1, "monitoring": 0.1},
        },
    }
    split = data_pipeline.split(context)
    features = data_pipeline.features({**context, "inputs": {"split": split}})
    return context, split, features


def test_p2_reads_p1_split_manifest(p1_features):
    context, split, features = p1_features
    source, manifest, partitions = inputs.load_inputs({**context, "inputs": {"features": features}})

    assert manifest["feature_columns"] == list(FEATURES)
    assert len(partitions["train"]) == 240
    assert len(partitions["validation"]) == 80
    protected = {str(i) for i in manifest["protected_ids"]}
    assert len(protected) == 80  # final test + monitoring
    assert not protected & set(partitions["train"]["ID"].astype(str))

    transformer = inputs.shared_preprocessor(source, manifest)
    assert not hasattr(transformer, "transformers_")  # handed over unfitted


def test_p2_trains_a_loadable_bundle_on_p1_output(p1_features, tmp_path, monkeypatch):
    context, _, features = p1_features
    monkeypatch.setattr(training_pipeline, "code_commit", lambda root: "a" * 40)
    lock = tmp_path / "requirements.lock"
    lock.write_text("fixture lock\n")
    context = {
        **context,
        "config": {
            **context["config"],
            "serving": {"model_name": "credit-default"},
            "training": {
                "tracking_uri": f"sqlite:///{tmp_path.as_posix()}/mlflow.db",
                "dependency_lock": "requirements.lock",
                "experiment_name": "p1-p2-contract",
            },
        },
        "inputs": {"features": features},
    }
    result = training_pipeline.train_baseline(context)

    path = tmp_path / result["artifact_uri"]
    assert sha256_file(path) == result["artifact_sha256"]
    bundle = load_bundle(path, result["artifact_sha256"], "credit-default-v1")
    instance = {name: _rows()[0][name] for name in FEATURES}
    prediction = predict_bundle(bundle, [instance])[0]
    assert prediction["label"] in (0, 1)
    assert 0 <= prediction["default_probability"] <= 1
