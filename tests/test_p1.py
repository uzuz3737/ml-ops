"""P1 hard failures, reproducibility, fitted-state parity and drift."""
# ruff: noqa: E402
# Optional data dependencies must be checked before importing their consumers.

import json
import os
from pathlib import Path

import pytest

joblib = pytest.importorskip("joblib")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression

from mlops_project.data import pipeline
from mlops_project.data.policy import FEATURES, TARGET, validate_rows
from mlops_project.features import build_pipeline, serving_frame
from mlops_project.monitoring.feature_drift import feature_drift, shifted_fixture
from mlops_project.pipelines.contracts import PipelineError


def rows():
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
        )
        row[TARGET] = i % 2
        result.append(row)
    return result


def context(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows()))
    source = {
        "contract_version": 1,
        "dataset_version": "fixture",
        "schema_version": "credit-default-v1",
        "data": pipeline._ref(path),
        "artifacts": [pipeline._ref(path)],
        "passed": True,
    }
    return {
        "run_id": "p1-test",
        "run_dir": str(tmp_path),
        "project_root": str(Path(__file__).parents[1]),
        "inputs": {"ingest": source, "validate": source},
        "config": {
            "project": {"seed": 42},
            "splits": {"train": 0.6, "validation": 0.2, "final_test": 0.1, "monitoring": 0.1},
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop("AGE"),
        lambda r: r.update(AGE="21"),
        lambda r: r.update(default_next_month=2),
        lambda r: r.update(SEX=9),
        lambda r: r.update(AGE=None),
    ],
)
def test_bad_data_alert_and_halt(tmp_path, change):
    ctx = context(tmp_path)
    data = rows()
    change(data[0])
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    ctx["inputs"]["ingest"]["data"] = pipeline._ref(path)
    with pytest.raises(PipelineError, match="Hard data validation failed"):
        pipeline.validate(ctx)
    assert json.loads((tmp_path / "data-validation-alert.json").read_text())["passed"] is False


def test_tfdv_evidence_and_missing_dependency(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(pipeline, "_tfdv", lambda *args: ([], []))
    assert pipeline.validate(ctx)["passed"]

    def unavailable(*args):
        raise ImportError

    monkeypatch.setattr(pipeline, "_tfdv", unavailable)
    with pytest.raises(PipelineError):
        pipeline.validate(ctx)


def test_split_disjoint_reproducible_and_features(tmp_path):
    ctx = context(tmp_path)
    first = pipeline.split(ctx)
    second = pipeline.split(ctx)
    assert first["split_manifest"] == second["split_manifest"]
    sets = [
        set(json.loads((tmp_path / v["ids"]["uri"]).read_text()))
        for v in first["partitions"].values()
    ]
    assert [len(s) for s in sets] == [240, 80, 40, 40]
    assert len(set.union(*sets)) == sum(map(len, sets)) == 400
    ctx["inputs"] = {"split": first}
    assert pipeline.features(ctx)["fit_partition"] == "train"


def test_saved_pipeline_parity_and_policy(tmp_path):
    frame = pd.DataFrame(rows())
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(
        frame[list(FEATURES)], frame[TARGET]
    )
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    raw = frame[list(FEATURES)].iloc[:5].to_dict("records")
    assert loaded.predict_proba(serving_frame(raw)) == pytest.approx(
        model.predict_proba(frame[list(FEATURES)].iloc[:5])
    )
    assert not validate_rows(rows())
    assert validate_rows([rows()[0], rows()[0]])
    with pytest.raises(ValueError):
        serving_frame([rows()[0]])


def test_drift_sample_rule_and_shift():
    frame = pd.DataFrame(rows())
    assert not feature_drift(frame, frame)["alert"]
    assert feature_drift(frame, shifted_fixture(frame))["alert"]
    assert feature_drift(frame, frame.iloc[:10])["status"] == "insufficient_samples"


def test_real_tfdv_schema_when_available(tmp_path):
    pytest.importorskip("tensorflow_data_validation")
    failures, refs = pipeline._tfdv(
        pd.DataFrame(rows()), Path(__file__).parents[1] / "schemas/credit_default.pbtxt", tmp_path
    )
    assert not failures
    assert refs


@pytest.mark.skipif(
    os.environ.get("P1_UCI_INTEGRATION") != "1", reason="opt-in UCI network integration"
)
def test_real_uci_stages(tmp_path):
    import tensorflow_data_validation  # noqa: F401

    ctx = context(tmp_path)
    ctx["config"]["dataset"] = {}
    acquired = pipeline.ingest(ctx)
    assert acquired["row_count"] == 30000
    ctx["inputs"] = {"ingest": acquired}
    checked = pipeline.validate(ctx)
    ctx["inputs"] = {"validate": checked}
    partitioned = pipeline.split(ctx)
    assert [p["row_count"] for p in partitioned["partitions"].values()] == [18000, 6000, 3000, 3000]
    ctx["inputs"] = {"split": partitioned}
    assert pipeline.features(ctx)["fit_partition"] == "train"


def snapshot_context(tmp_path):
    ctx = context(tmp_path)
    ctx["project_root"] = str(tmp_path)
    data = rows()
    refs = {}
    for name, records in (
        ("train", data[:240]),
        ("validation", data[240:320]),
        ("protected", [r["ID"] for r in data[320:]]),
    ):
        path = tmp_path / f"source-{name}.json"
        path.write_text(json.dumps(records))
        refs[name] = pipeline._ref(path)
    manifest = {
        "contract_version": 1,
        "dataset_version": "regime-v2",
        "schema_version": "credit-default-v1",
        "source": "synthetic test fixture",
        "license": "test-only",
        "attribution": "P1 test generator",
        "partitions": {k: refs[k] for k in ("train", "validation")},
        "protected_ids": refs["protected"],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(manifest))
    ctx["config"]["dataset"] = {"retraining_snapshots": {"regime-v2": pipeline._ref(path)}}
    ctx["retraining"] = {
        "candidate_dataset_version": "regime-v2",
        "candidate_dataset_approved": True,
        "labels_validated": True,
        "separate_training_validation": True,
        "final_test_excluded": True,
    }
    return ctx


def test_approved_snapshot_preserves_partitions(tmp_path):
    ctx = snapshot_context(tmp_path)
    result = pipeline.ingest(ctx)
    assert result["dataset_version"] == "regime-v2"
    assert result["row_count"] == 320
    ctx["inputs"] = {"validate": {**result, "passed": True}}
    split_result = pipeline.split(ctx)
    assert set(split_result["partitions"]) == {"train", "validation"}
    assert [p["row_count"] for p in split_result["partitions"].values()] == [240, 80]
    assert split_result["strategy"] == "approved-independent-snapshot"


@pytest.mark.parametrize(
    "fault", ["unapproved", "unknown_version", "checksum", "leakage", "invalid_rows"]
)
def test_snapshot_rejects_unapproved_changed_or_leaking_data(tmp_path, fault):
    ctx = snapshot_context(tmp_path)
    if fault == "unapproved":
        ctx["retraining"]["labels_validated"] = False
    elif fault == "unknown_version":
        ctx["retraining"]["candidate_dataset_version"] = "missing"
    else:
        path = tmp_path / "source-train.json"
        data = json.loads(path.read_text())
        if fault == "leakage":
            data[0]["ID"] = 400
        elif fault == "invalid_rows":
            data[0]["AGE"] = "21"
        else:
            data[0]["AGE"] = 22
        path.write_text(json.dumps(data))
        if fault != "checksum":
            manifest_path = tmp_path / "snapshot.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["partitions"]["train"] = pipeline._ref(path)
            manifest_path.write_text(json.dumps(manifest))
            ctx["config"]["dataset"]["retraining_snapshots"]["regime-v2"] = pipeline._ref(
                manifest_path
            )
    with pytest.raises(PipelineError):
        pipeline.ingest(ctx)


@pytest.mark.parametrize(
    "record", [None, [], {"ID": []}, {**rows()[0], "ID": []}, {**rows()[0], "AGE": 10**1000}]
)
def test_malformed_records_fail_without_validator_crash(record):
    assert validate_rows([record])
