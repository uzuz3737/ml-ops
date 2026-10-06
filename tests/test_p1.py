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
from mlops_project.monitoring.feature_drift import (
    calibrate_threshold,
    feature_drift,
    shifted_fixture,
)
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
    ctx["project_root"] = str(tmp_path)  # features writes project-relative URIs
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


def test_train_reference_calibration_is_reproducible_and_detects_shift():
    frame = pd.DataFrame(rows())
    report = calibrate_threshold(frame, window_rows=200, trials=20)
    assert report == calibrate_threshold(frame, window_rows=200, trials=20)
    assert not feature_drift(frame, frame, threshold=report["threshold"])["alert"]
    assert feature_drift(frame, shifted_fixture(frame), threshold=report["threshold"])["alert"]
    with pytest.raises(ValueError):
        calibrate_threshold(frame.iloc[:10])


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
    ctx["project_root"] = str(tmp_path)  # features writes project-relative URIs
    assert pipeline.features(ctx)["fit_partition"] == "train"
    train = pipeline._read(ctx, partitioned["partitions"]["train"]["data"])
    monitoring = pipeline._read(ctx, partitioned["partitions"]["monitoring"]["data"])
    calibration = calibrate_threshold(train, window_rows=len(monitoring))
    healthy = feature_drift(
        train, monitoring, minimum_rows=len(monitoring), threshold=calibration["threshold"]
    )
    shifted = feature_drift(
        train,
        shifted_fixture(monitoring),
        minimum_rows=len(monitoring),
        threshold=calibration["threshold"],
    )
    pipeline._write(ctx, "drift-calibration.json", calibration)
    pipeline._write(ctx, "healthy-drift.json", healthy)
    pipeline._write(ctx, "synthetic-shift-drift.json", shifted)
    assert not healthy["alert"]
    assert shifted["alert"]


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
    ctx["config"]["dataset"] = {
        "retraining_snapshots": {"regime-v2": pipeline._ref(path)},
        "protected_final_test_ids": refs["protected"],
    }
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
    assert (tmp_path / "data-validation-alert.json").is_file()


def test_snapshot_cannot_replace_trusted_holdout(tmp_path):
    ctx = snapshot_context(tmp_path)
    ctx["config"]["dataset"]["protected_final_test_ids"] = {"uri": "other.json", "sha256": "0" * 64}
    with pytest.raises(PipelineError, match="configured original final-test"):
        pipeline.ingest(ctx)


def test_real_snapshot_validation_when_tfdv_available(tmp_path):
    pytest.importorskip("tensorflow_data_validation")
    ctx = snapshot_context(tmp_path)
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    schema_dir.joinpath("credit_default.pbtxt").write_bytes(
        (Path(__file__).parents[1] / "schemas/credit_default.pbtxt").read_bytes()
    )
    ctx["inputs"] = {"ingest": pipeline.ingest(ctx)}
    checked = pipeline.validate(ctx)
    assert checked["passed"]
    assert "approved_partitions" in checked
    ctx["inputs"] = {"validate": checked}
    partitions = pipeline.split(ctx)
    ctx["inputs"] = {"split": partitions}
    ctx["project_root"] = str(tmp_path)  # features writes project-relative URIs
    assert pipeline.features(ctx)["fit_partition"] == "train"


@pytest.mark.parametrize(
    "record", [None, [], {"ID": []}, {**rows()[0], "ID": []}, {**rows()[0], "AGE": 10**1000}]
)
def test_malformed_records_fail_without_validator_crash(record):
    assert validate_rows([record])


def file_context(tmp_path, name, frame=None, text=None):
    """A pipeline-shaped run directory fed by an operator data_file."""
    (tmp_path / "data").mkdir(exist_ok=True)
    path = tmp_path / "data" / name
    if text is not None:
        path.write_text(text, encoding="utf-8")
    else:
        frame.to_csv(path, index=False)
    run = tmp_path / "artifacts" / "runs" / "file-run"
    run.mkdir(parents=True)
    schema = tmp_path / "schemas"
    schema.mkdir(exist_ok=True)
    (schema / "credit_default.pbtxt").write_text("")
    return {
        "run_id": "file-run",
        "run_dir": str(run),
        "project_root": str(tmp_path),
        "data_file": {"uri": f"data/{name}", "sha256": pipeline.sha256_file(path)},
        "config": {"project": {"seed": 42}},
    }


def uci_frame():
    # Kaggle/UCI spellings: no ID column, PAY_1 for PAY_0, dotted target header.
    frame = pd.DataFrame(rows()).drop(columns="ID").rename(columns={"PAY_0": "PAY_1"})
    return frame.rename(columns={TARGET: "default.payment.next.month"})


def test_data_file_with_uci_headers_is_ingested_and_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_tfdv", lambda *args: ([], []))
    ctx = file_context(tmp_path, "good.csv", uci_frame())
    ingested = pipeline.ingest(ctx)
    assert ingested["generated_ids"] is True and ingested["missing_columns"] == []
    assert ingested["source"] == "data_file:data/good.csv"
    ctx["inputs"] = {"ingest": ingested}
    assert pipeline.validate(ctx)["passed"] is True
    status = json.loads(
        (tmp_path / "artifacts/monitoring/exported/data_validation.json").read_text()
    )
    assert status["status"] == "ok" and status["failure_count"] == 0


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda f: f.assign(SEX=[3] + [1] * (len(f) - 1)), ("SEX", "type_or_domain")),
        (lambda f: f.assign(AGE=["-5"] + ["30"] * (len(f) - 1)), ("AGE", "type_or_domain")),
        (lambda f: f.astype(object).assign(LIMIT_BAL=["n/a"] * len(f)), ("LIMIT_BAL", None)),
        (lambda f: f.drop(columns="PAY_AMT6"), ("*", "columns")),
    ],
)
def test_bad_data_file_stops_at_validate_with_alert(tmp_path, monkeypatch, change, expected):
    monkeypatch.setattr(pipeline, "_tfdv", lambda *args: ([], []))
    ctx = file_context(tmp_path, "bad.csv", change(uci_frame()))
    ctx["inputs"] = {"ingest": pipeline.ingest(ctx)}  # ingest never judges the content
    with pytest.raises(PipelineError, match="Hard data validation failed"):
        pipeline.validate(ctx)
    alert = json.loads((Path(ctx["run_dir"]) / "data-validation-alert.json").read_text())
    assert alert["severity"] == "critical" and alert["passed"] is False
    field, rule = expected
    assert any(
        s["field"] == field and (rule is None or s["rule"] == rule)
        for s in alert["failure_summary"]
    )
    if rule == "columns":
        assert alert["missing_columns"] == ["PAY_AMT6"]
    shared = tmp_path / "artifacts/monitoring"
    assert json.loads((shared / "exported/data_validation.json").read_text())["status"] == "alert"
    assert (shared / "alerts/data_validation-file-run.json").is_file()


def test_unreadable_data_file_fails_ingest_with_alert(tmp_path):
    ctx = file_context(tmp_path, "broken.json", text="{not json")
    with pytest.raises(PipelineError, match="could not be parsed") as caught:
        pipeline.ingest(ctx)
    assert caught.value.code == "data_validation_failed"
    alert = json.loads((Path(ctx["run_dir"]) / "data-validation-alert.json").read_text())
    assert alert["failures"][0]["rule"] == "unreadable_file"


def completed_run(tmp_path, protected):
    """artifacts/runs/src with the final-test ID list a succeeded split stage recorded."""
    run = tmp_path / "artifacts/runs/src"
    run.mkdir(parents=True)
    ids = run / "final_test-ids.json"
    ids.write_text(json.dumps(protected))
    digest = pipeline.sha256_file(ids)
    (run / "split.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "result": {
                    "partitions": {
                        "final_test": {"ids": {"uri": "final_test-ids.json", "sha256": digest}}
                    }
                },
            }
        )
    )
    return {"uri": "artifacts/runs/src/final_test-ids.json", "sha256": digest}


def written_snapshot(tmp_path):
    from mlops_project.data.snapshots import write_snapshot

    tmp_path.mkdir(parents=True, exist_ok=True)
    data = rows()
    protected = completed_run(tmp_path, [r["ID"] for r in data[320:]])
    ctx = snapshot_context(tmp_path)
    del ctx["config"]["dataset"]  # nothing registered: found by content address only
    version = write_snapshot(
        ctx,
        data[:240],
        data[240:320],
        protected_ids=protected,
        provenance={"source": "test", "license": "test-only", "attribution": "fixture"},
    )
    ctx["retraining"]["candidate_dataset_version"] = version
    return ctx, version


def test_content_addressed_snapshot_needs_no_config_edit(tmp_path):
    ctx, version = written_snapshot(tmp_path)
    assert version.startswith("snapshot-")
    result = pipeline.ingest(ctx)
    assert result["dataset_version"] == version and result["row_count"] == 320
    again, same = written_snapshot(tmp_path / "again")
    assert same == version  # identical rows, identical version


@pytest.mark.parametrize("fault", ["tampered_rows", "forged_holdout", "unknown_version"])
def test_content_addressed_snapshot_rejects_tampering(tmp_path, fault):
    ctx, version = written_snapshot(tmp_path)
    directory = tmp_path / "artifacts/retraining/snapshots" / version
    if fault == "tampered_rows":
        data = json.loads((directory / "train.json").read_text())
        data[0]["AGE"] = 99
        (directory / "train.json").write_text(json.dumps(data))
    elif fault == "forged_holdout":
        other = tmp_path / "artifacts/runs/src/other-ids.json"
        other.write_text("[399]")
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["protected_ids"] = {
            "uri": "artifacts/runs/src/other-ids.json",
            "sha256": pipeline.sha256_file(other),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
    else:
        ctx["retraining"]["candidate_dataset_version"] = "snapshot-" + "0" * 32
    with pytest.raises(PipelineError) as caught:
        pipeline.ingest(ctx)
    assert caught.value.code in {"snapshot_checksum", "snapshot_leakage", "invalid_snapshot"}


def test_changed_data_file_is_refused(tmp_path):
    ctx = file_context(tmp_path, "good.csv", uci_frame())
    (tmp_path / "data/good.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(PipelineError, match="changed after the run started"):
        pipeline.ingest(ctx)
