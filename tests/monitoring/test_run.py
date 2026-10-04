"""Monitoring pass over API event files: healthy, feature drift and quality drop."""
# ruff: noqa: E402

import json
import random
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("prometheus_client")

from mlops_project.data.policy import FEATURES
from mlops_project.monitoring.run import run_once
from mlops_project.pipelines.runner import atomic_json

REPO = Path(__file__).resolve().parents[2]


def _row(rng):
    row = {name: rng.randint(0, 5000) for name in FEATURES}
    row.update(
        LIMIT_BAL=rng.randint(1, 50) * 10_000,
        SEX=rng.choice([1, 2]),
        EDUCATION=rng.randint(0, 6),
        MARRIAGE=rng.randint(0, 3),
        AGE=rng.randint(21, 70),
    )
    for name in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        row[name] = rng.randint(-2, 3)
    return row


@pytest.fixture
def project(tmp_path):
    for name in ("project.yaml", "monitoring.yaml"):
        (tmp_path / "configs").mkdir(exist_ok=True)
        shutil.copyfile(REPO / "configs" / name, tmp_path / "configs" / name)
    rng = random.Random(7)
    reference = [_row(rng) for _ in range(3000)]
    run_dir = tmp_path / "artifacts/runs/train-run"
    run_dir.mkdir(parents=True)
    (run_dir / "train.json").write_text(json.dumps(reference))
    bundle = run_dir / "train_baseline/bundle.joblib"
    bundle.parent.mkdir()
    bundle.write_bytes(b"not loaded by the monitor")
    atomic_json(
        bundle.parent / "manifest.json",
        {"threshold": 0.5, "metrics": {"average_precision": 0.95, "roc_auc": 0.95}},
    )
    atomic_json(
        tmp_path / "artifacts/deployments/active-model.json",
        {
            "model_version": "3",
            "run_id": "train-run",
            "artifact_uri": bundle.relative_to(tmp_path).as_posix(),
            "artifact_sha256": "c" * 64,
        },
    )
    return tmp_path, reference


def _write_events(root, rows, *, flip_labels=False):
    events = root / "artifacts/serving/events"
    events.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 10, 4, tzinfo=UTC)
    predictions, labels = [], []
    for i, features in enumerate(rows):
        truth = i % 4 == 0
        when = (start + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
        predictions.append(
            {
                "request_id": f"r{i}",
                "instance_index": 0,
                "prediction_time": when,
                "model_version": "3",
                "features": features,
                "label": int(truth),
                "default_probability": 0.8 if truth else 0.2,
            }
        )
        observed = (start + timedelta(days=1)).isoformat().replace("+00:00", "Z")
        label = int(truth) ^ int(flip_labels)
        labels.append(
            {
                "request_id": f"r{i}",
                "labels": [{"instance_index": 0, "label": label}],
                "observed_at": observed,
            }
        )
    for name, records in (("predictions.jsonl", predictions), ("labels.jsonl", labels)):
        (events / name).write_text("".join(json.dumps(r) + "\n" for r in records))


def _exported(root, name):
    return json.loads((root / "artifacts/monitoring/exported" / name).read_text())


def test_healthy_window_raises_nothing(project):
    root, reference = project
    _write_events(root, reference[:500])
    summary = run_once("configs/project.yaml", root, as_of="2026-10-06T00:00:00Z")

    assert summary["alerts"] == []
    assert _exported(root, "feature_drift.json")["status"] == "ok"
    assert _exported(root, "quality.json")["status"] == "ok"


def test_shifted_features_raise_a_drift_alert(project):
    root, reference = project
    shifted = [{**row, "LIMIT_BAL": row["LIMIT_BAL"] * 10} for row in reference[:500]]
    _write_events(root, shifted)
    summary = run_once("configs/project.yaml", root, as_of="2026-10-06T00:00:00Z")

    assert _exported(root, "feature_drift.json")["status"] == "alert"
    [alert_file] = [a for a in summary["alerts"] if a.startswith("data_drift")]
    alert = json.loads((root / "artifacts/monitoring/alerts" / alert_file).read_text())
    assert alert["feature"] == "LIMIT_BAL"
    assert alert["observed"] > alert["threshold"]


def test_flipped_labels_raise_a_quality_alert(project):
    root, reference = project
    _write_events(root, reference[:500], flip_labels=True)
    summary = run_once("configs/project.yaml", root, as_of="2026-10-06T00:00:00Z")

    assert _exported(root, "feature_drift.json")["status"] == "ok"
    assert _exported(root, "quality.json")["status"] == "alert"
    assert any(a.startswith("quality") for a in summary["alerts"])


def test_too_few_events_is_insufficient_data(project):
    root, reference = project
    _write_events(root, reference[:50])
    run_once("configs/project.yaml", root, as_of="2026-10-06T00:00:00Z")

    assert _exported(root, "feature_drift.json")["status"] == "insufficient_data"
    assert _exported(root, "quality.json")["status"] == "insufficient_data"
