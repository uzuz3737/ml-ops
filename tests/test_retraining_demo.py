"""Demo preparation keeps protected data out of the candidate and refuses reuse."""

import json
import runpy
import shutil
from pathlib import Path

import pytest
import yaml

from mlops_project.data.policy import FEATURES, TARGET
from mlops_project.data.snapshots import load_snapshot
from mlops_project.pipelines.contracts import PipelineError

ROOT = Path(__file__).resolve().parents[1]
prepare = runpy.run_path(str(ROOT / "scripts/prepare-retraining-demo.py"))["prepare"]


@pytest.fixture
def demo_root(tmp_path):
    shutil.copytree(
        ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("*.local.yaml")
    )
    source = tmp_path / "artifacts/runs/initial"
    source.mkdir(parents=True)
    rows = []
    for i in range(20):
        row = dict.fromkeys(FEATURES, 0)
        row.update(ID=i + 1, LIMIT_BAL=10000, AGE=30, SEX=1, EDUCATION=1, MARRIAGE=1)
        row[TARGET] = i % 2
        rows.append(row)
    for name, records in (("train", rows[:12]), ("validation", rows[12:])):
        (source / f"{name}.json").write_text(json.dumps(records))
    (source / "training-split-manifest.json").write_text(
        json.dumps({"protected_ids": [21, 22], "dataset_version": "original"})
    )
    active = tmp_path / "artifacts/deployments/active-model.json"
    active.parent.mkdir()
    active.write_text(
        json.dumps(
            {
                "run_id": "initial",
                "model_version": "1",
                "artifact_uri": "bundle.joblib",
                "artifact_sha256": "a" * 64,
            }
        )
    )
    quality = tmp_path / "artifacts/monitoring/exported/quality.json"
    quality.parent.mkdir(parents=True)
    quality.write_text(
        json.dumps(
            {
                "model_version": "1",
                "status": "alert",
                "threshold_crossed": True,
                "threshold": 0.12,
                "observed": 0.4,
                "sample_count": 500,
                "labeled_sample_count": 500,
                "positive_label_count": 250,
                "negative_label_count": 250,
                "label_coverage": 1,
            }
        )
    )
    return tmp_path


def test_demo_snapshot_is_loadable_protected_and_immutable(demo_root):
    result = prepare(demo_root, "initial", "demo-1")
    config = yaml.safe_load((demo_root / result["config"]).read_text())
    alert = json.loads((demo_root / result["alert"]).read_text())
    frame, partitions, _ = load_snapshot(
        {"project_root": str(demo_root), "config": config, "retraining": alert}
    )
    assert set(frame.ID) == set(range(1, 21))
    assert frame.loc[frame.ID == 1, TARGET].item() == 1
    assert set(partitions["train"]).isdisjoint(partitions["validation"])
    assert result["submitted"] is False and alert["demo_only"] is True
    with pytest.raises(ValueError, match="immutable"):
        prepare(demo_root, "initial", "demo-1")


def test_demo_rejects_protected_rows(demo_root):
    path = demo_root / "artifacts/runs/initial/train.json"
    rows = json.loads(path.read_text())
    rows[0]["ID"] = 21
    path.write_text(json.dumps(rows))
    with pytest.raises(PipelineError, match="protected holdout"):
        prepare(demo_root, "initial", "leaking")
    assert not (demo_root / "configs/retraining-leaking.local.yaml").exists()
