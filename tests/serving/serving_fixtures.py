"""Shared setup for serving tests: a tiny fake model and an on-disk project."""

import json
from pathlib import Path

import joblib
import yaml

from mlops_project.pipelines.contracts import sha256_file
from mlops_project.pipelines.runner import atomic_json
from mlops_project.serving.settings import Settings

REPO = Path(__file__).resolve().parents[2]
FEATURES = tuple(
    yaml.safe_load((REPO / "configs/data_schema.yaml").read_text(encoding="utf-8"))["features"]
)
EXAMPLE = json.loads((REPO / "examples/predict.json").read_text(encoding="utf-8"))


class FakeModel:
    """Probability grows with LIMIT_BAL; enough to tell versions apart."""

    classes_ = [0, 1]

    def __init__(self, scale=1_000_000):
        self.scale = scale

    def predict_proba(self, rows):
        rows = rows.values.tolist() if hasattr(rows, "values") else rows
        return [[1 - min(1.0, r[0] / self.scale), min(1.0, r[0] / self.scale)] for r in rows]


class BrokenModel(FakeModel):
    def predict_proba(self, rows):
        raise ValueError("boom")


def make_settings(root: Path, **overrides) -> Settings:
    values = dict(
        project_root=root,
        model_name="credit-default",
        schema_version="credit-default-v1",
        features=FEATURES,
        deployments_dir=root / "artifacts/deployments",
        events_dir=root / "artifacts/serving/events",
        monitoring_dir=root / "artifacts/monitoring/exported",
        smoke_fixture=EXAMPLE,
        poll_seconds=60,
    )
    values.update(overrides)
    return Settings(**values)


def publish_bundle(root: Path, version: str, model=None, threshold=0.5) -> dict:
    """Write a bundle plus a passing gate report and return a deploy manifest."""
    bundle = root / f"artifacts/models/v{version}/model.joblib"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model or FakeModel(), "threshold": threshold}, bundle)
    artifact_sha = sha256_file(bundle)
    report = root / f"artifacts/runs/run-{version}/gate-report.json"
    atomic_json(
        report,
        {"passed": True, "candidate_model_version": version, "artifact_sha256": artifact_sha},
    )
    return {
        "contract_version": 1,
        "action": "deploy",
        "deployment_id": f"deploy-v{version}",
        "model_name": "credit-default",
        "model_version": version,
        "schema_version": "credit-default-v1",
        "artifact_uri": bundle.relative_to(root).as_posix(),
        "artifact_sha256": artifact_sha,
        "gate_report_uri": report.relative_to(root).as_posix(),
        "gate_report_sha256": sha256_file(report),
    }


def request_deploy(settings: Settings, manifest: dict) -> None:
    atomic_json(settings.deployments_dir / "desired-model.json", manifest)


def read_ack(settings: Settings, deployment_id: str) -> dict:
    path = settings.deployments_dir / "acknowledgements" / f"{deployment_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
