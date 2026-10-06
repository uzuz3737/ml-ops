import json

import pytest

pytest.importorskip("mlflow")

from mlops_project.training.pipeline import _deployed_champion  # noqa: E402


def _context(root, model_version="2"):
    return {
        "project_root": str(root),
        "config": {"serving": {"deployments_dir": "artifacts/deployments"}},
        "retraining": {"model_version": model_version},
    }


def test_retraining_compares_against_the_alerted_deployed_bundle(tmp_path):
    deployments = tmp_path / "artifacts/deployments"
    deployments.mkdir(parents=True)
    (deployments / "active-model.json").write_text(
        json.dumps(
            {"model_version": "2", "artifact_uri": "artifacts/b/bundle.joblib",
             "artifact_sha256": "a" * 64}
        )
    )  # fmt: skip
    assert _deployed_champion(_context(tmp_path)) == {
        "uri": "artifacts/b/bundle.joblib",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="no longer the deployed champion"):
        _deployed_champion(_context(tmp_path, model_version="1"))


def test_no_deployment_means_no_champion(tmp_path):
    assert _deployed_champion(_context(tmp_path)) is None
