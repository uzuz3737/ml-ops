from pathlib import Path

import pytest
from mlflow import MlflowClient

from mlops_project.pipelines.contracts import validate_result
from mlops_project.registry.pipeline import register, retrieve_bundle
from mlops_project.training.bundle import load_bundle, predict_bundle


def test_real_registration_retry_and_immutable_retrieval(trained):
    context, _, evaluation = trained
    context = {**context, "inputs": {"evaluate": evaluation}}
    registered = register(context)
    assert registered["model_version"] == "1"
    validate_result(registered, "register", context["run_id"], Path(context["run_dir"]))
    assert register(context)["model_version"] == "1"
    client = MlflowClient(tracking_uri=context["config"]["training"]["tracking_uri"])
    assert len(client.search_model_versions("name = 'credit-default'")) == 1
    assert client.get_registered_model("credit-default").aliases == {}
    retrieved = retrieve_bundle(context, "credit-default", "1", "artifacts/retrieved.joblib")
    assert retrieved["artifact_sha256"] == evaluation["artifact_sha256"]
    bundle = load_bundle(
        Path(context["project_root"]) / retrieved["artifact_uri"], retrieved["artifact_sha256"]
    )
    assert len(predict_bundle(bundle, [{"a": 1, "b": 0, "c": 0}])) == 1
    with pytest.raises(ValueError, match="exact version"):
        retrieve_bundle(context, "credit-default", "champion", "artifacts/bad.joblib")


def test_forged_tracking_identity_blocked(trained):
    context, _, evaluation = trained
    client = MlflowClient(tracking_uri=context["config"]["training"]["tracking_uri"])
    client.set_tag(evaluation["mlflow_run_id"], "dataset_version", "changed")
    with pytest.raises(ValueError, match="Tracking/evaluation"):
        register({**context, "inputs": {"evaluate": evaluation}})
