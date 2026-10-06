"""Concurrent initial training reuses a newly created MLflow experiment."""

from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS

from mlops_project.training.pipeline import _experiment_id


class RacingClient:
    def __init__(self, code=RESOURCE_ALREADY_EXISTS):
        self.lookups = 0
        self.code = code

    def get_experiment_by_name(self, name):
        self.lookups += 1
        return None if self.lookups == 1 else SimpleNamespace(experiment_id="7")

    def create_experiment(self, name, artifact_location):
        raise MlflowException("created concurrently", error_code=self.code)


def test_uses_concurrently_created_experiment():
    client = RacingClient()
    assert _experiment_id(client, "credit-default", None) == "7"
    assert client.lookups == 2


def test_other_mlflow_errors_remain_failures():
    client = RacingClient(INVALID_PARAMETER_VALUE)
    with pytest.raises(MlflowException):
        _experiment_id(client, "credit-default", None)
    assert client.lookups == 1
