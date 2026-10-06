"""Lightweight DAG-definition contract tests; real DagBag testing runs in CI.

Airflow is deliberately not installed into the worker's environment. These
minimal constructor doubles exercise declared graph construction and dispatch;
they do not represent scheduler or container integration evidence.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def dag_module(monkeypatch):
    airflow = ModuleType("airflow")
    operators = ModuleType("airflow.operators")
    python_operators = ModuleType("airflow.operators.python")

    class DagDouble:
        def __init__(self, **kwargs):
            self.parameters = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    class OperatorDouble:
        def __init__(self, **kwargs):
            self.parameters = kwargs
            self.task_id = kwargs["task_id"]
            self.downstream = set()

        def __rshift__(self, other):
            self.downstream.add(other.task_id)
            return other

    airflow.DAG = DagDouble
    python_operators.PythonOperator = OperatorDouble
    monkeypatch.setitem(sys.modules, "airflow", airflow)
    monkeypatch.setitem(sys.modules, "airflow.operators", operators)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", python_operators)
    path = Path(__file__).resolve().parents[1] / "dags" / "credit_default_pipeline.py"
    spec = importlib.util.spec_from_file_location("tested_dag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_requires_data_gates_all_training_runs_and_approval(dag_module):
    tasks = dag_module.tasks
    assert tasks["ingest"].downstream == {"validate"}
    assert tasks["validate"].downstream == {"split"}
    assert tasks["split"].downstream == {"features"}
    assert tasks["features"].downstream == {
        "train_baseline",
        "train_candidate_1",
        "train_candidate_2",
    }
    for training in ("train_baseline", "train_candidate_1", "train_candidate_2"):
        assert tasks[training].downstream == {"evaluate"}
    assert tasks["evaluate"].downstream == {"register", "approve"}
    assert tasks["register"].downstream == {"benchmark", "approve"}
    assert tasks["benchmark"].downstream == {"approve"}
    assert tasks["approve"].downstream == {"deploy"}
    assert tasks["deploy"].downstream == {"verify"}
    assert tasks["verify"].downstream == set()
    assert all(task.parameters["trigger_rule"] == "all_success" for task in tasks.values())
    assert dag_module.dag.parameters["max_active_runs"] == 1
    assert dag_module.dag.parameters["schedule"] is None


def test_dispatch_preserves_worker_failure_and_normalizes_generated_run_id(dag_module, monkeypatch):
    received = []

    def fail(step, run_id, config):
        received.append((step, run_id, config))
        raise RuntimeError("validation failed")

    monkeypatch.setattr(dag_module, "execute_remote", fail)
    run = SimpleNamespace(conf={}, run_id="manual__2026-10-04T12:00:00+00:00")
    with pytest.raises(RuntimeError, match="validation failed"):
        dag_module.dispatch_step("validate", "", "configs/project.yaml", dag_run=run)
    assert received[0][0] == "validate"
    assert received[0][1].startswith("airflow-")
    assert ":" not in received[0][1]


def test_retrain_requires_same_deduplicated_trigger_identity(dag_module, monkeypatch):
    monkeypatch.setattr(dag_module, "execute_remote", lambda *_args: {"state": "succeeded"})
    run = SimpleNamespace(
        conf={"mode": "retrain", "trigger_id": "alert-1"}, run_id="retrain-alert-1"
    )
    assert (
        dag_module.dispatch_step("ingest", "retrain-alert-1", "configs/project.yaml", dag_run=run)[
            "state"
        ]
        == "succeeded"
    )
    with pytest.raises(ValueError, match="both run IDs"):
        dag_module.dispatch_step("ingest", "different", "configs/project.yaml", dag_run=run)


def test_data_file_conf_reaches_every_stage_but_not_retraining(dag_module, monkeypatch):
    received = []
    monkeypatch.setattr(
        dag_module,
        "execute_remote",
        lambda step, run_id, config, **kwargs: received.append(kwargs) or {"state": "succeeded"},
    )
    run = SimpleNamespace(conf={"data_file": "data/bad-domain.csv"}, run_id="manual__x")
    dag_module.dispatch_step("validate", "demo-1", "configs/project.yaml", dag_run=run)
    assert received == [{"data_file": "data/bad-domain.csv"}]
    retrain = SimpleNamespace(
        conf={"mode": "retrain", "trigger_id": "t", "data_file": "data/x.csv"}, run_id="retrain-t"
    )
    with pytest.raises(ValueError, match="approved snapshot"):
        dag_module.dispatch_step("ingest", "retrain-t", "configs/project.yaml", dag_run=retrain)
