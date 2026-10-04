"""P0 orchestration: explicit, fail-closed tasks dispatch to the shared worker.

This module imports no data validation or model libraries. Component adapters run
in the worker image, independently of Airflow's dependency environment.
"""

import hashlib
from datetime import UTC, datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from mlops_project.pipelines.client import execute_remote
from mlops_project.pipelines.contracts import DEPENDENCIES as STEP_DEPENDENCIES


def dispatch_step(step: str, run_id: str, config_path: str, **context):
    """Return only the small worker record; exceptions fail the Airflow task."""
    configuration = context["dag_run"].conf or {}
    if not run_id:
        # Airflow-generated IDs contain colons; journals use safe directory names.
        identity = context["dag_run"].run_id.encode("utf-8")
        run_id = "airflow-" + hashlib.sha256(identity).hexdigest()[:24]
    if configuration.get("mode") == "retrain":
        trigger_id = configuration.get("trigger_id")
        if not isinstance(trigger_id, str) or not trigger_id:
            raise ValueError("Retraining requires a nonempty deduplicated trigger_id")
        expected_run_id = "retrain-" + trigger_id
        if run_id != expected_run_id or context["dag_run"].run_id != expected_run_id:
            raise ValueError("Retraining must use retrain-<trigger_id> for both run IDs")
    return execute_remote(step, run_id, config_path)


with DAG(
    dag_id="credit_default_pipeline",
    description="Validated credit-default training, gated deployment and verification",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=3,
    dagrun_timeout=timedelta(hours=2),
    default_args={"owner": "P0", "retries": 0},
    tags=["credit-default", "P0", "integration"],
) as dag:
    tasks = {
        step: PythonOperator(
            task_id=step,
            python_callable=dispatch_step,
            op_kwargs={
                "step": step,
                "run_id": "{{ dag_run.conf.get('pipeline_run_id', '') }}",
                "config_path": "{{ dag_run.conf.get('config_path', 'configs/project.yaml') }}",
            },
            trigger_rule="all_success",
        )
        for step in STEP_DEPENDENCIES
    }
    for step, dependencies in STEP_DEPENDENCIES.items():
        for dependency in dependencies:
            tasks[dependency] >> tasks[step]
