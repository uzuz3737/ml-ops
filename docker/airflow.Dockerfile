# The official image supplies Airflow's matching constrained dependencies.
FROM apache/airflow:2.10.5-python3.11
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src
COPY --chown=airflow:root src/mlops_project /workspace/src/mlops_project
COPY --chown=airflow:root dags /opt/airflow/dags
COPY --chown=airflow:root scripts/airflow-init.sh /opt/airflow/bootstrap/airflow-init.sh
# DAG imports use stdlib-only HTTP dispatch; no TFDV/model dependencies here.
