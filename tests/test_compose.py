"""Static infrastructure invariants; Docker smoke is a separate integration job."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "compose.yaml").read_text(encoding="utf-8")
    )


def test_published_ports_are_local_only_and_all_services_have_pins(compose):
    services = compose["services"]
    published = {name for name, service in services.items() if "ports" in service}
    assert published == {"api", "mlflow", "airflow-webserver", "prometheus", "grafana"}
    for service in services.values():
        assert ":latest" not in service["image"]
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:")
    assert services["api"]["profiles"] == ["application"]
    assert "ports" not in services["postgres"]
    assert "ports" not in services["pipeline-worker"]


def test_airflow_waits_for_database_migrations_and_worker(compose):
    services = compose["services"]
    for name in ("airflow-webserver", "airflow-scheduler"):
        service = services[name]
        assert service["environment"]["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor"
        assert service["depends_on"]["postgres"]["condition"] == "service_healthy"
        assert (
            service["depends_on"]["airflow-init"]["condition"] == "service_completed_successfully"
        )
        assert service["depends_on"]["pipeline-worker"]["condition"] == "service_healthy"
    assert set(services["airflow-init"]["depends_on"]) == {"postgres"}
    assert services["airflow-init"]["restart"] == "no"
    assert services["mlflow"]["volumes"] == ["mlflow-store:/mlflow"]
    assert "sqlite:////mlflow/mlflow.db" in services["mlflow"]["command"]


def test_workers_share_data_and_artifacts_without_docker_socket(compose):
    services = compose["services"]
    worker_mounts = services["pipeline-worker"]["volumes"]
    api_mounts = services["api"]["volumes"]
    for mount in ("./data:/workspace/data", "./artifacts:/workspace/artifacts"):
        assert mount in worker_mounts and mount in api_mounts
    for service in services.values():
        assert "privileged" not in service
        assert not any("docker.sock" in mount for mount in service.get("volumes", []))
    assert "postgres:" not in services["mlflow"]["command"]
