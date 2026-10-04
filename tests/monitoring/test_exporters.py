import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry, generate_latest  # noqa: E402

from mlops_project.monitoring.exporters import (  # noqa: E402
    MonitoringCollector,
    publish_feature_drift,
    publish_quality,
)


def _scrape(directory, features=("LIMIT_BAL", "AGE")):
    registry = CollectorRegistry()
    registry.register(MonitoringCollector(directory, features))
    return generate_latest(registry).decode()


def test_nothing_published_yet_exports_no_values(tmp_path):
    text = _scrape(tmp_path)
    assert 'data_drift_status{status="ok"} 0.0' in text
    assert "data_drift_score{" not in text


def test_feature_drift_is_exported_for_schema_features_only(tmp_path):
    publish_feature_drift(
        {
            "status": "alert",
            "feature_scores": {"LIMIT_BAL": 0.31, "AGE": 0.02, "request_id_abc": 9.0},
        },
        tmp_path,
    )
    text = _scrape(tmp_path)
    assert 'data_drift_status{status="alert"} 1.0' in text
    assert 'data_drift_score{feature="LIMIT_BAL"} 0.31' in text
    assert "request_id_abc" not in text


def test_quality_metrics_are_exported(tmp_path):
    publish_quality(
        {
            "status": "insufficient_data",
            "metrics": {"average_precision": 0.42, "made_up": 1.0},
            "labeled_count": 120,
        },
        tmp_path,
    )
    text = _scrape(tmp_path)
    assert 'model_quality_status{status="insufficient_data"} 1.0' in text
    assert 'model_quality{metric="average_precision"} 0.42' in text
    assert "made_up" not in text
    assert "model_quality_labeled_samples 120.0" in text


def test_unknown_status_is_refused(tmp_path):
    with pytest.raises(ValueError):
        publish_quality({"status": "fine"}, tmp_path)
