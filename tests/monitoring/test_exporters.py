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
    # shape returned by P1 monitoring.feature_drift.feature_drift
    publish_feature_drift(
        {
            "contract_version": 1,
            "status": "measured",
            "alert": True,
            "threshold": 0.2,
            "metrics": {"LIMIT_BAL": 0.31, "AGE": 0.02, "request_id_abc": 9.0},
        },
        tmp_path,
    )
    text = _scrape(tmp_path)
    assert 'data_drift_status{status="alert"} 1.0' in text
    assert 'data_drift_score{feature="LIMIT_BAL"} 0.31' in text
    assert "data_drift_threshold 0.2" in text
    assert "request_id_abc" not in text


@pytest.mark.parametrize(
    ("result", "status"),
    [
        ({"status": "measured", "alert": False, "metrics": {}}, "ok"),
        ({"contract_version": 1, "status": "insufficient_samples", "alert": False},
         "insufficient_data"),
    ],
)  # fmt: skip
def test_p1_drift_statuses_map_to_contract(tmp_path, result, status):
    publish_feature_drift(result, tmp_path)
    assert f'data_drift_status{{status="{status}"}} 1.0' in _scrape(tmp_path)


def test_quality_metrics_are_exported(tmp_path):
    # shape returned by P2 monitoring.quality.evaluate_quality
    publish_quality(
        {
            "contract_version": 1,
            "status": "alert",
            "metrics": {"average_precision": 0.42, "made_up": 1.0},
            "labeled_sample_count": 120,
            "label_coverage": 0.6,
            "observed": 0.08,
        },
        tmp_path,
    )
    text = _scrape(tmp_path)
    assert 'model_quality_status{status="alert"} 1.0' in text
    assert 'model_quality{metric="average_precision"} 0.42' in text
    assert "made_up" not in text
    assert "model_quality_labeled_samples 120.0" in text
    assert "model_quality_label_coverage 0.6" in text
    assert "model_quality_degradation 0.08" in text


def test_unknown_status_is_refused(tmp_path):
    with pytest.raises(ValueError):
        publish_quality({"status": "fine"}, tmp_path)
    with pytest.raises(ValueError):
        publish_feature_drift({"status": "fine"}, tmp_path)
