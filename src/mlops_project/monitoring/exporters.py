"""Expose P1/P2 monitoring results as Prometheus metrics.

Drift and quality jobs run in the pipeline worker, not in the API. They call
publish_feature_drift / publish_quality, which write the latest result under
artifacts/monitoring/exported/. The API's /metrics reads those files on each
scrape, so Prometheus only has to scrape one target.

Field names follow the monitoring result in INTERFACE_CONTRACTS.md; adjust
_feature_scores / _quality_metrics once P1 and P2 hand over real output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from prometheus_client.core import GaugeMetricFamily

from ..pipelines.runner import atomic_json, utc_now

STATUSES = ("ok", "alert", "insufficient_data", "error")
QUALITY_METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1", "brier_score")
FEATURE_DRIFT_FILE = "feature_drift.json"
QUALITY_FILE = "quality.json"


def _check_status(result: dict) -> str:
    status = result.get("status")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    return status


def publish_feature_drift(result: dict, directory: Path) -> Path:
    _check_status(result)
    path = Path(directory) / FEATURE_DRIFT_FILE
    atomic_json(path, {**result, "exported_at": utc_now()})
    return path


def publish_quality(result: dict, directory: Path) -> Path:
    _check_status(result)
    path = Path(directory) / QUALITY_FILE
    atomic_json(path, {**result, "exported_at": utc_now()})
    return path


def _load(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _feature_scores(result: dict) -> dict:
    scores = result.get("feature_scores", {})
    return scores if isinstance(scores, dict) else {}


def _quality_metrics(result: dict) -> dict:
    metrics = result.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


class MonitoringCollector:
    def __init__(self, directory: Path, features: tuple[str, ...]):
        self.directory = Path(directory)
        self.features = set(features)

    def describe(self):
        return []

    def _status_family(self, name: str, help_text: str, result: dict | None):
        family = GaugeMetricFamily(name, help_text, labels=["status"])
        current = result.get("status") if result else None
        for status in STATUSES:
            family.add_metric([status], 1.0 if status == current else 0.0)
        return family

    def collect(self):
        drift = _load(self.directory / FEATURE_DRIFT_FILE)
        quality = _load(self.directory / QUALITY_FILE)

        yield self._status_family("data_drift_status", "Latest feature-drift result", drift)
        scores = GaugeMetricFamily(
            "data_drift_score", "Per-feature drift statistic from P1", labels=["feature"]
        )
        if drift:
            for feature, value in _feature_scores(drift).items():
                # only schema features become labels, so cardinality stays at 23
                if feature in self.features and _finite(value):
                    scores.add_metric([feature], float(value))
        yield scores

        yield self._status_family("model_quality_status", "Latest labeled-quality result", quality)
        metrics = GaugeMetricFamily(
            "model_quality", "Labeled quality metric from P2", labels=["metric"]
        )
        if quality:
            for name, value in _quality_metrics(quality).items():
                if name in QUALITY_METRICS and _finite(value):
                    metrics.add_metric([name], float(value))
        yield metrics
        labeled = GaugeMetricFamily("model_quality_labeled_samples", "Labels in the quality window")
        if quality and _finite(quality.get("labeled_count")):
            labeled.add_metric([], float(quality["labeled_count"]))
        yield labeled
