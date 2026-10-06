"""Expose P1/P2 monitoring results as Prometheus metrics.

Drift and quality jobs run in the pipeline worker, not in the API. They call
publish_feature_drift / publish_quality, which write the latest result under
artifacts/monitoring/exported/. The API's /metrics reads those files on each
scrape, so Prometheus only has to scrape one target.

Input shapes:
- P1 feature_drift(): {"status": "measured" | "insufficient_samples",
  "alert": bool, "metrics": {feature: psi}, "threshold": float}
- P2 evaluate_quality(): {"status": "ok" | "alert" | "insufficient_data",
  "metrics": {...}, "observed": float, "labeled_sample_count": int, ...}
Both are stored with a status from the contract set (ok/alert/insufficient_data/error).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from prometheus_client.core import GaugeMetricFamily

from ..pipelines.jsonio import atomic_json, utc_now

STATUSES = ("ok", "alert", "insufficient_data", "error")
QUALITY_METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1", "brier_score")
FEATURE_DRIFT_FILE = "feature_drift.json"
QUALITY_FILE = "quality.json"
# Written by the pipeline's validate stage (mlops_project.data.pipeline).
DATA_VALIDATION_FILE = "data_validation.json"


def _drift_status(result: dict) -> str:
    status = result.get("status")
    if status == "measured":
        return "alert" if result.get("alert") is True else "ok"
    if status == "insufficient_samples":
        return "insufficient_data"
    if status in STATUSES:
        return status
    raise ValueError(f"unknown feature-drift status: {status!r}")


def publish_feature_drift(result: dict, directory: Path) -> Path:
    path = Path(directory) / FEATURE_DRIFT_FILE
    atomic_json(
        path,
        {**result, "source_status": result.get("status"), "status": _drift_status(result),
         "exported_at": utc_now()},
    )  # fmt: skip
    return path


def publish_quality(result: dict, directory: Path) -> Path:
    if result.get("status") not in STATUSES:
        raise ValueError(f"quality status must be one of {STATUSES}")
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
    scores = result.get("metrics", {})
    return scores if isinstance(scores, dict) else {}


def _quality_metrics(result: dict) -> dict:
    metrics = result.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _epoch(result: dict | None) -> float | None:
    try:
        moment = datetime.fromisoformat(result["exported_at"].replace("Z", "+00:00"))
    except (TypeError, KeyError, AttributeError, ValueError):
        return None
    return moment.timestamp() if moment.tzinfo else None


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
        threshold = GaugeMetricFamily("data_drift_threshold", "PSI alert threshold from P1")
        if drift and _finite(drift.get("threshold")):
            threshold.add_metric([], float(drift["threshold"]))
        yield threshold

        yield self._status_family("model_quality_status", "Latest labeled-quality result", quality)
        metrics = GaugeMetricFamily(
            "model_quality", "Labeled quality metric from P2", labels=["metric"]
        )
        if quality:
            for name, value in _quality_metrics(quality).items():
                if name in QUALITY_METRICS and _finite(value):
                    metrics.add_metric([name], float(value))
        yield metrics
        for name, key, help_text in (
            ("model_quality_labeled_samples", "labeled_sample_count", "Labels in the window"),
            ("model_quality_label_coverage", "label_coverage", "Labeled share of predictions"),
            ("model_quality_degradation", "observed", "Worst metric drop vs reference"),
        ):
            family = GaugeMetricFamily(name, help_text)
            if quality and _finite(quality.get(key)):
                family.add_metric([], float(quality[key]))
            yield family

        # A stopped monitoring job leaves the last status frozen; export its age.
        last_run = GaugeMetricFamily(
            "monitoring_last_run_timestamp_seconds", "Unix time of the latest monitoring pass"
        )
        moments = [m for m in (_epoch(drift), _epoch(quality)) if m is not None]
        if moments:
            last_run.add_metric([], max(moments))
        yield last_run

        validation = _load(self.directory / DATA_VALIDATION_FILE)
        yield self._status_family(
            "data_validation_status", "Latest pipeline data-validation result", validation
        )
        failures = GaugeMetricFamily(
            "data_validation_failures", "Failed checks in the latest validated dataset"
        )
        if validation and _finite(validation.get("failure_count")):
            failures.add_metric([], float(validation["failure_count"]))
        yield failures
