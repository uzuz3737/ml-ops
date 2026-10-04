"""One monitoring pass over the API's event logs (P3 glue for P1/P2 computations).

    python -m mlops_project.monitoring.run --config configs/project.yaml

Reads the confirmed active deployment, takes the latest window of its
prediction events, runs P1 feature_drift against the train partition of the
run that produced the model and P2 evaluate_quality against the bundle's
validation metrics, then publishes both for /metrics and writes an alert
record for each check that fires. Run it in the pipeline worker, which has
the shared artifacts volume and the P1/P2 dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from ..data.policy import FEATURES
from ..pipelines.contracts import PipelineError, confined_path, load_config, project_directory
from ..pipelines.runner import atomic_json, read_record, utc_now
from .exporters import publish_feature_drift, publish_quality
from .feature_drift import feature_drift
from .quality import evaluate_quality


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _window(events: list[dict], model_version: str, size: int) -> list[dict]:
    mine = [e for e in events if e.get("model_version") == model_version]
    mine.sort(key=lambda e: (_time(e["prediction_time"]), e["request_id"], e["instance_index"]))
    return mine[-size:]


def _alert(directory: Path, source: str, result: dict, model_version: str, **fields) -> Path:
    alert_id = (
        f"{source}-{hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]}"
    )
    record = {
        "contract_version": 1,
        "alert_id": alert_id,
        "source": source,
        "severity": "critical" if source == "quality" else "warning",
        "model_version": model_version,
        "created_at": utc_now(),
        **fields,
    }
    path = directory / f"{alert_id}.json"
    atomic_json(path, record)
    return path


def run_once(config_path="configs/project.yaml", project_root=None, as_of: str | None = None):
    root = Path(project_root).resolve() if project_root else project_directory()
    config, _ = load_config(config_path, root)
    monitoring = yaml.safe_load(
        confined_path(
            config.get("monitoring_config", "configs/monitoring.yaml"), root, must_exist=True
        ).read_text(encoding="utf-8")
    )
    artifacts = confined_path(config["pipeline"].get("artifact_root", "artifacts"), root)
    deployments = confined_path(
        config.get("serving", {}).get("deployments_dir", "artifacts/deployments"), root
    )
    active_path = deployments / "active-model.json"
    if not active_path.exists():
        raise PipelineError("no_active_model", "No confirmed deployment to monitor yet.")
    active = read_record(active_path)
    version = active["model_version"]
    window_size = monitoring["minimum_samples"]

    events_dir = artifacts / "serving" / "events"
    window = _window(_jsonl(events_dir / "predictions.jsonl"), version, window_size)
    exported = artifacts / "monitoring" / "exported"
    alerts = artifacts / "monitoring" / "alerts"
    cutoff = _time(as_of) if as_of else datetime.now(UTC)
    summary = {"model_version": version, "window_rows": len(window), "alerts": []}

    # P1 data drift: reference is the train partition the deployed model was fitted on.
    reference = pd.read_json(artifacts / "runs" / active["run_id"] / "train.json")
    current = pd.DataFrame([e["features"] for e in window], columns=list(FEATURES))
    if len(current) < window_size:
        drift = {"contract_version": 1, "status": "insufficient_samples", "alert": False}
    else:
        drift = feature_drift(
            reference,
            current,
            minimum_rows=window_size,
            threshold=monitoring["data_drift_threshold"],
        )
    publish_feature_drift(drift, exported)
    summary["data_drift"] = drift.get("status"), drift.get("alert")
    if drift.get("alert") is True:
        worst = max(drift["metrics"], key=drift["metrics"].get)
        path = _alert(
            alerts,
            "data_drift",
            drift,
            version,
            reason="data_drift",
            statistic="max_feature_psi",
            feature=worst,
            observed=drift["metrics"][worst],
            threshold=monitoring["data_drift_threshold"],
            suggested_action="Inspect the drifted feature; retrain only with a new approved snapshot.",
        )
        summary["alerts"].append(path.name)

    # P2 labeled quality against the bundle's own validation metrics.
    bundle_manifest = read_record(
        confined_path(active["artifact_uri"], root, must_exist=True).parent / "manifest.json"
    )
    metrics = bundle_manifest["metrics"]
    reference_metrics = {
        "model_version": version,
        "reference_id": active["artifact_sha256"],
        **{k: metrics[k] for k in ("average_precision", "roc_auc", "brier_score") if k in metrics},
    }
    if window:
        # join_labels treats end as exclusive, so stop just after the newest event
        end = _time(window[-1]["prediction_time"]) + timedelta(microseconds=1)
        quality = evaluate_quality(
            window,
            _jsonl(events_dir / "labels.jsonl"),
            reference=reference_metrics,
            policy=monitoring,
            model_version=version,
            window_id=f"{version}-{window[0]['prediction_time']}-{len(window)}",
            start=window[0]["prediction_time"],
            end=_iso(end),
            as_of=_iso(max(cutoff, end)),
            threshold=bundle_manifest["threshold"],
        )
    else:
        quality = {"contract_version": 1, "status": "insufficient_data", "model_version": version}
    publish_quality(quality, exported)
    summary["quality"] = quality["status"]
    if quality["status"] == "alert":
        path = _alert(
            alerts,
            "quality",
            quality,
            version,
            reason="quality",
            statistic="max_quality_degradation",
            observed=quality["observed"],
            threshold=monitoring["quality_degradation_threshold"],
            components=quality["degradation_components"],
            labeled_sample_count=quality["labeled_sample_count"],
            label_coverage=quality["label_coverage"],
            suggested_action="Confirm labels, then retrain on an approved new snapshot.",
        )
        summary["alerts"].append(path.name)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--project-root")
    parser.add_argument("--as-of", help="ISO-8601 cutoff for labels (default: now)")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_once(args.config, args.project_root, args.as_of)))
        return 0
    except PipelineError as error:
        print(json.dumps({"error": error.as_dict()}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
