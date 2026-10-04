"""P2 label joins and quality signals; telemetry/exporting belongs to P3."""

import hashlib
from datetime import datetime

from mlops_project.pipelines.contracts import json_bytes
from mlops_project.training.metrics import classification_metrics


def _timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone-aware timestamps required")
    return parsed


def _label(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Labels must be strict integer 0/1")
    return value


def join_labels(predictions, feedback, *, model_version, start, end, as_of):
    """Join one exact version/window as of a cutoff; missing labels remain missing."""
    start, end, cutoff = map(_timestamp, (start, end, as_of))
    if start >= end or end > cutoff:
        raise ValueError("Require start < end <= as_of")
    events, all_keys = {}, set()
    for event in predictions:
        key = (event["request_id"], event["instance_index"])
        if not isinstance(key[0], str) or not key[0] or type(key[1]) is not int or key[1] < 0:
            raise ValueError("Invalid prediction join identity")
        if key in all_keys:
            raise ValueError("Duplicate prediction identity")
        all_keys.add(key)
        timestamp = _timestamp(event["prediction_time"])
        if event["model_version"] == model_version and start <= timestamp < end:
            _label(event["label"])
            probability = event["default_probability"]
            if type(probability) not in (int, float) or not 0 <= probability <= 1:
                raise ValueError("Invalid prediction probability")
            events[key] = event
    labels, duplicate_count, unmatched_count = {}, 0, 0
    label_times = {}
    for item in feedback:
        observed = _timestamp(item["observed_at"])
        if observed > cutoff:
            continue
        if not isinstance(item["request_id"], str) or not item["request_id"]:
            raise ValueError("Invalid feedback request identity")
        for entry in item["labels"]:
            index = entry["instance_index"]
            if type(index) is not int or index < 0:
                raise ValueError("Invalid feedback instance index")
            key = (item["request_id"], index)
            label = _label(entry["label"])
            if key not in events:
                unmatched_count += 1
                continue
            if observed < _timestamp(events[key]["prediction_time"]):
                raise ValueError("Feedback precedes prediction")
            if key in labels:
                if labels[key] != label:
                    raise ValueError("Conflicting feedback labels")
                duplicate_count += 1
            else:
                labels[key], label_times[key] = label, observed
    joined = [
        {
            **events[key],
            "ground_truth": value,
            "label_age_seconds": (
                label_times[key] - _timestamp(events[key]["prediction_time"])
            ).total_seconds(),
        }
        for key, value in labels.items()
    ]
    count, labeled = len(events), len(joined)
    return {
        "rows": joined,
        "sample_count": count,
        "labeled_sample_count": labeled,
        "positive_label_count": sum(r["ground_truth"] for r in joined),
        "negative_label_count": sum(r["ground_truth"] == 0 for r in joined),
        "label_coverage": labeled / count if count else 0.0,
        "unmatched_label_count": unmatched_count,
        "duplicate_label_count": duplicate_count,
    }


def evaluate_quality(
    predictions,
    feedback,
    *,
    reference,
    policy,
    model_version,
    window_id,
    start,
    end,
    as_of,
    threshold,
):
    """A labeled AP drop is a quality signal; concept drift needs controlled evidence."""
    for key in ("minimum_samples", "minimum_labeled_samples", "minimum_class_samples"):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise ValueError(f"Calibrated positive integer {key} required")
    for key in ("minimum_label_coverage", "quality_degradation_threshold"):
        if type(policy.get(key)) not in (int, float) or not 0 <= policy[key] <= 1:
            raise ValueError(f"Calibrated fraction {key} required")
    if reference["model_version"] != model_version or not reference.get("reference_id"):
        raise ValueError("Reference must be pinned to the same model version")
    ap = reference["average_precision"]
    if type(ap) not in (int, float) or not 0 <= ap <= 1:
        raise ValueError("Invalid reference average precision")
    joined = join_labels(
        predictions, feedback, model_version=model_version, start=start, end=end, as_of=as_of
    )
    report = {
        "contract_version": 1,
        "model_version": model_version,
        "window_id": window_id,
        "reference_id": reference["reference_id"],
        "policy_sha256": hashlib.sha256(json_bytes(policy)).hexdigest(),
        "start": start,
        "end": end,
        "as_of": as_of,
        **{k: v for k, v in joined.items() if k != "rows"},
        "status": "insufficient_data",
    }
    sufficient = (
        joined["sample_count"] >= policy["minimum_samples"]
        and joined["labeled_sample_count"] >= policy["minimum_labeled_samples"]
        and min(joined["positive_label_count"], joined["negative_label_count"])
        >= policy["minimum_class_samples"]
        and joined["label_coverage"] >= policy["minimum_label_coverage"]
    )
    if not sufficient:
        return report
    rows = joined["rows"]
    metrics = classification_metrics(
        [r["ground_truth"] for r in rows], [r["default_probability"] for r in rows], threshold
    )
    components = {
        "average_precision_drop": max(
            0.0, reference["average_precision"] - metrics["average_precision"]
        )
    }
    # AP is prevalence-sensitive. Optional ROC/Brier reference evidence detects
    # deteriorating ranking/calibration even when changed prevalence raises AP.
    for key in ("roc_auc", "brier_score"):
        if key in reference:
            value = reference[key]
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"Invalid reference {key}")
            delta = value - metrics[key] if key == "roc_auc" else metrics[key] - value
            components[f"{key}_degradation"] = max(0.0, delta)
    degradation = max(components.values())
    crossed = degradation > policy["quality_degradation_threshold"]
    return {
        **report,
        "metrics": metrics,
        "observed": degradation,
        "degradation_components": components,
        "threshold_crossed": crossed,
        "status": "alert" if crossed else "ok",
        "reason": "quality",
        "mean_label_age_seconds": sum(r["label_age_seconds"] for r in rows) / len(rows),
    }


def retraining_alert(
    report, *, alert_id, trigger_id, dataset_version, candidate_evidence, created_at
):
    """Build P0's signal only after independent dataset evidence is supplied by owners."""
    if report["status"] != "alert" or report.get("threshold_crossed") is not True:
        raise ValueError("Insufficient/non-actionable quality cannot trigger retraining")
    _timestamp(created_at)
    flags = (
        "candidate_dataset_approved",
        "labels_validated",
        "separate_training_validation",
        "final_test_excluded",
    )
    if any(candidate_evidence.get(k) is not True for k in flags):
        raise ValueError("Independent approved labeled candidate evidence required")
    version = candidate_evidence["candidate_dataset_version"]
    if not isinstance(version, str) or not version or version == dataset_version:
        raise ValueError("Candidate requires a new immutable dataset identity")
    return {
        "contract_version": 1,
        "alert_id": alert_id,
        "trigger_id": trigger_id,
        "model_version": report["model_version"],
        "dataset_version": dataset_version,
        "created_at": created_at,
        "reason": "quality",
        "observed": report["observed"],
        "threshold_crossed": True,
        **{
            k: report[k]
            for k in (
                "sample_count",
                "labeled_sample_count",
                "positive_label_count",
                "negative_label_count",
                "label_coverage",
            )
        },
        "candidate_dataset_version": version,
        **{k: True for k in flags},
    }
