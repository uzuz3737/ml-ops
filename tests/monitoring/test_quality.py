import copy

import pytest

from mlops_project.monitoring.quality import evaluate_quality, join_labels, retraining_alert


@pytest.fixture
def quality_inputs():
    predictions = [
        {
            "request_id": f"r{i}",
            "instance_index": 0,
            "model_version": "1",
            "prediction_time": "2026-10-04T01:00:00Z",
            "label": int(i >= 10),
            "default_probability": 0.9 if i >= 10 else 0.1,
        }
        for i in range(20)
    ]
    feedback = [
        {
            "request_id": f"r{i}",
            "labels": [{"instance_index": 0, "label": int(i >= 10)}],
            "observed_at": "2026-10-04T03:00:00Z",
        }
        for i in range(20)
    ]
    options = {
        "model_version": "1",
        "window_id": "window-1",
        "start": "2026-10-04T00:00:00Z",
        "end": "2026-10-04T02:00:00Z",
        "as_of": "2026-10-04T04:00:00Z",
        "threshold": 0.5,
        "reference": {
            "reference_id": "reference-1",
            "model_version": "1",
            "average_precision": 1.0,
        },
        "policy": {
            "minimum_samples": 20,
            "minimum_labeled_samples": 16,
            "minimum_class_samples": 5,
            "minimum_label_coverage": 0.8,
            "quality_degradation_threshold": 0.15,
        },
    }
    return predictions, feedback, options


def test_fixed_features_changed_label_relation_produces_quality_alarm(quality_inputs):
    predictions, feedback, options = quality_inputs
    original = copy.deepcopy(predictions)
    assert evaluate_quality(predictions, feedback, **options)["status"] == "ok"
    for item in feedback:
        item["labels"][0]["label"] = 1 - item["labels"][0]["label"]
    drift = evaluate_quality(predictions, feedback, **options)
    assert (
        predictions == original
    )  # Exact same observations/probabilities, changed label relationship.
    assert drift["status"] == "alert" and drift["observed"] > 0.15
    assert drift["metrics"]["average_precision"] == 0.5
    assert (
        drift["reason"] == "quality"
    )  # Observational signal does not assert proof of concept drift.


def test_absent_delayed_and_single_class_labels_are_insufficient(quality_inputs):
    predictions, feedback, options = quality_inputs
    assert evaluate_quality(predictions, [], **options)["status"] == "insufficient_data"
    assert evaluate_quality(predictions, feedback[:10], **options)["status"] == "insufficient_data"
    for item in feedback:
        item["observed_at"] = "2026-10-05T00:00:00Z"
    report = evaluate_quality(predictions, feedback, **options)
    assert report["labeled_sample_count"] == 0 and "observed" not in report


def test_idempotent_duplicates_unmatched_and_conflicts(quality_inputs):
    predictions, feedback, options = quality_inputs
    kwargs = {k: options[k] for k in ("model_version", "start", "end", "as_of")}
    feedback.append(copy.deepcopy(feedback[0]))
    feedback.append(
        {
            "request_id": "unknown",
            "observed_at": options["as_of"],
            "labels": [{"instance_index": 0, "label": 1}],
        }
    )
    joined = join_labels(predictions, feedback, **kwargs)
    assert joined["labeled_sample_count"] == 20
    assert joined["duplicate_label_count"] == joined["unmatched_label_count"] == 1
    feedback[-2]["labels"][0]["label"] = 1
    with pytest.raises(ValueError, match="Conflicting"):
        join_labels(predictions, feedback, **kwargs)


@pytest.mark.parametrize("invalid", [True, 0.0, "1", None, 2])
def test_invalid_feedback_label_rejected(quality_inputs, invalid):
    predictions, feedback, options = quality_inputs
    feedback[0]["labels"][0]["label"] = invalid
    with pytest.raises(ValueError, match="strict integer"):
        evaluate_quality(predictions, feedback, **options)


def test_version_isolation_and_calibration_required(quality_inputs):
    predictions, feedback, options = quality_inputs
    predictions[0]["model_version"] = "2"
    report = evaluate_quality(predictions, feedback, **options)
    assert report["sample_count"] == 19 and report["status"] == "insufficient_data"
    options["policy"]["minimum_samples"] = None
    with pytest.raises(ValueError, match="Calibrated"):
        evaluate_quality(predictions, feedback, **options)


def test_ap_increase_from_changed_prevalence_does_not_mask_ranking_drop():
    probabilities = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    original = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    from mlops_project.training.metrics import classification_metrics

    reference = classification_metrics(original, probabilities, 0.5)
    predictions = [
        {
            "request_id": f"r{i}",
            "instance_index": 0,
            "model_version": "1",
            "prediction_time": "2026-10-04T01:00:00Z",
            "label": int(p >= 0.5),
            "default_probability": p,
        }
        for i, p in enumerate(probabilities)
    ]
    labels = [
        {
            "request_id": f"r{i}",
            "labels": [{"instance_index": 0, "label": 1 - y}],
            "observed_at": "2026-10-04T03:00:00Z",
        }
        for i, y in enumerate(original)
    ]
    report = evaluate_quality(
        predictions,
        labels,
        reference={"reference_id": "r", "model_version": "1", **reference},
        policy={
            "minimum_samples": 5,
            "minimum_labeled_samples": 5,
            "minimum_class_samples": 1,
            "minimum_label_coverage": 1,
            "quality_degradation_threshold": 0.2,
        },
        model_version="1",
        window_id="w",
        start="2026-10-04T00:00:00Z",
        end="2026-10-04T02:00:00Z",
        as_of="2026-10-04T04:00:00Z",
        threshold=0.5,
    )
    assert report["status"] == "alert"
    assert report["metrics"]["average_precision"] > reference["average_precision"]
    assert report["degradation_components"]["average_precision_drop"] == 0
    assert report["degradation_components"]["roc_auc_degradation"] > 0.5


def test_alert_requires_independent_approved_dataset(quality_inputs):
    predictions, feedback, options = quality_inputs
    for item in feedback:
        item["labels"][0]["label"] = 1 - item["labels"][0]["label"]
    report = evaluate_quality(predictions, feedback, **options)
    args = {
        "alert_id": "quality-1",
        "trigger_id": "drift-1",
        "dataset_version": "data-v1",
        "created_at": options["as_of"],
    }
    with pytest.raises(ValueError, match="Independent"):
        retraining_alert(report, candidate_evidence={}, **args)
    evidence = {
        "candidate_dataset_version": "regime-v2",
        "candidate_dataset_approved": True,
        "labels_validated": True,
        "separate_training_validation": True,
        "final_test_excluded": True,
    }
    alert = retraining_alert(report, candidate_evidence=evidence, **args)
    assert alert["label_coverage"] == 1 and alert["positive_label_count"] == 10
