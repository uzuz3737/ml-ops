"""P2 component evidence on UCI XLS, using explicitly provisional P1 fixtures.

This harness is outside src and is not an ingestion, TFDV, or shared-feature
adapter. Replace its fixture manifests/factory with P1's reviewed output for
course integration. It never claims service performance or deployment evidence.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlops_project.monitoring.quality import evaluate_quality
from mlops_project.pipelines.contracts import json_bytes, sha256_file
from mlops_project.pipelines.runner import atomic_json
from mlops_project.registry.pipeline import register, retrieve_bundle
from mlops_project.training.bundle import load_bundle, probabilities
from mlops_project.training.pipeline import (
    EXPERIMENTS,
    evaluate,
    train_baseline,
    train_candidate_1,
    train_candidate_2,
)


def build_fixture_preprocessor(*, feature_columns):
    """Numeric-only verification fixture; P1 owns the production transformation."""
    return StandardScaler()


def context_for(
    root, directory, run_id, frames, features, dataset_version, protected_ids, *, retraining=False
):
    directory.mkdir(parents=True, exist_ok=True)
    refs = {}
    for name, frame in frames.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        refs[name] = {
            "uri": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "format": "csv",
        }
    manifest = {
        "contract_version": 1,
        "dataset_version": dataset_version,
        "schema_version": "credit-default-v1",
        "validation_id": f"{dataset_version}-validation",
        "feature_columns": features,
        "target_column": "default_next_month",
        "identifier": "ID",
        "partitions": refs,
        "protected_ids": protected_ids,
        "final_test_excluded": retraining,
        "evidence_scope": "P2 verification fixture; P1 TFDV/shared features pending",
    }
    path = directory / "split-manifest.json"
    atomic_json(path, manifest)
    config = {
        "project": {"seed": 42},
        "serving": {"model_name": "credit-default-p2-verification"},
        "training": {
            "tracking_uri": f"sqlite:///{(root / 'artifacts/p2-verification/mlflow.db').as_posix()}",
            "experiment_name": "p2-uci350-verification",
        },
    }
    result = {
        "contract_version": 1,
        "run_id": run_id,
        "project_root": str(root),
        "run_dir": str(directory),
        "config": config,
        "config_sha256": hashlib.sha256(json_bytes(config)).hexdigest(),
        "inputs": {
            "features": {
                "split_manifest": {
                    "uri": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                },
                "preprocessor_factory": "evidence:build_fixture_preprocessor",
            }
        },
    }
    if retraining:
        result["retraining"] = {
            "candidate_dataset_version": dataset_version,
            "evidence_scope": "direct component simulation; no Airflow authorization",
        }
    return result


def experiments(context):
    results = {}
    for stage, function in zip(
        EXPERIMENTS, (train_baseline, train_candidate_1, train_candidate_2), strict=True
    ):
        print(f"Training {context['run_id']}: {stage}", flush=True)
        results[stage] = function(context)
    evaluation = evaluate({**context, "inputs": results})
    registered = register({**context, "inputs": {"evaluate": evaluation}})
    return results, evaluation, registered


def quality_inputs(frame, bundle, features, model_version):
    scores = probabilities(bundle["pipeline"], frame[features])
    predictions, feedback = [], []
    for i, (score, target) in enumerate(zip(scores, frame["default_next_month"], strict=True)):
        predictions.append(
            {
                "request_id": f"replay-{i}",
                "instance_index": 0,
                "model_version": model_version,
                "prediction_time": "2026-10-04T01:00:00Z",
                "label": int(score >= bundle["manifest"]["threshold"]),
                "default_probability": float(score),
            }
        )
        feedback.append(
            {
                "request_id": f"replay-{i}",
                "observed_at": "2026-10-04T03:00:00Z",
                "labels": [{"instance_index": 0, "label": int(target)}],
            }
        )
    return predictions, feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / "artifacts/runs" / args.run_id
    if output.exists():
        raise ValueError("Choose a fresh run-id; verification outputs are immutable")
    with zipfile.ZipFile(args.archive) as archive:
        filename = next(name for name in archive.namelist() if name.endswith(".xls"))
        with archive.open(filename) as workbook:
            frame = pd.read_excel(workbook, header=1, engine="xlrd")
    frame = frame.rename(columns={"default payment next month": "default_next_month"})
    schema = yaml.safe_load((root / "configs/data_schema.yaml").read_text())
    features = schema["features"]
    assert set(frame.columns) == set(features + ["ID", "default_next_month"])
    assert len(frame) == 30000 and frame.ID.is_unique and not frame.isna().any().any()
    train, held = train_test_split(
        frame, test_size=0.4, random_state=42, stratify=frame.default_next_month
    )
    validation, held = train_test_split(
        held, test_size=0.5, random_state=42, stratify=held.default_next_month
    )
    final_test, replay = train_test_split(
        held, test_size=0.5, random_state=42, stratify=held.default_next_month
    )
    dataset_version = "uci350-verification-" + sha256_file(args.archive)[:12]
    context = context_for(
        root,
        output,
        args.run_id,
        {"train": train, "validation": validation, "final_test": final_test},
        features,
        dataset_version,
        [int(v) for v in pd.concat([final_test, replay]).ID],
    )
    atomic_json(output / "context.json", context)
    results, evaluation, registered = experiments(context)
    retrieved = retrieve_bundle(
        context,
        registered["model_name"],
        registered["model_version"],
        (output / "retrieved.joblib").relative_to(root).as_posix(),
    )
    bundle = load_bundle(root / retrieved["artifact_uri"], retrieved["artifact_sha256"])
    # Reserve monitoring rows into alarm / new-regime fit / new-regime validation
    # BEFORE creating the changed relationship. No frozen test/old fit rows enter.
    alarm, regime = train_test_split(
        replay, test_size=2 / 3, random_state=42, stratify=replay.default_next_month
    )
    regime_train, regime_validation = train_test_split(
        regime, test_size=0.3, random_state=42, stratify=regime.default_next_month
    )
    predictions, feedback = quality_inputs(alarm, bundle, features, registered["model_version"])
    from mlops_project.training.metrics import classification_metrics

    reference_metrics = classification_metrics(
        alarm.default_next_month.to_numpy(),
        np.array([p["default_probability"] for p in predictions]),
        bundle["manifest"]["threshold"],
    )
    policy = {
        "minimum_samples": 500,
        "minimum_labeled_samples": 400,
        "minimum_class_samples": 30,
        "minimum_label_coverage": 0.8,
        "quality_degradation_threshold": 0.05,
    }
    options = {
        "reference": {
            "reference_id": "controlled-same-input-reference",
            "model_version": registered["model_version"],
            **{k: reference_metrics[k] for k in ("average_precision", "roc_auc", "brier_score")},
        },
        "policy": policy,
        "model_version": registered["model_version"],
        "window_id": "controlled-alarm-window",
        "start": "2026-10-04T00:00:00Z",
        "end": "2026-10-04T02:00:00Z",
        "as_of": "2026-10-04T04:00:00Z",
        "threshold": bundle["manifest"]["threshold"],
    }
    normal = evaluate_quality(predictions, feedback, **options)
    for item in feedback:
        item["labels"][0]["label"] = 1 - item["labels"][0]["label"]
    changed = evaluate_quality(predictions, feedback, **options)
    for part in (regime_train, regime_validation):
        part["default_next_month"] = 1 - part["default_next_month"]
    retrain_context = context_for(
        root,
        root / "artifacts/runs" / f"{args.run_id}-regime",
        f"{args.run_id}-regime",
        {"train": regime_train, "validation": regime_validation},
        features,
        dataset_version + "-label-inversion",
        [int(v) for v in pd.concat([train, validation, final_test, alarm]).ID],
        retraining=True,
    )
    retrain_context["config"]["training"]["comparison_bundle"] = {
        "uri": registered["artifact_uri"],
        "sha256": registered["artifact_sha256"],
    }
    retrain_context["config_sha256"] = hashlib.sha256(
        json_bytes(retrain_context["config"])
    ).hexdigest()
    atomic_json(Path(retrain_context["run_dir"]) / "context.json", retrain_context)
    _, regime_evaluation, regime_registered = experiments(retrain_context)
    summary = {
        "scope": "P2 real-data component verification with provisional numeric-scaler fixture; not full production integration",
        "code_commit": evaluation["code_commit"],
        "source_tree_sha256": evaluation["source_tree_sha256"],
        "verification_harness_sha256": sha256_file(__file__),
        "archive_sha256": sha256_file(args.archive),
        "dataset_version": dataset_version,
        "source": "https://archive.ics.uci.edu/dataset/350",
        "license": "CC-BY-4.0",
        "credit": "Yeh, I. (2009). DOI:10.24432/C55S3H",
        "partition_counts": {
            "train": len(train),
            "validation": len(validation),
            "final_test": len(final_test),
            "monitoring": len(replay),
        },
        "experiments": evaluation["experiments"],
        "selected": evaluation["experiment_id"],
        "registered_version": registered["model_version"],
        "artifact_sha256": registered["artifact_sha256"],
        "immutable_retrieval_passed": True,
        "final_test_read": False,
        "normal_quality": normal,
        "controlled_label_inversion_quality": changed,
        "regime_counts": {
            "alarm": len(alarm),
            "train": len(regime_train),
            "validation": len(regime_validation),
        },
        "regime_experiments": regime_evaluation["experiments"],
        "regime_comparison": regime_evaluation["comparison"],
        "regime_selected_metrics": regime_evaluation["metrics"],
        "regime_version": regime_registered["model_version"],
        "gate_proposal": {"minimum_average_precision": 0.4, "max_regression": 0.02},
        "quality_policy_demo_only": policy,
        "production_gate_approval": False,
        "airflow_retraining_triggered": False,
    }
    atomic_json(output / "p2-evidence.json", summary)
    print(
        json.dumps(
            {
                "evidence": str(output / "p2-evidence.json"),
                "selected": summary["selected"],
                "AP": evaluation["metrics"]["average_precision"],
                "controlled_quality": changed["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
