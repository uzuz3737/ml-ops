"""Prepare an explicit synthetic label-inversion demo; never submit a DAG implicitly.

Run after monitoring-demo.py --scenario concept-drift. Only original training
and validation rows are transformed. Monitoring and final-test IDs stay protected.
The generated config must be selected when restarting the pipeline worker.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_project.artifacts import atomic_json, read_record, utc_now  # noqa: E402
from mlops_project.data.policy import TARGET  # noqa: E402
from mlops_project.data.snapshots import load_snapshot  # noqa: E402
from mlops_project.monitoring.quality import retraining_alert  # noqa: E402
from mlops_project.pipelines.contracts import (  # noqa: E402
    confined_path,
    load_config,
    sha256_file,
    validate_run_id,
)


def prepare(root, source_run, trigger_id):
    validate_run_id(source_run)
    validate_run_id(trigger_id)
    validate_run_id("retrain-" + trigger_id)
    config, _ = load_config("configs/project.yaml", root)
    artifacts = confined_path(config["pipeline"].get("artifact_root", "artifacts"), root)
    active = read_record(artifacts / "deployments/active-model.json")
    if active["run_id"] != source_run:
        raise ValueError("Source run must be the currently deployed initial model")
    source = artifacts / "runs" / source_run
    training = read_record(source / "training-split-manifest.json")
    report = read_record(artifacts / "monitoring/exported/quality.json")
    if report["model_version"] != active["model_version"]:
        raise ValueError("Quality evidence belongs to a different deployed model")
    if report.get("status") != "alert":
        raise ValueError("Run the concept-drift scenario and inspect its quality alert first")
    destination = root / "data/retraining" / trigger_id
    output_config = root / "configs" / f"retraining-{trigger_id}.local.yaml"
    if destination.exists() or output_config.exists():
        raise ValueError("Use a new trigger ID; existing demo snapshots are immutable")

    def reference(path):
        return {"uri": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}

    destination.mkdir(parents=True)
    protected = destination / "protected-ids.json"
    protected.write_text(json.dumps(training["protected_ids"]) + "\n", encoding="utf-8")
    partitions = {}
    for name in ("train", "validation"):
        rows = json.loads((source / f"{name}.json").read_text(encoding="utf-8"))
        for row in rows:
            row[TARGET] = 1 - row[TARGET]
        path = destination / f"{name}.json"
        path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
        partitions[name] = reference(path)
    version = f"synthetic-inverted-{trigger_id}"
    manifest_path = destination / "manifest.json"
    manifest = {
        "contract_version": 1,
        "dataset_version": version,
        "schema_version": "credit-default-v1",
        "source": f"Synthetic label inversion of UCI 350 train/validation from {source_run}",
        "license": "CC-BY-4.0",
        "attribution": "I-Cheng Yeh, Default of Credit Card Clients, UCI; labels modified for classroom demo only",
        "partitions": partitions,
        "protected_ids": reference(protected),
    }
    atomic_json(manifest_path, manifest)
    config["dataset"]["protected_final_test_ids"] = reference(protected)
    config["dataset"]["retraining_snapshots"] = {version: reference(manifest_path)}
    config.setdefault("training", {})["comparison_bundle"] = {
        "uri": active["artifact_uri"],
        "sha256": active["artifact_sha256"],
    }
    evidence = {
        "candidate_dataset_version": version,
        "candidate_dataset_approved": True,
        "labels_validated": True,
        "separate_training_validation": True,
        "final_test_excluded": True,
    }
    # Reuse the production validator before allowing this dataset into a DAG.
    load_snapshot({"retraining": evidence, "config": config, "project_root": str(root)})
    alert = retraining_alert(
        report,
        alert_id=f"demo-{trigger_id}",
        trigger_id=trigger_id,
        dataset_version=training["dataset_version"],
        candidate_evidence=evidence,
        created_at=utc_now(),
    )
    alert["demo_only"] = True
    atomic_json(destination / "alert.json", alert)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "config": str(output_config.relative_to(root)),
        "alert": str((destination / "alert.json").relative_to(root)),
        "candidate_dataset_version": version,
        "submitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--trigger-id", required=True)
    parser.add_argument(
        "--approve-synthetic-demo",
        action="store_true",
        required=True,
        help="Explicitly approve synthetic labels for this classroom demo only",
    )
    args = parser.parse_args()
    print(
        json.dumps(prepare(Path(__file__).resolve().parents[1], args.source_run, args.trigger_id))
    )


if __name__ == "__main__":
    main()
