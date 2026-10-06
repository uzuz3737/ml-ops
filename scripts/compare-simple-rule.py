"""Compare the selected model with a fixed repayment-delay rule on validation only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_project.artifacts import atomic_json, read_record  # noqa: E402
from mlops_project.pipelines.contracts import validate_run_id  # noqa: E402
from mlops_project.training.bundle import load_bundle, probabilities  # noqa: E402
from mlops_project.training.inputs import load_inputs, verified_path  # noqa: E402
from mlops_project.training.metrics import classification_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    validate_run_id(args.run_id)
    root = Path(__file__).resolve().parents[1]
    directory = root / "artifacts/runs" / args.run_id
    record = read_record(directory / "evaluate.json")
    if record["state"] != "succeeded":
        raise ValueError("A successful candidate evaluation is required")
    selected = record["result"]
    context = {
        "project_root": str(root),
        "inputs": {"features": {"split_manifest": selected["split_manifest"]}},
    }
    _, manifest, splits = load_inputs(context)
    rows = splits["validation"].sort_values("ID")
    labels = rows[manifest["target_column"]].to_numpy()
    bundle = load_bundle(
        verified_path(
            {"uri": selected["artifact_uri"], "sha256": selected["artifact_sha256"]}, root
        ),
        selected["artifact_sha256"],
        manifest["schema_version"],
    )
    model_scores = probabilities(bundle["pipeline"], rows[manifest["feature_columns"]])
    # A predeclared simple rule, not tuned on validation or the final test.
    rule_flags = (rows["PAY_0"].to_numpy() >= 2).astype(float)
    capacity = max(1, int(len(rows) * 0.20))

    def capacity_result(scores):
        chosen = np.argsort(-scores, kind="stable")[:capacity]
        captured = int(labels[chosen].sum())
        return {
            "reviewed": capacity,
            "defaults_found": captured,
            "precision": captured / capacity,
            "recall": captured / int(labels.sum()),
        }

    report = {
        "run_id": args.run_id,
        "dataset_version": selected["dataset_version"],
        "artifact_sha256": selected["artifact_sha256"],
        "validation_id": selected["validation_id"],
        "partition": "validation",
        "rule": "PAY_0 >= 2 (at least two months repayment delay)",
        "rule_metrics": classification_metrics(labels, rule_flags, 0.5),
        "model_metrics": classification_metrics(
            labels, model_scores, bundle["manifest"]["threshold"]
        ),
        "review_capacity_fraction": 0.20,
        "rule_capacity": capacity_result(rule_flags),
        "model_capacity": capacity_result(model_scores),
        "naive_capacity": capacity_result(np.zeros(len(labels))),
        "limitations": "Illustrative 20% review-budget proxy, not stakeholder-approved financial value. Rule scores are binary, not calibrated probabilities; do not interpret rule Brier score as calibrated risk. Ties use ascending client ID. No final-test labels used.",
    }
    atomic_json(directory / "simple-rule-comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
