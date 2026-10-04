"""Bootstrap a provisional quality threshold from permitted validation data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mlops_project.pipelines.runner import atomic_json
from mlops_project.training.bundle import load_bundle, probabilities
from mlops_project.training.inputs import verified_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    registered = json.loads(args.registration.read_text(encoding="utf-8"))
    manifest = json.loads(verified_path(registered["split_manifest"], root).read_text())
    ref = manifest["partitions"]["validation"]
    data = verified_path(ref, root)
    rows = pd.read_csv(data) if ref["format"] == "csv" else pd.read_parquet(data)
    bundle = load_bundle(
        verified_path(
            {"uri": registered["artifact_uri"], "sha256": registered["artifact_sha256"]}, root
        ),
        registered["artifact_sha256"],
        registered["schema_version"],
    )
    labels = rows[manifest["target_column"]].to_numpy()
    scores = probabilities(bundle["pipeline"], rows[manifest["feature_columns"]])
    if set(labels) != {0, 1}:
        raise ValueError("Calibration requires both classes")

    def measures(y, p):
        return np.array(
            [average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p)]
        )

    reference = measures(labels, scores)
    rng, degradations = np.random.default_rng(42), []
    for _ in range(300):
        indices = rng.integers(0, len(labels), size=400)
        if len(set(labels[indices])) != 2:
            raise ValueError("Insufficient class coverage for the declared bootstrap sample size")
        current = measures(labels[indices], scores[indices])
        degradations.append(
            max(
                0.0, reference[0] - current[0], reference[1] - current[1], current[2] - reference[2]
            )
        )
    quantile = float(np.quantile(degradations, 0.99))
    result = {
        "scope": "provisional validation bootstrap calibration; consumer review required",
        "dataset_version": registered["dataset_version"],
        "artifact_sha256": registered["artifact_sha256"],
        "validation_id": registered["validation_id"],
        "replicates": 300,
        "seed": 42,
        "labeled_samples_per_replicate": 400,
        "empirical_99th_percentile": quantile,
        "safety_margin": 0.01,
        "quality_degradation_threshold_proposal": quantile + 0.01,
        "minimum_samples_proposal": 500,
        "minimum_labeled_samples_proposal": 400,
        "minimum_class_samples_proposal": 30,
        "minimum_label_coverage_proposal": 0.8,
        "production_approved": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
