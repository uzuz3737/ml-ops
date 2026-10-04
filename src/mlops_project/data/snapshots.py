"""Immutable approved retraining snapshots with independent protected holdout IDs."""

import json
from pathlib import Path

import pandas as pd

from mlops_project.data.policy import validate_rows
from mlops_project.pipelines.contracts import PipelineError, confined_path, sha256_file


def load_snapshot(context):
    request = context["retraining"]
    version = request.get("candidate_dataset_version")
    if any(
        request.get(flag) is not True
        for flag in (
            "candidate_dataset_approved",
            "labels_validated",
            "separate_training_validation",
            "final_test_excluded",
        )
    ):
        raise PipelineError(
            "unapproved_retraining", "Snapshot requires approved controller evidence."
        )
    registry = context["config"]["dataset"].get("retraining_snapshots", {})
    ref = registry.get(version)
    root = Path(context["project_root"])

    def read(reference):
        if not isinstance(reference, dict) or not isinstance(reference.get("uri"), str):
            raise PipelineError("invalid_snapshot", "Snapshot references require uri and sha256.")
        path = confined_path(reference["uri"], root, must_exist=True)
        if sha256_file(path) != reference.get("sha256"):
            raise PipelineError("snapshot_checksum", "Approved snapshot content changed.")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            raise PipelineError("invalid_snapshot", "Snapshot must be valid UTF-8 JSON.") from None

    manifest = read(ref)
    if not isinstance(manifest, dict) or (
        manifest.get("contract_version") != 1
        or manifest.get("dataset_version") != version
        or manifest.get("schema_version") != "credit-default-v1"
    ):
        raise PipelineError(
            "invalid_snapshot", "Snapshot identity or schema does not match approval."
        )
    if any(
        not isinstance(manifest.get(key), str) or not manifest[key].strip()
        for key in ("source", "license", "attribution")
    ):
        raise PipelineError(
            "invalid_snapshot", "Snapshot requires source, license and attribution provenance."
        )
    if set(manifest.get("partitions", {})) != {"train", "validation"}:
        raise PipelineError(
            "invalid_snapshot",
            "Retraining snapshot needs independent train/validation partitions only.",
        )
    protected = read(manifest.get("protected_ids"))
    if (
        not isinstance(protected, list)
        or not protected
        or any(type(i) is not int or i <= 0 for i in protected)
    ):
        raise PipelineError(
            "invalid_snapshot",
            "Protected holdout requires a nonempty positive integer ID manifest.",
        )
    all_rows, ids, partitions = [], set(protected), {}
    for name in ("train", "validation"):
        rows = read(manifest["partitions"][name])
        if not isinstance(rows, list) or validate_rows(rows):
            raise PipelineError("invalid_snapshot", "Snapshot rows violate canonical data policy.")
        current = {row["ID"] for row in rows}
        if ids & current:
            raise PipelineError(
                "snapshot_leakage", "Snapshot overlaps another partition or protected holdout."
            )
        if {row["default_next_month"] for row in rows} != {0, 1}:
            raise PipelineError(
                "invalid_snapshot", "Each retraining partition requires both label classes."
            )
        ids.update(current)
        partitions[name] = sorted(current)
        all_rows.extend(rows)
    provenance = {
        "manifest": ref,
        "protected_ids": manifest["protected_ids"],
        "partitions": manifest["partitions"],
        **{key: manifest[key] for key in ("source", "license", "attribution")},
    }
    return pd.DataFrame(all_rows).sort_values("ID"), partitions, provenance
