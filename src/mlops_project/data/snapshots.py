"""Immutable approved retraining snapshots with independent protected holdout IDs.

A snapshot is found either in the configured registry
(dataset.retraining_snapshots) or, without editing configuration, as a
content-addressed directory <artifact_root>/retraining/snapshots/<version>/
written by write_snapshot(). There the version is derived from the partition
checksums, and the protected IDs must be the final-test IDs of a completed run.
"""

import hashlib
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mlops_project.data.policy import validate_rows
from mlops_project.pipelines.contracts import PipelineError, confined_path, sha256_file

SNAPSHOT_DIR = Path("retraining") / "snapshots"
_VERSION = re.compile(r"snapshot-[0-9a-f]{32}\Z")


def snapshot_version(train_sha256: str, validation_sha256: str) -> str:
    digest = hashlib.sha256(f"{train_sha256}\n{validation_sha256}".encode()).hexdigest()
    return "snapshot-" + digest[:32]


def _artifact_root(context, root: Path) -> Path:
    return confined_path(context["config"].get("pipeline", {}).get("artifact_root", "artifacts"), root)


def _relative(path: Path, root: Path) -> dict:
    return {"uri": path.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha256_file(path)}


def _content_addressed(context, version, root: Path) -> dict:
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise PipelineError(
            "invalid_snapshot", "Candidate dataset version is not a registered snapshot."
        )
    path = _artifact_root(context, root) / SNAPSHOT_DIR / version / "manifest.json"
    if not path.is_file():
        raise PipelineError("invalid_snapshot", "No snapshot exists for this dataset version.")
    return _relative(path, root)


def _completed_final_test(reference, context, root: Path) -> dict:
    """Accept only the final-test ID list recorded by a succeeded split stage."""
    runs = _artifact_root(context, root) / "runs"
    try:
        path = confined_path(reference["uri"], root, must_exist=True)
        record = json.loads((path.parent / "split.json").read_text(encoding="utf-8"))
        recorded = record["result"]["partitions"]["final_test"]["ids"]
    except (OSError, KeyError, TypeError, ValueError, PipelineError):
        raise PipelineError(
            "snapshot_leakage", "Snapshot must name the final-test IDs of a completed run."
        ) from None
    if (
        path.parent.parent != runs.resolve()
        or path.name != "final_test-ids.json"
        or record.get("state") != "succeeded"
        or recorded.get("sha256") != reference.get("sha256")
        or sha256_file(path) != reference.get("sha256")
    ):
        raise PipelineError(
            "snapshot_leakage", "Snapshot must name the final-test IDs of a completed run."
        )
    return reference


def write_snapshot(
    context, train: list[dict], validation: list[dict], *, protected_ids: dict, provenance: dict
) -> str:
    """Write an immutable, content-addressed retraining snapshot and return its version."""
    root = Path(context["project_root"]).resolve()
    target_root = _artifact_root(context, root) / SNAPSHOT_DIR
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target_root) as temporary:
        staging = Path(temporary)
        for name, rows in (("train", train), ("validation", validation)):
            (staging / f"{name}.json").write_text(
                json.dumps(rows, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
        version = snapshot_version(
            sha256_file(staging / "train.json"), sha256_file(staging / "validation.json")
        )
        directory = target_root / version
        if not directory.exists():
            shutil.copytree(staging, directory)
            manifest = {
                "contract_version": 1,
                "dataset_version": version,
                "schema_version": "credit-default-v1",
                **{key: provenance[key] for key in ("source", "license", "attribution")},
                "partitions": {
                    name: _relative(directory / f"{name}.json", root)
                    for name in ("train", "validation")
                },
                "protected_ids": protected_ids,
                "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
    return version


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
    dataset = context["config"].get("dataset", {})
    registry = dataset.get("retraining_snapshots", {})
    ref = registry.get(version)
    root = Path(context["project_root"])
    content_addressed = ref is None
    if content_addressed:
        ref = _content_addressed(context, version, root)

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
    if content_addressed:
        checksums = [
            reference.get("sha256") if isinstance(reference, dict) else None
            for reference in (manifest["partitions"]["train"], manifest["partitions"]["validation"])
        ]
        if version != snapshot_version(*checksums):
            raise PipelineError(
                "snapshot_checksum", "Snapshot content does not match its dataset version."
            )
        trusted_holdout = _completed_final_test(manifest.get("protected_ids"), context, root)
    else:
        trusted_holdout = dataset.get("protected_final_test_ids")
    if not trusted_holdout or manifest.get("protected_ids") != trusted_holdout:
        raise PipelineError(
            "snapshot_leakage", "Snapshot must use the configured original final-test ID manifest."
        )
    protected = read(trusted_holdout)
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
