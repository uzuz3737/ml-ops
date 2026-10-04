"""P0 callable stages. All downstream references are relative to the current run."""

import json
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_project.data.policy import FEATURES, TARGET, validate_rows
from mlops_project.pipelines.contracts import PipelineError, confined_path, sha256_file

SOURCE = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SOURCE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"


def _write(context, name, value):
    path = Path(context["run_dir"]) / name
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return _ref(path)


def _ref(path):
    return {"uri": path.name, "sha256": sha256_file(path)}


def _read(context, ref, *, check_policy=True):
    path = confined_path(ref["uri"], Path(context["run_dir"]), must_exist=True)
    if sha256_file(path) != ref["sha256"]:
        raise PipelineError("artifact_mismatch", "Data artifact checksum changed.")
    records = json.loads(path.read_text(encoding="utf-8"))
    if check_policy and validate_rows(records):
        raise PipelineError("invalid_data_artifact", "Canonical data artifact violates row policy.")
    return pd.DataFrame(records)


def _base(context):
    return {"contract_version": 1, "run_id": context["run_id"]}


def ingest(context):
    run = Path(context["run_dir"])
    archive = run / "source.zip"
    if context.get("retraining"):
        return _ingest_snapshot(context)
    local = context["config"]["dataset"].get("source_archive")
    if local:
        shutil.copyfile(
            confined_path(local, Path(context["project_root"]), must_exist=True), archive
        )
    else:
        with urllib.request.urlopen(SOURCE, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    if sha256_file(archive) != SOURCE_SHA256:
        raise PipelineError("source_checksum", "UCI source differs from the profiled snapshot.")
    with zipfile.ZipFile(archive) as bundle:
        names = [n for n in bundle.namelist() if n.endswith(".xls")]
        if len(names) != 1:
            raise PipelineError("source_format", "Expected one UCI spreadsheet.")
        frame = pd.read_excel(bundle.open(names[0]), header=1)
    frame = frame.rename(columns={"default payment next month": TARGET})
    failures = validate_rows(frame.to_dict("records"))
    if failures:
        _write(context, "ingestion-errors.json", failures)
        raise PipelineError("source_format", "Source fails canonical row policy.")
    path = run / "data.json"
    frame.to_json(path, orient="records")
    data = _ref(path)
    try:
        commit = subprocess.run(
            ["git", "-C", context["project_root"], "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None  # Git may be absent from the worker image; do not invent provenance.
    result = {
        **_base(context),
        "dataset_version": data["sha256"],
        "schema_version": "credit-default-v1",
        "data": data,
        "content_sha256": data["sha256"],
        "source": SOURCE,
        "source_sha256": SOURCE_SHA256,
        "ingest_code_commit": commit,
        "license": "CC-BY-4.0",
        "attribution": "Yeh, I. (2009). UCI Default of Credit Card Clients. doi:10.24432/C55S3H",
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": len(frame),
        "feature_columns": list(FEATURES),
        "target_column": TARGET,
        "profile": {
            "label_counts": frame[TARGET].value_counts().sort_index().to_dict(),
            "minimum": frame.min().to_dict(),
            "maximum": frame.max().to_dict(),
        },
        "artifacts": [data, _ref(archive)],
    }
    manifest = _write(context, "dataset-manifest.json", result)
    result["artifacts"].append(manifest)
    return result


def _ingest_snapshot(context):
    from mlops_project.data.snapshots import load_snapshot

    frame, partitions, provenance = load_snapshot(context)
    path = Path(context["run_dir"]) / "data.json"
    frame.to_json(path, orient="records")
    result = {
        **_base(context),
        "dataset_version": context["retraining"]["candidate_dataset_version"],
        "schema_version": "credit-default-v1",
        "content_sha256": sha256_file(path),
        "data": _ref(path),
        "row_count": len(frame),
        "feature_columns": list(FEATURES),
        "target_column": TARGET,
        "approved_partitions": partitions,
        "snapshot_provenance": provenance,
        **{key: provenance[key] for key in ("source", "license", "attribution")},
        "receipt_sha256": context["retraining"].get("receipt_sha256"),
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": [_ref(path)],
    }
    result["artifacts"].append(_write(context, "dataset-manifest.json", result))
    return result


def _tfdv(frame, schema_path, run):
    import tensorflow_data_validation as tfdv
    from google.protobuf import text_format

    schema = tfdv.load_schema_text(str(schema_path))
    stats = tfdv.generate_statistics_from_dataframe(frame.drop(columns="ID"))
    anomalies = tfdv.validate_statistics(stats, schema, environment="TRAINING")
    refs = []
    for name, proto in (("statistics.pb", stats), ("anomalies.pb", anomalies)):
        path = run / name
        path.write_bytes(proto.SerializeToString())
        refs.append(_ref(path))
    (run / "anomalies.pbtxt").write_text(text_format.MessageToString(anomalies), encoding="utf-8")
    refs.append(_ref(run / "anomalies.pbtxt"))
    return [
        {"field": name, "rule": "tfdv", "reason": entry.short_description}
        for name, entry in anomalies.anomaly_info.items()
    ], refs


def validate(context):
    source = context["inputs"]["ingest"]
    frame = _read(context, source["data"], check_policy=False)
    failures = validate_rows(frame.to_dict("records"))
    artifacts = list(source["artifacts"])
    backend = "not_run_row_failures"
    if not failures:
        schema = confined_path(
            "schemas/credit_default.pbtxt", Path(context["project_root"]), must_exist=True
        )
        try:
            extra, refs = _tfdv(frame, schema, Path(context["run_dir"]))
            failures.extend(extra)
            artifacts.extend(refs)
            backend = "tensorflow-data-validation"
        except ImportError:
            failures.append({"rule": "missing_tfdv_dependency"})
            backend = "unavailable"
    report = {
        **_base(context),
        "dataset_version": source["dataset_version"],
        "schema_version": source["schema_version"],
        "row_count": len(frame),
        "passed": not failures,
        "failures": failures,
        "validation_backend": backend,
    }
    artifacts.append(_write(context, "validation-report.json", report))
    if failures:
        _write(
            context,
            "data-validation-alert.json",
            {**report, "event": "data_validation_failed", "severity": "critical"},
        )
        raise PipelineError(
            "data_validation_failed", "Hard data validation failed; report and alert persisted."
        )
    return {**source, **report, "data": source["data"], "artifacts": artifacts}


def split(context):
    source = context["inputs"]["validate"]
    if source.get("passed") is not True:
        raise PipelineError("unvalidated_data", "Splitting requires passing validation.")
    frame = _read(context, source["data"]).sort_values("ID").reset_index(drop=True)
    config = context["config"]
    proportions = config["splits"]
    if any(
        proportions[k] != v
        for k, v in zip(
            ("train", "validation", "final_test", "monitoring"), (0.6, 0.2, 0.1, 0.1), strict=True
        )
    ):
        raise PipelineError("unsupported_split", "This split contract requires 60/20/10/10.")
    seed = config["project"]["seed"]
    if "approved_partitions" in source:
        approved = source["approved_partitions"]
        if not context.get("retraining") or set(approved) != {"train", "validation"}:
            raise PipelineError(
                "invalid_snapshot", "Approved partitions require controlled retraining."
            )
        parts = [(name, frame[frame.ID.isin(ids)]) for name, ids in approved.items()]
        if sum(len(part) for _, part in parts) != len(frame) or set(approved["train"]) & set(
            approved["validation"]
        ):
            raise PipelineError("snapshot_leakage", "Snapshot partition membership changed.")
        return _save_splits(context, source, parts, "approved-independent-snapshot")
    train, remaining = train_test_split(
        frame, train_size=0.6, random_state=seed, stratify=frame[TARGET]
    )
    validation, remaining = train_test_split(
        remaining, train_size=0.5, random_state=seed, stratify=remaining[TARGET]
    )
    final_test, monitoring = train_test_split(
        remaining, train_size=0.5, random_state=seed, stratify=remaining[TARGET]
    )
    return _save_splits(
        context,
        source,
        (
            ("train", train),
            ("validation", validation),
            ("final_test", final_test),
            ("monitoring", monitoring),
        ),
        "stratified-client-disjoint",
    )


def _save_splits(context, source, parts, strategy):
    partitions, artifacts = {}, list(source["artifacts"])
    for name, part in parts:
        part = part.sort_values("ID")
        path = Path(context["run_dir"]) / f"{name}.json"
        part.to_json(path, orient="records")
        ids = _write(context, f"{name}-ids.json", part.ID.tolist())
        partitions[name] = {
            "data": _ref(path),
            "ids": ids,
            "row_count": len(part),
            "label_counts": part[TARGET].value_counts().sort_index().to_dict(),
        }
        artifacts.extend([_ref(path), ids])
    manifest = {
        **_base(context),
        "dataset_version": source["dataset_version"],
        "schema_version": source["schema_version"],
        "seed": context["config"]["project"]["seed"],
        "strategy": strategy,
        "partitions": partitions,
    }
    ref = _write(context, "split-manifest.json", manifest)
    return {**manifest, "split_manifest": ref, "artifacts": [*artifacts, ref]}


def features(context):
    import joblib

    from mlops_project.features import build_transformer

    source = context["inputs"]["split"]
    train = _read(context, source["partitions"]["train"]["data"])
    transformer = build_transformer().fit(train[list(FEATURES)])
    path = Path(context["run_dir"]) / "transformer.joblib"
    joblib.dump(transformer, path)
    return {
        **source,
        **_base(context),
        "transformer": _ref(path),
        "feature_order": list(FEATURES),
        "fit_partition": "train",
        "output_features": transformer.get_feature_names_out().tolist(),
        "artifacts": [*source["artifacts"], _ref(path)],
    }
