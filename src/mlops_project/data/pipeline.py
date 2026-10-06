"""P0 callable stages. All downstream references are relative to the current run."""

import json
import math
import shutil
import subprocess
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_project.data.policy import FEATURES, TARGET, validate_rows
from mlops_project.pipelines.contracts import PipelineError, confined_path, sha256_file

SOURCE = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SOURCE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
# Headers found in copies of UCI 350: the original spreadsheet, the Kaggle CSV
# (PAY_1, dotted target) and ucimlrepo's generic X1..X23/Y names.
GENERIC_COLUMNS = {f"X{i}": name for i, name in enumerate(FEATURES, start=1)} | {"Y": TARGET}
TARGET_ALIASES = {"default payment next month", "default next month"}


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


def _publish_validation(context, report, *, alert):
    """Persist the stop/alert evidence and the status Prometheus reads via the API."""
    run = Path(context["run_dir"])
    record = {**report, "event": "data_validation_failed", "severity": "critical"}
    # Pipeline runs live in <artifact_root>/runs/<run_id>; isolated adapter calls
    # (tests) keep their evidence in run_dir only.
    if run.parent.name == "runs":
        artifacts = run.parents[1]
        status = {
            "contract_version": 1,
            "run_id": context["run_id"],
            "status": "alert" if alert else "ok",
            "failure_count": len(report.get("failures", [])),
            "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        exported = artifacts / "monitoring" / "exported"
        exported.mkdir(parents=True, exist_ok=True)
        (exported / "data_validation.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        if alert:
            alerts = artifacts / "monitoring" / "alerts"
            alerts.mkdir(parents=True, exist_ok=True)
            (alerts / f"data_validation-{context['run_id']}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
    return _write(context, "data-validation-alert.json", record) if alert else None


def _summary(failures):
    """Group per-cell failures so a 30,000-row file still reads as a few causes."""
    counts = Counter((f.get("field", "*"), f["rule"]) for f in failures)
    return [
        {"field": field, "rule": rule, "rows": count}
        for (field, rule), count in counts.most_common(25)
    ]


def _cell(value):
    """Keep what the file actually says; only unambiguous numerals become numbers."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        return value.item() if hasattr(value, "item") else value
    text = value.strip()
    if not text:
        return None
    for parse in (int, float):
        try:
            number = parse(text)
        except ValueError:
            continue
        return number if math.isfinite(number) else text
    return text


def _canonical(name):
    text = str(name).strip()
    if text.lower().replace(".", " ").replace("_", " ") in TARGET_ALIASES:
        return TARGET
    return GENERIC_COLUMNS.get(text, text)


def _load_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("instances", value.get("records"))
        if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
            raise ValueError("JSON data must be a list of row objects")
        return pd.DataFrame(value, dtype=object)
    raw = pd.read_excel(path, header=None, dtype=object)
    # The UCI spreadsheet has a generic X1..Y row above the descriptive header.
    header = next(
        (i for i in range(min(3, len(raw))) if "LIMIT_BAL" in set(raw.iloc[i].astype(str))), 0
    )
    frame = raw.iloc[header + 1 :].reset_index(drop=True)
    frame.columns = raw.iloc[header]
    return frame


def _ingest_file(context):
    """Ingest an operator file as-is; the validate stage decides whether it may be used."""
    reference = context["data_file"]
    root = Path(context["project_root"])
    source = confined_path(reference["uri"], root, must_exist=True)
    if sha256_file(source) != reference["sha256"]:
        raise PipelineError("artifact_mismatch", "data_file changed after the run started.")
    run = Path(context["run_dir"])
    copy = run / f"source{source.suffix.lower()}"
    shutil.copyfile(source, copy)
    try:
        frame = _load_table(copy)
    except Exception as error:  # unreadable input is bad data, not a worker crash
        report = {
            **_base(context),
            "data_file": reference,
            "passed": False,
            "failures": [{"rule": "unreadable_file", "reason": type(error).__name__}],
        }
        report["failure_summary"] = _summary(report["failures"])
        _publish_validation(context, report, alert=True)
        raise PipelineError("data_validation_failed", "data_file could not be parsed.") from None
    frame = frame.rename(columns=_canonical)
    if "PAY_1" in frame.columns and "PAY_0" not in frame.columns:
        frame = frame.rename(columns={"PAY_1": "PAY_0"})
    records = [{str(k): _cell(v) for k, v in row.items()} for row in frame.to_dict("records")]
    generated_ids = "ID" not in frame.columns
    if generated_ids:  # ID is lineage only; number rows so splits stay client-disjoint
        records = [{"ID": index, **row} for index, row in enumerate(records, start=1)]
    path = run / "data.json"
    path.write_text(json.dumps(records, allow_nan=False), encoding="utf-8")
    data = _ref(path)
    expected = {"ID", *FEATURES, TARGET}
    columns = set(records[0]) if records else set()
    result = {
        **_base(context),
        "dataset_version": data["sha256"],
        "schema_version": "credit-default-v1",
        "data": data,
        "content_sha256": data["sha256"],
        "source": "data_file:" + reference["uri"],
        "source_sha256": reference["sha256"],
        "data_file": reference,
        "generated_ids": generated_ids,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": len(records),
        "feature_columns": list(FEATURES),
        "target_column": TARGET,
        "missing_columns": sorted(expected - columns),
        "unexpected_columns": sorted(columns - expected),
        "artifacts": [data, _ref(copy)],
    }
    result["artifacts"].append(_write(context, "dataset-manifest.json", result))
    return result


def ingest(context):
    run = Path(context["run_dir"])
    archive = run / "source.zip"
    if context.get("retraining"):
        return _ingest_snapshot(context)
    if context.get("data_file"):
        return _ingest_file(context)
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
        report = {**_base(context), "passed": False, "failures": failures}
        _publish_validation(context, {**report, "failure_summary": _summary(failures)}, alert=True)
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

    try:
        frame, partitions, provenance = load_snapshot(context)
    except PipelineError as error:
        _write(
            context,
            "data-validation-alert.json",
            {
                **_base(context),
                "event": "snapshot_ingestion_failed",
                "severity": "critical",
                "passed": False,
                "error": error.as_dict(),
            },
        )
        raise
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
    # TFDV also profiles rows that broke the row policy, as long as the columns match,
    # so a bad file gets both the exact cells and the schema anomalies.
    shape_ok = not any(f["rule"] in {"columns", "empty_dataset"} for f in failures)
    if shape_ok:
        schema = confined_path(
            "schemas/credit_default.pbtxt", Path(context["project_root"]), must_exist=True
        )
        try:
            extra, refs = _tfdv(frame, schema, Path(context["run_dir"]))
            failures.extend(extra)
            artifacts.extend(refs)
            backend = "tensorflow-data-validation"
        except ImportError:
            if not failures:
                failures.append({"rule": "missing_tfdv_dependency"})
                backend = "unavailable"
        except Exception as error:  # mixed-type columns cannot be profiled
            failures.append({"rule": "tfdv_unprofilable", "reason": type(error).__name__})
            backend = "tensorflow-data-validation"
    report = {
        **_base(context),
        "dataset_version": source["dataset_version"],
        "schema_version": source["schema_version"],
        "row_count": len(frame),
        "passed": not failures,
        "failures": failures,
        "failure_summary": _summary(failures),
        "validation_backend": backend,
    }
    for key in ("data_file", "missing_columns", "unexpected_columns"):
        if source.get(key):
            report[key] = source[key]
    artifacts.append(_write(context, "validation-report.json", report))
    _publish_validation(context, report, alert=bool(failures))
    if failures:
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


def _training_manifest(context, source):
    """Split manifest in the shape P2's training.inputs.load_inputs reads.

    Partitions are written as CSV with project-relative URIs; monitoring rows are
    never written here, only listed as protected IDs together with final test.
    """
    root = Path(context["project_root"]).resolve()
    run = Path(context["run_dir"]).resolve()
    if not run.is_relative_to(root):
        raise PipelineError("path_outside_root", "run_dir must be inside the project root.")
    partitions, refs = {}, []
    for name in ("train", "validation", "final_test"):
        if name not in source["partitions"]:
            continue  # approved retraining snapshots only carry train/validation
        frame = _read(context, source["partitions"][name]["data"])
        path = run / f"{name}.csv"
        frame[["ID", *FEATURES, TARGET]].to_csv(path, index=False)
        partitions[name] = {
            "uri": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "format": "csv",
        }
        refs.append(_ref(path))
    protected = []
    for name in ("final_test", "monitoring"):
        if name in source["partitions"]:
            ids_path = run / source["partitions"][name]["ids"]["uri"]
            protected.extend(json.loads(ids_path.read_text(encoding="utf-8")))
    manifest = {
        "contract_version": 1,
        "dataset_version": source["dataset_version"],
        "schema_version": source["schema_version"],
        "validation_id": partitions["validation"]["sha256"],
        "feature_columns": list(FEATURES),
        "target_column": TARGET,
        "identifier": "ID",
        "partitions": partitions,
        "protected_ids": sorted(protected),
    }
    if context.get("retraining"):
        manifest["final_test_excluded"] = True
    path = run / "training-split-manifest.json"
    refs.append(_write(context, path.name, manifest))
    portable = {"uri": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
    return portable, refs


def features(context):
    import joblib

    from mlops_project.features import build_transformer

    source = context["inputs"]["split"]
    train = _read(context, source["partitions"]["train"]["data"])
    transformer = build_transformer().fit(train[list(FEATURES)])
    path = Path(context["run_dir"]) / "transformer.joblib"
    joblib.dump(transformer, path)
    training_manifest, refs = _training_manifest(context, source)
    return {
        **source,
        **_base(context),
        "data_split_manifest": source["split_manifest"],
        "split_manifest": training_manifest,
        "preprocessor_factory": "mlops_project.features:build_preprocessor",
        "transformer": _ref(path),
        "feature_order": list(FEATURES),
        "fit_partition": "train",
        "output_features": transformer.get_feature_names_out().tolist(),
        "artifacts": [*source["artifacts"], _ref(path), *refs],
    }
