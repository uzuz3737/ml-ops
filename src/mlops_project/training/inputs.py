"""P1 boundary: immutable raw splits and a shared unfitted transformer factory."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_project.pipelines.contracts import PipelineError, confined_path, sha256_file


def verified_path(reference, root):
    if not isinstance(reference, dict) or not {"uri", "sha256"} <= reference.keys():
        raise PipelineError("missing_data_contract", "Require a portable uri/sha256 reference")
    path = confined_path(reference["uri"], root, must_exist=True)
    if sha256_file(path) != reference["sha256"]:
        raise PipelineError("artifact_mismatch", "Input checksum mismatch")
    return path


def load_inputs(context):
    root = Path(context["project_root"])
    source = context["inputs"]["features"]
    path = verified_path(source["split_manifest"], root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if type(manifest.get("contract_version")) is not int or manifest["contract_version"] != 1:
        raise ValueError("Unsupported split manifest")
    for key in ("dataset_version", "schema_version", "validation_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"Missing {key}")
    retraining = context.get("retraining")
    if retraining and (
        manifest["dataset_version"] != retraining["candidate_dataset_version"]
        or manifest.get("final_test_excluded") is not True
    ):
        raise ValueError("Retraining must consume approved independent current-regime splits")
    features = manifest["feature_columns"]
    target, identifier = manifest["target_column"], manifest["identifier"]
    if (
        not features
        or len(set(features)) != len(features)
        or target in features
        or identifier in features
    ):
        raise ValueError("Invalid predictor signature or target/ID leakage")
    partitions, ids = {}, {}
    for name in ("train", "validation"):
        reference = manifest["partitions"][name]
        data_path = verified_path(reference, root)
        if reference.get("format") == "csv":
            frame = pd.read_csv(data_path)
        elif reference.get("format") == "parquet":
            frame = pd.read_parquet(data_path)
        else:
            raise ValueError("Explicit csv/parquet format required")
        if set(frame.columns) != set(features + [target, identifier]) or frame.empty:
            raise ValueError("Unexpected raw split columns or empty partition")
        if frame.isna().any().any() or not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite or missing data")
        if any(not pd.api.types.is_numeric_dtype(frame[f]) for f in features):
            raise ValueError("Features must be numeric")
        if not pd.api.types.is_integer_dtype(frame[target]) or set(frame[target]) != {0, 1}:
            raise ValueError(
                "Each fitting/evaluation partition requires strict binary labels and both classes"
            )
        if frame[identifier].duplicated().any():
            raise ValueError("Duplicate client identity")
        ids[name] = set(frame[identifier].astype(str))
        if sha256_file(data_path) != reference["sha256"]:
            raise ValueError("Split changed during read")
        partitions[name] = frame
    if ids["train"] & ids["validation"]:
        raise ValueError("Training/validation client overlap")
    # P1 forwards protected row identities, allowing checks without reading test labels.
    protected = manifest.get("protected_ids")
    if not isinstance(protected, list):
        raise ValueError(
            "Require protected final-test/monitoring IDs, including an explicit empty list for retraining"
        )
    if (ids["train"] | ids["validation"]) & {str(v) for v in protected}:
        raise ValueError("Protected evaluation/replay rows leaked into fitting/selection")
    return source, manifest, partitions


def shared_preprocessor(source, manifest):
    reference = source.get(
        "preprocessor_factory", "mlops_project.features.pipeline:build_preprocessor"
    )
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError("Preprocessor requires module:function reference")
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise PipelineError(
            "missing_adapter", "P1 must supply the shared unfitted preprocessor factory"
        ) from exc
    transformer = factory(feature_columns=manifest["feature_columns"])
    if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
        raise ValueError("P1 factory must return an unfitted sklearn-compatible transformer")
    return transformer
