"""P2's callable training/evaluation stages for the P0 runner."""

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_project.pipelines.contracts import confined_path, sha256_file
from mlops_project.pipelines.runner import atomic_json, file_lock
from mlops_project.training.bundle import environment, load_bundle, probabilities
from mlops_project.training.inputs import load_inputs, shared_preprocessor, verified_path
from mlops_project.training.metrics import classification_metrics, select_threshold

EXPERIMENTS = ("train_baseline", "train_candidate_1", "train_candidate_2")


def tracking_uri(context):
    uri = context["config"].get("training", {}).get("tracking_uri") or os.environ.get(
        "MLFLOW_TRACKING_URI"
    )
    if not uri:
        raise ValueError("Configure MLFLOW_TRACKING_URI or training.tracking_uri explicitly")
    return uri


def code_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _provenance(context):
    root = Path(context["project_root"])
    commit = context["config"].get("training", {}).get("code_commit") or os.environ.get(
        "MLOPS_CODE_COMMIT"
    )
    if not commit:
        try:
            commit = code_commit(root)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                "Container training requires training.code_commit or MLOPS_CODE_COMMIT"
            ) from exc
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise ValueError("Use an exact full lowercase Git commit")
    digest = hashlib.sha256()
    # Record actual source content too, including local changes. Container Git
    # metadata is deliberately excluded by the existing Dockerfile.
    for source in sorted((root / "src/mlops_project").rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    return {"code_commit": commit, "source_tree_sha256": digest.hexdigest()}


def _estimator(stage, seed):
    if stage == "train_baseline":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1500, random_state=seed)
    if stage == "train_candidate_1":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )


def _train(context, stage):
    source, manifest, splits = load_inputs(context)
    root = Path(context["project_root"])
    directory = confined_path(context["run_dir"], root) / stage
    directory.mkdir(parents=True, exist_ok=True)
    features, target = manifest["feature_columns"], manifest["target_column"]
    train, validation = splits["train"], splits["validation"]
    seed = context["config"]["project"]["seed"]
    model = Pipeline(
        [
            ("preprocessor", shared_preprocessor(source, manifest)),
            ("estimator", _estimator(stage, seed)),
        ]
    )
    model.fit(train[features], train[target])
    scores = probabilities(model, validation[features])
    threshold = select_threshold(validation[target].to_numpy(), scores)
    metrics = classification_metrics(validation[target].to_numpy(), scores, threshold)
    lock = confined_path(
        context["config"]
        .get("training", {})
        .get("dependency_lock", "src/mlops_project/training/requirements.lock"),
        root,
        must_exist=True,
    )
    identity = {
        "contract_version": 1,
        "run_id": context["run_id"],
        "experiment_id": stage,
        **_provenance(context),
        "dataset_version": manifest["dataset_version"],
        "schema_version": manifest["schema_version"],
        "split_manifest_sha256": source["split_manifest"]["sha256"],
        "validation_id": manifest["validation_id"],
        "config_sha256": context["config_sha256"],
    }
    metadata = {
        **identity,
        "task": "binary-credit-default",
        "positive_class": 1,
        "feature_columns": features,
        "feature_types": {f: str(train[f].dtype) for f in features},
        "target_column": target,
        "threshold": threshold,
        "random_seed": seed,
        "hyperparameters": model.named_steps["estimator"].get_params(),
        "preprocessor_factory": source.get(
            "preprocessor_factory", "mlops_project.features.pipeline:build_preprocessor"
        ),
        "environment": environment(),
        "dependency_lock_sha256": sha256_file(lock),
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "metrics": metrics,
    }
    path = directory / "bundle.joblib"
    joblib.dump({"pipeline": model, "manifest": metadata}, path, compress=3)
    checksum = sha256_file(path)
    atomic_json(directory / "manifest.json", {**metadata, "artifact_sha256": checksum})
    atomic_json(directory / "environment.json", metadata["environment"])
    smoke_scores = probabilities(model, validation[features].iloc[:2])
    atomic_json(
        directory / "smoke.json",
        {
            "instances": json.loads(validation[features].iloc[:2].to_json(orient="records")),
            "default_probabilities": smoke_scores.tolist(),
            "tolerance": 1e-8,
        },
    )
    mlflow.set_tracking_uri(tracking_uri(context))
    experiment_name = context["config"].get("training", {}).get("experiment_name", "credit-default")
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)
    if existing is None:
        artifact_location = None
        if tracking_uri(context).startswith("sqlite:"):
            artifact_location = (root / "artifacts/mlflow").resolve().as_uri()
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=f"{context['run_id']}-{stage}") as run:
        mlflow.set_tags(
            {
                **identity,
                "owner": "pairot230",
                "random_seed": seed,
                "dependency_lock_sha256": metadata["dependency_lock_sha256"],
                "artifact_sha256": checksum,
                "status": "trained",
            }
        )
        mlflow.log_params(metadata["hyperparameters"])
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
        mlflow.log_artifacts(str(directory), artifact_path="bundle")
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=infer_signature(train[features], model.predict_proba(train[features])),
            pip_requirements=[f"{k}=={v}" for k, v in environment()["packages"].items()],
        )
        mlflow_id = run.info.run_id
    result = {
        **identity,
        "status": "trained",
        "random_seed": seed,
        "hyperparameters": metadata["hyperparameters"],
        "metrics": metrics,
        "mlflow_run_id": mlflow_id,
        "mlflow_model_uri": f"runs:/{mlflow_id}/model",
        "artifact_uri": path.relative_to(root).as_posix(),
        "artifact_sha256": checksum,
        "environment_uri": (directory / "environment.json").relative_to(root).as_posix(),
        "split_manifest": source["split_manifest"],
        "artifacts": [
            {"uri": p.relative_to(Path(context["run_dir"])).as_posix(), "sha256": sha256_file(p)}
            for p in directory.iterdir()
            if p.is_file() and p.name != "result.json"
        ],
    }
    atomic_json(directory / "result.json", result)
    return result


def train_baseline(context):
    return _train(context, "train_baseline")


def train_candidate_1(context):
    return _train(context, "train_candidate_1")


def train_candidate_2(context):
    return _train(context, "train_candidate_2")


def evaluate(context):
    results = [context["inputs"][name] for name in EXPERIMENTS]
    baseline = results[0]
    for result in results:
        for key in (
            "run_id",
            "code_commit",
            "source_tree_sha256",
            "dataset_version",
            "schema_version",
            "validation_id",
            "split_manifest_sha256",
            "config_sha256",
        ):
            if result[key] != baseline[key]:
                raise ValueError(f"Experiments have incomparable {key}")
        if (
            result["run_id"] != context["run_id"]
            or result["config_sha256"] != context["config_sha256"]
        ):
            raise ValueError("Stale experiment evidence")
    # Reload raw validation and recompute, so selection never trusts reported scores alone.
    synthetic_context = {
        **context,
        "inputs": {"features": {"split_manifest": baseline["split_manifest"]}},
    }
    _, manifest, splits = load_inputs(synthetic_context)
    rows = splits["validation"]
    candidates = []
    for result in results:
        path = verified_path(
            {"uri": result["artifact_uri"], "sha256": result["artifact_sha256"]},
            Path(context["project_root"]),
        )
        bundle = load_bundle(path, result["artifact_sha256"], result["schema_version"])
        for key in (
            "run_id",
            "code_commit",
            "source_tree_sha256",
            "dataset_version",
            "schema_version",
            "validation_id",
            "config_sha256",
        ):
            if bundle["manifest"][key] != result[key]:
                raise ValueError("Bundle and experiment provenance differ")
        metrics = classification_metrics(
            rows[manifest["target_column"]].to_numpy(),
            probabilities(bundle["pipeline"], rows[manifest["feature_columns"]]),
            bundle["manifest"]["threshold"],
        )
        candidates.append({**result, "metrics": metrics})
    selected = max(candidates, key=lambda r: r["metrics"]["average_precision"])
    comparison = {
        "kind": "baseline",
        "validation_id": manifest["validation_id"],
        "average_precision": candidates[0]["metrics"]["average_precision"],
    }
    champion = context["config"].get("training", {}).get("comparison_bundle")
    if context.get("retraining") and not champion:
        raise ValueError(
            "Retraining requires current champion comparison on the same permitted validation"
        )
    if champion:
        bundle = load_bundle(
            verified_path(champion, Path(context["project_root"])),
            champion["sha256"],
            manifest["schema_version"],
        )
        if bundle["manifest"]["feature_columns"] != manifest["feature_columns"]:
            raise ValueError("Champion feature signature mismatch")
        comparison = {
            "kind": "current-model",
            "validation_id": manifest["validation_id"],
            "average_precision": classification_metrics(
                rows[manifest["target_column"]].to_numpy(),
                probabilities(bundle["pipeline"], rows[manifest["feature_columns"]]),
                bundle["manifest"]["threshold"],
            )["average_precision"],
            "artifact_sha256": champion["sha256"],
        }
    report = {
        **selected,
        "passed": True,
        "status": "validated",
        "comparison": comparison,
        "selection_reason": "Highest validation average precision; experiment order breaks exact ties",
        "experiments": [
            {
                "experiment_id": r["experiment_id"],
                "mlflow_run_id": r["mlflow_run_id"],
                "metrics": r["metrics"],
            }
            for r in candidates
        ],
    }
    path = Path(context["run_dir"]) / "evaluation.json"
    atomic_json(path, report)
    report["artifacts"] = [
        *selected["artifacts"],
        {"uri": "evaluation.json", "sha256": sha256_file(path)},
    ]
    mlflow.set_tracking_uri(tracking_uri(context))
    client = mlflow.MlflowClient()
    for result in candidates:
        client.set_tag(
            result["mlflow_run_id"],
            "selected",
            str(result["experiment_id"] == selected["experiment_id"]).lower(),
        )
    client.log_artifact(selected["mlflow_run_id"], str(path), "evaluation")
    return report


def evaluate_final_test(context, selected, *, locked_candidate_sha256):
    """One locked initial-candidate assessment; never used by selection/gates/retraining."""
    import pandas as pd

    if context.get("retraining") or context["run_id"].startswith("retrain-"):
        raise ValueError("Protected final test is forbidden for retraining")
    if selected.get("passed") is not True or locked_candidate_sha256 != selected["artifact_sha256"]:
        raise ValueError("Explicitly lock the evaluated candidate before final-test assessment")
    root = Path(context["project_root"])
    source = {"split_manifest": selected["split_manifest"]}
    _, manifest, splits = load_inputs({**context, "inputs": {"features": source}})
    reference = manifest["partitions"]["final_test"]
    path = verified_path(reference, root)
    identity = {
        "dataset_version": manifest["dataset_version"],
        "split_manifest_sha256": selected["split_manifest_sha256"],
        "artifact_sha256": locked_candidate_sha256,
        "final_test_sha256": reference["sha256"],
    }
    ledger_id = hashlib.sha256(manifest["dataset_version"].encode()).hexdigest()
    ledger_dir = root / "artifacts/final-test"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger_dir / f"{ledger_id}.json"
    with file_lock(ledger_dir / f".{ledger_id}.lock"):
        if ledger.exists():
            previous = json.loads(ledger.read_text(encoding="utf-8"))
            if previous["identity"] != identity:
                raise ValueError(
                    "Final test was already consumed by a different locked candidate/split"
                )
            if previous["state"] != "completed":
                raise ValueError(
                    "Final-test assessment interrupted; inspect preserved evidence before recovery"
                )
            return previous["report"]
        bundle = load_bundle(
            verified_path(
                {"uri": selected["artifact_uri"], "sha256": locked_candidate_sha256}, root
            ),
            locked_candidate_sha256,
            manifest["schema_version"],
        )
        # Record lock before touching held-out labels; a crash cannot permit a new candidate.
        for key in (
            "split_manifest_sha256",
            "run_id",
            "dataset_version",
            "schema_version",
            "code_commit",
            "config_sha256",
        ):
            if bundle["manifest"][key] != selected[key]:
                raise ValueError("Locked candidate provenance differs from final assessment")
        atomic_json(ledger, {"identity": identity, "state": "locked"})
        if reference["format"] == "csv":
            rows = pd.read_csv(path)
        elif reference["format"] == "parquet":
            rows = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported final-test format")
        features, target, identifier = (
            manifest["feature_columns"],
            manifest["target_column"],
            manifest["identifier"],
        )
        if set(rows.columns) != set(features + [target, identifier]) or rows.isna().any().any():
            raise ValueError("Invalid final-test columns/nulls")
        ids = set(rows[identifier].astype(str))
        used = set(splits["train"][identifier].astype(str)) | set(
            splits["validation"][identifier].astype(str)
        )
        if (
            rows[identifier].duplicated().any()
            or ids & used
            or not ids <= {str(v) for v in manifest["protected_ids"]}
        ):
            raise ValueError("Final-test IDs are not protected/disjoint")
        report = {
            "contract_version": 1,
            "purpose": "locked-initial-candidate-final-assessment",
            "run_id": context["run_id"],
            **identity,
            "metrics": classification_metrics(
                rows[target].to_numpy(),
                probabilities(bundle["pipeline"], rows[features]),
                bundle["manifest"]["threshold"],
            ),
        }
        atomic_json(Path(context["run_dir"]) / "final-test.json", report)
        atomic_json(ledger, {"identity": identity, "state": "completed", "report": report})
    mlflow.set_tracking_uri(tracking_uri(context))
    mlflow.MlflowClient().log_artifact(
        selected["mlflow_run_id"], str(Path(context["run_dir"]) / "final-test.json"), "final-test"
    )
    return report
