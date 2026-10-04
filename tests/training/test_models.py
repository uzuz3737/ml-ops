import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import rewrite_manifest
from mlflow import MlflowClient

from mlops_project.pipelines.contracts import sha256_file, validate_result
from mlops_project.training.bundle import load_bundle, predict_bundle
from mlops_project.training.inputs import load_inputs
from mlops_project.training.metrics import classification_metrics, select_threshold
from mlops_project.training.pipeline import evaluate


def test_genuine_three_runs_bundle_and_runner_contract(trained):
    context, results, evaluation = trained
    client = MlflowClient(tracking_uri=context["config"]["training"]["tracking_uri"])
    assert len({r["mlflow_run_id"] for r in results.values()}) == 3
    for stage, result in results.items():
        validate_result(result, stage, context["run_id"], Path(context["run_dir"]))
        run = client.get_run(result["mlflow_run_id"])
        assert run.info.status == "FINISHED"
        assert (
            run.data.params
            and run.data.metrics
            and run.data.tags["dataset_version"] == "synthetic-v1"
        )
        assert {"bundle", "model"} <= {
            a.path for a in client.list_artifacts(result["mlflow_run_id"])
        }
        bundle = load_bundle(
            Path(context["project_root"]) / result["artifact_uri"],
            result["artifact_sha256"],
            "fixture-v1",
        )
        smoke = json.loads(
            (Path(context["project_root"]) / result["artifact_uri"])
            .with_name("smoke.json")
            .read_text()
        )
        prediction = predict_bundle(bundle, smoke["instances"])
        np.testing.assert_allclose(
            [p["default_probability"] for p in prediction],
            smoke["default_probabilities"],
            atol=1e-8,
        )
    validate_result(evaluation, "evaluate", context["run_id"], Path(context["run_dir"]))
    assert evaluation["comparison"]["validation_id"] == evaluation["validation_id"]
    assert evaluation["metrics"]["average_precision"] == max(
        r["metrics"]["average_precision"] for r in results.values()
    )


def test_tampered_bundle_blocks_evaluation(trained):
    context, results, _ = trained
    path = Path(context["project_root"]) / results["train_baseline"]["artifact_uri"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(Exception, match="checksum"):
        evaluate({**context, "inputs": results})


def test_metric_definition_and_threshold():
    from sklearn.metrics import average_precision_score

    labels, scores = [0, 1, 0, 1], [0.1, 0.7, 0.5, 0.9]
    threshold = select_threshold(labels, scores)
    assert threshold == 0.7
    assert classification_metrics(labels, scores, threshold)[
        "average_precision"
    ] == average_precision_score(labels, scores)
    with pytest.raises(ValueError):
        classification_metrics([True, False], [0.1, 0.2], 0.5)


def test_protected_rows_and_overlap_fail_closed(training_context):
    context = training_context
    rewrite_manifest(context, lambda m: m.update(protected_ids=[0]))
    with pytest.raises(ValueError, match="Protected"):
        load_inputs(context)
    rewrite_manifest(context, lambda m: m.update(protected_ids=[]))
    root = Path(context["project_root"])
    validation = pd.read_csv(root / "validation.csv")
    validation.loc[0, "ID"] = 0
    validation.to_csv(root / "validation.csv", index=False)
    rewrite_manifest(
        context,
        lambda m: m["partitions"]["validation"].update(sha256=sha256_file(root / "validation.csv")),
    )
    with pytest.raises(ValueError, match="overlap"):
        load_inputs(context)


def test_missing_p1_factory_is_explicit():
    from mlops_project.training.inputs import shared_preprocessor

    with pytest.raises(Exception, match="P1 must supply"):
        shared_preprocessor(
            {"preprocessor_factory": "missing_p1_module:build"}, {"feature_columns": ["a"]}
        )


def test_retraining_requires_approved_new_data(training_context):
    context = {**training_context, "retraining": {"candidate_dataset_version": "new-regime"}}
    with pytest.raises(ValueError, match="approved independent"):
        load_inputs(context)
