import pytest

from mlops_project.training.pipeline import evaluate_final_test


def test_final_test_locks_candidate_and_retraining_cannot_consume(trained):
    context, _, selected = trained
    report = evaluate_final_test(
        context, selected, locked_candidate_sha256=selected["artifact_sha256"]
    )
    assert report["metrics"]["sample_count"] == 20
    assert (
        evaluate_final_test(context, selected, locked_candidate_sha256=selected["artifact_sha256"])
        == report
    )
    other = {**selected, "artifact_sha256": "c" * 64}
    with pytest.raises(ValueError, match="already consumed"):
        evaluate_final_test(context, other, locked_candidate_sha256="c" * 64)
    with pytest.raises(ValueError, match="forbidden"):
        evaluate_final_test(
            {**context, "retraining": {"candidate_dataset_version": "other"}},
            selected,
            locked_candidate_sha256=selected["artifact_sha256"],
        )
