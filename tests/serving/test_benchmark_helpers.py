import pytest

pytest.importorskip("httpx")

from mlops_project.pipelines.contracts import PipelineError  # noqa: E402
from mlops_project.serving import pipeline  # noqa: E402


def test_percentile_uses_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert pipeline._percentile(values, 0.50) == 50
    assert pipeline._percentile(values, 0.95) == 95
    assert pipeline._percentile([3.0], 0.95) == 3.0


def test_benchmark_refuses_incomplete_register_result(tmp_path):
    context = {
        "run_id": "run-1",
        "project_root": str(tmp_path),
        "run_dir": str(tmp_path / "run"),
        "config": {},
        "inputs": {"register": {"model_version": "1"}},
    }
    with pytest.raises(PipelineError) as error:
        pipeline.benchmark(context)
    assert error.value.code == "missing_gate_evidence"
