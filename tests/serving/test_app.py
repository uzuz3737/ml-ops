import copy
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("joblib")

from fastapi.testclient import TestClient  # noqa: E402
from serving_fixtures import (  # noqa: E402
    EXAMPLE,
    REPO,
    make_settings,
    publish_bundle,
    request_deploy,
)

from mlops_project.serving.app import create_app  # noqa: E402

JSON = {"Content-Type": "application/json"}


def _client(settings):
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path, max_batch_size=5, max_body_bytes=20_000)


@pytest.fixture
def client(settings):
    with _client(settings) as test_client:
        yield test_client


@pytest.fixture
def served(settings, tmp_path, client):
    request_deploy(settings, publish_bundle(tmp_path, "7"))
    client.app.state.watcher.check_once()
    return client


def _instance(**changes):
    item = copy.deepcopy(EXAMPLE["instances"][0])
    item.update(changes)
    return item


def test_health_works_before_any_model(client):
    assert client.get("/health").json() == {"status": "alive"}
    assert client.get("/ready").status_code == 503
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"


def test_ready_and_predict_report_loaded_version(served):
    ready = served.get("/ready").json()
    assert ready == {
        "status": "ready",
        "model_version": "7",
        "schema_version": "credit-default-v1",
    }

    body = {"instances": [_instance(LIMIT_BAL=100_000), _instance(LIMIT_BAL=900_000)]}
    response = served.post("/predict", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["model_version"] == "7"
    assert response.headers["X-Request-ID"] == result["request_id"]
    assert [p["label"] for p in result["predictions"]] == [0, 1]
    assert result["predictions"][0]["default_probability"] == pytest.approx(0.1)


def test_malformed_json_is_400(served):
    response = served.post(
        "/predict", content=b'{"instances": [', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_json"


def test_oversized_body_is_413(served):
    body = {"instances": [_instance()] * 200}
    assert served.post("/predict", json=body).status_code == 413


@pytest.mark.parametrize(
    ("body", "rule"),
    [
        ({"instances": []}, "envelope"),
        ({"instances": [_instance()], "extra": 1}, "envelope"),
        ({"instances": [_instance()] * 6}, "batch_size"),
        ({"instances": [_instance(SEX=None)]}, "null"),
        ({"instances": [_instance(AGE="35")]}, "type"),
        ({"instances": [_instance(AGE=35.5)]}, "not_integer"),
        ({"instances": [_instance(SEX=3)]}, "out_of_domain"),
        ({"instances": [_instance(PAY_0=10)]}, "out_of_domain"),
        ({"instances": [_instance(AGE=0)]}, "out_of_range"),
        ({"instances": [_instance(PAY_AMT1=-1)]}, "out_of_range"),
        ({"instances": [_instance(LIMIT_BAL=10**400)]}, "non_finite"),
        ({"instances": [_instance(PAY_0=True)]}, "type"),
        ({"instances": [_instance(default_next_month=1)]}, "unexpected_field"),
        ({"instances": [{k: v for k, v in _instance().items() if k != "LIMIT_BAL"}]},
         "missing_field"),
    ],
)  # fmt: skip
def test_invalid_input_is_422_with_rule(served, body, rule):
    response = served.post("/predict", json=body)
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert rule in {d["rule"] for d in details}


def test_nan_is_rejected(served):
    raw = json.dumps({"instances": [_instance()]}).replace("100000", "NaN", 1)
    response = served.post(
        "/predict", content=raw.encode(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["rule"] == "non_finite"


def test_one_bad_instance_rejects_whole_batch(served, settings):
    body = {"instances": [_instance(), _instance(SEX=None)]}
    response = served.post("/predict", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["instance_index"] == 1
    assert not (settings.events_dir / "predictions.jsonl").exists()


def test_predictions_are_logged_for_monitoring(served, settings):
    request_id = served.post("/predict", json=EXAMPLE).json()["request_id"]
    lines = (settings.events_dir / "predictions.jsonl").read_text().splitlines()
    event = json.loads(lines[-1])
    assert event["request_id"] == request_id
    assert event["model_version"] == "7"
    assert event["features"]["LIMIT_BAL"] == EXAMPLE["instances"][0]["LIMIT_BAL"]


def test_feedback_flow(served):
    request_id = served.post("/predict", json={"instances": [_instance()] * 2}).json()["request_id"]
    label = {
        "request_id": request_id,
        "labels": [{"instance_index": 0, "label": 1}],
        "observed_at": "2099-01-01T00:00:00Z",
    }

    first = served.post("/feedback", json=label)
    assert first.status_code == 202
    assert first.json()["stored"] == 1

    replay = served.post("/feedback", json=label)
    assert replay.status_code == 202
    assert replay.json()["stored"] == 0

    conflict = copy.deepcopy(label)
    conflict["labels"][0]["label"] = 0
    assert served.post("/feedback", json=conflict).status_code == 409

    out_of_range = copy.deepcopy(label)
    out_of_range["labels"][0]["instance_index"] = 2
    assert served.post("/feedback", json=out_of_range).status_code == 422

    unknown = label | {"request_id": "missing"}
    assert served.post("/feedback", json=unknown).status_code == 404

    no_tz = label | {"observed_at": "2026-10-04T00:00:00"}
    assert served.post("/feedback", json=no_tz).status_code == 422

    too_early = label | {"observed_at": "2000-01-01T00:00:00Z"}
    assert served.post("/feedback", json=too_early).json()["error"]["code"] == (
        "observed_before_prediction"
    )


def test_feedback_survives_restart(settings, tmp_path):
    request_deploy(settings, publish_bundle(tmp_path, "7"))
    with _client(settings) as first:
        request_id = first.post("/predict", json=EXAMPLE).json()["request_id"]

    with _client(settings) as second:
        response = second.post(
            "/feedback",
            json={
                "request_id": request_id,
                "labels": [{"instance_index": 0, "label": 0}],
                "observed_at": "2099-01-01T00:00:00Z",
            },
        )
    assert response.status_code == 202


def test_metrics_expose_request_counters(served):
    served.post("/predict", json=EXAMPLE)
    served.post("/predict", json={"instances": []})
    text = served.get("/metrics").text
    assert 'api_requests_total{endpoint="/predict",status="200"}' in text
    assert 'api_validation_failures_total{rule="envelope"}' in text
    assert "model_ready 1.0" in text
    assert "request_id" not in text


def test_event_files_match_p2_label_join_format(served, settings):
    request_id = served.post("/predict", json={"instances": [_instance()] * 2}).json()["request_id"]
    served.post(
        "/feedback",
        json={
            "request_id": request_id,
            "labels": [{"instance_index": 1, "label": 1}],
            "observed_at": "2099-01-01T00:00:00Z",
        },
    )
    prediction = json.loads((settings.events_dir / "predictions.jsonl").read_text().splitlines()[0])
    feedback = json.loads((settings.events_dir / "labels.jsonl").read_text().splitlines()[0])

    # fields read by monitoring.quality.join_labels
    for key in ("request_id", "instance_index", "prediction_time", "model_version", "label",
                "default_probability"):  # fmt: skip
        assert key in prediction
    assert feedback["request_id"] == request_id
    assert feedback["labels"] == [{"instance_index": 1, "label": 1}]
    assert feedback["observed_at"] == "2099-01-01T00:00:00Z"


def test_values_p1_allows_are_accepted(served):
    # negative bills and the extra EDUCATION/MARRIAGE codes are real in UCI 350
    body = {"instances": [_instance(BILL_AMT1=-5000, EDUCATION=6, MARRIAGE=0, PAY_0=-2, AGE=35.0)]}
    assert served.post("/predict", json=body).status_code == 200


INVALID_EXAMPLES = sorted((REPO / "examples/invalid").glob("*.json"))


@pytest.mark.parametrize("path", INVALID_EXAMPLES, ids=lambda p: p.stem)
def test_shipped_invalid_examples_are_rejected(served, path):
    response = served.post("/predict", content=path.read_bytes(), headers=JSON)
    assert response.status_code == 422


def test_shipped_batch_example_is_scored_in_order(served):
    body = json.loads((REPO / "examples/predict-batch.json").read_text())
    predictions = served.post("/predict", json=body).json()["predictions"]
    assert len(predictions) == len(body["instances"])
