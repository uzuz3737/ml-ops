import pytest

pytest.importorskip("joblib")
pytest.importorskip("prometheus_client")

from serving_fixtures import (  # noqa: E402
    BrokenModel,
    make_settings,
    publish_bundle,
    read_ack,
    request_deploy,
)

from mlops_project.serving.bundle import ModelSlot  # noqa: E402
from mlops_project.serving.watcher import DeploymentWatcher  # noqa: E402


@pytest.fixture
def setup(tmp_path):
    settings = make_settings(tmp_path)
    slot = ModelSlot()
    return settings, slot, DeploymentWatcher(settings, slot)


def test_deploy_loads_exact_version_and_acks(setup, tmp_path):
    settings, slot, watcher = setup
    request_deploy(settings, publish_bundle(tmp_path, "1"))
    watcher.check_once()

    ack = read_ack(settings, "deploy-v1")
    assert ack["contract_version"] == 1
    assert ack["status"] == "loaded"
    assert ack["model_version"] == "1"
    assert slot.get().model_version == "1"


def test_same_deployment_is_handled_once(setup, tmp_path):
    settings, slot, watcher = setup
    request_deploy(settings, publish_bundle(tmp_path, "1"))
    watcher.check_once()
    first = slot.get()
    watcher.check_once()
    assert slot.get() is first


def test_checksum_mismatch_keeps_current_model(setup, tmp_path):
    settings, slot, watcher = setup
    request_deploy(settings, publish_bundle(tmp_path, "1"))
    watcher.check_once()

    bad = publish_bundle(tmp_path, "2")
    with (tmp_path / bad["artifact_uri"]).open("ab") as handle:
        handle.write(b"tampered")
    request_deploy(settings, bad)
    watcher.check_once()

    ack = read_ack(settings, "deploy-v2")
    assert ack["status"] == "failed"
    assert ack["error"]["code"] == "artifact_mismatch"
    assert slot.get().model_version == "1"


def test_model_that_cannot_predict_is_rejected(setup, tmp_path):
    settings, slot, watcher = setup
    request_deploy(settings, publish_bundle(tmp_path, "3", model=BrokenModel()))
    watcher.check_once()

    assert read_ack(settings, "deploy-v3")["status"] == "failed"
    assert slot.get() is None


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "credit-default-v0"},
        {"model_name": "another-model"},
        {"model_version": "champion"},
        {"gate_report_sha256": "f" * 64},
    ],
)
def test_unapproved_or_incompatible_manifest_fails(setup, tmp_path, change):
    settings, slot, watcher = setup
    request_deploy(settings, publish_bundle(tmp_path, "1") | change)
    watcher.check_once()
    assert read_ack(settings, "deploy-v1")["status"] == "failed"
    assert slot.get() is None


def test_rollback_restores_previous_version(setup, tmp_path):
    settings, slot, watcher = setup
    first = publish_bundle(tmp_path, "1")
    request_deploy(settings, first)
    watcher.check_once()
    request_deploy(settings, publish_bundle(tmp_path, "2"))
    watcher.check_once()
    assert slot.get().model_version == "2"

    request_deploy(settings, first | {"action": "rollback", "deployment_id": "rollback-abc"})
    watcher.check_once()

    ack = read_ack(settings, "rollback-abc")
    assert ack["status"] == "loaded"
    assert ack["model_version"] == "1"
    assert slot.get().model_version == "1"


def test_unload_tombstone_clears_model(setup, tmp_path):
    settings, slot, watcher = setup
    request_deploy(settings, publish_bundle(tmp_path, "1"))
    watcher.check_once()

    request_deploy(
        settings,
        {
            "contract_version": 1,
            "action": "unload",
            "deployment_id": "cancel-123",
            "status": "unavailable",
            "reason": "first_deployment_failed",
        },
    )
    watcher.check_once()

    assert read_ack(settings, "cancel-123")["status"] == "unloaded"
    assert slot.get() is None


def test_restarted_watcher_reloads_current_manifest(setup, tmp_path):
    settings, _, watcher = setup
    request_deploy(settings, publish_bundle(tmp_path, "1"))
    watcher.check_once()

    fresh_slot = ModelSlot()
    DeploymentWatcher(settings, fresh_slot).check_once()
    assert fresh_slot.get().model_version == "1"
