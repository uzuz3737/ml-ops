"""Close the loop: quality alert -> new labeled snapshot -> Airflow retraining -> gates.

    docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario concept-drift
    docker compose exec pipeline-worker python scripts/retraining-demo.py --approve-snapshot

Policy (configs/monitoring.yaml): only a labeled quality alert above its calibrated
threshold may retrain; data drift alone means "investigate". The new data is the
part of the deployed run's monitoring partition that was NOT in the alarm window,
relabeled under the same controlled concept-drift regime (labels inverted), and
split 70/30 into independent train/validation. The original final test stays
excluded. Passing --approve-snapshot is the operator's approval of that data.

The script submits through the deduplicating controller (cooldown, one active
run), waits for the Airflow run retrain-<trigger_id> and reports whether the
gates deployed the new model or kept the old one.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlops_project.data.policy import FEATURES, TARGET  # noqa: E402
from mlops_project.data.snapshots import write_snapshot  # noqa: E402
from mlops_project.monitoring.quality import retraining_alert  # noqa: E402
from mlops_project.pipelines.contracts import PipelineError, load_config  # noqa: E402
from mlops_project.pipelines.retraining import AirflowClient, submit_retraining  # noqa: E402

CONFIG = "configs/project.yaml"
ATTRIBUTION = "Yeh, I. (2009). UCI Default of Credit Card Clients. doi:10.24432/C55S3H"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_quality_alert(artifacts: Path, version: str) -> dict:
    alerts = [read(p) for p in (artifacts / "monitoring/alerts").glob("quality-*.json")]
    mine = [a for a in alerts if a.get("model_version") == version]
    if not mine:
        raise SystemExit("No quality alert for the deployed model; run monitoring-demo first.")
    return max(mine, key=lambda a: a["created_at"])


def build_snapshot(config: dict, active: dict, args) -> str:
    run = ROOT / "artifacts/runs" / active["run_id"]
    pool = pd.read_json(run / "monitoring.json")
    # Same draw as monitoring-demo.py, so the alarm window never trains the new model.
    alarm = pool.sample(n=min(args.alarm_rows, len(pool)), random_state=args.alarm_seed)
    fresh = pool[~pool["ID"].isin(alarm["ID"])].copy()
    fresh[TARGET] = 1 - fresh[TARGET]  # the controlled concept-drift regime
    train, validation = train_test_split(
        fresh, test_size=0.3, random_state=42, stratify=fresh[TARGET]
    )
    columns = ["ID", *FEATURES, TARGET]
    rows = [
        part[columns].astype(int).sort_values("ID").to_dict("records")
        for part in (train, validation)
    ]
    ids = run / "final_test-ids.json"
    protected = {
        "uri": ids.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(ids.read_bytes()).hexdigest(),
    }
    provenance = {
        "source": f"monitoring partition of run {active['run_id']} outside the alarm window, "
        "labels under the controlled concept-drift regime",
        "license": "CC-BY-4.0",
        "attribution": ATTRIBUTION,
    }
    context = {"project_root": str(ROOT), "config": config}
    version = write_snapshot(context, *rows, protected_ids=protected, provenance=provenance)
    print(f"Snapshot {version}: {len(rows[0])} train / {len(rows[1])} validation rows")
    return version


def wait(dag_id: str, run_id: str, timeout: int) -> str:
    client = AirflowClient()
    path = (
        f"/api/v1/dags/{urllib.parse.quote(dag_id, safe='')}"
        f"/dagRuns/{urllib.parse.quote(run_id, safe='')}"
    )
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        state = client("GET", path).get("state")
        if state != last:
            print(f"Airflow {run_id}: {state}", flush=True)
            last = state
        if state in {"success", "failed"}:
            return state
        time.sleep(5)
    raise SystemExit(f"{run_id} is still running after {timeout}s; check Airflow.")


def report(run_id: str, previous: str) -> None:
    run = ROOT / "artifacts/runs" / run_id
    for step in ("evaluate", "approve", "deploy"):
        path = run / f"{step}.json"
        if not path.exists():
            continue
        record = read(path)
        if record.get("state") == "failed":
            print(f"Stopped at {step}: {record['error']['message']}")
            break
        result = record["result"]
        if step == "evaluate":
            print(
                f"Selected {result.get('experiment_id')}: validation AP "
                f"{result['metrics']['average_precision']:.4f} vs current model "
                f"{result['comparison']['average_precision']:.4f} on the same new data"
            )
    active = read(ROOT / "artifacts/deployments/active-model.json")
    if active["model_version"] != previous:
        print(f"Deployed model v{active['model_version']} (replaced v{previous}).")
    else:
        print(f"Model v{previous} stays deployed; the gates did not approve a replacement.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--approve-snapshot", action="store_true", required=True)
    parser.add_argument("--alarm-rows", type=int, default=600, help="monitoring-demo --rows")
    parser.add_argument("--alarm-seed", type=int, default=42, help="monitoring-demo --seed")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    config, _ = load_config(CONFIG, ROOT)
    monitoring = yaml.safe_load((ROOT / config["monitoring_config"]).read_text(encoding="utf-8"))
    artifacts = ROOT / "artifacts"
    active = read(artifacts / "deployments/active-model.json")
    quality = read(artifacts / "monitoring/exported/quality.json")
    if quality.get("status") != "alert" or quality.get("model_version") != active["model_version"]:
        print("Latest quality check is not an alert for the deployed model: nothing to retrain.")
        print("Policy: retrain only on a labeled quality alert (concept drift).")
        return 0
    alarm = latest_quality_alert(artifacts, active["model_version"])
    print(
        f"Quality alert {alarm['alert_id']} on v{active['model_version']}: degradation "
        f"{quality['observed']:.3f} > threshold {monitoring['quality_degradation_threshold']}"
    )

    version = build_snapshot(config, active, args)
    dataset_version = read(artifacts / "runs" / active["run_id"] / "ingest.json")["result"][
        "dataset_version"
    ]
    trigger = "quality-" + hashlib.sha256(f"{alarm['alert_id']}:{version}".encode()).hexdigest()
    alert = retraining_alert(
        quality,
        alert_id=alarm["alert_id"],
        trigger_id=trigger[:40],
        dataset_version=dataset_version,
        candidate_evidence={
            "candidate_dataset_version": version,
            "candidate_dataset_approved": True,
            "labels_validated": True,
            "separate_training_validation": True,
            "final_test_excluded": True,
        },
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    alert["threshold"] = monitoring["quality_degradation_threshold"]
    try:
        receipt = submit_retraining(alert, CONFIG, project_root=ROOT)
    except PipelineError as error:
        print(f"Retraining not submitted: {error.code} — {error.message}")
        return 1
    run_id = receipt["pipeline_run_id"]
    print(f"Triggered Airflow run {run_id} ({receipt['state']})")
    state = wait(monitoring["retraining"]["dag_id"], run_id, args.timeout)
    report(run_id, active["model_version"])
    return 0 if state == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
