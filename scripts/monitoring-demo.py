"""Replay held-out monitoring rows through the live API, send their labels, run monitoring.

    docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario healthy
    docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario feature-drift
    docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario concept-drift

healthy        real monitoring rows, real labels
feature-drift  LIMIT_BAL x10 (P1 shifted_fixture), real labels -> data drift alert
concept-drift  same feature distribution, labels inverted -> quality alert, no drift

Rows come from the monitoring partition of the run that produced the active
model, which never entered fitting, tuning or the final test. Each scenario
sends a fresh window of --rows instances, so windows never mix.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_project.data.policy import FEATURES, TARGET  # noqa: E402
from mlops_project.monitoring.feature_drift import shifted_fixture  # noqa: E402
from mlops_project.monitoring.run import run_once  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=("healthy", "feature-drift", "concept-drift"))
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    args = parser.parse_args()

    active = json.loads((ROOT / "artifacts/deployments/active-model.json").read_text())
    pool = pd.read_json(ROOT / "artifacts/runs" / active["run_id"] / "monitoring.json")
    rows = pool.sample(n=min(args.rows, len(pool)), random_state=args.seed)
    if args.scenario == "feature-drift":
        rows = shifted_fixture(rows)
    labels = rows[TARGET].astype(int).tolist()
    if args.scenario == "concept-drift":
        labels = [1 - y for y in labels]
    instances = rows[list(FEATURES)].astype(int).to_dict("records")

    sent = []
    with httpx.Client(base_url=args.api, timeout=30) as client:
        for start in range(0, len(instances), args.batch):
            chunk = instances[start : start + args.batch]
            response = client.post("/predict", json={"instances": chunk})
            response.raise_for_status()
            sent.append((response.json()["request_id"], labels[start : start + args.batch]))
        observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        for request_id, chunk_labels in sent:
            response = client.post(
                "/feedback",
                json={
                    "request_id": request_id,
                    "labels": [
                        {"instance_index": i, "label": y} for i, y in enumerate(chunk_labels)
                    ],
                    "observed_at": observed_at,
                },
            )
            response.raise_for_status()

    summary = run_once("configs/project.yaml", ROOT)
    print(json.dumps({"scenario": args.scenario, "model_version": active["model_version"],
                      "instances": len(instances), **summary}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
