"""Assess one explicitly locked initial candidate on the protected final test."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_project.artifacts import read_record  # noqa: E402
from mlops_project.pipelines.contracts import (  # noqa: E402
    configuration_snapshot,
    confined_path,
    load_config,
    validate_run_id,
)
from mlops_project.training.pipeline import evaluate_final_test  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--locked-candidate-sha256", required=True)
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()
    validate_run_id(args.run_id)
    root = Path(__file__).resolve().parents[1]
    config, path = load_config(args.config, root)
    config_hash, _ = configuration_snapshot(config, path, root)
    directory = confined_path(config["pipeline"].get("artifact_root", "artifacts"), root)
    run = directory / "runs" / args.run_id
    record = read_record(run / "evaluate.json")
    if record["state"] != "succeeded" or record["config_sha256"] != config_hash:
        raise ValueError("Require a successful evaluation with unchanged configuration")
    context = {
        "run_id": args.run_id,
        "run_dir": str(run),
        "project_root": str(root),
        "config": config,
        "config_sha256": config_hash,
    }
    result = evaluate_final_test(
        context, record["result"], locked_candidate_sha256=args.locked_candidate_sha256
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
