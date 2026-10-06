"""Run P2 stages using a saved P0-style context; no data/serving substitutes."""

import argparse
import json
from pathlib import Path

from mlops_project.artifacts import atomic_json
from mlops_project.training import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[*pipeline.EXPERIMENTS, "evaluate", "register", "final-test"],
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--locked-candidate-sha256")
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    if args.stage == "final-test":
        result = pipeline.evaluate_final_test(
            context,
            context["inputs"]["evaluate"],
            locked_candidate_sha256=args.locked_candidate_sha256,
        )
    elif args.stage == "register":
        from mlops_project.registry.pipeline import register

        result = register(context)
    else:
        result = getattr(pipeline, args.stage)(context)
    atomic_json(args.output, result)
    print(f"{args.stage}: saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
