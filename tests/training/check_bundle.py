"""Verify trusted bundle prediction parity in a consumer image."""

import argparse
import json
from pathlib import Path

import numpy as np

from mlops_project.training.bundle import load_bundle, predict_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--schema-version", required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle, args.checksum, args.schema_version)
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    predictions = predict_bundle(bundle, smoke["instances"])
    np.testing.assert_allclose(
        [p["default_probability"] for p in predictions],
        smoke["default_probabilities"],
        atol=smoke["tolerance"],
        rtol=0,
    )
    print("Trusted bundle checksum, schema, environment and prediction parity passed.")


if __name__ == "__main__":
    main()
