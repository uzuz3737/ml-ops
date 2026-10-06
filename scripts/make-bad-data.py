"""Write corrupted copies of the real UCI rows for the data-validation demo.

    docker compose exec pipeline-worker python scripts/make-bad-data.py
    docker compose exec pipeline-worker python scripts/airflow-run.py --data-file data/bad-domain.csv

bad-domain.csv          correct types, impossible values: SEX=3, AGE=-5, PAY_0=12,
                        EDUCATION=9 and target 2 (row policy and TFDV both fire)
bad-types.csv           text in LIMIT_BAL, blank PAY_AMT1 cells
bad-missing-column.csv  PAY_AMT6 column dropped
good-sample.csv         the same rows untouched, for contrast

Rows come from a completed run's canonical data.json (default: the run behind
the active deployment), so only the injected faults differ from real data.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlops_project.data.policy import FEATURES, TARGET  # noqa: E402

UCI_HEADER = {TARGET: "default payment next month"}  # as in the original spreadsheet


def source_rows(run_id: str | None) -> pd.DataFrame:
    runs = ROOT / "artifacts" / "runs"
    if not run_id:
        active = ROOT / "artifacts" / "deployments" / "active-model.json"
        if active.exists():
            run_id = json.loads(active.read_text(encoding="utf-8"))["run_id"]
    candidates = [runs / run_id / "data.json"] if run_id else sorted(runs.glob("*/data.json"))
    for path in candidates:
        if path.is_file():
            return pd.read_json(path)[["ID", *FEATURES, TARGET]]
    raise SystemExit("No completed run with data.json; run the pipeline once first.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="run whose data.json to copy (default: active model's)")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    rows = source_rows(args.run_id).sample(n=args.rows, random_state=args.seed)
    rows = rows.sort_values("ID").reset_index(drop=True)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    domain = rows.copy()
    domain.loc[0:9, "SEX"] = 3
    domain.loc[10:19, "AGE"] = -5
    domain.loc[20:29, "PAY_0"] = 12
    domain.loc[30:39, "EDUCATION"] = 9
    domain.loc[40:44, TARGET] = 2

    types = rows.copy().astype(object)
    types.loc[0:9, "LIMIT_BAL"] = "unknown"
    types.loc[10:19, "PAY_AMT1"] = ""

    missing = rows.drop(columns=["PAY_AMT6"])

    files = {
        "bad-domain.csv": domain,
        "bad-types.csv": types,
        "bad-missing-column.csv": missing,
        "good-sample.csv": rows,
    }
    for name, frame in files.items():
        frame.rename(columns=UCI_HEADER).to_csv(out / name, index=False)
        print(f"wrote {args.out_dir}/{name} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
