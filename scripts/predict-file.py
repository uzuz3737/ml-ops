"""Score a test-case file through the live API, one row per request.

    docker compose exec pipeline-worker python scripts/predict-file.py data/test-cases.csv

Accepts CSV, JSON (a list of rows or {"instances": [...]}) and XLS/XLSX with the
UCI headers, the Kaggle variants (PAY_1, default.payment.next.month) or X1..X23.
Each row is sent on its own so one bad row cannot hide the others' results.
A label column, if present, is not sent; it is printed next to the prediction.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlops_project.data.policy import TARGET  # noqa: E402
from mlops_project.data.tables import read_table  # noqa: E402


def post(url: str, body: dict) -> tuple[int, dict]:
    request = Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        try:
            return error.code, json.loads(error.read())
        finally:
            error.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--out", help="also write every response to this JSON file")
    args = parser.parse_args()

    rows = read_table(args.file)

    results, accepted, correct, labelled = [], 0, 0, 0
    print(f"{'row':>4} {'ID':>7} {'status':>6}  result")
    for number, row in enumerate(rows, start=1):
        expected = row.pop(TARGET, None)
        instance = {k: v for k, v in row.items() if v is not None or k != "ID"}
        try:
            status, body = post(args.api.rstrip("/") + "/predict", {"instances": [instance]})
        except URLError:
            print(f"API not reachable at {args.api}; is the api container up?", file=sys.stderr)
            return 2
        results.append({"row": number, "status": status, "response": body})
        if status == 200:
            accepted += 1
            prediction = body["predictions"][0]
            text = (
                f"label={prediction['label']} p(default)={prediction['default_probability']:.4f}"
                f" model=v{body['model_version']}"
            )
            if expected in (0, 1):
                labelled += 1
                correct += prediction["label"] == expected
                text += f" expected={expected}"
        else:
            problems = body.get("error", {}).get("details") or [body.get("error", {})]
            text = "; ".join(
                f"{p.get('field', '-')}: {p.get('rule', p.get('code'))}" for p in problems
            )
        print(f"{number:>4} {str(row.get('ID', '-')):>7} {status:>6}  {text}")

    print(f"\n{accepted}/{len(rows)} rows scored, {len(rows) - accepted} rejected with reasons.")
    if labelled:
        print(f"Agreement with the file's labels: {correct}/{labelled}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
