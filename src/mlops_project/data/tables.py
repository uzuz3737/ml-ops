"""Read an operator-supplied credit table into canonical row dicts.

Shared by the pipeline's data_file ingest and scripts/predict-file.py, so a file
the pipeline accepts is parsed the same way when it is sent to the API.

Values are kept as the file states them: only unambiguous numerals become
numbers and blanks become None, so the row policy and TFDV see the real faults
instead of pandas' coercions.
"""

import json
import math
from pathlib import Path

import pandas as pd

from mlops_project.data.policy import FEATURES, TARGET

# Headers found in copies of UCI 350: the original spreadsheet, the Kaggle CSV
# (PAY_1, dotted target) and ucimlrepo's generic X1..X23/Y names.
GENERIC_COLUMNS = {f"X{i}": name for i, name in enumerate(FEATURES, start=1)} | {"Y": TARGET}
TARGET_ALIASES = {"default payment next month", "default next month"}
SUFFIXES = (".csv", ".json", ".xls", ".xlsx")


def canonical_column(name) -> str:
    text = str(name).strip()
    if text.lower().replace(".", " ").replace("_", " ") in TARGET_ALIASES:
        return TARGET
    return GENERIC_COLUMNS.get(text, text)


def cell(value):
    """Blank -> None, numeral text -> int/float, anything else unchanged."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        return value.item() if hasattr(value, "item") else value
    text = value.strip()
    if not text:
        return None
    for parse in (int, float):
        try:
            number = parse(text)
        except ValueError:
            continue
        return number if math.isfinite(number) else text
    return text


def _frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("instances", value.get("records"))
        if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
            raise ValueError("JSON data must be a list of row objects")
        return pd.DataFrame(value, dtype=object)
    if suffix not in SUFFIXES:
        raise ValueError(f"Unsupported table format {suffix!r}")
    raw = pd.read_excel(path, header=None, dtype=object)
    # The UCI spreadsheet has a generic X1..Y row above the descriptive header.
    header = next(
        (i for i in range(min(3, len(raw))) if "LIMIT_BAL" in set(raw.iloc[i].astype(str))), 0
    )
    frame = raw.iloc[header + 1 :].reset_index(drop=True)
    frame.columns = raw.iloc[header]
    return frame


def read_table(path: str | Path) -> list[dict]:
    """Rows with canonical column names; raises on files that cannot be parsed at all."""
    frame = _frame(Path(path)).rename(columns=canonical_column)
    if "PAY_1" in frame.columns and "PAY_0" not in frame.columns:
        frame = frame.rename(columns={"PAY_1": "PAY_0"})
    return [{str(k): cell(v) for k, v in row.items()} for row in frame.to_dict("records")]
