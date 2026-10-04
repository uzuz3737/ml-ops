"""Strict lightweight row policy shared by training and serving."""

import math
from numbers import Real

FEATURES = (
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *(f"BILL_AMT{i}" for i in range(1, 7)),
    *(f"PAY_AMT{i}" for i in range(1, 7)),
)
TARGET = "default_next_month"
DOMAINS = {
    "SEX": (1, 2),
    "EDUCATION": tuple(range(7)),
    "MARRIAGE": tuple(range(4)),
    **dict.fromkeys(FEATURES[5:11], tuple(range(-2, 10))),
    TARGET: (0, 1),
}


def validate_rows(records, *, training=True):
    """Reject coercions, nulls and invalid domains; report safe row positions only."""
    required = set(FEATURES) | ({"ID", TARGET} if training else set())
    failures, ids = [], set()
    if not isinstance(records, list):
        return [{"rule": "records_must_be_list"}]
    for index, row in enumerate(records):
        if not isinstance(row, dict) or set(row) != required:
            failures.append({"row": index, "rule": "columns"})
            continue
        for name, value in row.items():
            valid = isinstance(value, Real) and not isinstance(value, bool)
            try:
                valid = valid and math.isfinite(value) and value == int(value)
            except (OverflowError, ValueError):
                valid = False
            if valid and name in DOMAINS:
                valid = value in DOMAINS[name]
            if valid and name in {"ID", "AGE", "LIMIT_BAL"}:
                valid = value > 0
            if valid and name.startswith("PAY_AMT"):
                valid = value >= 0
            if not valid:
                failures.append({"row": index, "field": name, "rule": "type_or_domain"})
        if training and isinstance(row["ID"], Real) and not isinstance(row["ID"], bool):
            if row["ID"] in ids:
                failures.append({"row": index, "field": "ID", "rule": "duplicate"})
            ids.add(row["ID"])
    if not records:
        failures.append({"rule": "empty_dataset"})
    return failures
