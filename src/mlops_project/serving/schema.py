"""Strict request checks for /predict.

TFDV is too heavy for the request path. The rules here are P1's serving row
policy (mlops_project.data.policy) spelled out per field, so a client gets the
exact field and reason; validate_rows() still runs last as the source of truth.
"""

from __future__ import annotations

import math

from ..data.policy import DOMAINS, validate_rows

POSITIVE = frozenset({"AGE", "LIMIT_BAL"})
# Rows copied from the UCI file carry their client ID. It is echoed back for
# matching but never reaches the model, exactly as in training.
IDENTIFIER = "ID"


def _finite(value) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:  # ints beyond float range
        return False


def _issue(rule: str, message: str, index: int | None = None, field: str | None = None) -> dict:
    issue = {"rule": rule, "message": message}
    if index is not None:
        issue["instance_index"] = index
    if field is not None:
        issue["field"] = field
    return issue


def check_instance(index: int, instance, features: tuple[str, ...]) -> list[dict]:
    if not isinstance(instance, dict):
        return [_issue("type", "Each instance must be a JSON object.", index)]
    problems = []
    for name in sorted(set(instance) - set(features) - {IDENTIFIER}):
        problems.append(_issue("unexpected_field", "Field is not a model input.", index, name))
    if IDENTIFIER in instance:
        value = instance[IDENTIFIER]
        if isinstance(value, bool) or type(value) is not int or value <= 0:
            problems.append(
                _issue("identifier", "ID must be a positive whole number.", index, IDENTIFIER)
            )
    for name in features:
        if name not in instance:
            problems.append(_issue("missing_field", "Required feature is missing.", index, name))
            continue
        value = instance[name]
        if value is None:
            problems.append(_issue("null", "Null is not allowed for this feature.", index, name))
        elif isinstance(value, bool) or not isinstance(value, int | float):
            problems.append(_issue("type", "Feature must be a JSON number.", index, name))
        elif not _finite(value):
            problems.append(_issue("non_finite", "Value must be a finite number.", index, name))
        elif value != int(value):
            problems.append(_issue("not_integer", "Feature must be a whole number.", index, name))
        elif name in DOMAINS and value not in DOMAINS[name]:
            problems.append(_issue("out_of_domain", "Code is not a known category.", index, name))
        elif name in POSITIVE and value <= 0:
            problems.append(_issue("out_of_range", "Value must be positive.", index, name))
        elif name.startswith("PAY_AMT") and value < 0:
            problems.append(_issue("out_of_range", "Payment cannot be negative.", index, name))
    if not problems and validate_rows(
        [{name: instance[name] for name in features}], training=False
    ):
        problems.append(_issue("policy", "Instance violates the data policy.", index))
    return problems


def check_predict_body(body, features: tuple[str, ...], max_batch: int) -> list[dict]:
    if not isinstance(body, dict):
        return [_issue("envelope", "Body must be an object with an instances list.")]
    extra = sorted(set(body) - {"instances"})
    if extra:
        return [_issue("envelope", f"Unexpected top-level keys: {', '.join(extra)}.")]
    instances = body.get("instances")
    if not isinstance(instances, list) or not instances:
        return [_issue("envelope", "instances must be a nonempty list.")]
    if len(instances) > max_batch:
        return [_issue("batch_size", f"At most {max_batch} instances per request.")]
    problems = []
    for index, instance in enumerate(instances):
        problems.extend(check_instance(index, instance, features))
    return problems
