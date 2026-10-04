"""Strict request checks for /predict.

TFDV is too heavy for the request path, so this mirrors the serving
environment of the reviewed schema: all 23 features, no target, no nulls.
Range/category limits are added once P1 freezes credit_default.pbtxt.
"""

from __future__ import annotations

import math

INTEGER_FEATURES = frozenset(
    {"SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}
)


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
    for name in sorted(set(instance) - set(features)):
        problems.append(_issue("unexpected_field", "Field is not a model input.", index, name))
    for name in features:
        if name not in instance:
            problems.append(_issue("missing_field", "Required feature is missing.", index, name))
            continue
        value = instance[name]
        if value is None:
            problems.append(_issue("null", "Null is not allowed for this feature.", index, name))
        elif isinstance(value, bool) or not isinstance(value, int | float):
            problems.append(_issue("type", "Feature must be a JSON number.", index, name))
        elif not math.isfinite(value):
            problems.append(_issue("non_finite", "NaN and infinity are rejected.", index, name))
        elif name in INTEGER_FEATURES and not isinstance(value, int):
            problems.append(_issue("type", "Feature must be an integer code.", index, name))
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
