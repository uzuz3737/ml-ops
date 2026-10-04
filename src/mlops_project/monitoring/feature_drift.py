"""Reference-binned PSI; demo thresholds require operational calibration."""

import numpy as np

from mlops_project.data.policy import DOMAINS, FEATURES, validate_rows


def feature_drift(reference, current, *, minimum_rows=200, threshold=0.2):
    if minimum_rows < 1 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid drift policy")
    for frame in (reference, current):
        if validate_rows(frame[list(FEATURES)].to_dict("records"), training=False):
            raise ValueError("Invalid drift window")
    if min(len(reference), len(current)) < minimum_rows:
        return {"contract_version": 1, "status": "insufficient_samples", "alert": False}
    metrics = {}
    for name in FEATURES:
        if name in DOMAINS:
            categories = DOMAINS[name]
            a = np.array([(reference[name] == v).sum() for v in categories])
            b = np.array([(current[name] == v).sum() for v in categories])
        else:
            inner = np.unique(np.quantile(reference[name], np.linspace(0, 1, 11)))
            bins = np.r_[-np.inf, inner, np.inf]
            a, b = np.histogram(reference[name], bins)[0], np.histogram(current[name], bins)[0]
        p = (a + 0.5) / (a.sum() + 0.5 * len(a))
        q = (b + 0.5) / (b.sum() + 0.5 * len(b))
        metrics[name] = float(np.sum((q - p) * np.log(q / p)))
    return {
        "contract_version": 1,
        "method": "reference-decile-psi-additive-0.5",
        "reference_rows": len(reference),
        "current_rows": len(current),
        "minimum_rows": minimum_rows,
        "threshold": threshold,
        "metrics": metrics,
        "alert": any(v >= threshold for v in metrics.values()),
        "status": "measured",
    }


def shifted_fixture(frame):
    """Synthetic feature shift; labels and IDs remain untouched (not concept drift)."""
    result = frame.copy()
    result["LIMIT_BAL"] = result["LIMIT_BAL"] * 10
    return result
