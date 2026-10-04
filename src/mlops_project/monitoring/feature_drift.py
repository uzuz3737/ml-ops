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
    metrics = _psi_metrics(reference, current)
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


def _psi_metrics(reference, current):
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
    return metrics


def shifted_fixture(frame):
    """Synthetic feature shift; labels and IDs remain untouched (not concept drift)."""
    result = frame.copy()
    result["LIMIT_BAL"] = result["LIMIT_BAL"] * 10
    return result


def calibrate_threshold(reference, *, window_rows=200, trials=100, quantile=0.99, seed=42):
    """Empirical null bootstrap of maximum feature PSI; reference must be train-only.

    Calibrates a demo window policy, not a production false-alarm guarantee.
    Future windows and final test must never be supplied to this function.
    """
    if type(window_rows) is not int or window_rows < 2 or len(reference) < 2 * window_rows:
        raise ValueError("Reference needs at least twice the calibration window size")
    if type(trials) is not int or trials < 20 or not 0.9 <= quantile < 1:
        raise ValueError("Use at least 20 trials and a quantile in [0.9, 1)")
    rng = np.random.default_rng(seed)
    # Bootstrap windows contain only previously validated reference rows.
    feature_drift(reference, reference, minimum_rows=window_rows)
    scores = []
    for _ in range(trials):
        current = reference.iloc[rng.integers(0, len(reference), size=window_rows)]
        scores.append(max(_psi_metrics(reference, current).values()))
    threshold = max(float(np.quantile(scores, quantile, method="higher")), 1e-6)
    return {
        "contract_version": 1,
        "method": "train-reference-bootstrap-max-feature-psi",
        "reference_rows": len(reference),
        "window_rows": window_rows,
        "trials": trials,
        "quantile": quantile,
        "seed": seed,
        "threshold": threshold,
        "null_scores": scores,
        "scope": "empirical-demo-calibration",
    }
