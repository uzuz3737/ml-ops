# Serving and monitoring (P3)

Owner: **@thanachaithongbai-hue**. Covers requirement rows S1–S5, O1, the export half of O2 and the serving half of E3 in [Requirements](REQUIREMENTS.md). Contracts are in [Interface contracts](INTERFACE_CONTRACTS.md); P0's delivery protocol is in [P0 handoff](P0_HANDOFF.md).

## What runs where

| Piece | Module | Runs in |
| --- | --- | --- |
| Prediction API | `mlops_project.serving.app:app` | `api` container, port 8000 |
| Deployment watcher (desired manifest → load → ACK) | `serving/watcher.py` | background thread inside the API |
| Candidate benchmark stage | `mlops_project.serving.pipeline:benchmark` | pipeline worker; starts its own uvicorn on a free local port |
| Monitoring pass (P1 drift + P2 quality on API events) | `python -m mlops_project.monitoring.run` | pipeline worker |
| Drift/quality → Prometheus | `monitoring/exporters.py` | read by the API on every `/metrics` scrape |
| Alert rules | `infra/prometheus/alerts.yml` | Prometheus |
| Dashboard | `infra/grafana/dashboards/serving.json` | Grafana, folder "Credit Default" |

The API never fits anything. It loads P2's `bundle.joblib` (`{"pipeline", "manifest"}`) only after checking the approved gate report, the artifact SHA-256, the schema version, the feature list and that Python/scikit-learn/pandas/numpy/joblib match the training environment. `requirements/serving.lock` pins the same versions as `src/mlops_project/training/requirements.lock` for that reason.

## Calling the API

```powershell
curl.exe -s http://localhost:8000/health
curl.exe -s http://localhost:8000/ready
curl.exe -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data "@examples/predict.json"
curl.exe -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data "@examples/predict-batch.json"
```

A response carries the exact loaded registry version and one prediction per instance, in order. `default_probability` is always for class 1 (default) and `label` applies the threshold stored in the bundle:

```json
{"request_id": "ec1c71d133474224938c636458243513", "model_version": "2",
 "predictions": [{"label": 0, "default_probability": 0.2765}, {"label": 1, "default_probability": 0.5248}]}
```

Delayed outcomes go to `/feedback`; `observed_at` needs a timezone and cannot precede the prediction:

```json
{"request_id": "ec1c71d133474224938c636458243513",
 "labels": [{"instance_index": 1, "label": 1}], "observed_at": "2026-11-04T00:00:00Z"}
```

### Abnormal input

Request checks follow P1's serving row policy (`mlops_project.data.policy`): all 23 features, whole numbers only, known category codes, positive `AGE`/`LIMIT_BAL`, non-negative `PAY_AMT*`; negative bill amounts and the undocumented `EDUCATION`/`MARRIAGE` codes found in UCI 350 are accepted. One bad instance rejects the whole batch and nothing is logged for drift.

| Status | When | Example file |
| --- | --- | --- |
| 400 | Body is not JSON | — |
| 413 | Body larger than `MLOPS_MAX_BODY_BYTES` (1 MB) | — |
| 422 | Missing feature, null, string number, unknown code, negative payment, target in input, empty or oversized batch | `examples/invalid/*.json` |
| 503 | No verified model loaded (`/health` stays 200) | — |

Each 422 lists `{rule, field, instance_index, message}` per problem, so the instructor's test cases can be explained live.

## Why real-time serving (S4)

The stakeholder use is a credit officer or an approval workflow asking about one client at the moment a limit or collection decision is made. That needs an answer in well under a second per request, the current model version for audit, and immediate rejection of malformed records — an online API. Batch scoring of the whole book is still possible through the same endpoint (up to `MLOPS_MAX_BATCH_SIZE`, default 1000 instances), and every scored request is logged for monitoring and delayed labels.

## Measured performance (S2, S3)

Measured by the `benchmark` stage of Airflow run `full-run-002` on the candidate it later deployed (model version 2, RandomForest), in its own uvicorn process inside the pipeline worker:

| Metric | SLO (`quality_gates.yaml`) | Measured | Result |
| --- | --- | --- | --- |
| p50 latency | — | 167.1 ms | — |
| p95 latency | ≤ 300 ms | 204.5 ms | pass |
| Throughput | ≥ 20 req/s | 60.0 req/s | pass |
| Error rate | ≤ 1% | 0% (0 of 17,998) | pass |

Workload: batch size 1, 10 concurrent clients, 300 s. Hardware: Docker Desktop WSL2 Linux 6.18, x86_64, 28 logical CPUs. Evidence: `artifacts/runs/full-run-002/benchmark-report.json` and `gate-report.json`.

The first full run (`full-run-001`) measured p95 318 ms and the gate correctly rejected that candidate. Profiling showed the model itself takes about 8 ms per row; the time went to ten threads running `predict_proba` at once and contending for the GIL. Scoring one batch at a time behind a lock brought p95 from 320 to 204 ms and throughput from 38 to 60 req/s without changing any prediction.

## Health, metrics and logs (S5, O1)

- `/health` is liveness only; `/ready` is 200 with `model_version`/`schema_version` only after a verified load, otherwise 503.
- `/metrics` exposes request counts by endpoint/status, a latency histogram with buckets around the 300 ms SLO, batch sizes, rejected inputs by rule, predictions by label, the probability distribution, watcher deployments by action/result, `model_ready`, plus the exported drift and quality gauges. No request IDs, model versions or raw features are used as labels.
- Logs are one JSON object per line with event, deployment ID, action, model version and safe error code.
- Grafana dashboard "Credit Default — serving and monitoring" shows all of the above.

## Monitoring pass and alerts (O1, O2)

```powershell
docker compose exec pipeline-worker python -m mlops_project.monitoring.run --config configs/project.yaml
```

It takes the newest `minimum_samples` (500) prediction events of the confirmed active model and

- runs P1 `feature_drift` against the train partition of the run that produced the model, with `data_drift_threshold` 0.0662 (P1 calibration on UCI train, 500-row windows, 99th percentile of 100 bootstrap windows);
- runs P2 `evaluate_quality` on the joined labels against the bundle's own validation metrics, with `quality_degradation_threshold` 0.123044 and the minimum label/class/coverage rules from `configs/monitoring.yaml`.

Both results are published for `/metrics`; every check that fires also writes `artifacts/monitoring/alerts/<alert_id>.json` with the statistic, observed value, threshold and suggested action. Prometheus rules in `infra/prometheus/alerts.yml`: `ApiDown`, `ModelNotReady`, `PredictErrorRateHigh`, `PredictLatencyP95High`, `InvalidInputSpike` (rate-based, not one alert per bad request), `DeploymentFailed`, `FeatureDrift`, `ModelQualityDegraded`.

### Live demo

`scripts/monitoring-demo.py` replays 600 rows from the monitoring partition of the active model's run (never used for fitting, tuning or the final test) through `/predict`, posts their labels to `/feedback` and runs one monitoring pass:

```powershell
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario healthy
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario feature-drift
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario concept-drift
```

Results against model version 2 on 4 October 2026:

| Scenario | What changes | Data drift | Labeled quality | Alert file |
| --- | --- | --- | --- | --- |
| healthy | nothing | ok | ok | — |
| feature-drift | `LIMIT_BAL` × 10, labels unchanged | **alert** | ok | `data_drift-*.json` |
| concept-drift | labels inverted, features unchanged | ok | **alert** (degradation 0.537 > 0.123) | `quality-*.json` |

Prometheus then reported `ModelQualityDegraded` as firing. Both drift scenarios are controlled replays of historical data, not observed future drift.

## Deployment, ACK and rollback (E3)

The watcher polls `artifacts/deployments/desired-model.json`. For `deploy`/`rollback` it verifies the passing gate report and artifact checksum, smoke-predicts `examples/predict.json` while the old model keeps serving, swaps atomically and writes `acknowledgements/<deployment_id>.json` with `status: loaded`. A failed load leaves the current model in place and ACKs `failed`. For P0's first-deployment `unload` tombstone it clears the model and ACKs `unloaded`. On restart it reloads the current desired manifest.

```powershell
docker compose exec pipeline-worker python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "demo rollback"
```

Done on 4 October 2026 after `full-run-003` deployed version 3 over version 2: the rollback CLI wrote a `rollback` manifest, the watcher loaded version 2 and ACKed, and `/ready` and `/predict` switched from `"model_version": "3"` to `"2"` without retraining (deployment `rollback-db30e0a9…`, audit under `artifacts/deployments/history/`).

## Tests

`tests/serving` (API contract, watcher, real P2 bundle end to end, benchmark on a real bundle), `tests/monitoring/test_exporters.py`, `tests/monitoring/test_run.py`. CI runs them in the "Serving API, watcher and benchmark" check with `requirements/serving.lock` installed.
