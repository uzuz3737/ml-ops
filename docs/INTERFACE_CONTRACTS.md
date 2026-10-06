# Interface contracts

P0–P3 components implement these boundaries. Exact adapter names are in `configs/project.yaml`; executable validation lives in `pipelines/contracts.py`, data policy and API schema. The selected stack is Docker, TFDV, MLflow tracking/registry, Airflow, FastAPI, Prometheus + Grafana, and GitHub Actions. Review coordinated contract changes and retain artifact/version evidence.

See [Architecture](ARCHITECTURE.md), [team runbook](TEAM_RUNBOOK_TH.md), and [evidence](evidence/INDEX.md). Stakeholder approval and external course acceptance still require team evidence.

## Decisions that must precede implementation

| Contract decision | Where to record it | Required result |
| --- | --- | --- |
| Task and user value | `PROJECT_BRIEF.md`, `configs/project.yaml` | Credit-card default classification, confirmed stakeholder, business metric, target/prediction meaning |
| Feature and label types | `schemas/credit_default.pbtxt`, referenced by `configs/data_schema.yaml` | Canonical names/types, required fields, reviewed ranges/categories, missing/outlier policy |
| Splitting | `configs/project.yaml` | Stratified 60/20/10/10 train/validation/final-test/replay; seed `42`; ID-disjoint |
| Prediction output | API schema and model signature | Binary default label and default probability per instance; threshold fixed per model |
| Model acceptance | `configs/quality_gates.yaml` | Optimizing metric/direction, absolute minimum quality, comparison rule, service gates |
| Monitoring | `configs/monitoring.yaml` | Reference/current windows, minimum rows/labels, thresholds, cooldown, retraining criteria |
| SLO and load | `configs/project.yaml` | p95 latency, error-rate/throughput targets, batch limit, workload, hardware |
| Privacy and data rights | `PROJECT_BRIEF.md` | Data origin/license, permitted storage, event fields, retention policy |

Use the canonical feature names defined below consistently even if UCI's loader returns `X1`–`X23`. These are planned examples with real feature names; they become callable only after the API is implemented. Numerical quality/SLO gates and business assumptions still need agreement and measurements.

## Credit dataset schema

The dataset is [UCI Default of Credit Card Clients, dataset 350](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients), with 30,000 clients and 23 predictors. Normalize the source target to `default_next_month`: integer `1` is default, `0` is no default. `ID` is provenance only and is excluded from model and request features. The original six-month history covers April–September 2005. Full mapping, attribution, profiling decisions, and limitations belong in [Dataset plan](DATASET.md).

| Canonical model/API feature | Source alias | Expected semantic type |
| --- | --- | --- |
| `LIMIT_BAL` | `X1` | Numeric credit amount in NT dollars |
| `SEX` | `X2` | Integer categorical code |
| `EDUCATION` | `X3` | Integer categorical code |
| `MARRIAGE` | `X4` | Integer categorical code |
| `AGE` | `X5` | Integer years |
| `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` | `X6`–`X11` | Integer repayment-status codes; preserve source naming |
| `BILL_AMT1`–`BILL_AMT6` | `X12`–`X17` | Numeric bill amounts in NT dollars |
| `PAY_AMT1`–`PAY_AMT6` | `X18`–`X23` | Numeric previous-payment amounts in NT dollars |

The source declares no missing values; initial validation rejects null/missing features. Profile actual category/repayment codes before freezing enums: the published descriptions do not necessarily list every raw code. Do not reject legitimate negative bill balances or redefine special repayment codes without a documented audit. Set plausible hard constraints separately from statistical outlier flags; rare values within the legitimate range are not automatically bad data.

Initial serving accepts all 23 predictors. Excluding sensitive attributes such as `SEX`, `EDUCATION`, or `MARRIAGE` from the model requires versioned internal feature selection, an updated bundle and slice-quality review; the public API may retain its stable 23-input contract. Removing fields from the public request requires a coordinated API/schema contract revision. These data alone do not establish applicability to a contemporary bank or new population.

## General rules

- Use UTF-8 text, JSON-compatible values, and UTC timestamps with a timezone suffix in stored events; display local dates in documentation where relevant.
- Use a unique `run_id` for each pipeline execution, `request_id` for each prediction request, and immutable dataset/model version identifiers.
- Every artifact includes `contract_version`; a component must reject an unsupported version explicitly.
- SHA-256 checksums identify exact data/artifact content. A path or Git branch name alone is not a version.
- Input schema and trained feature order are immutable within a model bundle. Changes require a new schema/model version and compatible deployment checks.
- Generated data, models, secrets, logs, and bulky evidence are ignored by Git unless a small safe fixture or reviewed evidence summary is intentionally committed.
- Validation, training, serving, and monitoring must agree on semantic types. Do not silently turn numeric strings into numbers or unknown categories into valid labels unless the approved schema explicitly allows that conversion.

## Data and split artifacts

`src/mlops_project/data/` produces a dataset manifest, TFDV statistics/schema/anomaly artifacts, a validation report, and a split manifest. Proposed file formats are JSON metadata and Parquet for tabular data; CSV is acceptable when parser rules and dtypes are explicit. Preserve the original UCI download checksum and record any spreadsheet/loader column normalization.

| Dataset manifest field | Contract |
| --- | --- |
| `contract_version`, `dataset_version` | Nonempty stable identifiers |
| `source`, `license` | Origin and allowed use; include retrieval/generation details |
| `content_sha256`, `schema_version` | Exact data identity and validation schema identity |
| `row_count`, `feature_columns`, `target_column` | Counts and agreed names; target is excluded from serving features |
| `created_at`, `ingest_code_commit` | Provenance |
| `data_uri` | Portable artifact location resolved through configuration |

The split manifest records dataset version, stratification method, seed `42`, train/validation/final-test/replay counts, and row-ID lists or their checksums. Proposed proportions are 60/20/10/10 (18,000/6,000/3,000/3,000 if all original rows survive validation). All client IDs are disjoint. The six monthly fields are features within each client record, not evidence that row order is a chronological split. Keep final test and monitoring replay examples out of all fitting/tuning. Version any cleaning that changes counts.

TFDV validates statistics against canonical `schemas/credit_default.pbtxt` with `TRAINING`/`SERVING` environments. The target is required in `TRAINING` and excluded from `SERVING`; provenance IDs are not predictor columns. `configs/data_schema.yaml` references the schema and raw-column mapping rather than duplicating a competing schema. Store the schema, statistics protobuf, and anomalies alongside the JSON report. A draft inferred from training statistics must be reviewed and frozen, not regenerated to accommodate every bad batch. `validate_statistics` returns anomalies; inspect `anomaly_info` and explicitly fail the task for configured hard anomalies. Aggregate checks need supplemental row-level rules for strict types, IDs, and non-finite values. [TFDV schema and validation](https://www.tensorflow.org/tfx/data_validation/get_started)

Validation returns a structured report with `run_id`, `dataset_version`, `schema_version`, `passed`, row count, artifact URIs/checksums, and failures grouped by field/rule. Failure entries identify safe row IDs, counts, and reasons without unnecessarily exposing raw sensitive values. Hard failures emit a persisted alert and raise a typed pipeline error; returning a report with `passed: false` while continuing training violates the contract.

| Input condition | Default behavior; customize only by approved schema |
| --- | --- |
| Required feature key/column absent | Fail |
| Present numeric feature with `null` and `imputable: true` in a reviewed later schema | Accept; imputer is fitted on training data; initial source contract rejects nulls |
| Null in non-imputable feature | Fail |
| Wrong type, infinity, NaN in JSON, invalid range/category | Fail |
| Outlier within valid schema range | Apply documented feature policy; learned bounds use training data only |
| Missing or invalid training label | Fail; do not impute labels |
| Target or unexpected feature in prediction input | Reject by default to expose schema mismatch/leakage |

Serving and training share the canonical feature policy through generated validators and separate schema environments: training requires the target and permitted metadata; prediction requires only the 23 features. Do not import the heavy TFDV stack into the API's per-request path; use a strict request validator generated from the same reviewed policy and periodically run TFDV on persisted serving windows. Monitoring metadata must never become a model feature by accident.

## Shared feature/model bundle

`src/mlops_project/features/` defines the pipeline builder. `training/` fits it; `serving/` loads the resulting fitted pipeline. The persisted bundle contains the transformer and estimator together, so raw validated features pass through exactly the transformations used for training. Do not fit a new scaler, encoder, imputer, selector, or estimator in the serving process. This follows [scikit-learn preprocessing and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

| Bundle item | Required content |
| --- | --- |
| Fitted pipeline artifact | Complete persisted preprocessing + model, with checksum |
| `manifest.json` | Contract/schema version, task, feature order/types, prediction type, model identity |
| Provenance | Git commit, dataset/split IDs, run ID, experiment configuration, dependency lock hash |
| Evaluation evidence | Validation metrics, comparison and gate report; final-test report only for the locked initial candidate. Retrained bundles use separate current-regime validation evidence, without reevaluating the protected final test. |
| Environment | Locked libraries/runtime, image digest when built, training hardware details |
| Smoke fixture | Safe raw input and expected prediction shape/type; numeric tolerance if needed |

Load only artifacts produced by the trusted project pipeline and verify their checksum and environment compatibility. Serializer-specific loading must be tested in the deployment image with the pinned dependencies. Reproducibility means the same inputs/configuration produce the same splits and equivalent predictions within a declared numeric tolerance; do not promise byte-identical serialized files when the library does not guarantee them.

## Training, evaluation, and gate contracts

Each experiment consumes immutable split manifests and a versioned configuration and returns an experiment result:

```text
contract_version, run_id, experiment_id, code_commit,
dataset_version, split_manifest_sha256, schema_version,
hyperparameters, random_seed, metrics, artifact_uri,
artifact_sha256, environment_uri, status
```

The result records all six required categories: code version, data version, hyperparameters, metrics, output artifacts, and environment. At least three comparable runs include a simple baseline. Proposed optimizing metric is positive-class (`1`) `average_precision`, higher is better; compute AP using its named definition rather than labeling a different PR-AUC calculation as AP. Report ROC AUC, recall, precision, F1, calibration/Brier score, and sample counts. Choose the classification threshold on validation data and persist it with the model; do not optimize it on test/replay labels. A comparison table uses the same split and metric definitions, with the selected run and written reason. Protect the test set for the locked final candidate.

The gate report is a JSON object with `candidate_model_version`, `code_commit`, `dataset_version`, `schema_version`, `artifact_sha256`, gate configuration checksum, timestamp, and a list of checks. Each check contains `name`, `threshold`, `observed`, `passed`, and linked evidence. Before approval, P3's isolated candidate-serving harness loads the exact bundle in the intended API image for real performance/smoke checks; P0 integrates this DAG step without changing the champion. Required checks cover:

- Schema and feature/output compatibility.
- Optimizing metric above/below the agreed absolute threshold according to its direction.
- Baseline/current-model comparison under the agreed regression tolerance; evaluate candidates and comparison models on the same permitted holdout/window.
- p50/p95 latency, throughput, error rate, artifact/resource size where configured, against a stated workload and SLO.
- Code quality, data correctness, and model-quality CI results linked to the exact candidate provenance.

`passed` is true only if every required gate passes. Missing measurements, unsupported schema, or stale evidence fail closed. A rejected candidate is retained for diagnosis and never replaces the current model. Numerical thresholds remain undecided until the team measures the credit-default baseline; no claim of passing gates is made in these docs. The protected final test is for the locked initial candidate's final report, not recurring retraining/promotion selection.

## Registry and deployment records

`src/mlops_project/registry/` stores immutable model versions plus application lifecycle metadata: `candidate → validated → approved → deployed`, with `rejected`, `retired`, and `rolled_back` outcomes. Use tags for the application state and `champion` / `previous` aliases or an equivalent persisted mapping. This design uses MLflow's alias/tag workflow and a database-backed registry; old registry stages are deprecated. [MLflow registry documentation](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

Approval records include candidate version, gate-report checksum, approver identity (person or versioned policy), time, and decision/reason. Deployment records include deployment ID, exact version/artifact checksum, previous version, image digest, health/smoke result, time, and rollback outcome where applicable.

P0's deploy operation is idempotent for the same deployment ID/version and serialized across competing changes. Its control channel is a shared-volume manifest watcher. P0 writes `artifacts/deployments/desired-model.json` by replacing a temporary file on the same filesystem. It contains `contract_version: 1`, `action: deploy` or `rollback`, `deployment_id`, registered model name, exact version, schema version, artifact URI/checksum, approval identity, gate-report project-relative URI/checksum, and run/code/data/config provenance. P3's API watcher validates this approved record and trusted artifact, stages it while the old model serves, atomically switches its active bundle, and writes `artifacts/deployments/acknowledgements/<deployment_id>.json` with integer `contract_version: 1`, `status: loaded`, matching deployment ID and model version, or a safe error. P0 waits with a timeout for that acknowledgement, `/ready` model/schema version, and valid exact-version prediction before calling P2's registry-finalization callback and recording success.

Each in-flight request retains the bundle it started with. P0 writes `active-model.json` only after confirmed serving and registry finalization; `previous-model.json` retains the prior confirmed healthy approved bundle. If acknowledgement/probes/registry updates fail, atomically request that prior model with a new rollback deployment ID, wait for matching prior-version acknowledgement/readiness/prediction, restore mappings, emit an alert, and report the original deployment failure. A desired manifest alone is not an eligible rollback target.

On first-deployment failure without a prior healthy artifact, P0 writes an unload tombstone: `contract_version: 1`, `action: unload`, a new `deployment_id`, `status: unavailable`, and `reason: first_deployment_failed`. P3 clears the model, writes ACK with `contract_version: 1` and `{status: unloaded, deployment_id}`, keeps `/health` at 200 and `/ready` at 503. P0 verifies those outcomes and calls P2 finalization with `action: unload`, `model_version: null` to clear aliases; the callback returns `contract_version: 1`, `passed: true`, `model_version: null`. If recovery cannot be confirmed, report recovery failure. Reconcile registry mapping and actual runtime version after restart. Restrict desired-manifest writes to P0's controller and acknowledgement writes to the API; no public reload endpoint is needed.

Changing an alias alone does not fulfill deployment. Acceptance requires `/ready` and `/predict` to return the expected exact version. Rollback restores a previously approved compatible artifact without retraining and records which failed deployment was undone. See [Architecture](ARCHITECTURE.md) for the sequence and scope limits.

The executable callback fields and rollback CLI are in [P0 handoff](P0_HANDOFF.md). Real loader, Registry and recovery behavior still require P2/P3 components and container acceptance evidence.

## Prediction API

The proposed API binds port `8000` and publishes it through Compose. Real request examples and commands will be added to [Getting started](GETTING_STARTED.md) when implemented. OpenAPI should expose the agreed concrete types.

| Endpoint | Success contract | Failure behavior |
| --- | --- | --- |
| `POST /predict` | `200`; validate all instances; one ordered prediction per instance | Reject the entire request on invalid input; never return partial success silently |
| `GET /health` | `200`, `{"status":"alive"}` if process can respond | Liveness is independent of model readiness |
| `GET /ready` | `200`, `{"status":"ready","model_version":"…","schema_version":"…"}` after bundle load/verification | `503` when no verified compatible model is loaded |
| `GET /metrics` | `200`; Prometheus text exposition for service/monitoring measurements | Record scrape failures in monitoring |
| `POST /feedback` | `202`; validated labels persisted for later joins | Input/join errors must be explicit; feedback cannot directly promote or mutate a model |

Prediction envelope:

```json
{
  "instances": [
    {
      "LIMIT_BAL": 200000,
      "SEX": 1,
      "EDUCATION": 2,
      "MARRIAGE": 1,
      "AGE": 35,
      "PAY_0": -1,
      "PAY_2": -1,
      "PAY_3": -1,
      "PAY_4": -1,
      "PAY_5": -1,
      "PAY_6": -1,
      "BILL_AMT1": 10000,
      "BILL_AMT2": 9000,
      "BILL_AMT3": 8000,
      "BILL_AMT4": 7000,
      "BILL_AMT5": 6000,
      "BILL_AMT6": 5000,
      "PAY_AMT1": 1000,
      "PAY_AMT2": 1000,
      "PAY_AMT3": 1000,
      "PAY_AMT4": 1000,
      "PAY_AMT5": 1000,
      "PAY_AMT6": 1000
    }
  ]
}
```

Response envelope; the probability/label below are illustrative, not measured model output:

```json
{
  "request_id": "unique-request-id",
  "model_version": "exact-loaded-registry-version",
  "predictions": [{"label": 1, "default_probability": 0.73}]
}
```

The API is not implemented yet. `instances` must be nonempty and no larger than the configured maximum batch size. `predictions.length == instances.length`, order is unchanged, and one bundle version serves the whole batch. Each prediction has an integer `label` in `{0,1}` and finite `default_probability` in `[0,1]`, where the probability always refers to positive/default class `1`. Apply the bundle's validation-selected threshold consistently; verify the estimator's class ordering rather than assuming probability-column order.

| HTTP status | Condition |
| --- | --- |
| `400` | Malformed JSON syntax |
| `413` | Payload exceeds configured byte limit |
| `422` | Valid JSON with invalid envelope, missing required feature, wrong type, forbidden null/category/range, or oversized instance count |
| `503` | No ready model, or required inference resource is unavailable |
| `500` | Unexpected internal failure; log details internally and return a safe message |

Use one documented error shape, for example `{"request_id":"…","error":{"code":"schema_violation","message":"…","details":[]}}`. Include field paths and instance indexes when useful. Override framework defaults as needed to achieve the documented status distinctions, and test them. Invalid inputs must not produce predictions or be included as valid feature samples in drift calculations.

## Prediction events and delayed labels

Prediction events contain `contract_version`, `request_id`, `instance_index`, prediction time, exact model/schema versions, prediction, and approved feature values or sufficient monitoring representations. Persist features only when the agreed data-use/retention policy allows them. Record request latency and status separately; do not place unique request IDs in Prometheus labels.

Proposed feedback envelope:

```json
{
  "request_id": "existing-request-id",
  "labels": [
    {"instance_index": 0, "label": 1}
  ],
  "observed_at": "2026-10-04T00:00:00Z"
}
```

Feedback requires an existing request and an in-range instance index. Validate labels strictly as integer `0` or `1` against the training target schema. Allow delayed partial feedback, but never treat absent labels as negative labels. Return `404` for an unknown retained request and `422` for an invalid index/type; replaying identical feedback is idempotent and returns `202`. Conflicting labels for the same prediction return `409` unless a reviewed correction workflow is implemented. Restrict label writes to trusted evaluators in a shared deployment and define an explicit local-demo mode.

Join on `(request_id, instance_index)` and retain the model version from the prediction event. Monitoring reports join counts, unmatched/duplicate labels, label age, and coverage. Retraining datasets are versioned snapshots of validated labeled observations, subject to the data/split policy; feedback is never immediately used to train inside the API request handler.

## Monitoring, alerts, and retraining

`src/mlops_project/monitoring/` consumes a fixed reference window, a current window, prediction events, label joins, and service metrics. Configuration defines window boundaries, minimum sample/label counts, feature tests, quality metrics, alert thresholds, persistence, and cooldown. Pin the reference to the deployed model/data version. Initial thresholds and sizes are provisional until calibrated; store the final values in `configs/monitoring.yaml`.

A monitoring result records `monitor_run_id`, model/reference/window IDs, sample and labeled counts, feature statistics, quality metrics, service/SLO results, threshold configuration checksum, and one of `ok`, `alert`, `insufficient_data`, or `error`. A feature-distribution change is data drift. A quality drop without reliable labels cannot establish concept drift; insufficient labels or inadequate positive/negative class coverage must result in `insufficient_data` for the affected quality check. Evaluate quality per exact model version so a deployment change does not confuse the comparison.

For concept-drift demo evidence, change the label-generating relationship while preserving similar feature distributions in the monitoring replay pool, show the corresponding labeled metric decline and alert, then show controlled retraining. Save the simulation manifest and a separate feature-data-drift example. Treat these as historical/synthetic replays, not real observed future drift. Unlabeled feature drift is not proof of concept drift.

Simulated-regime retraining consumes separately reserved training examples and separate simulated-regime validation examples; it cannot fit on alarm/evaluation replay rows or the frozen test set. Predeclare a current-regime promotion policy and retain original validation performance as regression evidence. If the candidate fails gates, demonstrate rejection and continued serving rather than manufacturing an improvement. Every new regime/window/split has a manifest/checksum so monitoring and model owners can prove independence.

Alert records include `alert_id`, source, severity, run/window/model IDs, statistic, observed value, threshold, creation time, reason, and suggested action. Alerts must be visible in a persisted local report/log plus a dashboard or CLI view for the demo. No external email/chat sending is assumed. Aggregate bad-request alerts by a configured rate/severity rule to avoid one alert per ordinary rejected request; corrupt raw-data runs alert immediately and stop.

A retraining request contains `trigger_id`, alert ID, policy/configuration version, candidate dataset version, reasons, and time. Retraining starts only when labels/sample size meet the policy, cooldown has elapsed, and no run with the same trigger is active. A deduplicated trigger invokes the same Airflow training DAG. Passing its promotion gates records approval and deployment; a failed run/gate records failure and retains the current serving model. Task dependencies must prevent deployment after an upstream failed task. Pass artifact references/checksums between tasks, with deployment mutations serialized. [Airflow DAGs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html)

## Service metrics and logs

Expose request counts by endpoint and status, request duration histogram, batch-size distribution, model-load/deployment failures, validation failures by bounded rule name, and current readiness. Histogram buckets must support the stated latency SLO. Compute p50/p95 from recorded durations/histograms and report throughput with its measurement interval and workload. Avoid identifiers, raw features, exception messages, or unbounded model versions as metric labels.

Structured logs include timestamp, level, component, event, request/run/deployment ID where relevant, actual model version, status, elapsed time, and a safe error code. Correlation IDs make the demonstration traceable; private data and secrets must not be logged. Keep monitoring artifacts with enough detail to explain an alert and compare pre/post-retraining quality.

## Contract acceptance and changes

Before connecting modules, owners exchange a valid data fixture, invalid fixture, fitted bundle fixture, experiment result, gate report, prediction response, and labeled event join. Fixtures use the approved schema and small safe data. Contract checks cover missing/null required fields, incompatible model schema, failed gates, unreadiness, label-join errors, and rollback version evidence. If a later schema enables numeric imputation, add the permitted-null case and verify its training/serving consistency.

For a contract change, update this document, concrete config/API schemas, consuming modules, and meaningful checks in the same PR or coordinated dependent PRs. Identify whether old bundles and event records remain readable. Cross-component acceptance scenarios and the instructor's normal/abnormal input preparation are listed in [Testing and demo](TESTING_AND_DEMO.md).
