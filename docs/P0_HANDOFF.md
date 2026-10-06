# P0 implementation and friend handoff

> **Status (6 October 2026):** written during development and kept as a record. Everything handed off here has since been delivered; "pending" items below are historical. Current behavior: [README](../README.md); results: [evidence index](evidence/INDEX.md).

Owner: **@uzuz3737 (you)**. Friends are assigned in the order requested: **@OuanEng (P1)**, **@pairot230 (P2)**, **@thanachaithongbai-hue (P3)**. This document records the integration increment and the concrete work each friend must connect. Course acceptance remains in [Requirements](REQUIREMENTS.md).

## What P0 implemented

| Work | Implemented behavior | Verification boundary |
| --- | --- | --- |
| Versioned configuration | Project/owner mapping, splits, schema pointer, gate/SLO proposals and disabled retraining policy | Numeric AP floor/regression allowance and reviewed TFDV schema are deliberately pending |
| Stage runner and HTTP worker/client | Dependency checks, bounded adapter subprocesses, timeouts, atomic reports, run/stage locking and idempotent retries | Tested with isolated adapters/localhost HTTP; no actual UCI/model execution |
| Evidence binding | Composite config digest includes project/policy/schema/monitor config and canonical schema; input and local artifact checksums detect changed evidence | A passing stale report cannot authorize a new run/config/artifact |
| Promotion policy | Requires matching run/code/data/schema/model identity, comparable validation AP and actual candidate service metrics/workload | Gate fixture tests verify policy behavior; actual baseline/gate values and measurements come from P2/P3 |
| Airflow DAG | `credit_default_pipeline`: ingest → validate → split → features → three training tasks → evaluate → register → benchmark → approve → deploy → verify | Static graph/contract checks pass; real Airflow image/DagBag/scheduler run requires Docker verification |
| Deployment controller | Atomic desired manifest, exact-version ACK/readiness/prediction, registry finalization and rollback to prior healthy version | P3 supplies actual loader/ACK API and P2 finalization; fixture checks do not prove a real deployed model |
| Compose and run wrappers | Pinned integration services, Linux container checks, bounded bootstrap/trigger/wait, exact-version ACK/readiness/prediction verification | Docker unavailable on implementation host; actual builds and full run remain unverified |
| CI/CD files | Code/docs suite; fixture data/failure/provenance checks; candidate-gate checks; manual infrastructure/full scopes | Infrastructure scope checks startup/DAG import only. Full lifecycle fails container preflight while real friend adapters/gates are absent |
| Collaboration docs | Named package board, source/test CODEOWNERS, PR template, contribution guide, report contribution fields | Repository administrator still configures collaborator access and required reviews |

Run the host commands in [Getting started](GETTING_STARTED.md). Use `ruff check .`, `ruff format --check .`, `pytest` and `python scripts/check_docs.py` for this increment. See the final commit/CI record for exact test counts; no real dataset/model/SLO result is claimed here.

## Exact adapters each friend supplies

`configs/project.yaml` is the executable mapping. Every function below has the signature `function(context: dict) -> dict`; no adapter uses a successful placeholder. Implement real behavior and tests under your own Git identity.

| Stage | Required callable | Owner | Direct upstream inputs |
| --- | --- | --- | --- |
| `ingest` | `mlops_project.data.pipeline:ingest` | @OuanEng | None |
| `validate` | `mlops_project.data.pipeline:validate` | @OuanEng | `ingest` |
| `split` | `mlops_project.data.pipeline:split` | @OuanEng | `validate` |
| `features` | `mlops_project.data.pipeline:features` | @OuanEng | `split` |
| `train_baseline` | `mlops_project.training.pipeline:train_baseline` | @pairot230 | `features` |
| `train_candidate_1` | `mlops_project.training.pipeline:train_candidate_1` | @pairot230 | `features` |
| `train_candidate_2` | `mlops_project.training.pipeline:train_candidate_2` | @pairot230 | `features` |
| `evaluate` | `mlops_project.training.pipeline:evaluate` | @pairot230 | All three training results |
| `register` | `mlops_project.registry.pipeline:register` | @pairot230 | `evaluate` |
| `benchmark` | `mlops_project.serving.pipeline:benchmark` | @thanachaithongbai-hue | `register` |
| `approve` | `mlops_project.pipelines.gates:approve` | @uzuz3737 (implemented) | `evaluate`, `register`, `benchmark` |
| `deploy` | `mlops_project.pipelines.deployment:deploy` | @uzuz3737 (implemented) | `approve` |
| `verify` | `mlops_project.pipelines.deployment:verify` | @uzuz3737 (implemented) | `deploy` |

P1's shared transform implementation belongs in `mlops_project.features`; the initial `features` entrypoint can delegate to it. P2/P3 import that implementation and fitted state. Monitoring boundaries are P1 `monitoring/feature_drift.py`, P2 `monitoring/quality.py`, P3 `monitoring/exporters.py`. These friend modules do not yet exist.

## Common context and result contract

The runner passes:

| Context field | Meaning |
| --- | --- |
| `contract_version` | Integer `1` |
| `run_id`, `step` | Current immutable run identity and configured stage |
| `config`, `config_path` | Parsed project config and its resolved file |
| `config_sha256` | Composite digest of the configuration snapshot |
| `project_root`, `run_dir` | Absolute project root and `artifacts/runs/<run_id>` |
| `inputs` | Mapping of **direct dependency stage names to their normalized result objects**, rather than entire stage records |
| `retraining` | Present only for a controller-approved `retrain-*` run; immutable approved candidate dataset, labels/split evidence and receipt checksum |

All results are finite JSON-compatible dictionaries with integer `contract_version: 1`. Include `run_id` matching the current run; if omitted, the runner normalizes it. Any `passed: false` or failed/error/rejected status blocks descendants. `validate`, `evaluate`, `benchmark`, `approve` and `verify` explicitly require boolean `passed: true` on success. An error must raise/return failure with a persisted report/alert rather than continuing downstream.

Optional `artifacts` is a list of `{uri, sha256}` entries. Each URI must resolve to a real file **within `run_dir`** with its lowercase 64-character SHA-256. Keep portable data/model references inside the report/manifest; do not put an out-of-run data/model path directly into this checked local-artifact list. The runner rechecks hashes during consumption/retry. MLflow remote artifacts need explicit portable identity/manifests; this local checker does not download them.

Store data manifests, TFDV reports, selected bundle manifest and measurements as real files under the run directory. Reports preserve the common [data/model contracts](INTERFACE_CONTRACTS.md). Direct inputs should forward portable downstream references/provenance so consumers need no friend's absolute path or unsafe rereading of unrelated runs.

The worker emits `<stage>.json` containing stage/run/state, timestamps, composite config and direct input hashes, plus normalized result or typed error. It writes `run.json` with the configuration snapshot. Reusing a run ID with changed config/evidence fails; use a new ID for a changed experiment/data run.

## P1 — @OuanEng

1. Implement the four data-stage callables, real UCI download attribution/checksum, canonical names, ID isolation and deterministic 60/20/10/10 seed-42 splits.
2. Supply reviewed `schemas/credit_default.pbtxt`, source/type/category/null decisions and TFDV statistics/anomalies. Hard anomalies must persist an alert and fail validation so training stops. Preflight currently reports the missing reviewed schema.
3. Provide the common train-only feature implementation and saved fitted-state contract reviewed by P2/P3. Demonstrate train/serve parity.
4. Implement feature-drift calculations and a controlled feature-shift fixture. Give P3 measured statistics/events and P0 a calibrated trigger signal.
5. Supply compatible hash-pinned Linux dependencies (`WORKER_REQUIREMENTS`), meaningful normal/failure tests and report section 3 plus feature-drift evidence.

Reviewer: @pairot230 for leakage/splits, @thanachaithongbai-hue for transform parity. First consumer handoff: a validated split/feature manifest and intentional bad-data failure.

## P2 — @pairot230

1. Implement baseline and two candidate callables, shared fitted transform/model bundles and at least three genuine MLflow runs with code/data/parameters/metrics/artifacts/environment.
2. Implement `evaluate` on validation, comparable baseline/current AP on the same population, and `register` with immutable MLflow version/model identity. Reserve final test from tuning, repeated gates and retraining.
3. Supply common evidence for `evaluate`, `register`, `benchmark`: `contract_version`, current `run_id`, full lowercase hexadecimal `code_commit` (40/64 characters), `dataset_version`, `schema_version` and model `artifact_sha256` (64 lowercase hexadecimal characters), all matching across the stages.
4. `evaluate` includes `passed: true`, `metrics.average_precision`, nonempty `validation_id`, and `comparison: {average_precision, validation_id}` on the identical population. `register` includes nonempty `model_name`, exact positive decimal-string MLflow `model_version` (for example `"2"`), loadable `artifact_uri` and matching `schema_version`; mutable aliases are rejected. P0 approval forwards these to deployment.
5. Measure baseline first; agree numeric `gates.metric.minimum` and `gates.comparison.max_regression` with P0, record rationale and change policy status from pending baseline. Undefined gates fail closed.
6. Supply the registry callback `mlops_project.registry.pipeline:finalize_deployment(context: dict) -> dict` for P0 to finalize exact deployed/rolled-back versions and aliases after live verification. Implement lifecycle/rollback artifact retrieval plus label joins/labeled quality and controlled concept-drift evidence; supply calibrated retraining policy inputs and separately reserved current-regime training/validation manifests. Preflight requires this callback.

Reviewer: @OuanEng for split/validation, @uzuz3737 for gates, @thanachaithongbai-hue for bundle/rollback. Report sections 4–5 and labeled-quality part of 7 belong to P2.

## P3 — @thanachaithongbai-hue

1. Implement `mlops_project.serving.app:app`, `/health`, `/ready`, `/predict`, `/metrics` and feedback/events using the frozen API contracts. Liveness must work before the first model; readiness is a separate gate.
2. Implement `mlops_project.serving.pipeline:benchmark(context: dict) -> dict`. It measures the **registered candidate in an isolated compatible API**, preserving current serving state. Return `passed: true`, matching common provenance/model version, `metrics: {p50_latency_ms, p95_latency_ms, throughput_rps, error_rate}`, nonempty `hardware`, and `workload: {batch_size, concurrency,measurement_seconds}`. Workload must match the declared gate config.
3. Implement the API watcher for P0's atomic desired-model manifest. Stage/verify the exact approved artifact/schema/checksum while the old model continues serving, then atomically activate it; handle prior-version manifests for rollback. P0 owns deploy/verify/controller callables and desired manifest; P3 owns actual serving behavior.
4. Write `artifacts/deployments/acknowledgements/<deployment_id>.json` with integer `contract_version: 1`, `status: loaded`, matching `deployment_id` and `model_version` after activation. `/ready` and prediction report that loaded exact version. P0's controller/launcher waits for matching ACK/readiness/prediction and invokes P2 finalization; failure restores the previous confirmed active model and verifies prior serving version.
5. Handle first-deployment cleanup when no prior healthy model exists: P0 writes a tombstone with `contract_version: 1`, `action: unload`, a new `deployment_id`, `status: unavailable` and `reason: first_deployment_failed`. Clear the loaded bundle, ACK with `contract_version: 1` and `{status: unloaded, deployment_id}`, return readiness 503 and keep health 200. P0 verifies recovery and asks P2's finalization callback to clear aliases with `action: unload`, `model_version: null`; its successful return has `contract_version: 1`, `passed: true`, `model_version: null`. A loaded candidate must not survive a failed first promotion.
6. Provide schema-valid `examples/predict.json`, API dependencies (`API_REQUIREMENTS`), loader compatibility/parity and actual rollback evidence. Root API base pins FastAPI/uvicorn; add actual model/runtime dependencies.
7. Extend starter Prometheus/Grafana provisioning under `infra/` with real service/quality/drift metrics, alerts and load results. P1/P2 supply computations; starter service-up panels do not prove model monitoring.

Reviewer: @uzuz3737 for delivery, @OuanEng for transforms, @pairot230 for model/labels. Report serving section 6 and dashboard/alerts section 7 belong to P3.

## P0 deployment protocol for P2/P3

P0 serializes deployment operations, revalidates the passing `gate-report.json` checksum and its exact run/config/model/schema/artifact provenance, then writes `artifacts/deployments/desired-model.json` atomically. This normal desired record has integer `contract_version: 1`, `action: deploy` or `rollback`, and these required fields:

```text
deployment_id, model_name, model_version, schema_version,
artifact_uri, artifact_sha256, approval_id,
gate_report_uri, gate_report_sha256,
run_id, code_commit, dataset_version, config_sha256,
policy_version, policy_sha256
```

`gate_report_uri` is project-relative and names the preserved passing gate report. P3 verifies the approved record and trusted artifact before loading. Its ACK is `artifacts/deployments/acknowledgements/<deployment_id>.json` with integer `contract_version: 1`, `status: loaded` and the same `deployment_id` and `model_version`. Live `/ready` must return `status: ready` plus the same model/schema version; `/predict` must return that model version, one ordered prediction per input, strict integer labels `0` or `1`, and finite probabilities in `[0,1]`.

P0 records `active-model.json` only after actual serving verification and registry finalization. `previous-model.json` stores the prior confirmed healthy approved manifest. A desired record alone is never a healthy rollback target. Failure preserves audit history and restores/verifies the prior confirmed model, or uses the first-deployment unload protocol above.

P2's configured `pipeline.registry_finalize` callback receives the ordinary context plus top-level `action` (`deploy`, `rollback`, or `unload`), `model_name`, `model_version`, `previous_model_version`, `deployment_id`, and `reason`. It runs after runtime verification, under a bounded subprocess deadline. Return integer `contract_version: 1`, boolean `passed: true`, and exact `model_version` matching the target; for unload that value is `null`. Make alias/tag updates idempotent for the deployment identity and restore mappings on rollback. P0 preflight requires this callable before changing a deployment.

The rollback CLI is `python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "demo rollback"`, with optional `--project-root`. Use it inside the worker to resolve Compose service URLs. Actual API loading, MLflow finalization and a real rollback remain P2/P3 integration evidence.

## Retraining and complete integration

P0's alert-to-Airflow controller is disabled until owners calibrate `configs/monitoring.yaml`. P1/P2/P3 supply actual measured alerts, labels/sample/class coverage and independent retraining split evidence. The controller enforces policy, cooldown, deduplication and active-run checks; it does not compute drift or invent outcomes. Reserved `retrain-*` run IDs require the controller's receipt bound to the same config digest; ordinary callers cannot invent a controlled retraining run. The runner adds `context.retraining` with the approved candidate dataset identity and labeled/independent-split evidence. P1 must ingest that approved new snapshot; P2 must evaluate the current-regime candidate without touching the protected final test. When configured:

```powershell
python -m mlops_project.pipelines.retraining --alert artifacts/example/alert.json --config configs/project.yaml
```

The alert path is an interface example, not an existing successful alert. Run the CLI inside the worker for Compose service URLs/credentials. Preserve complete detection → trigger → tracked candidate → gates → approval/rejection → deployment evidence.

The controller accepts alert `reason` values `data_drift`, `concept_drift`, or `quality`. `observed` is a nonnegative drift or quality-degradation score, and must be strictly greater than the matching calibrated policy threshold; `threshold_crossed: true` alone is insufficient. Include `contract_version: 1`, safe `trigger_id`, `alert_id`, affected `model_version`, current `dataset_version`, a different approved `candidate_dataset_version`, timezone-aware nonfuture `created_at`, and the boolean evidence flags `candidate_dataset_approved`, `labels_validated`, `separate_training_validation`, `final_test_excluded`, `threshold_crossed`, all true. Counts `sample_count`, `labeled_sample_count`, `positive_label_count`, `negative_label_count` and `label_coverage` must agree and meet the calibrated minima; coverage equals labeled/all samples. The controller checks active DAG runs, persists a pending receipt before POST, and reconciles ambiguous submission against the exact Airflow run ID. Resubmit the same alert to reconcile pending state; changing alert/policy evidence under the same trigger ID fails.

Airflow tasks use zero automatic retries so invalid data/model failures stop immediately. Once the cause is corrected, use a new run ID for changed evidence; an unchanged successful stage can be replayed idempotently. A pending controller receipt is not authorization for a worker to retrain.

After genuine handoffs, P0 reviews consumer examples and runs the Docker lifecycle; another member verifies a clean clone. Container integration's manual `infrastructure` default starts P0 services and imports the actual Airflow DAG, without model delivery. Select `full` for actual CD in an ephemeral GitHub Actions Compose environment; it preflights inside the built worker with component dependencies. Both scopes upload actual logs and stop runner services. The workflow does not connect to a teammate's laptop. An incomplete full lifecycle fails; a green framework/infrastructure run is not a real TFDV/model acceptance run.

Administrative evidence remains external work: actual stakeholder/topic approval, roster exception/additional members, user line-by-line understanding/review, completed report/demo and submission. Nobody's contributions or measurements are fabricated by these assignments.
