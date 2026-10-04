# P2 model, registry and labeled-quality handoff

Owner: **@pairot230**. The P2 implementation is ready for component review. It includes three genuine tracked experiments, fitted bundles, validation selection, immutable registration/retrieval, deployment finalization/recovery, a protected final-test assessment, delayed-label joins and labeled-quality signals. Real UCI component measurements are in [REPORT.md](REPORT.md).

The model implementation changes P2-owned source/test paths. At the owner's subsequent request, CI is split into ten independent checks and the existing P0 callback test fixture is adapted for the added registry directory and realistic child-process startup. P1 still owns production acquisition, TFDV, splitting and shared transforms; P3 owns serving, benchmarking, metrics export and dashboards; P0 owns integration, approval policy, orchestration and deployment verification. Shared runtime configuration and teammate implementation logic are unchanged.

## Install and test

From the repository root, with Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r src/mlops_project/training/requirements-dev.lock
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/training tests/registry tests/monitoring/test_quality.py -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
python scripts/check_docs.py
```

`requirements.lock` is the runtime lock; `requirements-dev.lock` adds pytest/Ruff. Both contain hashes and Windows/Linux markers. The original P0 dev lock pins `packaging==26.3`, which conflicts with MLflow 2.19's `packaging<25`; use the P2 dev lock for this component. P0/P1 must resolve the combined TFDV/worker lock before full integration. Pin compatibility with the reviewed P1 stack rather than combining incompatible independent locks.

The existing Dockerfile supports the runtime without edits:

```powershell
docker build --target integration --build-arg EXTRA_REQUIREMENTS=src/mlops_project/training/requirements.lock -t mlops-p2-verification:local .
```

Its image contains no Git metadata or Git executable. P0 must provide the real full lowercase Git SHA through `training.code_commit` in the frozen project configuration, or `MLOPS_CODE_COMMIT` in the worker environment. An absent/abbreviated commit fails; no synthetic commit is substituted. Every experiment also records a hash of the actual Python source tree and an environment inventory.

## P1 → P2 contract

The configured callables are already the names required by P0:

- `mlops_project.training.pipeline:train_baseline`
- `mlops_project.training.pipeline:train_candidate_1`
- `mlops_project.training.pipeline:train_candidate_2`
- `mlops_project.training.pipeline:evaluate`
- `mlops_project.registry.pipeline:register`
- `mlops_project.registry.pipeline:finalize_deployment`

The first three consume `context.inputs.features`. P1 supplies a project-relative split manifest reference and the common transformer factory:

```json
{
  "split_manifest": {"uri": "artifacts/runs/example/split-manifest.json", "sha256": "actual-64-character-lowercase-checksum"},
  "preprocessor_factory": "mlops_project.features:build_preprocessor"
}
```

The factory signature is `build_preprocessor(*, feature_columns: list[str]) -> sklearn-compatible transformer`. P2 clones its returned transformer, fits it only on training rows and persists it together with the estimator. P3 never refits it. The factory reference can be supplied by the trusted P1 result; it is not public API input. No production fallback transformer is supplied by P2.

The signed split manifest has this shape; the illustrative short checksum labels below must be replaced with actual SHA-256 values:

```json
{
  "contract_version": 1,
  "dataset_version": "immutable-dataset-version",
  "schema_version": "credit-default-v1",
  "validation_id": "immutable-validation-population",
  "feature_columns": ["LIMIT_BAL", "SEX", "...all remaining reviewed features..."],
  "target_column": "default_next_month",
  "identifier": "ID",
  "partitions": {
    "train": {"uri": "data/splits/train.csv", "sha256": "actual-sha256", "format": "csv"},
    "validation": {"uri": "data/splits/validation.csv", "sha256": "actual-sha256", "format": "csv"},
    "final_test": {"uri": "data/splits/final-test.csv", "sha256": "actual-sha256", "format": "csv"}
  },
  "protected_ids": ["all final-test and monitoring client identities"]
}
```

CSV and Parquet are supported, with the format declared explicitly. Each training/validation file must have exactly the reviewed features, target and ID; numeric finite features, strict integer binary labels, no nulls and both classes. IDs must be unique and disjoint between fit/validation and the protected pool. P1 remains responsible for canonical schema/category/range validation and dataset provenance. References are confined to the project and verified before reading.

For retraining, the new manifest must match `context.retraining.candidate_dataset_version`, include `final_test_excluded: true`, and name independent current-regime fit/validation data. `training.comparison_bundle: {uri, sha256}` is mandatory for retraining and points to the actual current champion bundle. P2 compares both models on that same new validation population. The final test is never read by training, selection, gates or retraining.

## Models and selection

All experiments use the same splits/factory and seed 42:

| Callable | Estimator | Main settings |
| --- | --- | --- |
| `train_baseline` | LogisticRegression | C=1, balanced classes, max_iter=1500 |
| `train_candidate_1` | RandomForestClassifier | 150 trees, depth=10, leaf minimum=5, balanced classes |
| `train_candidate_2` | HistGradientBoostingClassifier | 150 iterations, learning_rate=0.08, 15 leaves, L2=1 |

Each experiment selects its binary threshold by maximum validation F1 (largest threshold breaks a tie). Selection maximizes positive-class average precision; experiment order breaks an exact AP tie. Evaluation reloads checksummed bundles and recomputes validation metrics, rejecting stale/incomparable evidence. It records AP, ROC AUC, precision, recall, F1, Brier score, calibration bins and class/sample counts.

Every MLflow run records code/source/config hashes, data/split/schema identities, parameters, seed, metrics, artifact checksum and files, plus environment/lock/hardware evidence. Registration requires matching finished tracking evidence. A version is registered as `validated`, with no serving alias mutation. Local result artifact references conform to P0's run-directory checksum checker.

`evaluate_final_test(context, selected, locked_candidate_sha256=...)` is a separate reporting action, outside the DAG's selection/gate path. Its ledger locks one candidate for a dataset before reading held-out labels. Repeating an identical completed assessment returns its report; a different candidate/split or any retraining context fails. An interrupted ledger stays locked for inspection. The UCI component verification deliberately leaves the real final test unused until P1's reviewed dataset and initial candidate are frozen.

## P2 → P3 bundle contract

Each experiment writes `bundle.joblib`, `manifest.json`, `environment.json`, `smoke.json` and `result.json` under its run directory. The checksummed joblib object contains `pipeline` (shared fitted preprocessor + estimator) and `manifest` (immutable signature, feature types/order, threshold, provenance, metrics and environment). Its model version remains external registry/deployment identity; registering it does not rewrite the trained bytes.

```python
from mlops_project.training.bundle import load_bundle, predict_bundle

# Call only after verifying P0 approval and trusted provenance.
bundle = load_bundle(path, artifact_sha256, schema_version)
predictions = predict_bundle(bundle, validated_raw_instances)
```

P3 applies its canonical request/category/range validation before calling this utility, and wraps the ordered predictions with request ID and the exact loaded MLflow version. The utility rejects missing/extra keys, nulls, booleans and nonfinite numeric values, uses the estimator's actual class ordering, and applies the stored threshold. Python major/minor and serialization-library versions must match. Checksums do not make an untrusted pickle safe: load only approved project artifacts.

`retrieve_bundle(context, model_name, exact_decimal_version, destination)` retrieves the immutable `bundle/bundle.joblib` from its registered MLflow run, verifies checksum/schema/environment/provenance and atomically materializes it under the project root. Aliases are not accepted as version arguments. The tests cover actual SQLite registry creation/retries and retrieval, and Windows-to-Linux prediction parity.

## P0 → P2 deployment protocol

`finalize_deployment` accepts exactly the callback context documented in P0's handoff, validates it against `desired-model.json`, and runs after P0's live ACK/readiness/prediction verification. It never calls the serving API or claims to verify it.

The controller's approved checksummed gate report binds registry tags to run/config/model/data/schema/artifact identity. `record_gate_decision(context, manifest)` records actual approval/rejection; P0 should invoke it for rejected reports too, because the current P0 runner stops descendants immediately on a failed approval. Rejection never changes `champion`.

Deploy sets `champion` to the exact target and `previous` to the prior healthy version; the old champion becomes `retired`. Rollback requires a previously approved immutable target, restores its remembered prior mapping and marks the displaced champion `rolled_back`. Unload clears both aliases and confirms `model_version: null`. Pending/completed journals bind deployment ID and evidence, preserve alias mappings for compensation/retry and reject changed/stale callbacks. P0's controller serializes serving changes; actual service rollback remains P0/P3's integration evidence.

## Labeled quality and controlled retraining

`mlops_project.monitoring.quality` exposes `join_labels`, `evaluate_quality` and `retraining_alert`. P3 supplies prediction events and validated feedback envelopes; P2 joins `(request_id, instance_index)` for one exact model version/window as of a timezone-aware cutoff. Unknown/out-of-window labels are counted, identical duplicates are idempotent, conflicting labels fail, and labels after the cutoff are excluded. Missing labels remain missing. Feedback preceding a prediction fails.

Calibrated policy keys are `minimum_samples`, `minimum_labeled_samples`, `minimum_class_samples`, `minimum_label_coverage` and `quality_degradation_threshold`. Undefined thresholds fail; insufficient sample/coverage/class support returns `insufficient_data` without a trigger. P3 persists the returned report and exports bounded metrics; this module does not send notifications or own dashboards.

The reference is pinned to the same model version and reference ID and contains `average_precision`, optionally `roc_auc` and `brier_score`. `observed` is the maximum nonnegative AP drop, ROC AUC drop and Brier increase among supplied reference metrics. Each component is preserved in `degradation_components`. All metrics are bounded fractions, and a threshold crossing is strict `observed > threshold`. Supplying all three is recommended: AP alone can rise after a prevalence change while ranking/calibration deteriorate. An observational alarm has `reason: quality`; controlled fixtures separately establish a known changed label relationship.

`retraining_alert` requires an actionable report plus externally supplied approved candidate-dataset and independent-split evidence. It forwards P0's flags/counts/coverage contract and never invents approvals. The UCI harness reserves 1,000 alarm, 1,400 new-regime training and 600 new-regime validation rows from monitoring before label inversion; all are client-disjoint and exclude the frozen final test. These are component simulations, not actual future credit data or a claimed Airflow run.

## Integration actions for reviewers

1. **P1:** provide the reviewed split/factory contract above, agree worker dependency compatibility and confirm leakage/feature policy.
2. **P0:** set the real container commit, review CI's use of the P2 dependency lock, review measured model/quality policy proposals, and record rejected gate decisions. No shared policy was silently approved here.
3. **P3:** load verified bundles, run candidate benchmarks and exact-version watcher/ACK/rollback acceptance, consume the quality reports and export metrics.
4. **P0 test fixture:** the callback fixture now uses `mkdir(exist_ok=True)` and a 2-second callback budget, preserving its real subprocess and action assertions. This integration adjustment was made with the requested ten-check CI update; the earlier proposed patch has been superseded.

AI assistance: Codex assisted implementation, failure checks and verification. The owner remains @pairot230 and must review/explain the contribution and obtain the consumer reviews required by the course. No reviewer approval, individual understanding, full production lifecycle or external submission is fabricated.
