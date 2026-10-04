# P2 contribution and measured results — @pairot230

Verified on **4 October 2026 (Asia/Bangkok)**. P2 component implementation is complete and ready for consumer review. Full-course acceptance still requires the reviewed P1 dataset/transformer, real P3 serving/benchmarks, P0 integration/policy approval and the required human reviews.

## Scope and design

Implemented baseline/two candidate stages, six-field MLflow tracking, raw-input fitted bundles, comparable validation evaluation, immutable registry versions/retrieval, gate/lifecycle metadata, deployment finalization/rollback/unload, protected initial final-test reporting, label joins and quality signals. All committed paths belong to P2; no shared configuration, CI, root docs or teammate module/test was edited. [README.md](README.md) gives the exact contracts and commands.

Training uses a single cloned shared preprocessing instance per model and fits it only on training rows. ID and target are excluded. Thresholds maximize validation F1; candidate selection maximizes positive-class AP on the identical validation population. Saved probabilities use the estimator's actual class ordering. Every successful run records Git/source/config provenance, data/split/schema identities, hyperparameters, metrics, actual artifact checksums/files and a runtime/dependency/hardware inventory. Registration and deployment finalization reject mismatched evidence.

## UCI component experiments

Data: [UCI Default of Credit Card Clients, dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), credited to Yeh, I. (2009), DOI `10.24432/C55S3H`, CC BY 4.0. The downloaded archive checksum is `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.

The verification harness uses all 30,000 rows, 23 features, client-disjoint stratified splits with seed 42: 18,000 train, 6,000 validation, 3,000 frozen final test and 3,000 monitoring. These historical row splits do not measure future deployment performance. The fixture uses a provisional numeric StandardScaler so P2 can measure actual estimators before P1's shared transformation/TFDV handoff. It lives under tests, and is explicitly not a production P1 adapter.

| Model | Validation AP | ROC AUC | F1 | Precision | Recall | Brier score | Stored threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.475850 | 0.703373 | 0.494398 | 0.475000 | 0.515448 | 0.209126 | 0.544055 |
| Random forest | 0.531571 | 0.773346 | 0.540238 | 0.534267 | 0.546345 | 0.170501 | 0.500210 |
| Histogram gradient boosting | **0.534997** | **0.775306** | 0.535417 | 0.496458 | 0.581010 | **0.137944** | 0.239078 |

The selected model is histogram gradient boosting because it has the highest validation AP. The AP difference from random forest is small and does not establish a statistically significant advantage. Calibration bins and all unrounded measurements are saved with each run. No final-test model evaluation was performed: that assessment remains reserved for the initial reviewed/frozen candidate.

## Full pipeline run with P1 preprocessing (scikit-learn 1.6.1)

The table above came from the component harness (numeric-only scaler, scikit-learn 1.5.2). After the P1/P2 alignment the same three experiments ran inside the real DAG: P1 ingest, TFDV validation, seed-42 split and P1's shared `build_preprocessor` (StandardScaler + one-hot categories), with scikit-learn 1.6.1 / pandas 1.5.3 so the bundle loads in the serving image. Airflow run `full-run-002`, validation population `6cab3d86…` (6,000 rows):

| Model | Validation AP | ROC AUC | F1 | Precision | Recall | Brier score | Stored threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression (baseline) | 0.5147 | 0.7553 | 0.5163 | 0.5266 | 0.5064 | 0.1863 | 0.5431 |
| Random forest | **0.5375** | **0.7774** | **0.5394** | 0.5298 | 0.5494 | 0.1731 | 0.5068 |
| Histogram gradient boosting | 0.5358 | 0.7743 | 0.5326 | 0.5394 | 0.5260 | **0.1378** | 0.2889 |

With P1's one-hot encoding of the repayment-status and demographic codes, the baseline gains about 0.04 AP and random forest is selected (highest validation AP; the margin over gradient boosting is 0.002 and is not a significant difference). It was registered as `models:/credit-default/2`, passed every gate (AP 0.5375 ≥ 0.40, regression vs baseline −0.023 ≤ 0.02, p95 204.5 ms, 60 req/s, 0% errors) and was deployed and verified. Evidence: `artifacts/runs/full-run-002/evaluation.json`, `registration.json`, `gate-report.json`. The final test is still untouched.

Proposal for P0 review: initial AP floor **0.40**, absolute same-population regression allowance **0.02**. The floor is below the measured baseline AP 0.47585 but above positive prevalence approximately 0.221; this is an initial engineering proposal rather than a stakeholder-approved business metric. The regression allowance is not a confidence interval or permission to degrade a deployed system. Real candidate service gates cannot pass without P3's measurements. Shared policy remains unchanged and fail-closed.

## Controlled labeled-quality and retraining experiment

Before simulation, the 3,000 monitoring rows are reserved into **1,000 alarm**, **1,400 current-regime fit** and **600 current-regime validation** rows. All IDs are disjoint, and none comes from the frozen final test or original fitting/selection partitions. The controlled scenario keeps the feature rows identical and inverts their label relationship. This also changes positive prevalence; it is a declared synthetic regime, not observed future credit behavior.

On the same 1,000 alarm observations, normal ROC AUC is **0.758087**, falling to **0.241913** after inversion. Brier score worsens from **0.135967** to **0.594430**. AP rises from **0.548981** to **0.638240** because prevalence changes. This demonstrates why AP alone is insufficient for this quality alarm. The largest degradation is the ROC AUC drop **0.516174**, and the quality report changes from `ok` to `alert`. Components and label coverage/class counts are preserved in the report; observations without adequate labels return `insufficient_data` and cannot trigger training.

Three genuine new MLflow experiments were trained on the independent current-regime fit data. On the same 600 current-regime validation examples, the original bundle has AP **0.632479**; the selected retrained random forest has AP **0.912629**, ROC AUC **0.766289**, F1 **0.891070**, and Brier score **0.150982**. These figures belong to the synthetic label regime and are not interchangeable with original-task AP. Model comparison is performed on the same population, not between the two validation partitions. The new model is registered without deployment; no Airflow trigger, service benchmark or production promotion is claimed.

The demonstration alarm uses a 0.05 degradation threshold. Separately, validation bootstrap calibration with seed 42, 300 replicates and 400 labeled observations per replicate gives an empirical 99th-percentile normal degradation of **0.113044**. Adding a 0.01 margin proposes **0.123044** for owner review, with minimum 500 observations, 400 labels, 30 examples per class and coverage 0.8. The controlled 0.516174 degradation exceeds this proposal too. This finite-sample bootstrap does not validate real traffic stationarity, repeated-testing false-alarm rates or a production business threshold; P0/P1/P3 must review it before enabling retraining.

## Verification and limits

- **Windows, Python 3.11.9:** 27 P2 tests pass; Ruff checks/formatting and `pip check` pass.
- **Linux, Python 3.11.11 container:** the original Dockerfile builds with the P2 hash-pinned runtime lock, `pip check` passes with the P2 dev lock, and all 27 P2 tests pass.
- **Current repository full suite on Linux:** 149 tests pass in 72.17 seconds, including the real callback subprocess. The owner requested ten CI checks; CI now installs the P2 dev lock, and the P0 fixture supports the existing registry directory and a 2-second child startup budget. No tests are skipped to obtain passing checks.
- **Artifact consumer check:** the UCI bundle trained on Windows loads in Linux with its expected checksum/schema/library versions and smoke probabilities agree within absolute tolerance `1e-8`.
- **MLflow/Registry:** real SQLite-backed runs/versions are exercised, including retry without duplicate version creation, immutable artifact retrieval, rejected gates without alias changes, deploy/rollback/unload and recovery from an injected alias-update outage. These do not establish actual P3 serving rollback or remote registry authentication.
- **Data/evidence failures:** tests cover protected-row leakage, split overlap, missing P1 transformer, tampered bundles, forged tracking metadata, changed deployment identity, invalid/conflicting/late/missing labels, single-class/insufficient windows, model-version isolation, and final-test reuse/retraining rejection.

The real-data runs and full logs stay in ignored `artifacts/`; the small committed [evidence summary](evidence/uci350-summary.json) contains sanitized aggregate measurements and traceability, not raw clients or model binaries. No container API performance, complete Docker Compose lifecycle, hosted CI, reviewer sign-off or stakeholder approval is asserted.

## Reproduce the component evidence

After installing the P2 dev lock, install `xlrd==2.0.1` only for the XLS verification harness. Place the UCI archive in ignored artifacts and use a fresh run ID:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python tests/training/evidence.py --archive artifacts/p2-verification/uci350.zip --run-id p2-uci350-new
python tests/training/calibrate_quality.py --registration artifacts/runs/p2-uci350-new/registration.json --output artifacts/p2-verification/quality-calibration.json
```

The harness verifies the dataset shape/IDs, freezes fixture manifests, runs/tracks/selects/registers the three estimators, retrieves the immutable selected bundle, evaluates the controlled quality change and independently trains/registers three current-regime experiments. It supplies no serving success placeholders. Use P1's reviewed artifacts/factory for production acceptance; fixture results do not substitute for TFDV or shared-feature review.

The selected validation threshold is a technical F1 choice. Business costs, fairness, consent/retention and actual stakeholder success criteria still require the team's documented decisions. AI assistance: Codex assisted code, tests, experiments and this report; @pairot230 must review and be able to explain the implementation and its limits.
