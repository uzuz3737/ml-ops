# Dataset and feature policy

Owner: P1; reviewers P2/P3. **Source identified; not downloaded/profiled here.** These are implementation targets, not measured results.

## Source

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) lists 30,000 clients, 23 features, a binary label, no missing values, Taiwan repayment history from April–September 2005, and CC BY 4.0 licensing. Acquisition can use its `.xls` file or `ucimlrepo` dataset ID 350. Attribute **Yeh, I. (2009), Default of Credit Card Clients, UCI Machine Learning Repository, DOI: 10.24432/C55S3H** in metadata and report.

Verify headers, actual row count, class distribution and category codes after acquisition. Keep raw data unchanged; record SHA-256, retrieval method/date, attribution and all conversion/removal rules.

## Canonical mapping

Use descriptive uppercase source feature names in the initial API. Map generic X names once during ingestion if needed. Normalize target to `default_next_month`: 1 is default, 0 is no default. ID is lineage information, never a predictor.

| Source | Canonical feature/purpose |
| --- | --- |
| X1 | `LIMIT_BAL` |
| X2–X5 | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| X6–X11 | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` |
| X12–X17 | `BILL_AMT1` … `BILL_AMT6` |
| X18–X23 | `PAY_AMT1` … `PAY_AMT6` |
| ID | Unique client identifier; excluded from inference features |
| Raw target heading | `default_next_month`; verify original heading |

Notice PAY_0 followed by PAY_2, not PAY_1. Freeze mapping across data, training and serving. A different feature/API contract needs versioned coordinated fixtures.

## Schema and features

- Planned `schemas/credit_default.pbtxt` is the reviewed TFDV schema; `configs/data_schema.yaml` references it and the mapping.
- Training requires features, valid unique ID and target in {0,1}. Serving excludes ID/target and rejects accidental target inclusion.
- Profile actual category/status codes before defining domains; resolve undocumented codes explicitly. Do not silently drop them or copy incomplete metadata domains.
- Choose ranges from semantics and profiling. Do not impose blanket nonnegative bounds on all monetary fields without checking their meaning.
- Initial policy: missing required features, wrong types, invalid targets and disallowed values are hard errors. If legitimate nulls are later allowed, document each field and use the same train-fitted imputer at serving. Never impute target/ID.
- Fit learned imputation/scaling/encoding/outlier transforms on training only. Persist the complete preprocessing-plus-estimator bundle and selected feature list.
- Define an explicit policy for unseen legitimate categories. Version schema changes; do not reinfer a permissive schema on every incoming batch.
- Review SEX, EDUCATION, MARRIAGE and age inclusion with the stakeholder; justify retained/excluded features and slice limitations. A versioned model subset can be used behind a stable input contract.

TFDV provides schema statistics/anomalies and training/serving environments. Persist its reports and explicitly fail the task when anomalies violate policy; an anomalies result does not automatically stop the DAG. See [TFDV guide](https://www.tensorflow.org/tfx/data_validation/get_started) and [validation API](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/validate_statistics).

## Proposed split

Label-stratified, ID-disjoint **60% train / 20% validation / 10% final test / 10% monitoring replay**, seed 42. If all source rows remain valid, sizes are 18,000 / 6,000 / 3,000 / 3,000; save actual counts, sorted ID manifests and hashes.

Monthly feature columns are not per-row event dates. Do not claim a chronological split by sorting ID. Future repeated clients/event dates require an appropriate group/time strategy. Validation selects models/thresholds; final test stays outside retraining/monitoring. Never train on the same replay rows used for post-retrain evaluation.

## Drift replay

Version scenario seed, IDs, transformations, labels and affected population. These are simulations rather than real future defaults.

| Scenario | Construction | Expected evidence |
| --- | --- | --- |
| Healthy | Unmodified monitoring clients/historical labels | Reference quality, distributions, label coverage |
| Invalid | Missing column, wrong type, invalid label/domain | Alert, failed validation, no train/promotion |
| Data drift | Reweight clients or change approved feature distributions; preserve original labels | Distribution metric and threshold alert |
| Concept drift | Same features; deliberately changed label rule in documented subgroup | Known changed P(y\|X), labeled quality comparison and alert |
| Retrain | Apply defined regime to separate training/validation clients | New tracked candidate; disjoint evaluation and deploy/reject outcome |

Feature shifts cannot establish concept drift. Quality decline alone has other causes; the controlled label generator establishes the relationship change in this demo. Record comparable features, labels and sample size. Evaluate retrained candidates on separate simulated-regime validation, retain original-regime regression results, and agree gates beforehand. Retain the deployed model if a candidate fails.

## Planned request fixture

Synthetic shape example, not an extracted client, validated domain fixture or prediction result. P1/P3 must confirm it against the frozen schema. Response shape is in [contracts](INTERFACE_CONTRACTS.md).

```json
{
  "instances": [{
    "LIMIT_BAL": 100000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
    "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
    "PAY_AMT1": 3000, "PAY_AMT2": 3000, "PAY_AMT3": 3000,
    "PAY_AMT4": 3000, "PAY_AMT5": 3000, "PAY_AMT6": 3000
  }]
}
```
