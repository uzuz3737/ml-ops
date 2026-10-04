# Verification, operations and demonstration

Status: **complete-system acceptance specification; P0 host framework tests pass, real container/ML acceptance remains unverified**. Cases reference [requirement IDs](REQUIREMENTS.md). Fixture tests listed in [P0 handoff](P0_HANDOFF.md) do not establish the integrated outcomes below. Each owner implements meaningful checks for their component and saves actual evidence.

## Before running checks

Freeze schema, data/split versions, model gates and operational targets in [Project brief](PROJECT_BRIEF.md). Build compatible pinned images; validate sample fixtures. Use a separate demo workspace/volume so destructive test inputs do not overwrite preserved source data or the last healthy model. Reserve the final test for final evaluation; recurring CI/promotion uses deterministic validation fixtures.

## Acceptance cases

| Case | Procedure | Required outcome | Owner / requirements |
| --- | --- | --- | --- |
| T01 Fresh-machine/full DAG | Clean clone; documented single wrapper; no hidden manual prep | Images/services start, Airflow completes raw→serve, exact model ready, valid host prediction | P0 + independent friend / G1,G3,G4,S1 |
| T02 Data reproducibility | Acquire same checksum; split twice with same version/config | Identical ID manifests, disjoint partitions, count/label/profile reports | P1 / D1,D2,G4 |
| T03 Hard invalid training data | Missing feature, wrong type, invalid label and unapproved category, separately | Saved anomalies + visible alert; DAG fails before training/promotion; current deployment stays healthy | P1/P0 / D3 |
| T04 Missing/outlier policy | Valid boundary, approved nullable field if any, forbidden null/type | Documented accept/impute/reject behavior; never silent label imputation | P1/P3 / D4 |
| T05 Training-serving consistency | Same valid raw rows through saved training bundle and loaded API | Probabilities/labels agree within declared numerical tolerance; identical feature order/selected transforms | P1/P2/P3 / D4 |
| T06 Experiments/selection | Baseline + at least two distinct candidate/configuration rounds on same split | Three run IDs, all six metadata categories, comparable metrics and explained choice | P2 / M1,M2,E1 |
| T07 Reject bad candidate | Candidate below approved quality gate, then failed operational gate | Rejected status/gate evidence; no healthy-version replacement | P2/P0/P3 / E2,S3 |
| T08 Serving/observability | Host POST, liveness/readiness, scrape metrics, inspect correlated logs | Correct response/version; counters/latency/logs reflect requests; unavailable model makes readiness 503 | P3 / S1,S5 |
| T09 Performance/SLO | Fixed approved workload and duration on actual running API | p50, p95, successful throughput, errors, metadata and SLO pass/fail | P3 / S2,S3,S4 |
| T10 Data-drift alert | Healthy then controlled distribution-shift replay | Reference/current stats, adequate window, exported drift score and threshold alert | P1/P3 / O1,O2 |
| T11 Labeled concept-drift alert | Same features, known changed label rule, joined outcomes | Feature comparison, labeled quality change, label coverage and alarm; explicitly simulated relationship change | P2/P3 / O1,O2 |
| T12 Retrain cycle | Trigger from T10/T11 according to policy | Airflow retrain run, new candidate/version, separate-regime validation and gates; deploy if approved, otherwise retain old model | P0/all / O3,E2,G1 |
| T13 Actual rollback | Deploy v1 then approved v2; restore v1 through controlled mechanism | New rollback audit/ACK, readiness + prediction report loaded v1; registry/deployment state agree | P0/P2/P3 / E3 |
| T14 CI pass/fail | Run valid PR, then isolated code/data/model failures | Three enforced check categories, pass + failure logs, blocked invalid delivery | P0/all / O4,O5 |
| T15 Service fault | Stop/restart API or make candidate bundle unavailable | System alert/readiness failure; preserved artifacts; safe recovery/old bundle remains when possible | P0/P3 / S5,O1 |
| T16 Presentation-day inputs | Try valid unseen values/boundaries, empty/oversized batch, missing/extra features, invalid labels | Contract-consistent response/errors; no silent crash; record actual instructor cases separately | All / R2,R3 |

For source-model reproducibility, compare data/split hashes and model outcomes against predeclared tolerance; record environment differences instead of promising bitwise identity from a seed alone.

## Load measurement protocol

P3 records code/model/image/data versions, CPU/RAM/OS, container resources, batch size, payload distribution, client concurrency, timeout, warmup and measurement duration. Separate warmup from recorded requests. Use representative valid payloads and avoid reusing one trivial input exclusively.

- Measure client-observed end-to-end latency on successful requests; report p50/p95 in milliseconds. Report timeout/error counts separately so excluding failures does not hide poor service.
- Compute successful throughput as successful requests / elapsed measurement seconds. Also report submitted/completed totals and server failures/timeouts.
- For batched requests, report requests/second and predictions/second with batch size. Do not mix those units.
- Record the exact SLO population: valid requests, workload, environment, time interval and thresholds. Compare each observed value with its gate; do not change targets after seeing results to obtain a pass.
- Five-minute local measurements support a local-test claim, not long-term production availability.

## Drift/quality runbook

1. Check input validation, reference/window IDs and minimum sample count first. Preserve source and model versions.
2. For feature shifts, inspect the TFDV/statistical report and feature/domain changes; do not classify every change as concept drift.
3. For quality, join labels by request plus instance index/model version. Check missing/late/duplicate labels and coverage; insufficient labels mean quality/concept status is unavailable.
4. In the controlled concept scenario, document the changed label rule while holding features comparable. Report quality degradation against the reference and explicitly label the data simulated.
5. Save alert type, score/threshold, window, affected model and evidence. Console/structured logs plus a Grafana alert are enough for the local demonstration; do not claim a notification reached an external recipient without proof.
6. Trigger Airflow retraining only under the agreed policy, cooldown and concurrency limits. Use separately versioned training/validation examples, never final test or the post-retrain evaluation rows.
7. Evaluate/record new version and gates. Deployment requires loaded-version acknowledgement, readiness and smoke check. A rejected candidate leaves the prior version running and the unresolved alert recorded.

## Rollback/recovery runbook

1. Record the issue, active version and last healthy version; inspect logs/quality/performance before choosing rollback.
2. P0 runs the controlled deployment operation targeting the previous approved bundle. Use a new deployment ID and preserve the failed attempt's audit record.
3. Wait for the API's actual acknowledgement and exact-version readiness. An alias change alone is incomplete.
4. Send a schema-valid prediction and verify returned version. Check registry lifecycle/current/previous metadata against serving state.
5. Record outcome, elapsed recovery time, reason and remaining alert. If reload fails, keep/restore the healthy active bundle when possible and report failure explicitly.

P0's controller exposes this command inside the worker, after real P2/P3 handoffs:

```powershell
docker compose exec -T pipeline-worker python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "demo rollback"
```

Only a stored approved, confirmed healthy prior model is eligible. P0 verifies matching ACK, schema/version readiness, valid predictions and P2 registry finalization before recording rollback success. For a failed first deployment without a prior model, the controller requests unload, requires `unloaded` ACK/readiness 503/health 200, and clears registry aliases through P2. A failed recovery is reported explicitly. Follow the [deployment contract](INTERFACE_CONTRACTS.md) and [P0 handoff](P0_HANDOFF.md); no real rollback has been demonstrated here.

## CI/CD evidence

Keep one normal passing run and separate failing examples for lint/test, TFDV data policy and model quality. Use fixtures/PRs or a safely isolated test workflow; do not merge deliberate broken production code to manufacture evidence. Save failing run URLs/logs before closing demonstration PRs.

CI should run bounded deterministic checks and model/data fixtures; provide a reproducible full-data workflow for the full evidence. A green lint job alone is insufficient. CD must actually start/deploy the approved image and model in an isolated Compose/runner environment, then verify exact-version readiness and a prediction smoke test. Document the selected runner/service connectivity; do not assume GitHub-hosted runners can access your laptop. Image building or static validation alone is insufficient delivery evidence.

## Twelve-minute presentation rehearsal

| Time | Speaker | Content |
| --- | --- | --- |
| 0:00–1:15 | P0 | Stakeholder/problem, business value, Canvas and stack architecture |
| 1:15–3:00 | P1 | Data/splits, TFDV schema, bad-data stop/alert, shared transforms |
| 3:00–5:00 | P2 | Baseline + three experiments, metrics/selection, registry gates |
| 5:00–7:15 | P3 | Host prediction, health, measured p50/p95/throughput and SLO |
| 7:15–9:15 | P3 with P1/P2 evidence | System/quality dashboard, distinct data/concept drift alerts |
| 9:15–10:45 | P0 | Airflow retraining evidence, real deploy/rollback, CI pass/fail |
| 10:45–12:00 | All, P0 coordinating | Results, limitations, contributions and transition to questions |

Long training/label windows may use recorded *actual* runs while showing a live prediction/validation/rollback. State what is live and what is recorded; never show fabricated metrics. Rehearse on the presentation machine and keep a local evidence backup. Each person prepares to explain their submitted code, failure behavior, contracts and tradeoffs.

## Evidence record per case

Save case ID/requirement IDs, command/payload, expected outcome, actual output, pass/fail, timestamp, code SHA, data/schema/model version, hardware/environment and author/reviewer in `docs/evidence/INDEX.md`. The directory/index is created when real evidence exists, not filled with imaginary successful results.
