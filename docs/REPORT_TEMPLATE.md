# Report working template

Status: **fill with actual implementation and evidence before submission**. This is a proposed outline: the brief asks for a complete report but does not supply an exact heading list. Follow any additional instructor template when provided.

P0 assembles the report; every member owns the text/evidence for their workstream. Replace TBD fields and clearly mark unresolved limitations. Do not describe planned behavior as demonstrated behavior.

## 1. Project and contributors — all

- Project title, course CP413008, semester 1 / academic year 2569 (2026).
- Members, student IDs, repository URL and individual contributions/PRs.
- Record resolution of the confirmed four-person roster versus the course requirement of 5–7.
- Submission version/commit, date, supported demo platform and tested runtime.

| Role / GitHub member | Report sections | Contribution/evidence to fill |
| --- | --- | --- |
| P0 — @uzuz3737 (you) | 2, 6, 8; assemble final report | Integration/Airflow/gate/CI framework exists; fill reviewed commit, host tests, real container/CI results and stakeholder evidence |
| P1 — @OuanEng | 3; feature-drift part of 7 | Fill real UCI/TFDV/features/drift code, tests, manifests and reviewed PR |
| P2 — @pairot230 | 4, 5; labeled-quality part of 7 | Fill actual runs/metrics/registry/quality code, tests and reviewed PR |
| P3 — @thanachaithongbai-hue | Serving part of 6; dashboards part of 7 | Fill real API/watcher/ACK/performance/monitor code, measurements and reviewed PR |

Repository: [uzuz3737/ml-ops](https://github.com/uzuz3737/ml-ops). GitHub handles are supplied/inferred assignments; legal names and student IDs remain to fill.

## 2. Problem, stakeholder and AI Canvas — P0/all

Use [Project brief](PROJECT_BRIEF.md): actual stakeholder/workflow, error consequences, proposed action, value, ML suitability/rules comparison, topic approval/uniqueness, complete Canvas and scope/limitations. Attach honest stakeholder evidence; anonymize personal details where appropriate.

## 3. Data and validation — P1

Document UCI attribution/license from [Dataset](DATASET.md), acquisition/checksum, raw/canonical mapping, observed size/diversity/class distribution, schemas, missing/outlier/category decisions and excluded features. Explain split strategy/ID-disjoint manifests/seeds and protected final test.

Show an actual invalid-input case with TFDV anomaly report, failed task and alert. Explain why detecting an anomaly also requires explicit failure handling. Explain the saved preprocessing bundle and show training-serving consistency evidence.

## 4. Modeling and experiment comparison — P2

Define optimizing metric, gating metrics, review-budget business metric and threshold-selection policy. Explain the naive/rules baseline, candidate choices and feature reasoning. Compare at least three genuine experiment rounds on the same agreed evaluation population.

| Run/model | Code/data/split version | Parameters/features | AP / other task metrics | Review-budget/cost measure | Gate result | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline, TBD run ID | TBD | TBD | TBD | TBD | TBD | TBD |
| Experiment 2, TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Experiment 3, TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Explain the final choice using results, calibration, errors and relevant slice limitations. Distinguish measured business outcomes from assumed offline proxies. Add final-test evaluation only for the frozen selected candidate, and explain test isolation from ongoing retraining.

## 5. Tracking and registry — P2/P0

For **every** experiment, provide these six categories and an immutable run reference:

| Required category | What to record | Example location/value to fill |
| --- | --- | --- |
| Code version | Git SHA; record dirty/local changes if used | TBD |
| Data version | Raw checksum, dataset version, schema/split IDs | TBD |
| Hyperparameters | Estimator/feature/threshold configuration and seeds | TBD |
| Metrics | Named population, optimizing/gating/business/task values | TBD |
| Artifacts | Complete model bundle, comparisons, plots/reports and checksums | TBD |
| Environment | Python/library lock, image identity, runtime/hardware | TBD |

Show model versions/status tags, candidate gate decisions and approved/deployed/previous identities. Demonstrate actual rollback: record registry plus API-loaded versions before/after. Explain promotion acknowledgement/readiness behavior and failure recovery.

## 6. Architecture, tooling and serving — P0/P3

Include the final architecture diagram and actual Airflow DAG. Explain why each of the nine required functions uses its selected tool; see [Requirements](REQUIREMENTS.md). Record tested versions rather than unverified package claims.

Explain real-time serving choice versus batch for the user workflow, container/network deployment, API request/response/error contracts, health/readiness, metrics/logs and shared artifact loading.

| Load-test setting/result | Actual value |
| --- | --- |
| Hardware/OS/container limits, image/model/code | TBD |
| Payload, batch size, concurrency, warmup/duration | TBD |
| p50 / p95 end-to-end latency | TBD |
| Successful request throughput and prediction throughput if batched | TBD |
| Errors/timeouts/server failure ratio | TBD |
| Declared SLO and per-gate pass/fail | TBD |

## 7. Monitoring, drift and retraining — P1/P2/P3/P0

Show system and prediction-quality dashboards with actual values; label coverage/delay and sample requirements; reference/current window IDs; algorithms/thresholds and fired alerts. Distinguish feature distribution changes from a changed feature-to-label relationship. Identify simulated labels explicitly.

Include healthy, data-drift and labeled concept-drift cases. Record detection → Airflow trigger → new experiment/version → separate evaluation → approval/rejection → deploy/retention. Explain cooldown, failures and rollback policy. Do not imply TFDV/Prometheus/Grafana alone prove concept drift.

## 8. CI/CD, reproducibility and collaboration — P0/all

Link the actual GitHub Actions workflow and passing/failing evidence for code, data and model gates. Explain delivery runner/network choices, built image identity and smoke checks. Show branches/PRs and member review history.

Provide verified fresh-machine commands and an independent clean-clone result. Include dataset acquisition, compatible locks/images, split/metric rerun evidence, one-command full DAG, failure exit codes and any manual prerequisites.

## 9. Tests, results and limitations — all

Use [Testing and demo](TESTING_AND_DEMO.md) and the requirement/evidence index. Summarize actual passes/failures, incomplete areas, historical-data limits, stakeholder assumptions, sensitivity and future label/data acquisition. Document instructor normal/abnormal test cases after presentation when available.

## 10. AI assistance and individual contribution — all

The course permits AI code assistance, requires disclosure of which parts used it, and expects students to explain every submitted code line.

| Member | Tool/assistance used | Files/tasks affected | Human review/verification | Evidence/PR |
| --- | --- | --- | --- | --- |
| @uzuz3737 | Codex assisted with planning/docs and P0 framework | Integration, gates, Airflow, Compose/wrappers, CI/docs/tests | Host framework checks recorded in [P0 handoff](P0_HANDOFF.md); human line-by-line explanation/review and container evidence still to fill | Commit/PR to fill |
| @OuanEng | Fill actual tools used | Data/TFDV/features/feature drift | Fill real review and verification | Fill PR/evidence |
| @pairot230 | Fill actual tools used | Training/MLflow/registry/labeled quality | Fill real review and verification | Fill PR/evidence |
| @thanachaithongbai-hue | Fill actual tools used | API/watcher/ACK/performance/metrics/dashboard | Fill real review and verification | Fill PR/evidence |

Record AI-assisted documentation as well as any later code assistance for transparency. Do not claim AI use or verification that did not occur. Each member reviews generated code, explains it, and records their own contributions/peer assessment.

## Appendices and final checklist

- Filled Canvas/decision log, data/schema/split manifests, runtime lock/image list.
- Experiment/registry evidence, load results, dashboards/alerts, DAG/CI pass-fail logs.
- Verified README/setup, PR/contribution history, presentation/demo evidence backup.
- References, including dataset attribution and tool documentation used.
- Actual status of every [requirement](REQUIREMENTS.md), with remaining gaps made explicit.

The source lists submission at 5 October 2026, 23:59 Bangkok, and presentation on 12 October from 08:30 (12 minutes + 3-minute Q&A). Confirm any additional submission format/location with the instructor; none is specified in the supplied brief.
