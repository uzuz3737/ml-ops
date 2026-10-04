# Requirements and evidence checklist

Authority: [original course brief](../โครงงานรายวิชา%20CP413008%20Machine%20Learning%20Engineering%20for%20Production.docx.md). Row IDs below are team-defined for tracking, not official requirement numbers. **Course acceptance rows remain unchecked.** P0's implemented/tested integration framework is tracked in [P0 handoff](P0_HANDOFF.md); it does not establish real data/model/API or container acceptance.

Owners: **P0 @uzuz3737**, **P1 @OuanEng** (data/features), **P2 @pairot230** (models/registry/quality), **P3 @thanachaithongbai-hue** (API/monitoring). Each PR cites the row IDs it satisfies. See [test/demo cases](TESTING_AND_DEMO.md).

## Administrative constraints

- [ ] Four-person roster resolved: instructor approves exception or team grows to the required 5–7. P0 records evidence.
- [ ] Topic approved and distinct from other groups; actual stakeholder/value established.
- [ ] Submission completed before the brief's 5 October 2026, 23:59 Bangkok cutoff.
- [ ] Presentation ready for 12 October 2026 from 08:30: 12 minutes plus 3-minute Q&A.
- [ ] Every member has contribution history, peer assessment and a technical explanation/presentation role.

## Rubric mapping — 20 points

| ID | Required behavior/evidence | Owner | Suggested proof | Done |
| --- | --- | --- | --- | --- |
| F1 | Problem, real user, value, feasibility, reason ML beats simple rules | P0/all | Brief/Canvas, stakeholder evidence and rules comparison | [ ] |
| F2 | Complete AI Project Canvas | P0/all | Filled Canvas; instructor template if provided | [ ] |
| F3 | Optimizing/gating metrics linked to business success | P0/P2 | Numeric metric definitions, review-budget/business result or honest proxy | [ ] |
| D1 | Sufficient size/diversity; justified size if under recommended 5,000 | P1 | Profile, provenance and limitations | [ ] |
| D2 | Reproducible ingestion and rational train/validation/test split | P1 | Checksum, seeded ID manifests/hashes, leakage checks | [ ] |
| D3 | Schema detects actual bad data; pipeline stops AND alerts | P1/P0 | Bad fixture, TFDV report, failed task/CI, alert | [ ] |
| D4 | Appropriate missing/outlier policies and one shared train/serve transform | P1/P2/P3 | Policy, saved fitted pipeline, identical-prediction comparison | [ ] |
| M1 | Baseline and systematic comparison of at least three experiment rounds | P2 | Three run IDs, comparable results, selected-model reasoning | [ ] |
| M2 | Explain model performance and relevant limitations | P2 | Task metrics, error/slice/feature analysis and business connection | [ ] |
| E1 | Each experiment logs code version, data version, hyperparameters, metrics, artifacts, environment | P2 | Tracking records containing all six categories | [ ] |
| E2 | Versioned registry AND status, gate before approval | P2/P0 | Candidate/rejected/approved/deployed records plus gate report | [ ] |
| E3 | Actual rollback to prior loaded serving version | P0/P2/P3 | Before/after readiness/prediction and deployment audit | [ ] |
| S1 | API serves externally from container | P3/P0 | Host-to-container prediction with exact model version | [ ] |
| S2 | Actual p50/p95 latency AND throughput | P3 | Raw load results, hardware, workload, duration, errors | [ ] |
| S3 | Declared SLO compared with measurements | P0/P3 | Confirmed thresholds, calculation and pass/fail | [ ] |
| S4 | Suitable serving mode with justification | P3/P0 | Real-time vs batch reason for stakeholder's workflow | [ ] |
| S5 | Health endpoint, serving metrics AND logs | P3 | Liveness/readiness, scraped metrics, correlated logs | [ ] |
| O1 | System monitoring and prediction quality monitoring | P3/P2 | Dashboard with actual data, labels/coverage, errors/latency | [ ] |
| O2 | Data drift AND concept drift detection, clear thresholds/alerts | P1/P2/P3 | Separate controlled scenarios, labeled quality, thresholds and fired alert | [ ] |
| O3 | Retrain policy and detection-to-new-model cycle | P0/all | Trigger, new tracked/versioned candidate, gates, deployment/rejection | [ ] |
| O4 | Working CI/CD checks code quality, data validity, model quality | P0/P1/P2 | Three enforced jobs/gates and delivery smoke checks | [ ] |
| O5 | CI evidence for passing AND failing runs | P0/all | Linked run/log for valid input and separate code/data/model failures | [ ] |
| G1 | Actual rerunnable DAG from raw data through serving, one command/button | P0/all | DAG graph/task log, wrapper command, exact-version API response | [ ] |
| G2 | Clear repository, real branches and PRs | All | Layout, reviewed PRs and individual commits | [ ] |
| G3 | Dockerfile/Compose and fresh-machine README | P0/P3 | Independent clean-clone run using documented commands | [ ] |
| G4 | All dependency versions recorded; reproducible reruns | P0/P1/P2 | Locks/images/env, identical split hashes and declared metric tolerance | [ ] |
| R1 | Report complete, readable, architecture diagram | P0/all | Filled report with actual evidence and limitations | [ ] |
| R2 | Timed shared presentation/demo; technical Q&A | All | Rehearsal, individual speaking roles and component explanations | [ ] |
| R3 | Instructor normal/abnormal test cases on presentation day | All | Flexible validation/API handling; recorded day-of results | [ ] |
| R4 | AI assistance disclosed; explain every submitted code line | All | Assistance log, reviewed generated code, individual preparation | [ ] |

Scoring groups: F1–F3 framing/planning **2**; D1–D4 data **3**; M1–M2 modeling **3**; E1–E3 tracking/registry **2**; S1–S5 serving **3**; O1–O5 delivery/monitoring/CI **3**; G1–G4 DAG/reproducibility/governance **2**; R1–R3 report/presentation/testing **2**. R4 is a separate explicit course instruction. This mapping preserves the eight official groups; it does not award invented subpoint values.

## Tool-function coverage

| Required function | Selected tool | Implementation evidence |
| --- | --- | --- |
| Version control | Git/GitHub | Branches, reviewed PRs, member commits |
| Containerization | Docker | Built image(s), Compose, independent run |
| Data validation | TFDV | Reviewed schema, stats/anomalies, explicit stop/alert |
| Experiment tracking | MLflow | Three runs and six metadata categories |
| Model registry | MLflow Registry | Exact versions, lifecycle tags/aliases and gate/rollback |
| Pipeline orchestration | Airflow | Actual DAG dependency graph and repeat runs |
| Model serving | FastAPI | Container API, health/metrics/logs, loaded model |
| Monitoring | Prometheus + Grafana | Actual system/quality/drift metrics, dashboard and alert |
| CI/CD | GitHub Actions | Enforced three domains, pass/fail proof, delivery verification |

P1/P2 must implement drift/labeled-quality computations and P3 must export them: installing monitoring products alone satisfies neither drift requirement.

## Evidence index template

Create `docs/evidence/INDEX.md` when real evidence exists. Store small sanitized outputs in Git; link larger artifacts by immutable location/checksum. Use records with the following fields:

| Requirement IDs | Date/run ID | Code SHA | Data/schema/model versions | Artifact path/link | Expected/actual/pass-fail | Author/reviewer |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Record failures honestly and keep failing CI logs. Do not check rows based only on diagrams, screenshots of empty dashboards, a trigger request, or a registry alias change without serving acknowledgement.
