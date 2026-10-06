# Project brief, decisions and AI Canvas

Status: **all components implemented and the full lifecycle verified in Docker** (see [evidence index](evidence/INDEX.md)); stakeholder validation is still open. Owner: P0 @uzuz3737 with all members. See [team roles](TEAM_WORK.md), [P0 handoff](P0_HANDOFF.md) and [dataset guide](DATASET.md).

## Problem and scope

Build an end-to-end system estimating next-month credit-default probability from client history and supporting a limited-capacity review queue. The user selected [UCI dataset 350](https://archive.ics.uci.edu/dataset/350). Predictions should help an analyst choose which clients to review; document what action follows a prediction.

**Stakeholder to validate:** a credit-risk/review analyst is the proposed user role, not a claimed partner. P0 must identify an actual stakeholder or agreed representative, record their needs, and confirm instructor approval/uniqueness. Public data alone does not establish a real user.

Compare ML with a simple repayment-delay rule. The proposed reason for ML is that repayment status, balances and amounts may interact beyond one threshold. Test this against the rule baseline before claiming improvement. False negatives miss defaults; false positives consume review capacity and may cause unnecessary intervention. Ask about relative costs, review budget and human overrides; do not invent interviews, costs or savings.

## Launch decision sheet

This is the authoritative location for project-specific choices. Proposed values require team confirmation; TBD means evidence or a value is missing.

| Decision | Current value | Owner |
| --- | --- | --- |
| Topic | Credit-default prediction/review prioritization; binary classification | All |
| Real stakeholder and approval | TBD: representative, workflow evidence, topic approval and uniqueness | P0 |
| Dataset/mapping | UCI 350, source archive SHA-256 `56c885f8…`, canonical dataset version `d1bc5f13…`, 30,000 rows; [Dataset guide](DATASET.md) | P1 |
| Tools | Docker, TFDV, MLflow/Registry, Airflow, FastAPI, Prometheus/Grafana, GitHub Actions, Git/GitHub | User selected |
| Modeling | CPU scikit-learn: logistic-regression baseline, random forest, histogram gradient boosting; random forest selected (validation AP 0.5375) | P2 |
| Splits | Stratified 60/20/10/10 train/validation/final-test/monitoring (18,000/6,000/3,000/3,000), seed 42, disjoint IDs | P1/P2 |
| Features | Initial contract accepts 23 source features; selected model features versioned separately | P1/P2 |
| Optimizing metric | `average_precision` (AP), positive class 1; maximize | P2 |
| Decision threshold | Maximizes validation F1 and is stored in the model bundle (0.5068 for the deployed random forest); a review-budget threshold is future work | P0/P2 |
| Quality gates | AP ≥ 0.40 and at most 0.02 AP below the comparison model on the same validation rows ([quality_gates.yaml](../configs/quality_gates.yaml)) | P0/P2 |
| Business metric | Proposed recall/precision at review budget and explicitly assumed offline error-cost proxy | P0/P2 |
| Serving | Synchronous FastAPI, model loaded once per deployment, batches up to 1,000 rows ([Serving](SERVING.md)) | P3 |
| Labels | Delayed labels posted to `/feedback` per request/instance; the demo replays held-out monitoring rows | P1/P3 |
| SLO | p95 ≤ 300 ms, ≥ 20 req/s, ≤ 1% errors at batch 1 × 10 clients; measured 204.5 ms / 60 req/s / 0% in `full-run-002` | P0/P3 |
| Monitoring/retrain | 500-row windows; data drift PSI > 0.0662; quality degradation > 0.123; retrain only on a labeled quality alert with a new approved snapshot, 1 h cooldown ([monitoring.yaml](../configs/monitoring.yaml)) | All |
| Promotion/rollback | Exact approved version, deployment ACK/readiness/smoke; preserve last healthy version | P0/P2/P3 |
| Runtime | Python 3.11 containers/CI, Airflow 2.10.5, MLflow 2.19.0, hash-locked dependencies; verified in Docker on 4 October 2026 | P0/all |
| Team | P0 @uzuz3737, P1 @OuanEng, P2 @pairot230, P3 @thanachaithongbai-hue; student IDs/availability and exception/additional members needed | P0 |

## AI Project Canvas — proposed working template

The brief requires a complete Canvas but supplies no exact component list. Use the instructor's exact template if later supplied.

| Component | Initial content / information to complete |
| --- | --- |
| Problem/current process | Prioritize review queue; document actual process and rules baseline |
| Users/stakeholders | Proposed analyst; record real stakeholder and people affected by errors |
| Value proposition | Better use of limited review capacity; quantify workload/default coverage |
| ML suitability | Test whether combined history improves on repayment-delay rules |
| Prediction/action | Class-1 probability, decision/ranking; action and override TBD |
| Data/labels | Historical UCI snapshots; acquisition, permission, future label availability |
| Model approach | Baseline plus at least two experiment rounds; comparison/explanation/calibration |
| Success measures | AP, numeric gates, review-budget business measure, serving SLO |
| Serving constraints | Real-time container API; traffic, latency, host-callable demo |
| Monitoring/lifecycle | System and labeled quality; both drift types; alerts/retrain/rollback |
| Risks/controls | Error consequences, leakage, historical generalizability, sensitive features, label coverage |
| People/resources | Four roles, hardware/time, team-size exception or added members |
| Delivery/validation | One-command Airflow workflow, evidence, fresh-machine test, report/presentation |

## Metrics

Report AP, ROC AUC, precision, recall, F1, confusion matrix and a calibration measure such as Brier score. AP is the proposed optimizing metric; it is not identical to trapezoidal PR-curve area. Choose prediction thresholds using validation only.

At an agreed review budget K, report `recall@K = defaults in top K / all defaults` and `precision@K = defaults in top K / K`. Compare model, simple rule and naive baseline on identical clients/budgets. Explain why K represents user capacity.

An optional cost proxy is `FN × missed_default_cost + FP × unnecessary_review_cost + reviews × review_cost`. Define costs without double-counting. Report assumptions/sensitivity; historical estimates are not observed bank savings.

**Operational discussion targets, not results or course numbers:** batch size 1, ten concurrent clients, p95 ≤ 300 ms, ≥20 successful requests/second, server-failure ratio ≤1% over a five-minute local test. P0/P3 confirm or change these for the use case/hardware before testing. Always report p50. A short load test does not establish long-term availability.

Each gate needs a numeric threshold, direction, evaluation population and pass/fail rule. Set model gates from training/validation baseline results before candidate selection. Preserve final test for documented final evaluation, outside tuning, repeated promotion and retraining.

## Decision/change log

| Date (Bangkok) | Decision | Reason/evidence | Owner/reviewer | Affected docs |
| --- | --- | --- | --- | --- |
| 4 October 2026 | UCI 350, named stack and four-person roster | User instruction | P0/team | All |
| 4 October 2026 | Named role ownership and P0 integration framework | User supplied three friend handles; P0 from existing Git remote/identity | @uzuz3737; component reviewers | README, team board, handoff, CODEOWNERS |
| TBD | Stakeholder, topic/roster approval, gates and real container verification | Pending actual evidence | P0/all | Brief, runtime, requirements |

Schema, label, selected-feature, optimizing-metric or API changes require coordinated contract updates. Routine implementation choices belong in the component PR.
