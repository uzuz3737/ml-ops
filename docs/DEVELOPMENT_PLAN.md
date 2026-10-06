# Development plan

> **Status (6 October 2026):** written during development and kept as a record. Everything handed off here has since been delivered; "pending" items below are historical. Current behavior: [README](../README.md); results: [evidence index](evidence/INDEX.md).

This plan follows the supplied course requirements and selected [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350). P0's integration framework is now implemented; actual P1–P3 components, container verification and full-course evidence remain open. Real stakeholder, topic approval, business measure, numeric model gates and SLO still need confirmation in [PROJECT_BRIEF.md](PROJECT_BRIEF.md). See [P0 handoff](P0_HANDOFF.md) for current implementation status.

The roster is **P0 @uzuz3737 (you), P1 @OuanEng, P2 @pairot230 and P3 @thanachaithongbai-hue**. The course requires **5–7 members**, so four people need additional members or an instructor-approved exception. See [TEAM_WORK.md](TEAM_WORK.md) for named assignments/reviewers and expansion options.

## Dates and scope

All times below use **Asia/Bangkok (UTC+7)**.

- Submission: **before 5 October 2026, 23:59**. Aim to submit by 23:30 to retain a buffer.
- Presentation: **12 October 2026, from 08:30**, with **12 minutes presenting and 3 minutes of questions** per group. The group's exact slot is not provided.
- Planning begins on **4 October 2026**. Time windows below are targets, not extra course deadlines. If a window has elapsed, use the next available session and update the plan honestly.
- The selected stack is **Git/GitHub, Docker, TFDV, MLflow tracking and registry, Airflow, FastAPI, Prometheus/Grafana, and GitHub Actions**. P0 pins framework/integration versions; component compatibility and container testing remain.
- W0/W8/W9/W10 have P0 framework/docs increments. All acceptance criteria below still require actual integrated evidence; workflow files do not establish hosted CI or delivery success.

The submission window is short. Keep one narrow prediction task, one containerized serving path, and small repeatable demonstrations of every assessed requirement. Remove unrelated UI/features and infrastructure expansion from the sprint. Documentation alone cannot satisfy implementation or demonstration requirements.

## First checkpoint: decisions that unblock implementation

P0 coordinates these decisions with all members. Record them in the linked brief, architecture and contracts before separate implementations diverge.

| Decision | Required record | Why it matters |
| --- | --- | --- |
| Topic and stakeholder | Approval/nonduplication status, real beneficiary, value and consequence of a wrong prediction | A selected dataset does not establish a real stakeholder or course topic approval. |
| Dataset version and target | Source/access instructions, permission/license, checksum/version, predictor/target definitions | Every run needs traceable inputs and consistent labels. Use [DATASET.md](DATASET.md). |
| Split and leakage control | Planned 60/20/10/10 train/validation/final-test/monitor-replay split, stratified seed 42 and disjoint IDs | Exclude identifier and target from predictors; fit preprocessing only on training data; reserve final test for final evaluation. |
| Business objective, optimizing metric and gates | Proposed average precision for positive label 1, validation-selected decision threshold, numerical quality gates pending baseline, proposed service SLO | Do not invent an accuracy threshold or claim a gate passes while undefined. Approve proposals against real stakeholder needs. |
| Serving scope | Request schema, response contract, expected traffic/latency, serving pattern and rationale | Establishes a testable service rather than a model notebook alone. |
| Monitoring and labels | Reference/current windows, label join and delay, sample sizes, drift/quality thresholds | Prediction quality and concept drift need outcomes or an explicitly controlled labeled scenario. |
| Versions and environment | Supported Python/container versions and a verified TFDV/Airflow/MLflow compatibility matrix | Test high-risk dependencies early; use Docker as the agreed reproducible execution environment. |
| Interfaces and ownership | Data manifest, model/transform bundle, registry record, request event, trigger and DAG step contracts | Producers and consumers must agree on the same artifacts. |

Complete the AI Project Canvas in [PROJECT_BRIEF.md](PROJECT_BRIEF.md), establish numerical model gates after a baseline, and record unresolved decisions with an owner and due time. The UCI dataset is historical; synthetic replay and drift scenarios must be labeled as demonstrations, without claims of live production outcomes or validated business impact.

The initial **proposed**, unmeasured SLO is p95 latency at most 300 ms, 20 requests/second with 10 clients over 5 minutes, and server failures at most 1%. P3 must approve the test workload with the team and report actual p50/p95, throughput and failure rate. These numbers are project proposals, not course-mandated thresholds or observed results.

## Lifecycle and dependencies

```text
Approved problem + data/metrics/SLO/contracts agreed
  -> data acquisition/versioning + TFDV validation + reproducible splits
  -> shared transformations fitted only on training data
  -> at least three fully recorded MLflow experiments, including baseline
  -> candidate evaluation against agreed gates on validation
  -> MLflow registry version/status + auditable approval/promotion
  -> Docker FastAPI service + health + logs/Prometheus metrics
  -> Grafana/system alerts + feature drift + labeled quality/concept drift
  -> retraining trigger -> Airflow training DAG -> new candidate
  -> gate -> approve/promote or reject; rollback when required
```

P0 integrates callable steps supplied by P1–P3 into the implemented **Airflow DAG**. `scripts/run-all.ps1 -Config configs/project.yaml` and its Bash equivalent build/start services, trigger/wait for the DAG and check deployment evidence/readiness. The full runtime is unverified without Docker and real adapters. Initial full preflight intentionally fails; infrastructure-only startup does not train/deploy. See [GETTING_STARTED.md](GETTING_STARTED.md).

A retraining trigger produces a candidate; it does not authorize a failing candidate to replace the current model. Preserve auditable approval/promotion decisions. Never use the final-test partition for training, tuning or promotion gates. Controlled retraining uses separate simulated training/validation data, without reusing monitor-trigger replay rows to train the candidate.

Critical path: contracts/data → baseline artifact → loadable API/container → integrated DAG/gates → complete evidence. Start CI/container scaffolding, monitoring fixtures and report sections in parallel once contracts are accepted.

## Work packages and acceptance criteria

Each package needs a reviewed PR, meaningful normal/failure checks, consumer handoff, and reproducible evidence. A file's existence does not establish completion.

| ID / lead | Work and proposed paths | Dependencies | Acceptance criteria / handoff |
| --- | --- | --- | --- |
| W0 — P0, team decisions | AI Project Canvas, metrics/SLO, decisions and integration configuration; `docs/PROJECT_BRIEF.md`, `configs/` | Topic/stakeholder and team agreement | Canvas has real information; ML rationale, failure consequences, business/ML link, split rules, SLO and gates documented. Pending approval/team-size issue is not presented as resolved. |
| W1 — P1 | Ingestion/versioning, TFDV schema/anomalies, deterministic splits; `src/mlops_project/data/`, `schemas/credit_default.pbtxt`, `configs/data_schema.yaml`, tests | W0 data/target contract; compatible TFDV runtime | Provenance/checksum recorded; planned 60/20/10/10 seed-42 split reproducible and ID-disjoint; normal data passes; malformed data generates an observable error/alert and stops dependent training/deployment. Freeze reviewed TFDV schema; convert anomalies into failure behavior, not merely a displayed report. |
| W2 — P1 | Shared preprocessing and missing/outlier handling; `src/mlops_project/features/`, tests | W1 schema/split; P2/P3 artifact agreement | Fit training data only; save fitted transform state with model; training and serving import one implementation; parity test passes for same raw record; category/missing/outlier policy documented/tested. |
| W3 — P2 | Baseline, experiments/evaluation and MLflow tracking; `src/mlops_project/training/`, configs/tests | W1/W2 and metric decisions | At least three recorded runs including baseline compared; each records code version, data version, hyperparameters, metrics, artifacts and environment; actual results justify selection; final test held out from tuning and promotion. |
| W4 — P2; P3/P0 integrate | MLflow registry, status/aliases, gates, approval and rollback metadata; `src/mlops_project/registry/`, tests | W3/gates; P3 loading contract | Version/status trace to tracked run; passing candidate promoted with decision record; failing candidate rejected; actual service rollback verified by prior version and prediction. A registry-label change alone is insufficient. |
| W5 — P3 | Docker FastAPI, request validation, external access, readiness, logs/metrics; `src/mlops_project/serving/`, Docker files/tests | API/model contract; W2/W4 | External client can call documented exposed port; valid prediction and clear malformed-input failure; `/health` measures process liveness and `/ready` verifies loaded-model readiness; responses/logs trace model versions; clean-container smoke and transform parity pass. Explain serving choice. |
| W6 — P3 | Performance harness and SLO comparison; scripts/fixtures, evidence | W5 and approved SLO | Actual p50/p95 latency, throughput and errors measured under stated hardware, request mix, concurrency, duration and warmup; compare with SLO; preserve commands/results. No fabricated benchmark. |
| W7a — P1 | Feature/data drift metrics and controlled data-drift fixture; `src/mlops_project/monitoring/`, fixtures/tests | W1 reference split; monitoring contract | Method, reference/current distributions, threshold and sample rule recorded; controlled feature shift generates expected alert; scenario explicitly synthetic. |
| W7b — P2 | Label join, quality metrics and controlled concept-drift evaluation; monitoring code/tests | W3 metrics; P3 versioned requests | Evaluate labels joined to prediction/model versions; change input-to-target relationship with fixed inputs in controlled scenario and detect labeled quality degradation; document delay/windows/minimum counts. Feature shift alone is not proof of concept drift. |
| W7c — P3; P0 policy integration | Prometheus/Grafana system/quality dashboards, alerts and retraining policy | W5 metrics; W7a/W7b; threshold decisions | System and quality/drift signals observable; alerts have reasons/thresholds; policy defines retraining trigger and candidate gate. P1/P2 deliver computations so P3 is not the sole monitoring implementer. |
| W8 — P0; leads supply steps | Airflow DAG, reruns, triggered retraining and gate-to-deploy wiring; `src/mlops_project/pipelines/`, orchestration configs/tests | W1–W5 contracts; W7 trigger | One command/button runs raw data → validation → training → evaluation → registry/approval → serving; bad data halts descendants; drift produces tracked candidate with promotion/rejection; reruns/retries retain traceability. Controlled retraining uses separate simulated train/validation data. |
| W9 — P0; P1/P2 gates, P3 serving handoff | GitHub Actions CI/CD, pinned dependencies, deployment workflow | W1/W3 gates; W5 deploy contract; W8 smoke | Real CI checks code, data and model quality; retained pass and intentional fail examples; failed gates block delivery; actual successful delivery verified; fresh environment reproduces workflow. |
| W10 — P0 + everyone | Integration, architecture, report, evidence index, startup verification and demo | All packages | Independent clean run follows startup guide; all requirements map to code/test/evidence; report has actual results/AI-use disclosure; normal/abnormal demo rehearsed; every member contributes/explains code. |

When a contract changes, update it and notify producers/consumers in the same PR. Do not complete a package while its consumer cannot use its output.

## Deadline sprint: 4–5 October

If the initial 4 October session starts late, prioritize the first integrated prediction and move incomplete targets to explicit 5 October owners. The schedule does not guarantee feasibility or excuse missing requirements.

| Bangkok date/time | Checkpoint | Parallel work / outcome |
| --- | --- | --- |
| 4 Oct, first available 60–90 minutes | C0: decisions, roles, compatibility | P0 records brief/contracts and checks Airflow/CI runtime; P1 proves TFDV in pinned container; P2/P3 agree artifact/API. Resolve topic approval, roster and thresholds promptly. |
| 4 Oct, remaining available session | C1: first vertical slice | P1 supplies validated sample/split/transform; P2 creates baseline artifact and fully tracked run; P3 serves in Docker with health/version; P0 wires Airflow steps and CI skeleton. P1/P2 also prepare distinct feature/labeled drift fixtures. |
| 5 Oct, 08:30–12:00 | C2: functional components | Complete validation failure/alert, three experiments, registry gates, external API, system metrics and shared preprocessing. P0 integrates raw-data-to-prediction. Give every requirement gap a lead/due time. |
| 5 Oct, 12:00–16:00 | C3: full lifecycle | P3 captures performance/Grafana evidence; P1/P2 supply drift/quality evidence; P0 runs detection → Airflow retraining → candidate → gate → promotion/rejection. P2/P3 verify actual rollback. Exercise intentional CI failures. |
| 5 Oct, 16:00–19:00 | C4: independent clean run | Another member follows startup docs in fresh environment. Exercise normal/malformed data, pipeline halt, SLO comparison, registry rejection/rollback, both drift scenarios, retraining, and all three CI gates. Fix integration gaps. |
| 5 Oct, 19:00–21:30 | C5: report/demo freeze | Every member supplies measurements, reproducible evidence, contributions and AI-use disclosure. P0 assembles report; peers check claims. Freeze features; repair only blocking failures/inaccurate docs. |
| 5 Oct, 21:30–23:30 | C6: submission/buffer | Confirm submission method/files, tested release commit, repository/data access and permitted assets. Submit before 23:59 and retain confirmation. |

Keep each mandatory item open in [REQUIREMENTS.md](REQUIREMENTS.md) until observed passing. If an item remains incomplete near submission, record the true status and consult the instructor through the team's normal channel; do not fabricate evidence or assume an extension.

## Integration and release checks

Each checkpoint handoff includes a usable output, exact verification command/request, tested commit/data/model versions, observed outcome and remaining gaps. Use [TEAM_WORK.md](TEAM_WORK.md)'s PR template.

An integration increment passes when contracts align, consumer examples work, meaningful normal/failure checks pass, versions are traceable and evidence is linked. Undefined gates and planned commands do not count as verification.

Before submission, additionally verify:

- Real problem/Canvas and business/ML metrics; topic approval and roster status accurately stated.
- Reproducible ingestion/splits, TFDV rejection with halt/alert, and common training-serving transform.
- Baseline and three fully tracked runs; held-out final evaluation and justified choice.
- Version/status registry, rejection, recorded approval/promotion and actual serving rollback.
- External Docker API, health, logs/metrics, serving rationale, actual p50/p95/throughput versus SLO.
- System/labeled quality monitoring; distinct data/concept drift, thresholds and retraining lifecycle.
- One-command/button Airflow DAG and actual CI/CD code/data/model pass/fail checks.
- Pinned environment and independently verified fresh startup; meaningful branch/PR history.
- Architecture/report/AI disclosure, every member's evidence and presentation segment.
- Submission files/receipt confirmed before deadline.

Use [TESTING_AND_DEMO.md](TESTING_AND_DEMO.md) for scenarios and [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) for lead contributions. Repeat checks when changes or unresolved failures justify them.

## Risks and responses

| Risk | Response / owner |
| --- | --- |
| Four members versus course 5–7 | P0 arranges another member or instructor exception; keep status visible until resolved. |
| Topic/stakeholder approval pending | P0 resolves immediately; dataset selection does not answer real-user/value questions. |
| TFDV/Airflow/version incompatibility | P1/P0 test early in Docker and pin supported versions; do not silently replace selected tools. |
| P0/P3 overloaded | P1 owns feature drift; P2 owns labeled quality/concept drift; P3 owns dashboards/service adapter; all supply callable steps and their own tests/report evidence. Expand roster if possible. |
| Schema/artifact mismatch | Producers/consumers review common contracts; integrate one real prediction early. |
| Delayed labels | P2 documents delay and controlled labeled demo; unlabeled alarms remain suspected change. |
| Candidate regression | P2 rejects with gates; P3/P0 retain previous artifact/deployment and verify rollback. |
| Hidden local state | P0 pins workflow; P3 containers runtime; another member verifies fresh startup. |
| Claims exceed evidence | Leads/reviewers link observed results to tested commit; P0 audits requirement index. |

## Presentation preparation: 6–12 October

Follow instructor rules for changes after submission; do not assume later work can replace submitted artifacts.

- **6–8 Oct:** rerun submitted version, rehearse failures, capture backup evidence and cross-train reviewers.
- **9–10 Oct:** rehearse a 12-minute script with each member speaking/demonstrating, plus technical questions.
- **11 Oct:** verify data/images/dependencies, live demo, normal/malformed fixtures and rollback; retain submitted commit and backup evidence.
- **12 Oct, before 08:30:** be ready for first slot; confirm exact team slot separately. Prepare for instructor normal/abnormal cases and 3-minute questions.

Return to the [README](../README.md) for the documentation index and [GETTING_STARTED.md](GETTING_STARTED.md) for the startup checklist.
