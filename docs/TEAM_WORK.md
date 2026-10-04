# Team work and handoffs

The confirmed starting team is **@uzuz3737 (you), @OuanEng, @pairot230 and @thanachaithongbai-hue**. Friends are assigned in the order supplied by the user. P0's handle is inferred from the existing Git remote and Git identity. Availability and student IDs remain to be recorded.

The course requires **5–7 members**. The confirmed four-person roster does not meet that requirement unless an instructor exception is granted. P0 should arrange another member or resolve that exception promptly. Expansion options below preserve the same work packages.

The selected project is credit-default prediction using the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350). The selected stack is Docker, TFDV, MLflow tracking/registry, Airflow, FastAPI, Prometheus/Grafana and GitHub Actions with Git/GitHub. P0's integration framework is implemented; real P1–P3 components and container acceptance remain pending. See [P0 handoff](P0_HANDOFF.md).

Use [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for deadlines and package acceptance, [INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md) for boundaries, and [REQUIREMENTS.md](REQUIREMENTS.md) for evidence coverage.

## Four-person assignments

| Role | GitHub member | Implementation ownership | Reviewer / backup | Personal evidence, report and demo |
| --- | --- | --- | --- | --- |
| **P0 — user: integration, Airflow and CI/CD** | **@uzuz3737** | Canvas/decisions with team; integration configuration/glue; Airflow DAG and retraining orchestration; GitHub Actions gate integration; deployment wiring; release and report assembly | @thanachaithongbai-hue reviews runtime/delivery; @pairot230 backs up DAG/model-gate integration | Framework code and failure tests; real full-run/hosted CI evidence pending; automation/integration report section; demonstrate one-command lifecycle. |
| **P1 — data, TFDV and shared features** | **@OuanEng** | Acquisition/provenance, TFDV schema/rejection, deterministic split, missing/outlier policy, common preprocessing; feature drift computation/fixture | @pairot230 reviews leakage/model input; @thanachaithongbai-hue reviews serving parity | Data/features/drift tests; bad-data halt/alert, repeatable split and parity evidence; dataset/validation report section; invalid-data/feature-drift demo. |
| **P2 — models, MLflow and quality** | **@pairot230** | Baseline/three experiments, evaluation, MLflow tracking/registry, model gate/status metadata; label join and labeled quality/concept-drift computation/fixture | @OuanEng reviews split; @uzuz3737 reviews gates; @thanachaithongbai-hue reviews loading/rollback | Training/registry/quality tests; six tracking fields, comparisons, rejection and labeled-drift evidence; model/registry/quality report section. |
| **P3 — FastAPI, Docker, SLO and dashboards** | **@thanachaithongbai-hue** | API/request contract, readiness, logs/metrics, model loading/swap, performance harness; Prometheus/Grafana dashboards/alerts consuming P1/P2 metrics | @uzuz3737 reviews watcher/ACK and benchmarks; @OuanEng reviews transforms; @pairot230 reviews version/labels | Serving/performance/monitor tests; external API, measured p50/p95/throughput, health/alerts and actual rollback; serving/monitoring report section. |

P0 is the user by default because the request asks for work for the user and friends. The team can reassign leads by agreement. Every package still has one accountable lead and a consumer reviewer.

P0 owns the atomic deployment/rollback controller and orchestration. P1 implements feature drift; P2 implements labeled quality/concept drift and registry finalization; P3 implements the serving watcher/load acknowledgement and candidate benchmark. P1–P3 supply independently callable component stages and their own tests, leaving P0 to integrate those actual components.

## Fair contribution

Each member needs substantive implementation, meaningful testing, reproducible evidence, report material and a speaking/demo segment. The course uses commits, peer assessment and technical answers when adjusting individual marks.

- Record authored commits and reviewed PRs honestly. Pair work records each contribution and uses co-author attribution where appropriate.
- Balance effort and dependency pressure, not file counts. Move a concrete subtask when a lead is overloaded; update its owner and consumer.
- The component lead tests/documents their code. P0 assembles report sections but does not write every person's explanation.
- Each critical component has a second person able to run and explain it. Review and handoff are real work.
- Record availability at the first checkpoint. Do not leave a member with only slides, screenshots or cosmetic work.
- Each lead discloses where AI assisted and verifies they can explain all submitted code.

P2 supplies model evidence consumed by P0's gate policy; P1 supplies data-gate/TFDV checks; P3 supplies the actual serving watcher/ACK consumed by P0's deployment controller. Keep shared wiring under P0 review.

## Expand to the course's 5–7 people

Adding members changes ownership, not requirements or acceptance criteria. Name the additional person and a backup at the first checkpoint after joining.

| Team size | Assignment change |
| --- | --- |
| **5 members** | Add **P4 — monitoring/retraining policy**. Transfer Prometheus/Grafana alert/dashboard integration and retraining policy from P3/P0 to P4. P1/P2 retain feature/labeled-quality definitions and computation, working with P4. P4 owns monitor integration tests/evidence/report and the drift demo. |
| **6 members** | Keep P4; add **P5 — orchestration/CI/CD**. Transfer Airflow DAG, GitHub Actions and delivery wiring from P0 to P5. P0 owns framing/integration, cross-component checks, release and report assembly with substantive integration code/tests. Component leads still supply callable steps/gate scripts. |
| **7 members** | Keep P4/P5; add **P6 — registry/release lifecycle**. Transfer registry finalization from P2 and deployment/rollback coordination from P0 to P6 with P3. P2 keeps baseline, experiments/tracking, evaluation and labeled quality. P6 owns lifecycle tests/evidence/report/demo. |

With six people, the split becomes P0 integration, P1 data/features, P2 models/tracking/registry, P3 API/performance, P4 monitoring/policy and P5 orchestration/CI. If approved to remain at four, document the exception rather than claiming the course permits four.

## Repository ownership

P0 paths exist; P1–P3 create their assigned modules. [CODEOWNERS](../.github/CODEOWNERS) uses P0 as the default owner and overrides each friend's paths, including separate monitoring modules. GitHub collaborator access and review enforcement require administrator setup.

| Area | Lead | Contract / review boundary |
| --- | --- | --- |
| `src/mlops_project/data/`, `schemas/credit_default.pbtxt`, `configs/data_schema.yaml` | P1 | Frozen reviewed TFDV schema; anomalies cause explicit failure/alert; manifest records source/version and disjoint partitions. |
| `src/mlops_project/features/` | P1 | P2/P3 import one transform implementation; fitted state travels with model; neither fits serving/replay inputs. |
| `src/mlops_project/training/`, `src/mlops_project/registry/` | P2 | P3 consumes agreed model/transform bundle; P0 calls train/evaluate/register steps; gate definitions reviewed by P0. |
| `src/mlops_project/serving/`, API dependency lock | P3 | P0 writes desired deployment manifest and waits for P3 watcher ACK/version/readiness; P3 benchmarks candidates; P1 reviews parity. |
| `src/mlops_project/monitoring/feature_drift.py`, `quality.py`, `exporters.py` | P1 / P2 / P3 respectively | P0 consumes a defined retraining trigger; P3 exports P1/P2 computations. Do not treat the whole monitoring directory as one person's work. |
| `src/mlops_project/pipelines/`, Airflow configuration, run wrappers, GitHub Actions | P0 | Implemented framework/gate policy; leads supply real callable stages and measurements. Wrappers wait for DAG and deployment verification; Docker acceptance pending. |
| Dependency pins / `configs/` | P0 coordinates; relevant lead authors | Verify compatible versions with P1/P3. One nominated editor handles each shared file; record thresholds/seeds/versions. |
| `tests/` | Component lead; P0 cross-component tests | Meaningful normal/failure checks; lead names a consumer reviewer. |
| `docs/` and root `README.md` | Leads author sections; P0 assembles | Link requirements, code/tests and evidence; only verified commands/results become final claims. |
| `artifacts/`, data and runtime output | Producer defined in contracts | Ignore large models/data, databases, generated outputs and secrets in Git; document retrieval/reproduction and manifests. |
| `docs/evidence/` | All leads, indexed by P0 | Optional small sanitized evidence during implementation; this plan does not itself create the folder or logs. |

Accept predictor/target mapping, planned 60/20/10/10 stratified seed-42 split, ID separation, bundle metadata, event fields, label join, trigger and step interfaces in [INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md). Exclude ID/target from predictors and keep final test out of training, tuning and promotion gates.

## Branch and PR workflow

Follow repository-specific rules if later added. Until then:

1. Start from the agreed current base branch. P0's integration branch is `codex/p0-integration`; friends can use `codex/p1-data-validation`, `codex/p2-model-tracking` or `codex/p3-serving`. Friend branch names are suggestions until created.
2. Keep each PR a coherent working increment with a clear consumer and acceptance criteria. Make it small enough to review before the deadline.
3. Update contracts before or with a breaking change. Notify producing and consuming leads; do not silently change field names, types, paths, metrics or artifact layouts.
4. Attach actual tests/run evidence and report notes using the template below. The reviewer checks the consumer example and meaningful failure behavior.
5. P0 merges integration-sensitive work after review and required checks pass. Fix failures on the branch; a pending/failed check cannot be reported as passed.
6. P0 runs affected integration checks after merge. Record the commit actually demonstrated by the evidence.

Avoid concurrent edits to shared files. Agree an editor or submit a small patch for assembly. Never overwrite another person's uncommitted work.

## PR and daily handoff template

Copy this into each PR or the team's usual handoff channel. Replace placeholders with actual evidence.

```markdown
Work package: W... / lead: ... / reviewer: ...
Requirement(s): links to docs/REQUIREMENTS.md entries
Problem and resulting behavior: ...
Changes: ...
Inputs / outputs / contract version: ...
Consumer example: verified command/request + expected result
Validation: exact command(s), environment, actual pass/fail and log
Failure case: fixture, expected stopping/alert behavior, observed result
Traceability: commit SHA, data version, run ID, model version, config/seed
Evidence: links to actual CI run, artifact, log, screenshot or output
Report contribution: section and actual measured findings
AI assistance: tool, assisted parts, verification/explanation owner
Dependencies / remaining gaps: ...
Next owner action and due time in Asia/Bangkok: ...
```

Each daily checkpoint includes a reviewed increment, usable consumer output, actual normal/failure result, remaining requirement gaps and the next dependency. A blocked handoff includes the exact contract/error and a minimal reproduction.

The deadline sprint on 4–5 October needs several integration checkpoints, not just a final merge. See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the timetable.

## Handoff readiness

| Handoff | Producer → consumer | Ready when |
| --- | --- | --- |
| Data/schema/split | P1 → P2/P0 | Version manifest, TFDV schema, access, split and fixtures usable; bad-data failure/alert contract explicit. |
| Shared transforms | P1 → P2/P3 | Same implementation/fitted state loadable; raw-record parity verified. |
| Model artifact | P2 → P3/P0 | Bundle has fitted transform, input signature, metadata and version; consumer loads/predicts. |
| Registry/promotion/rollback | P2 → P3/P0 | Gate/status/approval semantics agreed; old/current artifact retrieval works; actual service version/prediction confirms rollback. |
| Request and operational metrics | P3 → P2/P1/P0 | Versioned request IDs/timestamps and approved fields available; outcomes join correctly. |
| Feature drift | P1 → P3/P0 | Reference/current windows, method/threshold/sample rule and repeated feature-shift fixture produce agreed metric/event. |
| Labeled quality/concept drift | P2 → P3/P0 | Labels, model version, windows/minimum count and fixed-input/changed-label-relation scenario support quality claim. |
| Trigger and retraining | P3 policy + P1/P2 signals → P0/P2 | Reason/threshold and candidate policy defined; Airflow runs independent simulated training/validation data and gates candidate. |
| Callable pipeline steps | P1–P3 → P0 | Inputs/outputs, errors and rerun behavior documented; independently runnable; no copying of component logic into DAG. |
| Release material | Everyone → P0 | Evidence references tested commit/config; report section and demo segment ready; claimed requirements traceable. |

Concept drift concerns the relationship between inputs and targets. A feature-distribution alarm alone is insufficient proof. The controlled concept scenario uses fixed inputs with a known changed label relationship; label/quality evidence demonstrates the effect. Synthetic scenarios and separate retraining partitions must be identified honestly.

## Report and 12-minute presentation

Every lead writes their section in [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md), covering design rationale, real measurements, failures/limitations, evidence and AI use. P0 combines sections; a peer verifies every result against evidence. P0 should not carry all report writing.

Use the canonical [twelve-minute rehearsal script](TESTING_AND_DEMO.md#twelve-minute-presentation-rehearsal). It assigns each person a technical speaking/demo segment; update that script after rehearsals rather than maintaining competing schedules.

The source allows **3 further minutes for questions**. With extra members, transfer corresponding segments to P4/P5/P6 so everyone contributes. Rehearse transitions and use backup evidence only with clear disclosure. All members must explain their code and at least one neighboring component.

## Assignment board

Update status from observed evidence. A merged PR can still lack integration or proof.

| Package | Named lead | Named reviewer/backup | Status | Next handoff / Bangkok due time | PR / evidence |
| --- | --- | --- | --- | --- | --- |
| W0 framing/integration | @uzuz3737 | @pairot230 | In progress: config/contracts exist; stakeholder evidence pending | Resolve decisions at C0 | [P0 handoff](P0_HANDOFF.md) |
| W1 data/TFDV | @OuanEng | @pairot230 | Not started | Ingest/validate/split adapters by C1 | — |
| W2 shared features | @OuanEng | @thanachaithongbai-hue | Not started | Shared fitted-feature contract by C1 | — |
| W3 models/tracking | @pairot230 | @OuanEng | Not started | Baseline and two candidates by C2 | — |
| W4 registry/lifecycle | @pairot230 | @uzuz3737 | Not started | Register/evaluation evidence by C2 | — |
| W5 API/Docker | @thanachaithongbai-hue | @uzuz3737 | Not started | API watcher/ACK and candidate benchmark by C1–C2 | — |
| W6 performance/SLO | @thanachaithongbai-hue | @uzuz3737 | Not started | Candidate benchmark by C3 | — |
| W7a feature drift | @OuanEng | @thanachaithongbai-hue | Not started | Feature-shift scenario by C3 | — |
| W7b labeled quality/concept | @pairot230 | @thanachaithongbai-hue | Not started | Labeled scenario by C3 | — |
| W7c dashboard/policy | @thanachaithongbai-hue; @uzuz3737 integrates | @OuanEng / @pairot230 | Not started: starter service-up panels only | Actual metrics/alerts by C3 | — |
| W8 Airflow/retraining | @uzuz3737 | @pairot230 | In progress: framework implemented; real adapters/container run pending | Integrate handoffs C1–C3 | [P0 handoff](P0_HANDOFF.md) |
| W9 CI/CD/reproducibility | @uzuz3737 | @thanachaithongbai-hue | In progress: workflows/locks exist; hosted/full-run evidence pending | Full pass/fail and clean run C3–C4 | [P0 handoff](P0_HANDOFF.md) |
| W10 report/demo/release | @uzuz3737 + all | All | In progress: docs/roster; real results/report/demo pending | Everyone supplies sections at C5 | [Report template](REPORT_TEMPLATE.md) |

Use **not started**, **in progress**, **blocked with reason**, **ready for review**, or **verified done**. Only verified done means acceptance, consumer integration and evidence are checked.

Return to the [README](../README.md) for startup instructions and the documentation index.
