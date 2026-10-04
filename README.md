# Credit Default MLOps — CP413008

Team project for building and demonstrating an end-to-end ML production system: raw data → validation → training → evaluation → approval → containerized API → monitoring → retraining and rollback.

**Current status: all four components integrated and the full lifecycle runs in Docker.** On 4 October 2026 `scripts/run-all.sh` took raw UCI 350 data through TFDV validation, P1 splits and shared preprocessing, three tracked MLflow experiments, registry, P3's candidate benchmark, the approval gate and a verified deployment to the FastAPI container (run `full-run-002`: AP 0.5375, p95 204.5 ms, 60 req/s, 0% errors). Rollback to the previous version, feature-drift and concept-drift alerts were also demonstrated. See the [evidence index](docs/evidence/INDEX.md), [serving and monitoring](docs/SERVING.md), [P1 handoff](docs/P1_HANDOFF.md), [P2 handoff](src/mlops_project/training/README.md) and [P0 handoff](docs/P0_HANDOFF.md).

The original [course requirements](./โครงงานรายวิชา%20CP413008%20Machine%20Learning%20Engineering%20for%20Production.docx.md) are the authority. The project docs translate them into work items; they do not replace the brief.

## Dates and team

- Submission: **5 October 2026, 23:59, Asia/Bangkok**, as stated in the brief. Plan to submit earlier.
- Presentation: **12 October 2026, from 08:30, Asia/Bangkok**. Each group has 12 minutes plus 3 minutes of questions.
- Your confirmed team: **you plus three friends, four people total**. The brief requires **5–7**; confirm an instructor exception or add members. The plan remains usable for your current four-person team.

## Start here

1. Read [Project brief and AI Canvas](docs/PROJECT_BRIEF.md) and [Dataset guide](docs/DATASET.md). Confirm stakeholder evidence, success metrics, topic approval and uniqueness with the instructor.
2. Find your named assignment and reviewer in [Team work](docs/TEAM_WORK.md) and follow [Contributing](CONTRIBUTING.md).
3. Freeze the first data/API/model contracts in [Interface contracts](docs/INTERFACE_CONTRACTS.md). This lets everyone implement in parallel.
4. Follow [Development plan](docs/DEVELOPMENT_PLAN.md), including the deadline sprint for 4–5 October. Work toward an early complete path through the system.
5. Use [Getting started](docs/GETTING_STARTED.md) to distinguish setup possible now from the runtime setup the team must implement.
6. Track completion against [Requirements and evidence](docs/REQUIREMENTS.md). A feature is complete when its behavior is demonstrated and the evidence is saved.

## Your work and your friends' work

The three friends are assigned in the order you supplied their handles. P0's handle comes from the existing repository remote and Git identity. Every member writes code, reviews a PR, contributes report evidence, and presents their component.

| Role | Main responsibility | First deliverable |
| --- | --- | --- |
| **P0 — you, @uzuz3737** | Integration, Airflow DAG, GitHub Actions, deployment/rollback controller and release | Integration framework implemented; finish real integration after adapter handoffs |
| **P1 — @OuanEng** | UCI ingestion, reproducible splits, TFDV schema/validation, shared features and data-drift statistics | Data manifest, bad-data fixture, fitted-feature contract and callable data stages |
| **P2 — @pairot230** | Baseline, three experiments, MLflow tracking/registry, model gates and labeled quality evaluation | Versioned model bundle, evaluation, registry states and callable model stages |
| **P3 — @thanachaithongbai-hue** | FastAPI, artifact watcher/ACK, performance, Prometheus/Grafana, prediction/label collection and alerts | Container API, exact-version load/ACK, candidate benchmark and dashboards |

See [Team work](docs/TEAM_WORK.md) for file ownership, handoffs, review pairs and expansion to five–seven people. P1/P2 supply monitoring computations so P3 can focus on collection, metrics and dashboards. Each owner integrates their component with P0.

[CODEOWNERS](.github/CODEOWNERS) marks their source/test paths. Repository administrator configuration is still needed for collaborator access and required reviews.

## Documentation

| Document | What it answers |
| --- | --- |
| [P0 handoff](docs/P0_HANDOFF.md) | What is implemented, what each friend must connect, and what remains unverified? |
| [Project brief](docs/PROJECT_BRIEF.md) | What problem are we solving, for whom, and how will success be measured? |
| [Dataset guide](docs/DATASET.md) | How do UCI columns, labels, splits and drift simulations work? |
| [Development plan](docs/DEVELOPMENT_PLAN.md) | What should be built first, what can run in parallel, and when is each milestone complete? |
| [Team work](docs/TEAM_WORK.md) | Who owns each deliverable and what must they hand over? |
| [Getting started](docs/GETTING_STARTED.md) | How should a new member set up, contribute, and eventually run the system? |
| [Architecture](docs/ARCHITECTURE.md) | How do all required tool functions and services fit together? |
| [Interface contracts](docs/INTERFACE_CONTRACTS.md) | What data, artifacts, API responses and deployment events must components exchange? |
| [Requirements](docs/REQUIREMENTS.md) | Which course requirement does each implementation and piece of evidence satisfy? |
| [Testing and demo](docs/TESTING_AND_DEMO.md) | How will we prove normal behavior, failures, drift, retraining and rollback? |
| [Serving and monitoring](docs/SERVING.md) | How do I call the API, what does it reject, how fast is it, and how are drift/quality alerts raised? |
| [Report template](docs/REPORT_TEMPLATE.md) | What should we submit and explain, including AI assistance and individual contributions? |

## Selected stack

Your selected tools are **Docker, TFDV, MLflow, MLflow Registry, Airflow, FastAPI, Prometheus + Grafana, and GitHub Actions**, with Git/GitHub for version control. A CPU-based scikit-learn model is the proposed initial modeling choice. P0 pins framework dependencies and integration images; friends must pin/verify component compatibility. The [architecture](docs/ARCHITECTURE.md) explains each tool's responsibility.

The course requires nine tool functions, a real stakeholder, reproducible data and model work, at least three experiment rounds, an externally callable API, measured performance, both data and concept drift, working CI/CD and rollback. All are covered in the [requirements checklist](docs/REQUIREMENTS.md).

## Run status

Python **3.11** is the declared container/CI version; P0 host checks ran with **3.12.14**. These commands check the integration framework:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements/dev.lock
ruff check .
ruff format --check .
pytest
python scripts/check_docs.py
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m mlops_project.pipelines.preflight --config configs/project.yaml --project-root .
```

Preflight is **expected to fail** on this initial handoff with missing friend adapters/unset numerical gates. Do not replace them with adapters returning dummy success.

When Docker is available, copy `.env.example` to ignored `.env` and run `./scripts/run-all.ps1 -InfrastructureOnly` for integration services. The full PowerShell command is `./scripts/run-all.ps1 -Config configs/project.yaml`; Bash uses `bash scripts/run-all.sh --config configs/project.yaml`. Full mode requires real adapters and approved gates, waits for the actual DAG terminal state, and verifies the deployment. Container behavior is still unverified here. Follow [Getting started](docs/GETTING_STARTED.md).

[Framework CI](.github/workflows/ci.yml) checks code and fixture-based data/model contracts. [Container integration](.github/workflows/full-pipeline.yml) accepts manual `infrastructure` (default) or `full` scope. Infrastructure mode checks P0 service startup and real Airflow DAG import. Select full for the actual lifecycle; its container preflight fails when friend adapters/gates are missing. Delivery uses the GitHub runner's own Compose environment. Real TFDV/model/course acceptance needs actual component and full-run evidence.
