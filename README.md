# Credit Default MLOps — CP413008

End-to-end ML production system for **credit-card default prediction** ([UCI dataset 350](https://archive.ics.uci.edu/dataset/350), 30,000 clients, 23 features).

```text
raw UCI data → TFDV validation → split → shared preprocessing → 3 tracked experiments (MLflow)
   → registry → candidate benchmark → approval gate → deploy to FastAPI → monitoring → rollback / retraining
```

One command runs the whole lifecycle in Docker, orchestrated by Airflow.

**Status (4 Oct 2026):** full lifecycle verified in Docker — run `full-run-002` passed every gate (AP 0.5375, p95 204.5 ms, 60 req/s, 0% errors) and was deployed; rollback, feature-drift and concept-drift alerts were demonstrated. Evidence: [docs/evidence/INDEX.md](docs/evidence/INDEX.md).

## Quick start (Docker)

**Needs:** Docker Desktop (Linux containers) with Compose 2.20+, Git, ~20 GB free disk, internet for images and the UCI download.

```bash
git clone https://github.com/uzuz3737/ml-ops.git
cd ml-ops
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
bash scripts/run-all.sh --config configs/project.yaml --run-id my-first-run
```

On Windows use Git Bash for that last command and prefix it with `MSYS_NO_PATHCONV=1`, or run `.\scripts\run-all.ps1 -Config configs/project.yaml -RunId my-first-run` in PowerShell.

The script builds the images, checks every stage exists (preflight), starts all services, triggers the Airflow DAG and waits until it finishes. The first run takes about 20–30 minutes (image builds plus a 5-minute load test). It ends with `Airflow run state: success` and the API serving the new model.

| Service | URL | Login |
| --- | --- | --- |
| Prediction API | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | `airflow` / `airflow-local-demo` |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | `admin` / `grafana-local-demo` |
| Prometheus | http://localhost:9090 | — |

Logins come from `.env`. If port 5000 is blocked (Windows sometimes reserves it), set `MLFLOW_HOST_PORT=5050` in `.env`.

## Try it

```bash
curl -s http://localhost:8000/ready
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/predict.json
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/invalid/unknown-category.json   # 422
```

PowerShell: use `curl.exe` instead of `curl`.

Demo scenarios (run after a successful pipeline run):

```bash
# roll back to the previous approved model, no retraining
docker compose exec pipeline-worker python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "demo rollback"

# replay held-out rows through the API, send labels, run monitoring
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario healthy
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario feature-drift   # data drift alert
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario concept-drift   # quality alert
```

Rollback needs two successful runs (a current and a previous model). Run outputs are in `artifacts/runs/<run-id>/`, deployments in `artifacts/deployments/`, alerts in `artifacts/monitoring/alerts/`.

Stop everything with `docker compose --profile application down` (add `-v` to also delete MLflow/Airflow/Grafana data).

## Develop and test (no Docker)

Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
pip install --require-hashes -r src/mlops_project/training/requirements-dev.lock
pip install --require-hashes -r requirements/serving.lock
ruff check . && ruff format --check .
pytest
python scripts/check_docs.py
```

TFDV only installs on Linux, so its tests are skipped on Windows/macOS; CI and the Docker worker run them.

## How it fits together

| Step | Tool | Code |
| --- | --- | --- |
| Ingest, validate, split, features, data drift | TFDV, pandas, scikit-learn | `src/mlops_project/data/`, `features/`, `monitoring/feature_drift.py` |
| Experiments, registry, labeled quality | MLflow tracking + registry | `src/mlops_project/training/`, `registry/`, `monitoring/quality.py` |
| API, deployment watcher, benchmark, metrics export | FastAPI, Prometheus, Grafana | `src/mlops_project/serving/`, `monitoring/exporters.py`, `monitoring/run.py`, `infra/` |
| Orchestration, gates, deploy/rollback, CI | Airflow, GitHub Actions, Docker Compose | `src/mlops_project/pipelines/`, `dags/`, `compose.yaml`, `.github/workflows/` |

Configuration lives in `configs/`: `project.yaml` (stages, paths), `quality_gates.yaml` (AP floor 0.40, max regression 0.02, p95 ≤ 300 ms, ≥ 20 req/s, ≤ 1% errors), `monitoring.yaml` (drift and quality thresholds, retraining policy), `data_schema.yaml`.

## Team

| Role | Member | Owns |
| --- | --- | --- |
| P0 | @uzuz3737 | Integration, Airflow DAG, gates, deployment/rollback controller, CI/CD |
| P1 | @OuanEng | UCI ingestion, TFDV schema/validation, splits, shared features, feature drift |
| P2 | @pairot230 | Baseline + experiments, MLflow tracking/registry, labeled quality |
| P3 | @thanachaithongbai-hue | FastAPI serving, deployment watcher, benchmark/SLO, Prometheus/Grafana, monitoring |

File ownership is in [CODEOWNERS](.github/CODEOWNERS); how to contribute is in [CONTRIBUTING](CONTRIBUTING.md).

## Documentation

| Document | What it answers |
| --- | --- |
| [Evidence index](docs/evidence/INDEX.md) | Which runs prove which requirement? |
| [Serving and monitoring](docs/SERVING.md) | API usage and errors, SLO results, alerts, rollback and drift demos |
| [Getting started](docs/GETTING_STARTED.md) | Detailed setup, service addresses, troubleshooting |
| [Architecture](docs/ARCHITECTURE.md) | How the nine tool functions and services fit together |
| [Interface contracts](docs/INTERFACE_CONTRACTS.md) | Data, artifact, API and deployment formats between components |
| [Requirements](docs/REQUIREMENTS.md) | Course rubric mapped to owners and evidence |
| [Project brief](docs/PROJECT_BRIEF.md) / [Dataset guide](docs/DATASET.md) | Problem, stakeholder, metrics; UCI columns and splits |
| [P0](docs/P0_HANDOFF.md) / [P1](docs/P1_HANDOFF.md) / [P2](src/mlops_project/training/README.md) handoffs | What each component implements and expects |
| [P2 results](src/mlops_project/training/REPORT.md) | Experiment metrics and model selection |
| [Testing and demo](docs/TESTING_AND_DEMO.md) / [Report template](docs/REPORT_TEMPLATE.md) | Demo plan; what to submit |
| [Team work](docs/TEAM_WORK.md) / [Development plan](docs/DEVELOPMENT_PLAN.md) | Work packages and schedule |

Course brief: [CP413008 requirements](./โครงงานรายวิชา%20CP413008%20Machine%20Learning%20Engineering%20for%20Production.docx.md). Submission **5 Oct 2026 23:59**, presentation **12 Oct 2026 08:30** (Asia/Bangkok).
