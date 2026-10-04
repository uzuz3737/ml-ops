# Getting started

**P0 integration code exists; the complete credit-default system is pending P1–P3 adapters.** Host checks cover framework behavior. Docker was unavailable on the implementation host, so container instructions require a real Docker-machine run before claiming runtime acceptance. See [P0 handoff](P0_HANDOFF.md) and [named assignments](TEAM_WORK.md).

## Clone and contribute

```powershell
git clone https://github.com/uzuz3737/ml-ops.git
Set-Location ml-ops
git switch -c codex/p1-data-validation
```

Use your assigned branch suggestion from [Contributing](../CONTRIBUTING.md), your own Git identity and your own clone. Read [Dataset](DATASET.md), [contracts](INTERFACE_CONTRACTS.md) and the callable-stage details in [P0 handoff](P0_HANDOFF.md). Supply a minimal genuine component handoff early; record missing inputs and the responsible member.

## Host development

Python **3.11** is the declared container/CI version. P0 host checks ran on Python **3.12.14**. The hash-pinned `requirements/dev.lock` includes the framework's runtime and development tools; `requirements/runtime.lock` is for the integration worker. Friends pin their own heavier TFDV/model/API dependencies and verify their Linux images.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements/dev.lock
$env:PYTHONPATH = (Resolve-Path .\src).Path
ruff check .
ruff format --check .
pytest
python scripts/check_docs.py
python -m mlops_project.pipelines.preflight --config configs/project.yaml --project-root .
```

Ruff, pytest and the docs checker should pass. **Preflight should fail in the initial checkout** with missing adapters/unconfigured numerical gates. This protects the runtime from reporting an incomplete project as a successful pipeline. Tests use isolated fixtures; they do not acquire UCI data or train real models.

On Bash, set `PYTHONPATH=src` before a direct module invocation. Pytest obtains `src` through `pyproject.toml` configuration. Keep `.venv`, data, artifacts, secrets and caches ignored.

## Docker startup

Prerequisites: Docker Engine with Linux containers and Docker Compose **2.20+**, available ports, image/package network access, and enough resources for Airflow plus the component images. Check `docker version` and `docker compose version` first.

Copy `.env.example` to ignored `.env` and review the local classroom configuration: Airflow/Grafana login, Postgres credentials and Linux container user IDs. Services publish on localhost by default. For Windows Docker Desktop, use a working Linux-container backend; TFDV is supplied by P1 in its compatible Linux image.

```powershell
Copy-Item .env.example .env
.\scripts\run-all.ps1 -InfrastructureOnly -TimeoutSeconds 1800
docker compose ps
```

Infrastructure-only startup starts integration services and omits the API/full DAG. It does not establish valid data, a trained model or successful deployment. Pins include Airflow **2.10.5 / Python 3.11**, MLflow **2.19.0** and Postgres **16.6**. They are configuration choices awaiting build/run verification, rather than a tested TFDV compatibility matrix.

After every assigned adapter imports successfully and gates are approved:

```powershell
.\scripts\run-all.ps1 -Config configs/project.yaml -TimeoutSeconds 7200
```

```bash
bash scripts/run-all.sh --config configs/project.yaml --timeout-seconds 7200
```

Both wrappers accept a run ID (`-RunId` / `--run-id`) for traceability and reruns. They preflight, build/start Compose, trigger the Airflow DAG, wait for terminal success/failure and check stored final verification. A trigger/run ID alone does not count as success. Do not bypass preflight or replace adapters with successful dummy outputs.

The full wrapper defaults to 7,200 seconds; infrastructure examples allow 1,800 seconds. Each adapter has a configured 1,800-second subprocess deadline, and the worker client allows 1,910 seconds for that stage, receipt/controller handoff and journal-lock overhead. The manual full workflow allows 150 minutes for setup, the two-hour lifecycle budget, logs and cleanup. Timeout results are failures and may require inspecting an Airflow run that is still active.

P1/P2 can supply a hash-pinned worker dependency lock through `WORKER_REQUIREMENTS` in `.env`; P3 supplies its API lock through `API_REQUIREMENTS`. Match those build arguments to real committed lock files and verify compatible dependencies before a full run.

## Layout and ownership

```text
src/mlops_project/pipelines/   # P0 runner, worker/client, preflight and retraining
src/mlops_project/pipelines/gates.py # P0 policy; P2/P3 supply real measurements
src/mlops_project/pipelines/deployment.py # P0 deployment/rollback controller
dags/                         # P0 Airflow graph
configs/                      # Shared versioned config and pending gate values
docker/                       # Integration images; P1–P3 extend component images
compose.yaml                  # Local integration topology
scripts/                      # Full-run wrappers and documentation checker
.github/workflows/            # Framework checks and manual actual lifecycle
requirements/                 # Hash-pinned framework locks
tests/                        # Framework contract/failure tests
docs/                         # Planning, contracts, status and report template
```

P1 creates `data/`, `features/`, schemas and feature-drift modules under the package. P2 creates `training/`, `registry/` and labeled-quality modules. P3 creates `serving/`, performance and metric-export modules. Their directories are ownership boundaries, not proof of existing implementations. Generated data and `artifacts/` outputs are ignored.

## Service addresses and evidence

| Service | Default host URL | Verification |
| --- | --- | --- |
| MLflow | `http://localhost:5000` | Actual tracking runs/artifacts and registry versions |
| Airflow | `http://localhost:8080` | Login from `.env`; lifecycle DAG task states and logs |
| FastAPI (P3/application profile) | `http://localhost:8000` | `/health`, exact-version `/ready`, prediction, metrics and feedback |
| Prometheus | `http://localhost:9090` | Real serving/quality/drift targets |
| Grafana | `http://localhost:3000` | Login from `.env`; P3-provisioned actual dashboards |

The P0 worker is an internal integration service. Its health is not model/API readiness. Reports are under `artifacts/runs/<run_id>/` with `run.json` and stage JSON files. Never treat a worker-health result, empty dashboard or successful fixture as serving acceptance.

```powershell
docker compose --profile application logs --no-color
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready
docker compose --profile application down
```

P3 adds a schema-valid prediction example and records the tested POST response once FastAPI exists. Preserve real logs/manifests in a sanitized evidence record before cleanup; `down` stops services without requesting volume deletion.

## Controlled rollback

After P2's finalization callback and P3's watcher/API are implemented, P0 can restore the previous confirmed healthy deployment:

```powershell
docker compose exec -T pipeline-worker python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "demo rollback"
```

The CLI uses the stored approved previous model, writes a new desired manifest, verifies ACK/readiness/prediction, finalizes registry metadata, and preserves an audit record. It does not retrain. Read [Rollback/recovery](TESTING_AND_DEMO.md) and [P0 deployment handoff](P0_HANDOFF.md) before exercising it. The current checkout lacks the real API and callback required for this command to succeed.

## Checks and troubleshooting

| Symptom | Action |
| --- | --- |
| Preflight names missing adapter | Implement the assigned module/callable from [P0 handoff](P0_HANDOFF.md); do not skip it |
| Preflight reports unset model gates | P2 measures baseline; P0/P2 approve values in gate config |
| Bad validation or `passed: false` | Inspect report/alert, repair input; downstream stages must stop |
| Stale report or retry input mismatch | Use a new run ID for changed inputs/config; investigate provenance |
| Docker not found/build fails | Enable Docker on the chosen machine; check pinned-image/dependency network access |
| API readiness/version failure | P3 checks desired model, load/ACK and probes; preserve prior serving version |
| Candidate rejected | Keep current model; inspect actual evaluation and performance gate evidence |
| Missing quality metrics | Inspect label joins/minimum coverage; report insufficient data |
| Workflow green but no full run | Framework CI verifies fixtures; dispatch the full lifecycle after adapter handoff |

Before submission, someone other than P0 follows these docs from a clean Docker clone and records image identity, OS/hardware, source/split/model versions, full-run logs, API request/readiness and dashboard access. Remaining verification is tracked in [Requirements](REQUIREMENTS.md).

