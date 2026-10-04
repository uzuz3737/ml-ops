# Credit Default MLOps — CP413008

**ภาษาไทย** · [English](#english)

---

## ภาษาไทย

ระบบ ML สำหรับใช้งานจริงแบบครบวงจร ใช้**ทำนายว่าลูกค้าบัตรเครดิตจะผิดนัดชำระเดือนถัดไปหรือไม่** ข้อมูลมาจาก [UCI dataset 350](https://archive.ics.uci.edu/dataset/350) (ลูกค้า 30,000 ราย, 23 feature)

```text
ข้อมูลดิบ UCI → ตรวจข้อมูลด้วย TFDV → แบ่ง split → preprocessing ร่วม → ทดลอง 3 โมเดล (MLflow)
   → registry → วัดความเร็ว candidate → gate อนุมัติ → deploy ขึ้น FastAPI → monitoring → rollback / retrain
```

ใช้**คำสั่งเดียว**รันทั้งระบบใน Docker โดยมี Airflow เป็นตัวควบคุมขั้นตอน

**สถานะ (4 ต.ค. 2026):** รันครบทั้งเส้นใน Docker ผ่านแล้ว run `full-run-002` ผ่าน gate ทุกข้อ (AP 0.5375, p95 204.5 ms, 60 req/s, error 0%) และ deploy สำเร็จ ทดสอบ rollback, alert ของ data drift และ concept drift แล้วด้วย ดูหลักฐานที่ [docs/evidence/INDEX.md](docs/evidence/INDEX.md)

### วิธีรัน (Docker)

**ต้องมี:** Docker Desktop (Linux containers) ที่มี Compose 2.20 ขึ้นไป, Git, พื้นที่ว่างประมาณ 20 GB และอินเทอร์เน็ตสำหรับโหลด image กับข้อมูล UCI

```bash
git clone https://github.com/uzuz3737/ml-ops.git
cd ml-ops
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
bash scripts/run-all.sh --config configs/project.yaml --run-id my-first-run
```

**บน Windows** ให้รันคำสั่งสุดท้ายใน Git Bash แบบนี้: `MSYS_NO_PATHCONV=1 bash scripts/run-all.sh --config configs/project.yaml --run-id my-first-run` (หลักฐานทั้งหมดรันด้วยวิธีนี้) ถ้าจะใช้ PowerShell ก็มี `.\scripts\run-all.ps1 -Config configs/project.yaml -RunId my-first-run` ให้ แต่ห้ามสั่ง redirect output ด้วย `*>` เพราะ Windows PowerShell 5.1 จะนับข้อความ progress ของ Docker เป็น error แล้วหยุดทันที

สคริปต์จะ build image, ตรวจว่ามีครบทุกขั้น (preflight), เปิดทุก service, สั่ง Airflow DAG แล้วรอจนจบ รอบแรกใช้เวลาประมาณ 20–30 นาที (build image + วัดความเร็วอีก 5 นาที) ถ้าเห็น `Airflow run state: success` แปลว่า API กำลังเสิร์ฟโมเดลใหม่อยู่

| Service | URL | Login |
| --- | --- | --- |
| Prediction API | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | `airflow` / `airflow-local-demo` |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | `admin` / `grafana-local-demo` |
| Prometheus | http://localhost:9090 | — |

รหัสผ่านทั้งหมดตั้งไว้ใน `.env` ถ้าใช้ port 5000 ไม่ได้ (Windows บางเครื่องจองไว้) ให้ใส่ `MLFLOW_HOST_PORT=5050` ใน `.env`

### ลองใช้งาน

```bash
curl -s http://localhost:8000/ready
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/predict.json
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/invalid/unknown-category.json   # ได้ 422
```

ถ้าใช้ PowerShell ให้พิมพ์ `curl.exe` แทน `curl` ตัวอย่าง request ที่ผิดแบบต่างๆ อยู่ใน `examples/invalid/` ทุกไฟล์จะได้ 422 พร้อมบอกว่าผิดที่ field ไหน

**เดโม** (ต้องรัน pipeline ผ่านแล้วอย่างน้อยหนึ่งรอบ):

```bash
# rollback กลับไปใช้โมเดลก่อนหน้าที่อนุมัติแล้ว โดยไม่ต้อง train ใหม่ (ต้องรันผ่านมาแล้ว 2 รอบ)
docker compose exec pipeline-worker python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "demo rollback"

# ส่งข้อมูลที่กันไว้สำหรับ monitoring ผ่าน API, ส่ง label กลับ แล้วรัน monitoring
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario healthy         # ปกติ ไม่มี alert
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario feature-drift   # มี data drift alert
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario concept-drift   # มี quality alert
```

ผลแต่ละ run อยู่ที่ `artifacts/runs/<run-id>/` ข้อมูล deploy อยู่ที่ `artifacts/deployments/` ส่วน alert อยู่ที่ `artifacts/monitoring/alerts/`

ปิดทุกอย่างด้วย `docker compose --profile application down` (ถ้าเติม `-v` จะลบข้อมูลของ MLflow/Airflow/Grafana ทิ้งด้วย)

### พัฒนาและรันเทสต์ (ไม่ใช้ Docker)

ใช้ Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
pip install --require-hashes -r src/mlops_project/training/requirements-dev.lock
pip install --require-hashes -r requirements/serving.lock
ruff check . && ruff format --check .
pytest
python scripts/check_docs.py
```

TFDV ติดตั้งได้เฉพาะบน Linux เลยทำให้เทสต์ของ TFDV ถูกข้ามบน Windows/macOS แต่ CI กับ worker ใน Docker จะรันเทสต์เหล่านี้ให้

### ส่วนต่างๆ ของระบบ

| ขั้นตอน | Tool | โค้ด |
| --- | --- | --- |
| โหลด, ตรวจ, แบ่งข้อมูล, feature, data drift | TFDV, pandas, scikit-learn | `src/mlops_project/data/`, `features/`, `monitoring/feature_drift.py` |
| ทดลองโมเดล, registry, quality ที่วัดจาก label | MLflow tracking + registry | `src/mlops_project/training/`, `registry/`, `monitoring/quality.py` |
| API, watcher สำหรับ deploy, benchmark, ส่ง metrics | FastAPI, Prometheus, Grafana | `src/mlops_project/serving/`, `monitoring/exporters.py`, `monitoring/run.py`, `infra/` |
| ควบคุมขั้นตอน, gate, deploy/rollback, CI | Airflow, GitHub Actions, Docker Compose | `src/mlops_project/pipelines/`, `dags/`, `compose.yaml`, `.github/workflows/` |

ค่า config อยู่ใน `configs/`
- `project.yaml`: ขั้นตอนของ pipeline และ path ต่างๆ
- `quality_gates.yaml`: AP ≥ 0.40, ลดลงได้ไม่เกิน 0.02, p95 ≤ 300 ms, ≥ 20 req/s, error ≤ 1%
- `monitoring.yaml`: threshold ของ drift และ quality, นโยบาย retrain
- `data_schema.yaml`: รายการ feature และ schema

### ทีม

| Role | สมาชิก | รับผิดชอบ |
| --- | --- | --- |
| P0 | @uzuz3737 | รวมระบบ, Airflow DAG, gate, deploy/rollback, CI/CD |
| P1 | @OuanEng | โหลดข้อมูล UCI, TFDV, แบ่ง split, preprocessing ร่วม, feature drift |
| P2 | @pairot230 | baseline + การทดลอง, MLflow tracking/registry, quality จาก label |
| P3 | @thanachaithongbai-hue | FastAPI, watcher สำหรับ deploy, benchmark/SLO, Prometheus/Grafana, monitoring |

ดูว่าใครดูแลไฟล์ไหนได้ใน [CODEOWNERS](.github/CODEOWNERS) วิธีร่วมพัฒนาอยู่ใน [CONTRIBUTING](CONTRIBUTING.md)

### เอกสาร

| เอกสาร | เนื้อหา |
| --- | --- |
| [Evidence index](docs/evidence/INDEX.md) | run ไหนเป็นหลักฐานของเกณฑ์ข้อไหน |
| [Serving and monitoring](docs/SERVING.md) | วิธีเรียก API, error แต่ละแบบ, ผล SLO, alert, rollback และเดโม drift |
| [Getting started](docs/GETTING_STARTED.md) | วิธีติดตั้งแบบละเอียด, ที่อยู่ของ service, การแก้ปัญหา |
| [Architecture](docs/ARCHITECTURE.md) | tool ทั้ง 9 หน้าที่ต่อกันอย่างไร |
| [Interface contracts](docs/INTERFACE_CONTRACTS.md) | รูปแบบข้อมูลที่ส่งต่อกันระหว่างแต่ละส่วน |
| [Requirements](docs/REQUIREMENTS.md) | เกณฑ์คะแนนของวิชา ใครรับผิดชอบ และใช้อะไรเป็นหลักฐาน |
| [Project brief](docs/PROJECT_BRIEF.md) / [Dataset guide](docs/DATASET.md) | โจทย์, ผู้ใช้, metric / คอลัมน์และ split ของ UCI |
| [P0](docs/P0_HANDOFF.md) / [P1](docs/P1_HANDOFF.md) / [P2](src/mlops_project/training/README.md) handoffs | แต่ละส่วนทำอะไร และต้องการอะไรจากส่วนอื่น |
| [P2 results](src/mlops_project/training/REPORT.md) | ผลการทดลองและเหตุผลที่เลือกโมเดล |
| [Testing and demo](docs/TESTING_AND_DEMO.md) / [Report template](docs/REPORT_TEMPLATE.md) | แผนเดโม / สิ่งที่ต้องส่ง |
| [Team work](docs/TEAM_WORK.md) / [Development plan](docs/DEVELOPMENT_PLAN.md) | การแบ่งงานและตารางเวลา |

โจทย์วิชา: [CP413008 requirements](./โครงงานรายวิชา%20CP413008%20Machine%20Learning%20Engineering%20for%20Production.docx.md) ส่งงาน **5 ต.ค. 2026 เวลา 23:59** นำเสนอ **12 ต.ค. 2026 เวลา 08:30** (เวลาไทย)

---

## English

End-to-end ML production system for **credit-card default prediction** ([UCI dataset 350](https://archive.ics.uci.edu/dataset/350), 30,000 clients, 23 features).

```text
raw UCI data → TFDV validation → split → shared preprocessing → 3 tracked experiments (MLflow)
   → registry → candidate benchmark → approval gate → deploy to FastAPI → monitoring → rollback / retraining
```

One command runs the whole lifecycle in Docker, orchestrated by Airflow.

**Status (4 Oct 2026):** full lifecycle verified in Docker — run `full-run-002` passed every gate (AP 0.5375, p95 204.5 ms, 60 req/s, 0% errors) and was deployed; rollback, feature-drift and concept-drift alerts were demonstrated. Evidence: [docs/evidence/INDEX.md](docs/evidence/INDEX.md).

### Quick start (Docker)

**Needs:** Docker Desktop (Linux containers) with Compose 2.20+, Git, ~20 GB free disk, internet for images and the UCI download.

```bash
git clone https://github.com/uzuz3737/ml-ops.git
cd ml-ops
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
bash scripts/run-all.sh --config configs/project.yaml --run-id my-first-run
```

**On Windows** run the last command in Git Bash as `MSYS_NO_PATHCONV=1 bash scripts/run-all.sh --config configs/project.yaml --run-id my-first-run` (this is how the evidence runs were made). `.\scripts\run-all.ps1 -Config configs/project.yaml -RunId my-first-run` also exists for PowerShell; do not redirect its output with `*>`, which makes Windows PowerShell 5.1 treat Docker's progress messages as errors.

The script builds the images, checks every stage exists (preflight), starts all services, triggers the Airflow DAG and waits until it finishes. The first run takes about 20–30 minutes (image builds plus a 5-minute load test). It ends with `Airflow run state: success` and the API serving the new model.

| Service | URL | Login |
| --- | --- | --- |
| Prediction API | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | `airflow` / `airflow-local-demo` |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | `admin` / `grafana-local-demo` |
| Prometheus | http://localhost:9090 | — |

Logins come from `.env`. If port 5000 is blocked (Windows sometimes reserves it), set `MLFLOW_HOST_PORT=5050` in `.env`.

### Try it

```bash
curl -s http://localhost:8000/ready
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/predict.json
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/invalid/unknown-category.json   # 422
```

PowerShell: use `curl.exe` instead of `curl`. Every file in `examples/invalid/` returns 422 with the failing field.

**Demo scenarios** (after at least one successful pipeline run):

```bash
# roll back to the previous approved model, no retraining (needs two successful runs)
docker compose exec pipeline-worker python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "demo rollback"

# replay held-out monitoring rows through the API, send labels, run monitoring
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario healthy         # no alert
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario feature-drift   # data drift alert
docker compose exec pipeline-worker python scripts/monitoring-demo.py --scenario concept-drift   # quality alert
```

Run outputs are in `artifacts/runs/<run-id>/`, deployments in `artifacts/deployments/`, alerts in `artifacts/monitoring/alerts/`.

Stop everything with `docker compose --profile application down` (add `-v` to also delete MLflow/Airflow/Grafana data).

### Develop and test (no Docker)

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

### How it fits together

| Step | Tool | Code |
| --- | --- | --- |
| Ingest, validate, split, features, data drift | TFDV, pandas, scikit-learn | `src/mlops_project/data/`, `features/`, `monitoring/feature_drift.py` |
| Experiments, registry, labeled quality | MLflow tracking + registry | `src/mlops_project/training/`, `registry/`, `monitoring/quality.py` |
| API, deployment watcher, benchmark, metrics export | FastAPI, Prometheus, Grafana | `src/mlops_project/serving/`, `monitoring/exporters.py`, `monitoring/run.py`, `infra/` |
| Orchestration, gates, deploy/rollback, CI | Airflow, GitHub Actions, Docker Compose | `src/mlops_project/pipelines/`, `dags/`, `compose.yaml`, `.github/workflows/` |

Configuration lives in `configs/`:
- `project.yaml`: pipeline stages and paths
- `quality_gates.yaml`: AP ≥ 0.40, max regression 0.02, p95 ≤ 300 ms, ≥ 20 req/s, ≤ 1% errors
- `monitoring.yaml`: drift and quality thresholds, retraining policy
- `data_schema.yaml`: features and schema

### Team

| Role | Member | Owns |
| --- | --- | --- |
| P0 | @uzuz3737 | Integration, Airflow DAG, gates, deployment/rollback controller, CI/CD |
| P1 | @OuanEng | UCI ingestion, TFDV schema/validation, splits, shared features, feature drift |
| P2 | @pairot230 | Baseline + experiments, MLflow tracking/registry, labeled quality |
| P3 | @thanachaithongbai-hue | FastAPI serving, deployment watcher, benchmark/SLO, Prometheus/Grafana, monitoring |

File ownership is in [CODEOWNERS](.github/CODEOWNERS); how to contribute is in [CONTRIBUTING](CONTRIBUTING.md).

### Documentation

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
