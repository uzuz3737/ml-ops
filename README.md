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

### วิธีรันบนเครื่องเปล่า (Docker)

**ต้องมี**

- Docker Desktop (Windows ใช้ WSL 2 backend, macOS ได้ทั้ง Intel และ Apple Silicon) หรือ Docker Engine + Compose 2.20 ขึ้นไปบน Linux ใน Docker Desktop → Settings → Resources ให้ **memory อย่างน้อย 8 GB** และพื้นที่ดิสก์ประมาณ 20 GB
- Git (Windows ใช้ Git for Windows ซึ่งมี Git Bash มาด้วย)
- อินเทอร์เน็ต สำหรับโหลด image, Python package และชุดข้อมูล UCI (pipeline โหลดเองและตรวจ SHA-256 ทุกครั้ง)
- port 8000, 8080, 5000, 3000 และ 9090 ต้องว่าง

ไม่ต้องลงอะไรเพิ่มบนเครื่อง ทั้ง Python, Airflow, MLflow, TFDV และโมเดลรันอยู่ใน container ทั้งหมด

**1. โหลดโค้ดและตั้งค่า**

```bash
git clone https://github.com/uzuz3737/ml-ops.git
cd ml-ops
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
```

ค่าเริ่มต้นใน `.env` ใช้เดโมบนเครื่องได้เลยไม่ต้องแก้

**2. รันทั้งระบบด้วยคำสั่งเดียว** (เลือกตาม shell ที่ใช้)

| Shell | คำสั่ง |
| --- | --- |
| macOS / Linux | `bash scripts/run-all.sh --run-id first-run` |
| Windows, Git Bash (หลักฐานทั้งหมดรันด้วยวิธีนี้) | `MSYS_NO_PATHCONV=1 bash scripts/run-all.sh --run-id first-run` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File scripts\run-all.ps1 -RunId first-run` |

สคริปต์จะ build image, ตรวจว่ามีครบทุกขั้น (preflight), เปิดทุก service, สั่ง Airflow DAG แล้วรอจนจบ: ข้อมูลดิบ UCI → ตรวจด้วย TFDV → split → ทดลอง 3 โมเดล → registry → load test → gate → deploy รอบแรกใช้เวลาประมาณ 20–40 นาที (build image, เทรน 3 โมเดล และ load test 5 นาที ถ้าเป็น Apple Silicon จะช้ากว่านี้ เพราะ image ของ worker ต้องรันแบบจำลอง x86-64) ถ้าสำเร็จจะจบด้วย

```text
Airflow run state: success
Verified deployment deploy-… running model 1.
```

รันรอบใหม่ทุกครั้งต้องใช้ `--run-id` ใหม่ ถ้าใช้ PowerShell ห้ามสั่ง redirect output ด้วย `*>` เพราะ Windows PowerShell 5.1 จะนับข้อความ progress ของ Docker เป็น error แล้วหยุดทันที

**3. เช็คว่าใช้งานได้**

```bash
curl -s http://localhost:8000/ready        # {"status": "ready", "model_version": "1", ...}
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/predict.json
```

(ถ้าใช้ PowerShell ให้พิมพ์ `curl.exe` แทน `curl`)

| Service | URL | Login |
| --- | --- | --- |
| Prediction API | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | `airflow` / `airflow-local-demo` |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | `admin` / `grafana-local-demo` |
| Prometheus | http://localhost:9090 | — |

รหัสผ่านทั้งหมดตั้งไว้ใน `.env`

**ถ้ามีปัญหา**

| อาการ | วิธีแก้ |
| --- | --- |
| `Docker is required` หรือ engine ไม่พร้อม | เปิด Docker Desktop แล้วรอจนขึ้นว่า engine running |
| `Create .env from .env.example …` | ทำข้อ 1 ในโฟลเดอร์โปรเจกต์ |
| `port is already allocated` | ปิดโปรแกรมที่ใช้ port นั้นอยู่ ถ้าเป็น MLflow ให้ใส่ `MLFLOW_HOST_PORT=5050` ใน `.env` (AirPlay ของ macOS และ Windows บางเครื่องจอง 5000 ไว้) |
| build หรือ stage ถูก kill (exit code 137) | เพิ่ม memory ให้ Docker Desktop เป็น 8 GB ขึ้นไป |
| `ingest` ล้มด้วย `source_checksum` หรือ network error | โหลดข้อมูลจาก UCI ไม่สำเร็จ เช็คอินเทอร์เน็ตแล้วรันใหม่ด้วย `--run-id` ใหม่ |
| `This run ID already belongs to a different configuration` | ใช้ `--run-id` ใหม่ |
| อยากเริ่มใหม่จากศูนย์ | `docker compose --profile application down -v` แล้วลบโฟลเดอร์ `artifacts/` |

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

# วงจรเทรนใหม่: หลัง concept-drift alert สร้างชุดข้อมูลใหม่ที่ติด label แล้ว สั่ง Airflow เทรนใหม่ แล้วผ่าน gate
docker compose exec pipeline-worker python scripts/retraining-demo.py --approve-snapshot

# ข้อมูลเสีย: สร้างไฟล์เสียจากข้อมูลจริง แล้วป้อนเข้า pipeline ต้องหยุดที่ขั้น validate พร้อม alert
docker compose exec pipeline-worker python scripts/make-bad-data.py
docker compose exec pipeline-worker python scripts/airflow-run.py --data-file data/bad-domain.csv

# test case จากอาจารย์ (CSV/JSON/XLSX วางไว้ใน data/) ส่งเข้า API ทีละแถวแล้วพิมพ์ผล
docker compose exec pipeline-worker python scripts/predict-file.py data/test-cases.csv
```

ไฟล์ข้อมูลใดก็ได้ใน `data/` ที่ใช้หัวคอลัมน์ของ UCI (รวมถึงแบบ Kaggle `PAY_1`/`default.payment.next.month` หรือ `X1..X23`) ส่งเข้า pipeline ได้ด้วย `--data-file` (หรือ conf `{"data_file": "data/x.csv"}` ตอนกด Trigger DAG ในหน้าเว็บ Airflow) ถ้าข้อมูลผิดจะหยุดที่ `validate` และเขียนเหตุผลแยกตามคอลัมน์ลงใน `artifacts/runs/<run-id>/validation-report.json` ส่วน `/predict` จะรับคอลัมน์ `ID` ได้ แต่ไม่ส่งเข้าโมเดล และส่ง `ID` คืนมาพร้อมผลทำนาย

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

---

## English

End-to-end ML production system for **credit-card default prediction** ([UCI dataset 350](https://archive.ics.uci.edu/dataset/350), 30,000 clients, 23 features).

```text
raw UCI data → TFDV validation → split → shared preprocessing → 3 tracked experiments (MLflow)
   → registry → candidate benchmark → approval gate → deploy to FastAPI → monitoring → rollback / retraining
```

One command runs the whole lifecycle in Docker, orchestrated by Airflow.

**Status (4 Oct 2026):** full lifecycle verified in Docker — run `full-run-002` passed every gate (AP 0.5375, p95 204.5 ms, 60 req/s, 0% errors) and was deployed; rollback, feature-drift and concept-drift alerts were demonstrated. Evidence: [docs/evidence/INDEX.md](docs/evidence/INDEX.md).

### Quick start on a clean machine (Docker)

**You need**

- Docker Desktop (Windows with the WSL 2 backend, or macOS Intel/Apple Silicon), or Docker Engine with Compose 2.20+ on Linux. In Docker Desktop → Settings → Resources allow **at least 8 GB memory** and about 20 GB of disk.
- Git (on Windows, Git for Windows, which includes Git Bash).
- Internet access for the images, the Python packages and the UCI dataset, which the pipeline downloads and checks against a fixed SHA-256.
- Free local ports 8000, 8080, 5000, 3000 and 9090.

Nothing else is installed on the host: Python, Airflow, MLflow, TFDV and the models all run in containers.

**1. Get the code and the settings**

```bash
git clone https://github.com/uzuz3737/ml-ops.git
cd ml-ops
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
```

The defaults in `.env` work unchanged for a local demo.

**2. Run everything with one command** (pick your shell)

| Shell | Command |
| --- | --- |
| macOS / Linux | `bash scripts/run-all.sh --run-id first-run` |
| Windows, Git Bash (used for all recorded evidence) | `MSYS_NO_PATHCONV=1 bash scripts/run-all.sh --run-id first-run` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File scripts\run-all.ps1 -RunId first-run` |

The script builds the images, checks that every stage exists (preflight), starts all services, triggers the Airflow DAG and waits for it: raw UCI data → TFDV validation → split → three experiments → registry → load test → gate → deployment. The first run takes about 20–40 minutes (image builds, three models and a 5-minute load test; Apple Silicon is slower because the worker image runs under x86-64 emulation). It ends with:

```text
Airflow run state: success
Verified deployment deploy-… running model 1.
```

Every new run needs a new `--run-id`. In PowerShell do not redirect the output with `*>`; Windows PowerShell 5.1 then treats Docker's progress messages as errors.

**3. Check that it works**

```bash
curl -s http://localhost:8000/ready        # {"status": "ready", "model_version": "1", ...}
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @examples/predict.json
```

(PowerShell: type `curl.exe` instead of `curl`.)

| Service | URL | Login |
| --- | --- | --- |
| Prediction API | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | `airflow` / `airflow-local-demo` |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | `admin` / `grafana-local-demo` |
| Prometheus | http://localhost:9090 | — |

Logins come from `.env`.

**If something goes wrong**

| Symptom | Fix |
| --- | --- |
| `Docker is required` or the engine is unavailable | Start Docker Desktop and wait until it reports that the engine is running |
| `Create .env from .env.example …` | Do step 1 in the project folder |
| `port is already allocated` | Stop the program using that port. For MLflow, set `MLFLOW_HOST_PORT=5050` in `.env` (macOS AirPlay and some Windows setups hold port 5000) |
| A build or stage is killed (exit code 137) | Give Docker Desktop more memory (8 GB or more) |
| `ingest` fails with `source_checksum` or a network error | The UCI download failed; check the internet connection and rerun with a new `--run-id` |
| `This run ID already belongs to a different configuration` | Use a new `--run-id` |
| Start again from nothing | `docker compose --profile application down -v`, then delete the `artifacts/` folder |

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

# retraining loop: after the concept-drift alert, build the new labeled snapshot, trigger Airflow, pass the gates
docker compose exec pipeline-worker python scripts/retraining-demo.py --approve-snapshot

# bad data: corrupt real rows, feed them to the pipeline; it must stop at validate with an alert
docker compose exec pipeline-worker python scripts/make-bad-data.py
docker compose exec pipeline-worker python scripts/airflow-run.py --data-file data/bad-domain.csv

# instructor test cases (CSV/JSON/XLSX placed in data/), scored row by row through the API
docker compose exec pipeline-worker python scripts/predict-file.py data/test-cases.csv
```

Any file in `data/` with UCI headers (including the Kaggle `PAY_1`/`default.payment.next.month` spelling or `X1..X23`) can be fed to the pipeline with `--data-file` (or conf `{"data_file": "data/x.csv"}` when triggering the DAG in the Airflow UI). Bad data stops at `validate` with per-column reasons in `artifacts/runs/<run-id>/validation-report.json`. `/predict` also accepts an `ID` column; it is not sent to the model and is echoed back with each prediction.

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
