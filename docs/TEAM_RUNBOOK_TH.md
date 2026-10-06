# คู่มือปฏิบัติสำหรับทีม

ใช้ PowerShell จากโฟลเดอร์ repository คำสั่งด้านล่างใช้ Docker Linux containers
สเปกและเงื่อนไขส่งงานให้ยึด [โจทย์อาจารย์](../โครงงานรายวิชา%20CP413008%20Machine%20Learning%20Engineering%20for%20Production.docx.md)
และ [รายการ requirement](REQUIREMENTS.md) ส่วนหลักฐานผลรันดู [evidence](evidence/INDEX.md)

## 1. ติดตั้งและเปิดระบบ

ติดตั้ง Git และ Docker Desktop เปิด Linux containers แล้วตรวจว่า Docker Engine ทำงาน
ใช้ Python 3.11 เมื่อต้องการแก้โค้ดและทดสอบบนเครื่อง Windows

```powershell
git clone --branch feature/final-project-readiness https://github.com/uzuz3737/ml-ops.git
Set-Location ml-ops
Copy-Item .env.example .env
docker version
docker compose version
```

แก้รหัสผ่านใน `.env` สำหรับเครื่องของตนเองก่อนเปิดระบบ ห้าม commit ไฟล์นี้
ถ้า port 5000 ถูกใช้งาน เปลี่ยน `MLFLOW_HOST_PORT=5050`

```powershell
.\scripts\run-all.ps1 -RunId team-initial-001
```

รอจนคำสั่งยืนยัน deployment สำเร็จ: ระบบดาวน์โหลดข้อมูล ตรวจ schema แบ่งข้อมูล
ฝึก Logistic Regression / Random Forest / HistGradientBoosting บันทึก MLflow
ลงทะเบียน candidate วัด API 300 วินาที ตรวจ gate แล้ว deploy และตรวจรุ่นที่ API โหลดจริง
การสร้าง image ครั้งแรกใช้เวลานานเพราะ TFDV/TensorFlow
ห้ามถือว่า DAG ถูก trigger แล้วเท่ากับสำเร็จ

## 2. ตรวจผลและใช้ API

```powershell
docker compose --profile application ps
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready
curl.exe -H "Content-Type: application/json" --data-binary "@examples/predict.json" http://localhost:8000/predict
curl.exe http://localhost:8000/metrics
```

`/ready` ต้องพร้อมใช้งาน และ `/predict` ต้องตอบ `model_version` ตรงกับรุ่นที่ deploy
ดูผลรายขั้นใน `artifacts/runs/team-initial-001/` เช่น `evaluate.json`, `benchmark.json`,
`approve.json`, `verify.json` และ `launch.json` เก็บ run ID ไว้อ้างอิงเสมอ
ตัวอย่างข้อมูลผิดอยู่ใน `examples/invalid/` ควรได้ HTTP 422

| หน้าใช้งาน | URL ปกติ | สิ่งที่ตรวจ |
| --- | --- | --- |
| Airflow | http://localhost:8080 | DAG `credit_default_pipeline`, task log, สถานะ failed/success |
| MLflow | http://localhost:5000 | experiments, params, metrics, artifacts, model versions |
| API | http://localhost:8000/docs | schema และทดลอง request |
| Prometheus | http://localhost:9090 | target API ต้อง UP และมี metrics |
| Grafana | http://localhost:3000 | dashboard latency/error/drift/quality |

Airflow/Grafana ใช้บัญชีใน `.env` MLflow ใช้ port ที่ตั้งไว้
ใน MLflow เปรียบเทียบทั้งสาม experiment ด้วย validation ชุดเดียวกัน
ตรวจ code commit, dataset version, dependency lock, model bundle และ metric ก่อนอนุมัติรุ่น
อย่าเปลี่ยน alias ใน MLflow เพื่อข้าม deployment gate

เปรียบเทียบ ML กับกฎง่าย `PAY_0 >= 2` บน validation เดียวกัน:

```powershell
docker compose exec -T pipeline-worker python scripts/compare-simple-rule.py --run-id team-initial-001
```

ผล `simple-rule-comparison.json` เปรียบเทียบ AP และจำนวน default ที่พบเมื่อกำหนด
งบตรวจ 20% เท่ากัน ตัวเลขนี้เป็นตัวแทนสำหรับสาธิต ต้องคุยกับ stakeholder
ก่อนอ้างเป็นคุณค่าหรือผลประหยัดของระบบจริง

## 3. ข้อมูลและการฝึกใหม่

ข้อมูลเริ่มต้นคือ UCI Default of Credit Card Clients 30,000 ราย
pipeline ตรึง checksum ของ ZIP และ seed 42; แบ่ง train/validation/final-test/monitoring
เป็น 60/20/10/10 พร้อม ID ที่ไม่ทับกัน

หากเครือข่ายดาวน์โหลด UCI ไม่ได้ ให้วาง ZIP ต้นฉบับที่ `data/raw/uci350.zip`
แล้วเพิ่ม `source_archive: data/raw/uci350.zip` ใต้ `dataset` ใน config
ไฟล์ต้องมี checksum ตรงกับต้นฉบับ ห้ามแก้ checksum เพียงเพื่อให้ผ่าน

การเปลี่ยนข้อมูลจริงต้องสร้าง snapshot JSON ที่ผ่าน `data.policy.validate_rows`:
มี `ID`, features ทั้ง 23 ตัว และ `default_next_month` เป็นจำนวนเต็ม 0/1
แยก train/validation และกัน ID ของ final test รวมถึง monitoring ออกจากข้อมูลใหม่
manifest ต้องมี `contract_version`, `dataset_version`, `schema_version`, `source`,
`license`, `attribution`, `partitions` และ `protected_ids` พร้อม SHA-256 ของทุกไฟล์
ลงทะเบียน manifest ใน `dataset.retraining_snapshots` และกำหนด
`dataset.protected_final_test_ids` เป็นรายการ ID ที่สงวนจากชุดเดิม
ตรวจ label และให้เจ้าของข้อมูลรับรองก่อนตั้ง approval flags
ดูโค้ดอ้างอิงที่ `src/mlops_project/data/snapshots.py`

หากเปลี่ยน feature/target ต้องแก้ policy, schema, preprocessing, API contract และ tests
ร่วมกัน การนำ CSV รูปแบบอื่นมาวางแทนไฟล์เดิมยังไม่ใช่วิธีเปลี่ยน dataset ที่รองรับ

ฝึกข้อมูลเริ่มต้นอีกรอบด้วย run ID ใหม่:

```powershell
.\scripts\run-all.ps1 -RunId team-initial-002
```

อย่าใช้ผล final test เลือกโมเดลหรือปรับพารามิเตอร์ ให้เลือกด้วย validation และ gate เท่านั้น

หลังเลือก initial candidate แน่นอนแล้ว ให้ล็อก checksum ก่อนเปิดผล final test:

```powershell
$selected = (Get-Content artifacts/runs/team-initial-001/evaluate.json -Raw | ConvertFrom-Json).result
docker compose exec -T pipeline-worker python scripts/assess-final-test.py --run-id team-initial-001 --locked-candidate-sha256 $selected.artifact_sha256
```

ผลอยู่ใน `final-test.json` ของ run และ MLflow ระบบบันทึก ledger ป้องกันการเปลี่ยน
candidate แล้วกลับมาวัด final test เดิมซ้ำ ใช้คำสั่งนี้กับ retraining ไม่ได้

## 4. Monitoring และสาธิต retraining

ต้องมี initial deployment สำเร็จก่อน แต่ละคำสั่งส่งข้อมูล monitoring ที่กันไว้อย่างน้อย
500 ราย พร้อม label แล้วคำนวณผลใหม่

```powershell
docker compose exec -T pipeline-worker python scripts/monitoring-demo.py --scenario healthy
docker compose exec -T pipeline-worker python scripts/monitoring-demo.py --scenario feature-drift
docker compose exec -T pipeline-worker python scripts/monitoring-demo.py --scenario concept-drift
```

ตรวจ `artifacts/monitoring/exported/` และ `artifacts/monitoring/alerts/`
feature-drift เปลี่ยน LIMIT_BAL ส่วน concept-drift กลับ label โดยคง features เดิม
ผลนี้เป็นการจำลองในชั้นเรียน ไม่ใช่หลักฐานว่าลูกค้าจริงเกิด concept drift
label ที่ไม่พอจะได้ `insufficient_data` และไม่ควรสั่ง retrain

เตรียมข้อมูลจำลองสำหรับ retraining โดยใช้ run ID ของ **initial model ที่กำลังใช้งาน**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r src/mlops_project/training/requirements-dev.lock -r requirements/serving.lock
.\.venv\Scripts\python.exe scripts/prepare-retraining-demo.py --source-run team-initial-001 --trigger-id demo-001 --approve-synthetic-demo
```

คำสั่งสร้าง snapshot ใหม่จาก train/validation เดิมที่กลับ label
ไม่ใช้แถว monitoring/final test ฝึกโมเดล และไม่ส่ง DAG อัตโนมัติ
config ที่สร้างอยู่ใน `configs/retraining-demo-001.local.yaml`
และ request อยู่ใน `data/retraining/demo-001/alert.json` ทั้งสองเป็นไฟล์ local ที่ไม่ commit
หากเตรียมไม่สำเร็จ ให้ตรวจ error และใช้ trigger ID ใหม่หลังแก้ปัญหา

เลือก config นี้ให้ worker แล้วส่ง request:

```powershell
$env:PIPELINE_CONFIG = 'configs/retraining-demo-001.local.yaml'
$env:MLOPS_CODE_COMMIT = git rev-parse HEAD
docker compose up -d --wait pipeline-worker
docker compose exec -T pipeline-worker python -m mlops_project.pipelines.retraining --config configs/retraining-demo-001.local.yaml --alert data/retraining/demo-001/alert.json
```

เปิด Airflow ดู run `retrain-demo-001` รอจนจบ ตรวจ candidate รุ่นใหม่ใน MLflow
และ `artifacts/runs/retrain-demo-001/` ระบบเทียบ champion กับ candidate บน validation
ชุดใหม่เดียวกัน รุ่นใหม่ต้องผ่าน model/performance gate จึง deploy
ถ้า gate ปฏิเสธ ให้บันทึกเหตุผลและคงรุ่นเก่า ห้ามลด threshold เพื่อให้ demo ผ่าน
ส่ง trigger ID เดิมซ้ำจะไม่สร้าง DAG ซ้ำ; trigger ใหม่ต้องรอ cooldown 3,600 วินาที
เมื่อจบสาธิตให้คืนค่า worker:

```powershell
Remove-Item Env:PIPELINE_CONFIG -ErrorAction SilentlyContinue
docker compose up -d --wait pipeline-worker
```

สคริปต์ monitoring-demo ใช้ monitoring partition ของ initial run
หลัง deploy โมเดล retrain ซึ่งไม่มี partition นี้ ให้ใช้ feedback จริง หรือ rollback
กลับ initial model ก่อนสาธิตชุดเดิมอีกครั้ง

## 5. อัปเดตและ rollback โมเดล

การรัน pipeline สำเร็จจะอัปเดตโมเดลผ่าน deployment manifest และ watcher
ตรวจ `/ready` และ `/predict` ทุกครั้ง อย่าคัดลอกไฟล์โมเดลทับรุ่นที่กำลังทำงาน
rollback ใช้ได้เมื่อมี current และ previous deployment ที่ยืนยันแล้ว:

```powershell
docker compose exec -T pipeline-worker python -m mlops_project.pipelines.deployment --rollback --config configs/project.yaml --reason "restore previous model after demo"
curl.exe http://localhost:8000/ready
```

ตรวจประวัติ `artifacts/deployments/history/` และรุ่นที่ API โหลดจริง

## 6. แก้โค้ด ทดสอบ และ push

```powershell
git switch feature/final-project-readiness
git pull --ff-only
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r src/mlops_project/training/requirements-dev.lock -r requirements/serving.lock
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_docs.py
```

Windows จะข้าม TFDV tests ที่ต้องใช้ Linux; ตรวจใน worker image เพิ่มก่อนสรุปว่าครบ
หากแก้ dependencies ต้องอัปเดต hash lock ที่เกี่ยวข้องและ build ใหม่
หากแก้โค้ด API ให้ restart API; การรัน pipeline ใหม่ใช้ wrapper เพื่อบันทึก commit ที่ถูกต้อง

```powershell
git diff
git add <ไฟล์ที่แก้จริง>
git commit -m "fix: explain the behavior corrected"
git push origin feature/final-project-readiness
docker compose --profile application up -d --build api
.\scripts\run-all.ps1 -RunId team-update-001
```

ทำทีละส่วนที่ทดสอบผ่าน ห้าม push `.env`, raw data, model binaries หรือ artifacts
branch นี้ยังไม่ merge เข้า `main` จนกว่าเจ้าของโปรเจกต์จะสั่ง

## 7. แก้ปัญหาที่พบบ่อย

| อาการ | วิธีตรวจ/แก้ |
| --- | --- |
| Build/download ล้มเหลว | ตรวจ Docker Engine และเครือข่าย แล้ว build ซ้ำ; ห้ามปิด hash verification |
| Port ถูกจอง | เปลี่ยน host port ใน `.env`/Compose แล้วเปิดใหม่ |
| `/health` ผ่านแต่ `/ready` 503 | ตรวจ DAG/deployment และ watcher log; ยังไม่มีโมเดลที่ยืนยัน |
| DAG failed | ดู task log และ JSON ขั้นที่ล้มเหลวใน run directory; แก้สาเหตุแล้วใช้ run ID ใหม่ |
| `config_not_allowed` | ตั้ง `PIPELINE_CONFIG` ให้ตรงกับ config ใน request และ recreate worker |
| `snapshot_checksum` / `snapshot_leakage` | ห้ามแก้ manifest หลบการตรวจ; สร้าง snapshot ที่ถูกต้องรุ่นใหม่ |
| Quality `insufficient_data` | เพิ่มข้อมูลและ label จริงให้ครบจำนวน/สัดส่วนขั้นต่ำ |
| p95 ไม่ผ่าน | ตรวจ CPU/RAM และงานอื่นบนเครื่อง เก็บผลล้มเหลวไว้แล้ววัดใหม่ตาม workload เดิม |
| `cooldown` / `trigger_conflict` | รอ cooldown หรือใช้ request เดิมตาม receipt; อย่าลบ ledger เพื่อข้าม policy |
| rollback unavailable | ต้องมี previous deployment ที่ยืนยันก่อน |

```powershell
docker compose --profile application logs --tail 100 pipeline-worker api airflow-scheduler
docker compose --profile application down
```

`down` หยุดระบบโดยคง named volumes ไว้ อย่าใช้ `down -v` หากยังต้องเก็บ MLflow/Airflow
ข้อมูลและ artifacts local ต้องสำรองแยกจาก Git
