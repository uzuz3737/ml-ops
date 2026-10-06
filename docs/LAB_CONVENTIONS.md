# แนวทางจาก Lab ที่ใช้กับ Final Project

ศึกษาตัวโค้ด 41 ไฟล์ใน 5 โปรเจกต์ของ `MLeng` รวม configuration, CI, tests
และการส่งข้อมูลระหว่างไฟล์ ก่อนเลือกคงโครงสร้างเดิมของ Final Project
การวิเคราะห์นี้เป็นการอ่านโค้ด ไม่ได้อ้างว่าได้รัน Lab ทุกชุดผ่านใหม่

| กลุ่ม Lab | Logic และความสัมพันธ์ | นำมาใช้ | สิ่งที่ไม่ตาม |
| --- | --- | --- | --- |
| `mlops_pipeline` (Wine) | validation → preprocessing artifacts → train โดยอ้าง MLflow run ID → registry → predict จาก alias | fitted preprocessing กับ model อยู่ใน bundle เดียว, tracking/registry และแยก CI code/data/model | validation และ preprocessing โหลดข้อมูลคนละรอบโดยไม่ผูก checksum; test ฝึก pipeline สำเนา |
| `mlops_cancer` | ดัดแปลงโครง Wine เป็น Breast Cancer โดยเปลี่ยน loader/feature/experiment | ใช้ workflow เดียวผ่าน configuration | คัดลอกทั้ง pipeline เมื่อเปลี่ยน dataset และ preprocessing ก่อนแบ่งข้อมูล |
| `lab_10/monitoring_lab/monitoring_lab` | training/API → logs → monitoring reports/dashboard → retraining policy | แยก monitoring config, เก็บ reference และ alert | training Bank แต่ serving Telco, ขอบเขตสัปดาห์ไม่ตรงกัน, อัปเดตเวลาโดยไม่ได้คำนวณใหม่, retrain ที่เป็นเพียงข้อความ |
| `airflow_lab/airflow_lab` | DAG เรียกขั้น train/evaluate และตัดสินใจตามผล | DAG บาง ใช้ module จริง แยก orchestration ออกจาก ML logic | เทียบ metrics คนละชุดข้อมูล, retrain ข้อมูลเก่าโดยไม่มี dataset version ใหม่, ไม่มีหลักฐาน API โหลดรุ่นใหม่ |
| TFX project | components ส่ง artifacts ผ่าน validation/transform/train/evaluate/push | artifact lineage, quality gate และ preprocessing สอดคล้องกัน | wrapper กลืน exit code และ validation ที่ไม่หยุดเมื่อพบ anomaly; ไม่เพิ่ม TFX ซ้ำกับ Airflow/TFDV ที่มีอยู่ |

## โครงสร้างที่ใช้

```text
src/mlops_project/
  artifacts.py       # atomic JSON, lock และการอ่าน artifact ร่วม
  data/              # ingestion, row policy, split, approved snapshots
  features/          # preprocessing factory ใช้ train/serve ร่วมกัน
  training/          # experiments, metrics, bundle, final assessment
  registry/          # version และ lifecycle
  serving/           # API, loaded bundle, watcher, events, benchmark
  monitoring/        # drift, delayed-label quality, exporters
  pipelines/         # contracts, runner, gates, deploy/rollback, retrain
dags/                # dependency graph เรียก worker
configs/             # policy ที่ตรวจสอบและ version ได้
schemas/             # schema สำหรับ TFDV
scripts/             # คำสั่งปฏิบัติ; logic ที่ใช้ซ้ำอยู่ใน package
tests/               # เรียก implementation จริง รวม failure cases
docs/                # คู่มือ, requirement, evidence
```

ใช้ `snake_case` สำหรับไฟล์/ฟังก์ชัน/ตัวแปร, `PascalCase` สำหรับ class
และ `UPPER_CASE` สำหรับค่าคงที่ ตั้งชื่อตามหน้าที่ ไม่สร้าง utility รวมทุกอย่าง
fit preprocessing จาก train เท่านั้นและส่ง fitted state ไป serving
ทุกขอบเขตงานส่ง artifact URI/checksum/version แทนการอาศัยชื่อไฟล์ล่าสุด
invalid data, gate failure และ timeout ต้องส่งผลล้มเหลวจริง
ทดสอบผ่าน public behavior และข้อมูลผิด ไม่เขียน model/policy สำเนาใน tests

ปรับเฉพาะส่วนที่ช่วยลด dependency ข้ามหน้าที่ เช่นย้ายการจัดเก็บ artifact
ออกจาก runner ให้ training/serving/monitoring ใช้ร่วมกันได้
ไม่ย้ายไฟล์เพียงเพื่อความสวย และไม่เพิ่ม framework ที่ซ้ำหน้าที่เดิม
