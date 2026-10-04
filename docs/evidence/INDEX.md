# Evidence index

Runs below were executed on 4 October 2026 with `bash scripts/run-all.sh --config configs/project.yaml --run-id <id>` on Docker Desktop (WSL2 Linux, x86_64, 28 logical CPUs). Artifacts live under `artifacts/` (ignored by Git); copy the listed files when assembling the report. Dataset version `d1bc5f133c49…` is the SHA-256 of the canonical UCI 350 data (source archive `56c885f8…`, 30,000 rows, TFDV validation passed).

| Requirement IDs | Date/run ID | Code SHA | Data/schema/model versions | Artifact path | Expected / actual | Author |
| --- | --- | --- | --- | --- | --- | --- |
| D2, D3, M1, E1, E2, S2, S3, G1 | 2026-10-04 `full-run-001` | `405f8d6` | `d1bc5f13…` / credit-default-v1 / model 1 | `artifacts/runs/full-run-001/gate-report.json`, `benchmark-report.json` | Gate must block an SLO miss / p95 318 ms > 300 ms, candidate rejected, nothing deployed | @thanachaithongbai-hue |
| D2–D4, M1, E1, E2, S1–S3, S5, G1, G3 | 2026-10-04 `full-run-002` | `cec24f6` | `d1bc5f13…` / credit-default-v1 / model 2 | `artifacts/runs/full-run-002/` (`validation-report.json`, `evaluation.json`, `registration.json`, `benchmark-report.json`, `gate-report.json`) | Raw data to verified deployment / DAG success; AP 0.5375; p95 204.5 ms, 60 req/s, 0% errors; `/ready` and host `/predict` return version 2 | @thanachaithongbai-hue |
| E2, G1 | 2026-10-04 `full-run-003` | `0859d87` | `d1bc5f13…` / credit-default-v1 / model 3 | `artifacts/runs/full-run-003/gate-report.json`, `artifacts/deployments/previous-model.json` | Second approved deployment / version 3 active, version 2 kept as previous | @thanachaithongbai-hue |
| E3 | 2026-10-04 `rollback-db30e0a9…` | `0859d87` | model 3 → 2 | `artifacts/deployments/history/rollback-db30e0a96f5a4220908919a8a01a31b9/`, `acknowledgements/` | Restore prior version without retraining / ACK loaded, `/ready` and `/predict` report 2 | @thanachaithongbai-hue |
| O1, O2 (data drift) | 2026-10-04 `monitoring-demo --scenario feature-drift` | `9371c2b` | model 2, 500-row window | `artifacts/monitoring/alerts/data_drift-*.json`, `artifacts/monitoring/exported/feature_drift.json` | `LIMIT_BAL` × 10 raises PSI alert / alert on `LIMIT_BAL`, quality ok | @thanachaithongbai-hue |
| O1, O2 (concept drift) | 2026-10-04 `monitoring-demo --scenario concept-drift` | `9371c2b` | model 2, 500 labeled rows | `artifacts/monitoring/alerts/quality-*.json`, Prometheus `ModelQualityDegraded` | Inverted labels raise quality alert without feature drift / degradation 0.537 > 0.123, drift ok, Prometheus alert firing | @thanachaithongbai-hue |
| O1 (no false alarm) | 2026-10-04 `monitoring-demo --scenario healthy` | `9371c2b` | model 2 | `artifacts/monitoring/exported/` | Real monitoring rows and labels stay quiet / no alerts | @thanachaithongbai-hue |

Not yet recorded here: hosted GitHub Actions pass/fail runs (O4, O5), a fresh-clone run by another member (G3), and the detection → retraining → new candidate cycle (O3).
