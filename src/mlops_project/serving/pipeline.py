"""`benchmark` pipeline stage: load-test the registered candidate.

The candidate runs in its own uvicorn process on a free local port, so the
live API and its champion model are never touched. The workload comes from
gates.service in quality_gates.yaml, which is what approve() checks against.
"""

from __future__ import annotations

import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import yaml

from ..pipelines.contracts import PipelineError, confined_path, sha256_file
from ..pipelines.runner import atomic_json, read_record, utc_now

_PROVENANCE = (
    "model_name",
    "model_version",
    "artifact_uri",
    "artifact_sha256",
    "code_commit",
    "dataset_version",
    "schema_version",
)
READY_TIMEOUT_SECONDS = 60
WARMUP_REQUESTS = 20


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _workload(context: dict, root: Path) -> dict:
    policy_path = confined_path(
        context["config"].get("quality_gates", "configs/quality_gates.yaml"), root, must_exist=True
    )
    service = yaml.safe_load(policy_path.read_text(encoding="utf-8"))["gates"]["service"]
    workload = {}
    for name in ("batch_size", "concurrency", "measurement_seconds"):
        value = service.get(name)
        if type(value) is not int or value < 1:
            raise PipelineError("invalid_policy", f"gates.service.{name} must be a positive int.")
        workload[name] = value
    return workload


def _payload(root: Path, context: dict, batch_size: int) -> dict:
    fixture_path = confined_path(
        context["config"]["serving"].get("smoke_fixture", "examples/predict.json"),
        root,
        must_exist=True,
    )
    instances = read_record(fixture_path).get("instances") or []
    if not instances:
        raise PipelineError("invalid_fixture", "Benchmark needs a nonempty smoke fixture.")
    return {"instances": [instances[i % len(instances)] for i in range(batch_size)]}


def _hardware() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()}; {cpu}; {os.cpu_count()} logical CPUs"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[rank]


def _wait_ready(client: httpx.Client, model_version: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise PipelineError("candidate_start_failed", "Candidate API exited during startup.")
        try:
            response = client.get("/ready")
            if response.status_code == 200 and response.json().get("model_version") == (
                model_version
            ):
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise PipelineError("candidate_start_failed", "Candidate API never became ready.")


def _load(base_url: str, payload: dict, workload: dict) -> tuple[list[float], int, int, float]:
    latencies: list[float] = []
    errors = 0
    total = 0
    lock = threading.Lock()
    deadline = time.monotonic() + workload["measurement_seconds"]

    def worker():
        nonlocal errors, total
        with httpx.Client(base_url=base_url, timeout=10) as client:
            while time.monotonic() < deadline:
                started = time.perf_counter()
                try:
                    ok = client.post("/predict", json=payload).status_code == 200
                except httpx.HTTPError:
                    ok = False
                elapsed_ms = (time.perf_counter() - started) * 1000
                with lock:
                    total += 1
                    if ok:
                        latencies.append(elapsed_ms)
                    else:
                        errors += 1

    started = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(workload["concurrency"])]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return latencies, errors, total, time.monotonic() - started


def benchmark(context: dict) -> dict:
    root = Path(context["project_root"]).resolve()
    run_dir = confined_path(context["run_dir"], root)
    registered = context.get("inputs", {}).get("register", {})
    for name in _PROVENANCE:
        if not isinstance(registered.get(name), str) or not registered[name]:
            raise PipelineError("missing_gate_evidence", f"register result lacks {name}.")

    workload = _workload(context, root)
    payload = _payload(root, context, workload["batch_size"])

    candidate_path = run_dir / "benchmark-candidate.json"
    atomic_json(
        candidate_path,
        {"deployment_id": f"benchmark-{context['run_id']}"}
        | {name: registered[name] for name in _PROVENANCE},
    )
    port = _free_port()
    env = {
        **os.environ,
        "MLOPS_PROJECT_ROOT": str(root),
        "MLOPS_CONFIG": str(context.get("config_path", "configs/project.yaml")),
        "MLOPS_CANDIDATE_MANIFEST": str(candidate_path),
        "PYTHONPATH": os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    command = [
        sys.executable, "-m", "uvicorn", "mlops_project.serving.app:app",
        "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
    ]  # fmt: skip
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=10) as client:
            _wait_ready(client, registered["model_version"], process)
            for _ in range(WARMUP_REQUESTS):
                client.post("/predict", json=payload)
        latencies, errors, total, elapsed = _load(base_url, payload, workload)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    if not latencies:
        raise PipelineError("benchmark_failed", "No successful request during the benchmark.")
    metrics = {
        "p50_latency_ms": round(_percentile(latencies, 0.50), 3),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
        "throughput_rps": round(len(latencies) / elapsed, 3),
        "error_rate": round(errors / total, 6),
    }
    report = {
        "contract_version": 1,
        "run_id": context["run_id"],
        "measured_at": utc_now(),
        "model_version": registered["model_version"],
        "metrics": metrics,
        "workload": workload,
        "requests": {"total": total, "errors": errors, "elapsed_seconds": round(elapsed, 3)},
        "hardware": _hardware(),
    }
    report_path = run_dir / "benchmark-report.json"
    atomic_json(report_path, report)
    return {
        "contract_version": 1,
        "run_id": context["run_id"],
        "passed": True,
        **{name: registered[name] for name in _PROVENANCE},
        "metrics": metrics,
        "hardware": report["hardware"],
        "workload": workload,
        "artifacts": [{"uri": report_path.name, "sha256": sha256_file(report_path)}],
    }
