#!/usr/bin/env bash
# P0 Linux/macOS bootstrap; Docker provides Python, YAML parsing and REST polling.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
config=configs/project.yaml
infrastructure_only=false
timeout_seconds=7200
run_id=
while (($#)); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || { printf 'Missing --config value\n' >&2; exit 2; }; config=$2; shift 2 ;;
    --infrastructure-only) infrastructure_only=true; shift ;;
    --timeout-seconds) [[ $# -ge 2 ]] || { printf 'Missing timeout value\n' >&2; exit 2; }; timeout_seconds=$2; shift 2 ;;
    --run-id) [[ $# -ge 2 ]] || { printf 'Missing run ID\n' >&2; exit 2; }; run_id=$2; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$timeout_seconds" =~ ^[0-9]+$ ]] && ((timeout_seconds >= 30 && timeout_seconds <= 7200)) || {
  printf 'Timeout must be an integer from 30 through 7200 seconds.\n' >&2; exit 2;
}
[[ "$config" =~ ^configs/[A-Za-z0-9._/-]+\.ya?ml$ ]] && [[ "/$config/" != *'/../'* ]] && [[ -f "$config" ]] || {
  printf 'Config must be an existing relative YAML path inside configs/.\n' >&2; exit 2;
}
command -v docker >/dev/null 2>&1 || { printf 'Docker is required; install and start its Linux engine.\n' >&2; exit 1; }
[[ -f .env ]] || { printf 'Create .env from .env.example and set local credentials first.\n' >&2; exit 1; }
export PIPELINE_CONFIG="$config"
if [[ "$(uname -s)" == Linux ]]; then
  # A clean Linux clone is writable by its host user, including CI's runner UID.
  export APP_UID="${APP_UID:-$(id -u)}"
fi
engine_type=$(docker info --format '{{.OSType}}')
[[ "$engine_type" == linux ]] || { printf 'This project requires Docker Linux containers.\n' >&2; exit 1; }
docker compose version
mkdir -p data artifacts schemas examples
docker compose config --quiet
docker compose build pipeline-worker airflow-init airflow-webserver airflow-scheduler
docker compose up -d --wait --wait-timeout 300 pipeline-worker
if [[ "$infrastructure_only" == false ]]; then
  docker compose exec -T pipeline-worker python -m mlops_project.pipelines.preflight --config "$config" --project-root /workspace
fi
docker compose up -d --wait --wait-timeout 300 airflow-webserver airflow-scheduler prometheus grafana pipeline-worker
if [[ "$infrastructure_only" == true ]]; then
  printf 'Infrastructure is running. No training DAG was triggered and no model was deployed.\n'
  exit 0
fi
docker compose --profile application build api
docker compose --profile application up -d --wait --wait-timeout 300 api
launch_args=(--config "$config" --timeout-seconds "$timeout_seconds")
if [[ -n "$run_id" ]]; then launch_args+=(--run-id "$run_id"); fi
docker compose exec -T pipeline-worker python /workspace/scripts/airflow-run.py "${launch_args[@]}"
