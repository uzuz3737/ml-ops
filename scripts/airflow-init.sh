#!/usr/bin/env bash
# P0: migration and first local administrator. No shell tracing of credentials.
set -euo pipefail
: "${AIRFLOW_ADMIN_USERNAME:?AIRFLOW_ADMIN_USERNAME is required}"
: "${AIRFLOW_ADMIN_PASSWORD:?AIRFLOW_ADMIN_PASSWORD is required}"
airflow db migrate

user_file=$(mktemp)
trap 'rm -f "$user_file"' EXIT
airflow users list --output json > "$user_file"
if python - "$user_file" "$AIRFLOW_ADMIN_USERNAME" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    users = json.load(handle)
raise SystemExit(0 if any(user["username"] == sys.argv[2] for user in users) else 1)
PY
then
  printf 'Airflow administrator already exists; credentials unchanged.\n'
else
  airflow users create \
    --username "$AIRFLOW_ADMIN_USERNAME" \
    --password "$AIRFLOW_ADMIN_PASSWORD" \
    --firstname Project --lastname Admin \
    --role Admin --email local-admin@example.invalid
fi
