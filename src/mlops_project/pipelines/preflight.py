"""Report real integration blockers before starting the full ML pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .contracts import (
    CONTRACT_VERSION,
    STAGES,
    PipelineError,
    confined_path,
    load_config,
    project_directory,
    resolve_adapter,
)

REQUIRED_THRESHOLDS = (
    ("metric", "minimum"),
    ("comparison", "max_regression"),
    ("service", "p95_latency_ms_max"),
    ("service", "error_rate_max"),
    ("service", "throughput_rps_min"),
)


def preflight(
    config_path: str | Path = "configs/project.yaml", *, project_root: str | Path | None = None
) -> dict:
    checks = []
    root = Path(project_root).resolve() if project_root else project_directory()
    try:
        config, _ = load_config(config_path, root)
        checks.append({"name": "project_config", "passed": True})
    except PipelineError as error:
        return {
            "contract_version": CONTRACT_VERSION,
            "passed": False,
            "checks": [{"name": "project_config", "passed": False, "error": error.as_dict()}],
        }
    for step in STAGES:
        try:
            resolve_adapter(config["pipeline"]["adapters"][step], step)
            checks.append({"name": f"adapter:{step}", "passed": True})
        except PipelineError as error:
            checks.append({"name": f"adapter:{step}", "passed": False, "error": error.as_dict()})
        except Exception:
            checks.append(
                {
                    "name": f"adapter:{step}",
                    "passed": False,
                    "error": {
                        "code": "adapter_import_failed",
                        "message": "Adapter import raised an internal error; its owner must investigate.",
                    },
                }
            )
    try:
        callback = config["pipeline"].get("registry_finalize")
        if callback is None:
            raise PipelineError(
                "adapter_unavailable",
                "@pairot230 must configure pipeline.registry_finalize for registry lifecycle and alias updates.",
            )
        resolve_adapter(callback, "register")
        checks.append({"name": "registry_finalize", "passed": True})
    except PipelineError as error:
        checks.append({"name": "registry_finalize", "passed": False, "error": error.as_dict()})
    except Exception:
        checks.append(
            {
                "name": "registry_finalize",
                "passed": False,
                "error": {
                    "code": "adapter_import_failed",
                    "message": "@pairot230 must repair the registry finalization callback import.",
                },
            }
        )
    if "schema_config" in config:
        try:
            import yaml

            schema_config = confined_path(config["schema_config"], root, must_exist=True)
            schema = yaml.safe_load(schema_config.read_text(encoding="utf-8"))
            if not isinstance(schema, dict) or not isinstance(schema.get("schema_path"), str):
                raise PipelineError(
                    "invalid_schema_config",
                    "Schema configuration must name the reviewed TFDV schema_path.",
                )
            confined_path(schema["schema_path"], root, must_exist=True)
            checks.append({"name": "reviewed_schema", "passed": True})
        except PipelineError as error:
            checks.append(
                {
                    "name": "reviewed_schema",
                    "passed": False,
                    "error": {
                        "code": error.code,
                        "message": "@OuanEng must provide the reviewed canonical TFDV schema and its project-local configuration.",
                    },
                }
            )
        except Exception:
            checks.append(
                {
                    "name": "reviewed_schema",
                    "passed": False,
                    "error": {
                        "code": "invalid_schema_config",
                        "message": "@OuanEng must provide valid YAML schema configuration.",
                    },
                }
            )
    try:
        import yaml

        quality_path = config.get("quality_gates", "configs/quality_gates.yaml")
        path = confined_path(quality_path, root, must_exist=True)
        quality = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(quality, dict) or not isinstance(quality.get("gates"), dict):
            raise PipelineError(
                "invalid_quality_gates", "Quality configuration must contain a gates mapping."
            )
        for group, key in REQUIRED_THRESHOLDS:
            values = quality["gates"].get(group, {})
            value = values.get(key) if isinstance(values, dict) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise PipelineError(
                    "unset_quality_gate",
                    f"Agree and set numeric gates.{group}.{key}; unset gates block approval.",
                )
            if value < 0 or (key in {"minimum", "max_regression", "error_rate_max"} and value > 1):
                raise PipelineError(
                    "invalid_quality_gates", f"gates.{group}.{key} is outside its allowed range."
                )
            if key in {"p95_latency_ms_max", "throughput_rps_min"} and value <= 0:
                raise PipelineError(
                    "invalid_quality_gates", f"gates.{group}.{key} must be positive."
                )
        checks.append({"name": "quality_gates", "passed": True})
    except PipelineError as error:
        checks.append({"name": "quality_gates", "passed": False, "error": error.as_dict()})
    except Exception:
        checks.append(
            {
                "name": "quality_gates",
                "passed": False,
                "error": {
                    "code": "invalid_quality_gates",
                    "message": "Quality-gate configuration could not be read as valid YAML.",
                },
            }
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--project-root")
    args = parser.parse_args()
    report = preflight(args.config, project_root=args.project_root)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
