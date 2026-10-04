"""P2 Linux checks plus full-suite verification in a temporary P0-fixture copy.

The original P0 test is never edited. The temporary copy adapts directory
creation and allows two seconds for its real subprocess startup.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            "src/mlops_project/training/requirements-dev.lock",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(["python", "-m", "pip", "check"], check=True)
    print("P2 component suite on unchanged repository", flush=True)
    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "tests/training",
            "tests/registry",
            "tests/monitoring/test_quality.py",
            "-q",
        ],
        cwd=root,
        check=True,
    )
    original = (root / "tests/test_deployment.py").read_bytes()
    with tempfile.TemporaryDirectory(prefix="p2-full-suite-") as temporary:
        copy = Path(temporary) / "repository"
        shutil.copytree(
            root,
            copy,
            ignore=shutil.ignore_patterns(
                "artifacts",
                "mlruns",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                ".git",
            ),
        )
        test = copy / "tests/test_deployment.py"
        content = test.read_text()
        assert content.count("    registry.mkdir()") == 1
        content = content.replace("    registry.mkdir()", "    registry.mkdir(exist_ok=True)")
        marker = '    monkeypatch.setattr(deployment, "_run_adapter", runner._run_adapter)'
        assert content.count(marker) == 1
        content = content.replace(
            marker,
            """    callback_context = context()
    callback_context["config"]["pipeline"]["deployment_timeout_seconds"] = 2
    Path(callback_context["config_path"]).write_text(yaml.safe_dump(callback_context["config"]), encoding="utf-8")
"""
            + marker,
        )
        test.write_text(content)
        print(
            "Full suite with P0 fixture directory/startup adaptations in a temporary copy",
            flush=True,
        )
        subprocess.run(
            ["python", "-m", "pytest", "-q"],
            cwd=copy,
            env={**os.environ, "PYTHONPATH": str(copy / "src")},
            check=True,
        )
    assert (root / "tests/test_deployment.py").read_bytes() == original
    print("Original P0 test remained unchanged.", flush=True)


if __name__ == "__main__":
    main()
