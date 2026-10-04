"""Verify the current repository in Linux with P2's locked dependencies."""

import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    commands = [
        [
            "python",
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            "src/mlops_project/training/requirements-dev.lock",
        ],
        ["python", "-m", "pip", "check"],
        ["python", "-m", "ruff", "check", ".", "--no-cache"],
        ["python", "-m", "ruff", "format", "--check", ".", "--no-cache"],
        ["python", "scripts/check_docs.py"],
        ["python", "-m", "pytest", "-p", "no:cacheprovider", "-q"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
