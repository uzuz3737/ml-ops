"""Exercise documentation validation using isolated fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_docs.py"
SPEC = importlib.util.spec_from_file_location("check_docs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_encoded_local_link_and_external_links(tmp_path: Path) -> None:
    (tmp_path / "report name.md").write_text("Report", encoding="utf-8")
    document = tmp_path / "README.md"
    document.write_text(
        "[report](report%20name.md#results) [remote](https://example.org/missing) "
        "[section](#section)",
        encoding="utf-8",
    )
    assert CHECKER.check_document(document) == []


def test_missing_link_reports_source_line(tmp_path: Path) -> None:
    document = tmp_path / "README.md"
    document.write_text("Title\n\n[broken](missing.md)\n", encoding="utf-8")
    errors = CHECKER.check_document(document)
    assert len(errors) == 1
    assert ":3: missing local link: missing.md" in errors[0]


@pytest.mark.parametrize("example", ['{"passed":}', '{"latency": NaN}'])
def test_invalid_json_examples_fail(tmp_path: Path, example: str) -> None:
    document = tmp_path / "README.md"
    document.write_text(f"```json\n{example}\n```\n", encoding="utf-8")
    assert "invalid JSON example" in CHECKER.check_document(document)[0]


def test_code_block_links_are_not_treated_as_prose(tmp_path: Path) -> None:
    document = tmp_path / "README.md"
    document.write_text(
        '```markdown\n[example](placeholder.md)\n```\n```json\n{"passed": true}\n```\n',
        encoding="utf-8",
    )
    assert CHECKER.check_document(document) == []


def test_repository_docs_are_valid() -> None:
    assert CHECKER.check_repository(SCRIPT.parents[1]) == []
