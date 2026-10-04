"""Check local Markdown links and JSON examples without network access."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

FENCES = re.compile(r"^```([^\n]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
LINKS = re.compile(r"!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant: {value}")


def check_document(path: Path) -> list[str]:
    """Return actionable errors for one document; URLs are not fetched."""
    content = path.read_text(encoding="utf-8-sig")
    errors: list[str] = []
    for match in FENCES.finditer(content):
        if match.group(1).strip().lower() == "json":
            try:
                json.loads(match.group(2), parse_constant=_reject_constant)
            except ValueError as exc:
                line = content.count("\n", 0, match.start()) + 1
                errors.append(f"{path}:{line}: invalid JSON example: {exc}")

    # Code examples may intentionally contain placeholder Markdown paths.
    prose = FENCES.sub(lambda match: "\n" * match.group(0).count("\n"), content)
    for match in LINKS.finditer(prose):
        target = match.group(1).strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        linked_path = path.parent / unquote(parsed.path)
        if not linked_path.exists():
            line = prose.count("\n", 0, match.start()) + 1
            errors.append(f"{path}:{line}: missing local link: {target}")
    return errors


def check_repository(root: Path) -> list[str]:
    paths = [root / "README.md", root / "CONTRIBUTING.md"]
    paths.extend(sorted((root / "docs").rglob("*.md")))
    errors: list[str] = []
    for path in paths:
        if path.exists():
            errors.extend(check_document(path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    errors = check_repository(arguments.root.resolve())
    if errors:
        print("\n".join(errors))
        return 1
    print("Documentation links and JSON examples passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
