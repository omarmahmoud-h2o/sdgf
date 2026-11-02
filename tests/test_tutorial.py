"""docs/tutorial.md: its generated reference tables match the code, and its links resolve."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "docs" / "tutorial.md"


def anchor(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_the_reference_tables_are_generated_from_the_code():
    pytest.importorskip("mcp")
    proc = subprocess.run(
        [sys.executable, "docs/tutorial_reference.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_the_tutorials_links_resolve():
    text = TUTORIAL.read_text()
    headings = {anchor(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE)}
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith("#"):
            assert target[1:] in headings, target
        elif not target.startswith("http"):
            assert (TUTORIAL.parent / target.split("#")[0]).exists(), target
