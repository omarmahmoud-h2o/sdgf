"""README.md stays in step with the code: every spec section and key, CLI subcommand,
optional extra and backend it should document, and the mock FAG demo runs as written."""

import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel

from sdgf.cli import build_parser
from sdgf.models.registry import REGISTRY as MODELS
from sdgf.spec.schema import OPTIONAL_SECTIONS, REQUIRED_SECTIONS, TaskSpec

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def section(title: str) -> str:
    start = README.index(title)
    end = README.find("\n## ", start + 1)
    return README[start : end if end != -1 else None]


def fields_of(model: type[BaseModel]) -> list[str]:
    return list(model.model_fields)


def test_every_spec_section_has_a_reference_heading():
    ref = section("## `task.yaml` reference")
    for name in REQUIRED_SECTIONS + OPTIONAL_SECTIONS:
        assert f"### `{name}`" in ref, name


@pytest.mark.parametrize("name", REQUIRED_SECTIONS + OPTIONAL_SECTIONS)
def test_every_section_key_is_documented(name):
    ref = section("## `task.yaml` reference")
    annotation = TaskSpec.model_fields[name].annotation
    model = getattr(annotation, "__args__", (annotation,))[0]  # list[ToolUse] -> ToolUse
    missing = [key for key in fields_of(model) if f"`{key}`" not in ref]
    assert not missing, f"{name}: {missing}"


def test_every_threshold_is_listed():
    ref = section("### `thresholds`")
    for key in fields_of(TaskSpec.model_fields["thresholds"].annotation):
        assert f"`{key}`" in ref, key


def test_every_subcommand_is_shown():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    for name in sub.choices:
        assert f"sdgf.cli {name}" in README, name


def test_every_optional_extra_is_listed():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    adapters = section("## Optional adapters")
    for extra in extras:
        if extra != "dev":
            assert f"| `{extra}` |" in adapters, extra


def test_every_registered_backend_is_named():
    ref = section("### `models`")
    for name in MODELS.names():
        assert f"`{name}`" in ref, name


def demo_commands(store_root: Path) -> list[list[str]]:
    block = section("## Running the FAG example").split("```bash", 1)[1].split("```", 1)[0]
    lines = block.replace("\\\n", " ").splitlines()
    commands = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("export ", "S=", "cd ")):
            continue
        commands.append(shlex.split(line.replace("$S", str(store_root))))
    return commands


def test_the_mock_demo_runs_as_written(tmp_path):
    commands = demo_commands(tmp_path)
    assert [c[3] for c in commands] == ["validate-spec", "plan", "run", "evaluate", "release"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])}
    outputs = {}
    for cmd in commands:
        assert cmd[:3] == ["python", "-m", "sdgf.cli"]
        proc = subprocess.run(
            [sys.executable, *cmd[1:]], cwd=ROOT, env=env, capture_output=True, text=True
        )
        outputs[cmd[3]] = (proc.returncode, json.loads(proc.stdout))
    assert outputs["validate-spec"][0] == 0
    assert outputs["run"][0] == 0 and outputs["run"][1]["accepted"] == 20
    # Without a calibration or embedder, the gate can't pass on its own; the README says so.
    assert outputs["evaluate"][0] == 1
    code, release = outputs["release"]
    assert code == 0 and release["released"]
    names = {p.name for p in Path(release["path"]).iterdir()}
    listed = section("A passing release").split("A failing gate", 1)[0]
    artefacts = re.findall(r"`(\w+\.(?:jsonl|json|md))`", listed)
    assert artefacts and set(artefacts) <= names, set(artefacts) - names
