"""Load a task directory: task.yaml, the optional hooks.py, and the seeds file.

Seed paths in task.yaml are resolved relative to the directory holding task.yaml.
Loading only reads and parses; checks that need the whole spec live in compile.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sdgf.spec.hooks import TaskHooks, load_hooks
from sdgf.spec.schema import TaskSpec, parse_spec

SPEC_FILENAME = "task.yaml"


class SpecLoadError(ValueError):
    """task.yaml or the seeds file could not be read or parsed."""


@dataclass(frozen=True)
class LoadedTask:
    spec: TaskSpec
    hooks: TaskHooks
    seeds: tuple[dict[str, Any], ...]
    task_dir: Path
    spec_path: Path
    seeds_path: Path
    seeds_bytes: bytes  # raw seeds file content, hashed into spec_version


def resolve_spec_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / SPEC_FILENAME
    if not p.is_file():
        raise SpecLoadError(f"task spec not found: {p}")
    return p


def read_spec(spec_path: Path) -> TaskSpec:
    try:
        data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SpecLoadError(f"{spec_path}: invalid YAML: {e}") from None
    return parse_spec(data)


def parse_seeds(raw: bytes, source: Path) -> tuple[dict[str, Any], ...]:
    """Parse JSONL seeds; blank lines are skipped, every other line must be a JSON object."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SpecLoadError(f"{source}: seeds are not valid UTF-8: {e}") from None
    seeds: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SpecLoadError(f"{source}:{lineno}: invalid JSON: {e.msg}") from None
        if not isinstance(obj, dict):
            raise SpecLoadError(f"{source}:{lineno}: each seed must be a JSON object")
        seeds.append(obj)
    return tuple(seeds)


def load_task(path: str | Path) -> LoadedTask:
    """Load a task from its directory or its task.yaml path."""
    spec_path = resolve_spec_path(path)
    task_dir = spec_path.parent
    spec = read_spec(spec_path)
    hooks = load_hooks(task_dir)

    seeds_path = (task_dir / spec.seeds.path).resolve()
    if not seeds_path.is_file():
        raise SpecLoadError(f"seeds.path: file not found: {seeds_path}")
    seeds_bytes = seeds_path.read_bytes()
    seeds = parse_seeds(seeds_bytes, seeds_path)

    return LoadedTask(
        spec=spec,
        hooks=hooks,
        seeds=seeds,
        task_dir=task_dir,
        spec_path=spec_path,
        seeds_path=seeds_path,
        seeds_bytes=seeds_bytes,
    )
