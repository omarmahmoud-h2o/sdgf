"""Stage 0 output: the frozen CompiledSpec and its spec_version (FRAMEWORK_DESIGN.md §6.1).

spec_version is a sha256 over three parts: the validated spec (as canonical JSON, so
comments and key order in task.yaml don't matter), the hooks.py source, and the raw
seeds bytes. Each part is length-prefixed so content can't shift between parts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdgf.spec.hooks import TaskHooks
from sdgf.spec.loader import load_task
from sdgf.spec.schema import TaskSpec


@dataclass(frozen=True)
class CompiledSpec:
    spec: TaskSpec
    hooks: TaskHooks
    seeds: tuple[dict[str, Any], ...]
    spec_version: str
    task_dir: Path

    @property
    def name(self) -> str:
        return self.spec.task.name


def canonical_spec_json(spec: TaskSpec) -> str:
    return json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def compute_spec_version(spec: TaskSpec, hooks_source: str, seeds_bytes: bytes) -> str:
    h = hashlib.sha256()
    for label, part in (
        (b"spec", canonical_spec_json(spec).encode("utf-8")),
        (b"hooks", hooks_source.encode("utf-8")),
        (b"seeds", seeds_bytes),
    ):
        h.update(label + b":" + str(len(part)).encode("ascii") + b":" + part)
    return h.hexdigest()


def compile_spec(path: str | Path) -> CompiledSpec:
    """Load and compile a task from its directory or task.yaml path."""
    loaded = load_task(path)
    return CompiledSpec(
        spec=loaded.spec,
        hooks=loaded.hooks,
        seeds=loaded.seeds,
        spec_version=compute_spec_version(loaded.spec, loaded.hooks.source, loaded.seeds_bytes),
        task_dir=loaded.task_dir,
    )
