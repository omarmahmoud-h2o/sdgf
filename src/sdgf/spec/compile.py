"""Stage 0 output: the frozen CompiledSpec and its spec_version (FRAMEWORK_DESIGN.md §6.1).

spec_version is a sha256 over three parts: the validated spec (as canonical JSON, so
comments and key order in task.yaml don't matter), the hooks.py source, and the raw
seeds bytes. Each part is length-prefixed so content can't shift between parts.

compile_spec also runs the stage 0 gates, so a bad spec fails before any spend:
every seed is re-scanned with the PII and toxicity scanners of the task's governance
profile, every listed tool must exist in the tool registry, and every release threshold
must be set (no silent gate defaults), and L6 must not be set up to cast K identical
votes. Judge-only worked examples (rubric.examples) are scanned like seeds and may hold
only fields the judge is allowed to see. Problems are collected and raised together.

L6 with consistency_k > 1 and every validation.consistency temperature 0 is rejected
rather than warned about: L6 votes on one model stage, so at temperature 0 its K votes
are one verdict repeated K times, at K times the cost, and confirm a wrong L5 verdict
instead of challenging it (§6.4, §11). Nothing is gained by running it that way, and a
warning is easy to miss in a long run; consistency_k: 1 is the explicit single vote.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Container, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdgf.governance.pii import build_pii_scanner
from sdgf.governance.profile import GovernanceProfileError, profile_for
from sdgf.governance.toxicity import build_toxicity_scanner
from sdgf.spec.hooks import TaskHooks
from sdgf.spec.loader import load_task
from sdgf.spec.schema import TaskSpec
from sdgf.tasktypes.base import TaskTypeError
from sdgf.tasktypes.registry import REGISTRY


class Stage0Error(ValueError):
    """The spec failed a stage 0 gate. `problems` lists every failure found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("stage 0 rejected the spec:\n  " + "\n  ".join(self.problems))


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


def _seed_label(i: int, seed: dict[str, Any]) -> str:
    sid = seed.get("id")
    return f"seed {i}" + (f" ({sid})" if isinstance(sid, str) else "")


def seed_gate_problems(
    spec: TaskSpec,
    seeds: Sequence[dict[str, Any]],
    *,
    pii_engines: Sequence[str] = ("regex",),
    toxicity_engines: Sequence[str] = ("keywords",),
) -> list[str]:
    """Re-scan every seed with the profile's PII and toxicity scanners (§6.1 step 3).

    Problems name the seed, scanner, rule and path, never the matched text.
    """
    try:
        profile = profile_for(spec)
    except GovernanceProfileError as e:
        return [f"governance: {e}"]
    scanners = _scanners(profile, pii_engines, toxicity_engines)
    problems = []
    for i, seed in enumerate(seeds):
        for kind, scanner in scanners:
            for f in scanner.scan_record(seed):
                problems.append(
                    f"seeds: {_seed_label(i, seed)} fails {kind} scan: rule {f.rule} at {f.path}"
                )
    return problems


def _scanners(profile: Any, pii_engines: Sequence[str], toxicity_engines: Sequence[str]):
    return (
        ("pii", build_pii_scanner(profile, pii_engines)),
        ("toxicity", build_toxicity_scanner(profile, toxicity_engines)),
    )


def example_gate_problems(
    spec: TaskSpec,
    *,
    pii_engines: Sequence[str] = ("regex",),
    toxicity_engines: Sequence[str] = ("keywords",),
) -> list[str]:
    """Judge-only worked examples are fictional and judge-visible only.

    Each rubric.examples entry is scanned like a seed (record and note, never the
    matched text in a problem), and its record may hold only the task type's
    judge_fields, so an example can't show the judge the label or its spans.
    """
    examples = spec.rubric.examples
    if not examples:
        return []
    problems = []
    try:
        fields = set(REGISTRY.resolve(spec.task).judge_fields())
    except TaskTypeError as e:
        problems.append(f"rubric.examples: can't check judge fields: {e}")
        fields = None
    try:
        scanners = _scanners(profile_for(spec), pii_engines, toxicity_engines)
    except GovernanceProfileError:
        scanners = ()  # seed_gate_problems already reports the governance problem
    for i, ex in enumerate(examples):
        if fields is not None:
            extra = sorted(set(ex.record) - fields)
            if extra:
                problems.append(
                    f"rubric.examples[{i}]: record has fields the judge may not see {extra}; "
                    f"allowed: {sorted(fields)}"
                )
        view = {"record": ex.record, "note": ex.note}
        for kind, scanner in scanners:
            for f in scanner.scan_record(view):
                problems.append(
                    f"rubric.examples[{i}] fails {kind} scan: rule {f.rule} at {f.path}"
                )
    return problems


def tool_gate_problems(spec: TaskSpec, tool_registry: Container[str]) -> list[str]:
    """Every tool the task lists must exist in the tool registry (§6.1 step 4)."""
    return [
        f"tools: {t.name!r} is not in the tool registry"
        for t in spec.tools
        if t.name not in tool_registry
    ]


def threshold_gate_problems(spec: TaskSpec) -> list[str]:
    """Every release threshold must be set (§6.1 step 5)."""
    return [f"thresholds.{name}: release threshold is not set" for name in spec.thresholds.unset()]


def consistency_gate_problems(spec: TaskSpec) -> list[str]:
    """L6 must not cast K identical votes: K > 1 at temperature 0 only (§6.4, §11)."""
    v = spec.validation
    if "L6" not in v.layers or v.consistency_k == 1:
        return []
    if any(t > 0 for t in v.consistency.temperatures):
        return []
    return [
        f"validation.consistency.temperatures: every L6 vote is at temperature 0, so the "
        f"{v.consistency_k} votes (consistency_k) would repeat one verdict; give a "
        "temperature above 0, or set consistency_k: 1 for a single vote"
    ]


def stage0_problems(
    spec: TaskSpec,
    seeds: Sequence[dict[str, Any]],
    *,
    tool_registry: Container[str] = frozenset(),
    pii_engines: Sequence[str] = ("regex",),
    toxicity_engines: Sequence[str] = ("keywords",),
) -> list[str]:
    return [
        *seed_gate_problems(
            spec, seeds, pii_engines=pii_engines, toxicity_engines=toxicity_engines
        ),
        *example_gate_problems(spec, pii_engines=pii_engines, toxicity_engines=toxicity_engines),
        *tool_gate_problems(spec, tool_registry),
        *threshold_gate_problems(spec),
        *consistency_gate_problems(spec),
    ]


def compile_spec(
    path: str | Path,
    *,
    tool_registry: Container[str] = frozenset(),
    pii_engines: Sequence[str] = ("regex",),
    toxicity_engines: Sequence[str] = ("keywords",),
) -> CompiledSpec:
    """Load and compile a task from its directory or task.yaml path.

    `tool_registry` is anything supporting `name in registry`; until the M6 tool registry
    exists the default is empty, so a spec listing tools must pass one explicitly.
    Raises Stage0Error if any stage 0 gate fails.
    """
    loaded = load_task(path)
    problems = stage0_problems(
        loaded.spec,
        loaded.seeds,
        tool_registry=tool_registry,
        pii_engines=pii_engines,
        toxicity_engines=toxicity_engines,
    )
    if problems:
        raise Stage0Error(problems)
    return CompiledSpec(
        spec=loaded.spec,
        hooks=loaded.hooks,
        seeds=loaded.seeds,
        spec_version=compute_spec_version(loaded.spec, loaded.hooks.source, loaded.seeds_bytes),
        task_dir=loaded.task_dir,
    )
