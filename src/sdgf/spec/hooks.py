"""Task hook points (FRAMEWORK_DESIGN.md §4.3) and the loader for an optional hooks.py.

A task may define any subset of four module-level functions next to its task.yaml:

    label_rule(record) -> label                  correct label from checkable facts
    sampler_constraints(cell, rng) -> cell|None  make a cell's sampled fields consistent;
                                                 None marks the combination invalid
    extra_validators(record) -> list[str]        task-specific L2 errors (empty = pass)
    post_process(record) -> record               derived fields added after validation

sampler_constraints takes an explicit random.Random because its conditioning draws are
random and every draw in the framework must be seeded.

Stage 0 step 6: every present hook must be callable with exactly its expected
positional arguments; anything else raises HookError naming the hook.
"""

from __future__ import annotations

import importlib.util
import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

Record = dict[str, Any]
Cell = dict[str, Any]

HOOKS_FILENAME = "hooks.py"


@runtime_checkable
class LabelRule(Protocol):
    def __call__(self, record: Record) -> Any: ...


@runtime_checkable
class SamplerConstraints(Protocol):
    def __call__(self, cell: Cell, rng: random.Random) -> Cell | None: ...


@runtime_checkable
class ExtraValidators(Protocol):
    def __call__(self, record: Record) -> list[str]: ...


@runtime_checkable
class PostProcess(Protocol):
    def __call__(self, record: Record) -> Record: ...


# hook name -> expected positional parameter names
HOOK_SIGNATURES: dict[str, tuple[str, ...]] = {
    "label_rule": ("record",),
    "sampler_constraints": ("cell", "rng"),
    "extra_validators": ("record",),
    "post_process": ("record",),
}


class HookError(ValueError):
    """A hooks.py failed to load or a hook has the wrong signature."""


@dataclass(frozen=True)
class TaskHooks:
    label_rule: LabelRule | None = None
    sampler_constraints: SamplerConstraints | None = None
    extra_validators: ExtraValidators | None = None
    post_process: PostProcess | None = None
    source: str = ""  # hooks.py text, hashed into spec_version; "" when absent
    path: Path | None = None

    def present(self) -> list[str]:
        return [name for name in HOOK_SIGNATURES if getattr(self, name) is not None]


def check_signature(name: str, fn: Any) -> None:
    """Raise HookError unless `fn` is callable with exactly the hook's positional args."""
    expected = HOOK_SIGNATURES[name]
    if not callable(fn):
        raise HookError(f"hook {name!r} must be a function, got {type(fn).__name__}")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        raise HookError(f"hook {name!r}: cannot inspect signature ({e})") from None

    want = f"{name}({', '.join(expected)})"
    positional = [
        p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required_kw = [
        p.name for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY and p.default is p.empty
    ]
    has_varargs = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
    required_pos = [p for p in positional if p.default is p.empty]

    if required_kw:
        raise HookError(
            f"hook {name!r} has required keyword-only parameter(s) {required_kw}; expected {want}"
        )
    if len(required_pos) > len(expected) or (len(positional) < len(expected) and not has_varargs):
        raise HookError(f"hook {name!r} has signature {name}{sig}; expected {want}")


def load_hooks(path: str | Path | None) -> TaskHooks:
    """Load a hooks.py file (or a task directory containing one).

    A missing file is not an error: hooks are optional, so an empty TaskHooks is returned.
    """
    if path is None:
        return TaskHooks()
    p = Path(path)
    if p.is_dir():
        p = p / HOOKS_FILENAME
    if not p.exists():
        return TaskHooks()

    source = p.read_text(encoding="utf-8")
    module_name = f"sdgf_task_hooks_{abs(hash(str(p.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise HookError(f"cannot import hooks from {p}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise HookError(f"error importing {p}: {type(e).__name__}: {e}") from e

    found: dict[str, Callable[..., Any]] = {}
    for name in HOOK_SIGNATURES:
        if hasattr(module, name):
            fn = getattr(module, name)
            check_signature(name, fn)
            found[name] = fn
    return TaskHooks(**found, source=source, path=p)
