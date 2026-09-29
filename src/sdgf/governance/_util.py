"""Helpers shared by the governance scanners."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from types import ModuleType
from typing import Any


class GovernanceEngineError(RuntimeError):
    """An optional governance engine is missing or misconfigured."""


def lazy_import(module: str, extra: str) -> ModuleType:
    """Import an optional engine when its adapter is built, with an install hint if missing."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise GovernanceEngineError(
            f"adapter needs the optional package {module!r}; install it with: pip install {extra}"
        ) from e


def iter_strings(
    value: Any, path: str = "", *, skip_private: bool = True
) -> Iterator[tuple[str, str]]:
    """Yield (path, text) for every string in a record, with paths like messages[1].content.

    Keys starting with "_" are pipeline-private (e.g. _provenance) and skipped by default."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            key = str(key)
            if skip_private and key.startswith("_"):
                continue
            yield from iter_strings(
                item, f"{path}.{key}" if path else key, skip_private=skip_private
            )
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from iter_strings(item, f"{path}[{i}]", skip_private=skip_private)
