"""Helpers shared by the governance scanners."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar


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


@dataclass(frozen=True)
class Finding:
    """One governance hit. `scanner` says which scanner produced it (pii, toxicity, secrets,
    entities) and `rule` which of its rules or categories fired."""

    scanner: ClassVar[str] = ""

    rule: str
    text: str
    start: int
    end: int
    path: str = ""
    engine: str = "regex"
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanner": self.scanner,
            "rule": self.rule,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "path": self.path,
            "engine": self.engine,
            "score": self.score,
        }


def sort_findings(findings: list[Any]) -> list[Any]:
    return sorted(findings, key=lambda f: (f.start, f.end, f.rule, f.engine))


class TextScanner(ABC):
    """Shared shape of every governance scanner: scan one string, or every string in a
    record (skipping "_"-private keys by default)."""

    engine: str = ""

    @abstractmethod
    def scan_text(self, text: str, path: str = "") -> list[Any]: ...

    def scan_record(self, record: Mapping[str, Any], *, skip_private: bool = True) -> list[Any]:
        findings: list[Any] = []
        for path, text in iter_strings(record, skip_private=skip_private):
            findings.extend(self.scan_text(text, path))
        return findings
