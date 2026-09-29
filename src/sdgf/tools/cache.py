"""Tool-response cache (FRAMEWORK_DESIGN.md §7.4, §7.8).

Responses are keyed by tool name plus canonical arguments (JSON with sorted keys, so
{"a": 1, "b": 2} and {"b": 2, "a": 1} hit the same entry). The cache saves repeat calls
and lets a record be replayed exactly from its trace.

A cache with a path appends each new entry to a JSONL file as it is stored, so a killed
run keeps what it fetched; for_store() puts that file in the spec_version's shared
artefact area, so every run of the spec reuses it. Only JSON-serialisable results can
be cached, and lookups return a copy, so a caller can't change a stored response.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from sdgf.store.artefacts import ArtefactStore, JsonlWriter, iter_jsonl
from sdgf.store.provenance import ToolTraceEntry

CACHE_STREAM = "tool_cache"


class ToolCacheError(RuntimeError):
    """A response can't be cached, or the cache file is inconsistent."""


def canonical_arguments(arguments: Any) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        raise ToolCacheError(f"tool arguments are not JSON-serialisable: {e}") from None


def cache_key(tool: str, arguments: Any) -> str:
    payload = json.dumps({"tool": tool, "arguments": canonical_arguments(arguments)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolCache:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._entries: dict[str, Any] = {}
        self._writer: JsonlWriter | None = None
        if self.path is not None:
            for lineno, entry in enumerate(iter_jsonl(self.path), 1):
                try:
                    key, tool, arguments = entry["key"], entry["tool"], entry["arguments"]
                    result = entry["result"]
                except (KeyError, TypeError):
                    raise ToolCacheError(f"{self.path}:{lineno}: malformed cache entry") from None
                if key != cache_key(tool, arguments):
                    raise ToolCacheError(f"{self.path}:{lineno}: key does not match its call")
                self._entries[key] = result

    @classmethod
    def from_trace(cls, trace: Iterable[ToolTraceEntry | Mapping[str, Any]]) -> ToolCache:
        """An in-memory cache holding every successful call in a record's tool trace.

        With it (and no handlers), a gateway answers the record's calls exactly as the
        original run did, so the record can be replayed from its provenance alone.
        """
        cache = cls()
        for entry in trace:
            e = asdict(entry) if isinstance(entry, ToolTraceEntry) else entry
            if e.get("error") is None:
                cache.put(e["tool"], e.get("arguments", {}), e.get("result"))
        return cache

    @classmethod
    def for_store(cls, store: ArtefactStore, spec_version: str) -> ToolCache:
        return cls(store.shared_jsonl_path(spec_version, CACHE_STREAM))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, tool: str, arguments: Any) -> tuple[bool, Any]:
        """(hit, result); result is a copy of the stored response."""
        key = cache_key(tool, arguments)
        if key not in self._entries:
            return False, None
        return True, copy.deepcopy(self._entries[key])

    def put(self, tool: str, arguments: Any, result: Any) -> Any:
        """Store a response; returns a copy of it as stored (JSON form, e.g. tuples as lists)."""
        key = cache_key(tool, arguments)
        try:
            stored = json.loads(json.dumps(result, ensure_ascii=False))
        except (TypeError, ValueError) as e:
            raise ToolCacheError(f"tool {tool!r}: result is not JSON-serialisable: {e}") from None
        if key in self._entries:
            return copy.deepcopy(self._entries[key])
        self._entries[key] = stored
        if self.path is not None:
            if self._writer is None:
                self._writer = JsonlWriter(self.path)
            self._writer.write({"key": key, "tool": tool, "arguments": arguments, "result": stored})
        return copy.deepcopy(stored)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
