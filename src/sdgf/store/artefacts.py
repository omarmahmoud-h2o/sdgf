"""Artefact store: saved outputs of every stage, keyed by spec_version (FRAMEWORK_DESIGN.md §7.8).

Layout under the store root:

    <root>/<spec_version>/shared/<stage>.json        reused by every run of this spec_version
    <root>/<spec_version>/runs/<run_id>/run.json     manifest (spec_version, run_id, created_at)
    <root>/<spec_version>/runs/<run_id>/<stage>.json stage artefacts for one run
    <root>/<spec_version>/runs/<run_id>/<name>.jsonl record streams (accepted, drops, ...)

Stage artefacts are written atomically (temp file + rename), so an artefact that exists
is complete and a resumed run can skip that stage. JSONL streams flush after every record,
so a killed run keeps what it wrote; a partial last line left by a crash is dropped.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

MANIFEST = "run.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ArtefactError(RuntimeError):
    """An artefact is missing, corrupt, or belongs to a different spec_version."""


def _check_name(kind: str, name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name) or ".." in name:
        raise ArtefactError(f"invalid {kind} {name!r}: use letters, digits, '_', '-', '.'")
    return name


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ArtefactError(f"artefact not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ArtefactError(f"{path}: corrupt artefact: {e}") from None


def _trim_partial_line(path: Path) -> None:
    """Drop a trailing line with no newline, i.e. a record a crash cut short."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        f.truncate(data.rfind(b"\n") + 1)


class JsonlWriter:
    """Append-only JSONL writer that flushes after each record."""

    def __init__(self, path: Path, fsync: bool = False):
        self.path = path
        self.fsync = fsync
        path.parent.mkdir(parents=True, exist_ok=True)
        _trim_partial_line(path)
        self._f: TextIO | None = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._f is None:
            raise ArtefactError(f"{self.path}: writer is closed")
        self._f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    def write_many(self, records: Iterable[dict[str, Any]]) -> None:
        for r in records:
            self.write(r)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records; a missing file yields nothing, an unterminated last line is skipped."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.endswith("\n"):
                return  # partial record from a killed run
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ArtefactError(f"{path}:{lineno}: corrupt record: {e}") from None


def _write_artefact(path: Path, spec_version: str, stage: str, data: Any) -> None:
    payload = {"spec_version": spec_version, "stage": stage, "data": data}
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _read_artefact(path: Path, spec_version: str) -> Any:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("spec_version") != spec_version:
        raise ArtefactError(f"{path}: artefact belongs to a different spec_version")
    return payload["data"]


class RunDir:
    """One run's directory, plus the spec_version-wide shared area."""

    def __init__(self, path: Path, shared: Path, spec_version: str, run_id: str, resumed: bool):
        self.path = path
        self.shared_path = shared
        self.spec_version = spec_version
        self.run_id = run_id
        self.resumed = resumed

    def _stage_path(self, stage: str, shared: bool) -> Path:
        base = self.shared_path if shared else self.path
        return base / f"{_check_name('stage', stage)}.json"

    def has_stage(self, stage: str, shared: bool = False) -> bool:
        return self._stage_path(stage, shared).is_file()

    def write_stage(self, stage: str, data: Any, shared: bool = False) -> Path:
        path = self._stage_path(stage, shared)
        _write_artefact(path, self.spec_version, stage, data)
        return path

    def read_stage(self, stage: str, shared: bool = False) -> Any:
        return _read_artefact(self._stage_path(stage, shared), self.spec_version)

    def stage(self, stage: str, compute: Callable[[], Any], shared: bool = False) -> Any:
        """Resume point: return the saved artefact if present, else compute, save and return it."""
        if self.has_stage(stage, shared):
            return self.read_stage(stage, shared)
        data = compute()
        self.write_stage(stage, data, shared)
        return data

    def jsonl_path(self, name: str) -> Path:
        return self.path / f"{_check_name('stream', name)}.jsonl"

    def jsonl(self, name: str, fsync: bool = False) -> JsonlWriter:
        return JsonlWriter(self.jsonl_path(name), fsync=fsync)

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        return list(iter_jsonl(self.jsonl_path(name)))


class ArtefactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _version_dir(self, spec_version: str) -> Path:
        return self.root / _check_name("spec_version", spec_version)

    def _shared_path(self, spec_version: str, stage: str) -> Path:
        return self._version_dir(spec_version) / "shared" / f"{_check_name('stage', stage)}.json"

    # Shared artefacts outside any run (e.g. a judge calibration), the same files a
    # RunDir of this spec_version reads with shared=True.
    def shared_path(self, spec_version: str, stage: str) -> Path:
        return self._shared_path(spec_version, stage)

    def has_shared(self, spec_version: str, stage: str) -> bool:
        return self._shared_path(spec_version, stage).is_file()

    def write_shared(self, spec_version: str, stage: str, data: Any) -> Path:
        path = self._shared_path(spec_version, stage)
        _write_artefact(path, spec_version, stage, data)
        return path

    def read_shared(self, spec_version: str, stage: str) -> Any:
        return _read_artefact(self._shared_path(spec_version, stage), spec_version)

    def shared_jsonl_path(self, spec_version: str, name: str) -> Path:
        """A shared record stream (e.g. the tool-response cache) reused across runs."""
        return self._version_dir(spec_version) / "shared" / f"{_check_name('stream', name)}.jsonl"

    def open_run(self, spec_version: str, run_id: str | None = None) -> RunDir:
        """Create a new run, or resume `run_id` if it already exists for this spec_version."""
        vdir = self._version_dir(spec_version)
        run_id = _check_name("run_id", new_run_id() if run_id is None else run_id)
        path = vdir / "runs" / run_id
        manifest = path / MANIFEST
        resumed = manifest.is_file()
        if resumed:
            meta = _read_json(manifest)
            if meta.get("spec_version") != spec_version or meta.get("run_id") != run_id:
                raise ArtefactError(f"{manifest}: manifest does not match run {run_id!r}")
        else:
            if path.exists() and any(path.iterdir()):
                raise ArtefactError(f"{path}: run directory exists without a manifest")
            meta = {
                "spec_version": spec_version,
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write_text(manifest, json.dumps(meta, sort_keys=True, indent=2))
        (vdir / "shared").mkdir(parents=True, exist_ok=True)
        return RunDir(path, vdir / "shared", spec_version, run_id, resumed)

    def runs(self, spec_version: str) -> list[str]:
        runs_dir = self._version_dir(spec_version) / "runs"
        if not runs_dir.is_dir():
            return []
        return sorted(p.name for p in runs_dir.iterdir() if (p / MANIFEST).is_file())

    def latest_run(self, spec_version: str) -> str | None:
        """Most recently created run for this spec_version, for `resume` without an id."""
        best: tuple[str, str] | None = None
        for run_id in self.runs(spec_version):
            created = _read_json(self._version_dir(spec_version) / "runs" / run_id / MANIFEST)
            key = (str(created.get("created_at", "")), run_id)
            if best is None or key > best:
                best = key
        return best[1] if best else None
