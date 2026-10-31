"""Drafts: task specs an agent writes, which a person approves before they run for real.

A draft lives in a workspace apart from the approved task directories:

    <root>/<name>/spec/      task.yaml, seeds.jsonl, hooks.py: what the agent edits
    <root>/<name>/draft.json what happened to it: submitted, rejected, approved
    <root>/_store/           the artefact store for dry runs, apart from real runs

A draft's hooks.py is code an agent wrote, so it never runs in the caller's process:
validate() compiles the draft in a subprocess (python -m sdgf.drafts inspect) with a
timeout and only ENV_KEEP of the environment, so the code can't read API keys. That is
not a sandbox: it can still use the network and the file system. allow_hooks=False keeps
agents from writing hooks.py at all; a draft copied from an approved task keeps its
hooks.py, which a person approved before.

The states follow from draft.json and a digest of the three files' bytes:

    editing    changed since anything below, or new
    submitted  validated and submitted, unchanged since
    rejected   a person rejected this exact content, with a reason for the agent
    approved   a person approved this exact content and it was copied into the tasks

Any edit after submitting, rejecting or approving makes the draft editing again, so a
person only ever approves the bytes they saw. Approval copies the spec into the approved
task root, where the registry (and so a paid run) can see it; nothing else can.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sdgf.jobs import child_env
from sdgf.service import ServiceError
from sdgf.service.tasks import NAME_RE

SPEC_FILES = ("task.yaml", "seeds.jsonl", "hooks.py")
HOOKS = "hooks.py"
SEEDS = "seeds.jsonl"
SPEC_DIR = "spec"
META_FILE = "draft.json"
STORE_DIR = "_store"  # can't clash with a draft: names start with a letter or digit
WORK_DIR = "_work"  # the validation subprocess's working directory

MAX_FILE_BYTES = 512 * 1024
MAX_SEEDS_PER_CALL = 1000
SEED_PREVIEW = 5
DEFAULT_VALIDATE_TIMEOUT_S = 60.0

EDITING = "editing"
SUBMITTED = "submitted"
REJECTED = "rejected"
APPROVED = "approved"


class DraftError(ServiceError):
    """A draft request that can't be done."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def spec_digest(spec_dir: Path) -> str:
    """sha256 over the three spec files' bytes, a missing file counting as absent."""
    h = hashlib.sha256()
    for name in SPEC_FILES:
        path = spec_dir / name
        h.update(name.encode() + b"\0")
        h.update(path.read_bytes() if path.is_file() else b"\0absent\0")
        h.update(b"\0")
    return h.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class DraftStore:
    def __init__(
        self,
        root: str | Path,
        *,
        approved_root: str | Path | None = None,
        allow_hooks: bool = True,
        validate_timeout_s: float = DEFAULT_VALIDATE_TIMEOUT_S,
        env: Mapping[str, str] | None = None,
    ):
        self.root = Path(root).resolve()
        self.approved_root = Path(approved_root).resolve() if approved_root else None
        self.allow_hooks = allow_hooks
        self.validate_timeout_s = validate_timeout_s
        self.env = dict(env or {})  # added to the validation subprocess's environment

    @property
    def store_root(self) -> Path:
        return self.root / STORE_DIR

    # ── reading ──────────────────────────────────────────────────

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            d.name for d in self.root.iterdir() if NAME_RE.fullmatch(d.name) and self._exists(d)
        )

    def list(self) -> list[dict[str, Any]]:
        return [self._summary(name) for name in self.names()]

    def spec_dir(self, name: str) -> Path:
        return self._draft_dir(name) / SPEC_DIR

    def state(self, name: str) -> str:
        meta = self._meta(name)
        digest = spec_digest(self.spec_dir(name))
        for key, state in (
            ("approval", APPROVED),
            ("submitted", SUBMITTED),
            ("rejected", REJECTED),
        ):
            entry = meta.get(key)
            if entry and entry["digest"] == digest:
                return state
        return EDITING

    def read(self, name: str) -> dict[str, Any]:
        """The draft's files (seeds as a count and a preview) and what happened to it."""
        spec = self.spec_dir(name)
        files: dict[str, Any] = {}
        for file in SPEC_FILES:
            path = spec / file
            if not path.is_file():
                files[file] = None
            elif file == SEEDS:
                lines = path.read_text(encoding="utf-8").splitlines()
                files[file] = {
                    "count": sum(1 for line in lines if line.strip()),
                    "preview": lines[:SEED_PREVIEW],
                }
            else:
                files[file] = path.read_text(encoding="utf-8")
        return {**self._summary(name), "files": files, "hooks_editable": self.allow_hooks}

    def review(self, name: str) -> dict[str, Any]:
        """For the person approving: the full files, and a diff against the approved task
        of the same name when there is one."""
        spec = self.spec_dir(name)
        current = self._approved_dir(name)
        diffs: dict[str, str] = {}
        full: dict[str, str | None] = {}
        for file in SPEC_FILES:
            path = spec / file
            text = path.read_text(encoding="utf-8") if path.is_file() else None
            full[file] = text
            if current is not None and current.is_dir():
                old_path = current / file
                old = old_path.read_text(encoding="utf-8") if old_path.is_file() else ""
                diff = difflib.unified_diff(
                    old.splitlines(keepends=True),
                    (text or "").splitlines(keepends=True),
                    f"approved/{name}/{file}",
                    f"draft/{name}/{file}",
                )
                diffs[file] = "".join(diff)
        return {
            **self._summary(name),
            "files": full,
            "diff_against_approved": diffs if diffs else None,
        }

    # ── editing ──────────────────────────────────────────────────

    def create(self, name: str, *, from_dir: Path | None = None) -> dict[str, Any]:
        """A new draft, empty or a copy of a task's spec files (from_dir)."""
        if not NAME_RE.fullmatch(name):
            raise DraftError(
                f"invalid draft name {name!r}",
                code="bad_name",
                hint="use letters, digits, '_', '-' and '.', starting with a letter or digit",
            )
        draft = self.root / name
        if draft.exists():
            raise DraftError(f"draft {name!r} already exists", code="draft_exists")
        (draft / SPEC_DIR).mkdir(parents=True)
        if from_dir is not None:
            for file in SPEC_FILES:
                if (from_dir / file).is_file():
                    shutil.copyfile(from_dir / file, draft / SPEC_DIR / file)
        meta = {
            "name": name,
            "created_at": _now(),
            "source": from_dir.name if from_dir is not None else None,
            "submitted": None,
            "rejected": None,
            "approval": None,
        }
        _write_atomic(draft / META_FILE, json.dumps(meta, indent=2).encode())
        return self.read(name)

    def write_file(self, name: str, file: str, content: str) -> dict[str, Any]:
        self._check_writable(name, file)
        data = content.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise DraftError(
                f"{file} would be {len(data)} bytes; the limit is {MAX_FILE_BYTES}",
                code="too_large",
            )
        _write_atomic(self.spec_dir(name) / file, data)
        return self._summary(name)

    def append_seeds(self, name: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        self._check_writable(name, SEEDS)
        if not records or len(records) > MAX_SEEDS_PER_CALL:
            raise DraftError(
                f"append 1 to {MAX_SEEDS_PER_CALL} seeds at a time; got {len(records)}",
                code="bad_seeds",
            )
        if not all(isinstance(r, Mapping) for r in records):
            raise DraftError("each seed must be a JSON object", code="bad_seeds")
        path = self.spec_dir(name) / SEEDS
        old = path.read_bytes() if path.is_file() else b""
        if old and not old.endswith(b"\n"):
            old += b"\n"
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        data = old + lines.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise DraftError(
                f"{SEEDS} would be {len(data)} bytes; the limit is {MAX_FILE_BYTES}",
                code="too_large",
            )
        _write_atomic(path, data)
        return self._summary(name)

    def delete_file(self, name: str, file: str) -> dict[str, Any]:
        if file not in SPEC_FILES:
            raise DraftError(f"{file!r} is not a spec file", code="bad_file", hint=str(SPEC_FILES))
        self.spec_dir(name).joinpath(file).unlink(missing_ok=True)
        return self._summary(name)

    # ── checking ─────────────────────────────────────────────────

    def validate(self, name: str) -> dict[str, Any]:
        """Compile the draft in a subprocess and report it, or the error that stopped it."""
        work = self.root / WORK_DIR
        work.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "sdgf.drafts", "inspect", str(self.spec_dir(name))]
        try:
            proc = subprocess.run(
                cmd,
                check=False,  # a failed compile is a report, read below
                capture_output=True,
                text=True,
                cwd=work,
                env=child_env(self.env, inherit=False),
                timeout=self.validate_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return _invalid(
                "validation_timeout",
                f"compiling the draft took over {self.validate_timeout_s:g}s",
                "look for slow or blocking code at the top level of hooks.py",
            )
        lines = proc.stdout.strip().splitlines()
        try:
            return json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            tail = proc.stderr.strip().splitlines()[-5:]
            return _invalid(
                "validation_crashed",
                f"the validator exited {proc.returncode} without a report: {' | '.join(tail)}",
                None,
            )

    def submit(self, name: str) -> dict[str, Any]:
        """Validate, then mark this exact content as waiting for a person."""
        report = self.validate(name)
        if not report["ok"]:
            error = report["error"]
            raise DraftError(
                f"the draft doesn't compile: {error['message']}",
                code="draft_invalid",
                hint=error.get("hint") or "fix it, then validate_draft and submit again",
            )
        meta = self._meta(name)
        meta["submitted"] = {
            "digest": spec_digest(self.spec_dir(name)),
            "spec_version": report["spec_version"],
            "warnings": report["warnings"],
            "at": _now(),
        }
        self._save_meta(name, meta)
        return {**self._summary(name), "validation": report}

    # ── a person's decisions ─────────────────────────────────────

    def approve(
        self, name: str, *, reviewer: str, note: str | None = None, replace: bool = False
    ) -> dict[str, Any]:
        """Copy the submitted draft into the approved task root as task <name>."""
        if not reviewer:
            raise DraftError("approving needs a reviewer", code="no_reviewer")
        if self.approved_root is None:
            raise DraftError("no approved task root is configured", code="no_task_root")
        self._require_submitted(name)
        dest = self.approved_root / name
        if dest.exists() and not replace:
            raise DraftError(
                f"task {name!r} already exists in {self.approved_root}",
                code="task_exists",
                hint="approve with replace to overwrite it, or copy the draft under a new name",
            )
        spec = self.spec_dir(name)
        dest.mkdir(parents=True, exist_ok=True)
        for file in SPEC_FILES:
            if (spec / file).is_file():
                shutil.copyfile(spec / file, dest / file)
            else:
                (dest / file).unlink(missing_ok=True)
        meta = self._meta(name)
        meta["approval"] = {
            "digest": meta["submitted"]["digest"],
            "spec_version": meta["submitted"]["spec_version"],
            "reviewer": reviewer,
            "note": note,
            "at": _now(),
            "promoted_to": str(dest),
        }
        self._save_meta(name, meta)
        return self._summary(name)

    def reject(self, name: str, *, reviewer: str, reason: str) -> dict[str, Any]:
        """Send a submitted draft back to the agent with a reason."""
        if not reviewer or not reason:
            raise DraftError("rejecting needs a reviewer and a reason", code="no_reviewer")
        self._require_submitted(name)
        meta = self._meta(name)
        meta["rejected"] = {
            "digest": meta["submitted"]["digest"],
            "reviewer": reviewer,
            "reason": reason,
            "at": _now(),
        }
        meta["submitted"] = None
        self._save_meta(name, meta)
        return self._summary(name)

    # ── helpers ──────────────────────────────────────────────────

    def _draft_dir(self, name: str) -> Path:
        draft = self.root / name
        if not NAME_RE.fullmatch(name) or not self._exists(draft):
            raise DraftError(f"unknown draft {name!r}", code="unknown_draft")
        return draft

    @staticmethod
    def _exists(draft: Path) -> bool:
        return (draft / META_FILE).is_file() and (draft / SPEC_DIR).is_dir()

    def _approved_dir(self, name: str) -> Path | None:
        return self.approved_root / name if self.approved_root is not None else None

    def _meta(self, name: str) -> dict[str, Any]:
        return json.loads((self._draft_dir(name) / META_FILE).read_text(encoding="utf-8"))

    def _save_meta(self, name: str, meta: Mapping[str, Any]) -> None:
        _write_atomic(self._draft_dir(name) / META_FILE, json.dumps(meta, indent=2).encode())

    def _summary(self, name: str) -> dict[str, Any]:
        meta = self._meta(name)
        state = self.state(name)
        summary = {"name": name, "state": state, "created_at": meta["created_at"]}
        if meta.get("source"):
            summary["copied_from"] = meta["source"]
        if state == REJECTED:
            summary["rejected"] = {k: meta["rejected"][k] for k in ("reviewer", "reason", "at")}
        if state == APPROVED:
            approval = meta["approval"]
            summary["approved"] = {
                k: approval[k] for k in ("reviewer", "note", "at", "spec_version")
            }
        if state == SUBMITTED:
            summary["submitted"] = {k: meta["submitted"][k] for k in ("spec_version", "at")}
            summary["hint"] = "waiting for a person to approve or reject it"
        return summary

    def _check_writable(self, name: str, file: str) -> None:
        self._draft_dir(name)
        if file not in SPEC_FILES:
            raise DraftError(f"{file!r} is not a spec file", code="bad_file", hint=str(SPEC_FILES))
        if file == HOOKS and not self.allow_hooks:
            raise DraftError(
                "this server doesn't let agents write hooks.py",
                code="hooks_not_allowed",
                hint="use the task type's built-in rules, or ask a person to write the hooks",
            )

    def _require_submitted(self, name: str) -> None:
        state = self.state(name)
        if state != SUBMITTED:
            meta = self._meta(name)
            changed = meta.get("submitted") is not None
            raise DraftError(
                f"draft {name!r} is {state}"
                + (", and changed since it was submitted" if changed else ", not submitted"),
                code="not_submitted",
                hint="the agent must submit the current content first",
            )


def _invalid(code: str, message: str, hint: str | None) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, "hint": hint}}


# ── the validation subprocess ────────────────────────────────────


def inspect(spec_dir: Path) -> dict[str, Any]:
    """Compile a spec and describe it, with warnings a person approving it should see.
    Runs in its own process: this is where a draft's hooks.py is imported."""
    from sdgf.service import Service

    service = Service()
    try:
        described = service.describe_task(spec_dir).data
        compiled = service.compile(spec_dir)
        warnings = []
        spec = compiled.spec
        if "few_shot" in spec.seeds.uses and compiled.seeds:
            generator = spec.models.generator
            hosting = generator.hosting or _default_hosting().get(generator.backend)
            if hosting != "local":
                where = (
                    "an external provider" if hosting else "a model whose hosting isn't declared"
                )
                warnings.append(
                    {
                        "code": "seeds_sent_to",
                        "message": f"few-shot seed text goes word for word to "
                        f"{generator.backend}:{generator.model}, {where}; obfuscate real seeds",
                    }
                )
    except Exception as e:  # noqa: BLE001 - every failure is a report for the agent
        return _invalid("invalid_spec", f"{type(e).__name__}: {e}", None)
    return {
        "ok": True,
        **described,
        "hooks": (spec_dir / HOOKS).is_file(),
        "warnings": warnings,
    }


def _default_hosting() -> dict[str, str | None]:
    """Each registered backend's default hosting, read from its class: building a backend
    can need its API key, which the validation subprocess doesn't have."""
    import sdgf.models.registry  # noqa: F401 - imports and registers the backends
    from sdgf.models.base import ModelBackend

    found: dict[str, str | None] = {}
    todo = list(ModelBackend.__subclasses__())
    while todo:
        cls = todo.pop()
        todo.extend(cls.__subclasses__())
        if isinstance(getattr(cls, "name", None), str):
            found.setdefault(cls.name, cls.default_hosting)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sdgf.drafts")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="compile a spec directory and report it as JSON")
    p.add_argument("spec_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(inspect(args.spec_dir), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
