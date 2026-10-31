"""Task specs by name: the directories under one or more roots that hold a task.yaml.

Agents name a task ("fag"); they never pass a filesystem path. A name is one path
component, and a task directory that resolves outside its root (e.g. a symlink) is not
found, so a name can't reach a spec the registry wasn't given.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class TaskRegistry:
    def __init__(self, roots: Iterable[str | Path]):
        self.roots = [Path(r).resolve() for r in roots]

    def find(self, name: str) -> Path | None:
        """The task directory for name, from the first root that has it, else None."""
        if not NAME_RE.fullmatch(name):
            return None
        for root in self.roots:
            task_dir = root / name
            if (task_dir / "task.yaml").is_file() and task_dir.resolve().parent == root:
                return task_dir
        return None

    def names(self) -> list[str]:
        found = {
            d.name
            for root in self.roots
            if root.is_dir()
            for d in root.iterdir()
            if self.find(d.name) is not None
        }
        return sorted(found)
