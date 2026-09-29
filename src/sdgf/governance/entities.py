"""Real-entity deny and allow lists (FRAMEWORK_DESIGN.md §7.5).

Synthetic records must not name real organisations or people the profile denies. Names are
matched case-insensitively on word boundaries, with any run of whitespace between words, so
"Example  Bank" matches a deny entry "Example Bank".

The allow list takes precedence where the two overlap in the text: a denied match that falls
inside an allowed match is not a finding. That lets a task deny a short name while allowing a
longer fictional name that contains it (deny "Acme", allow "Acme Test Pty Ltd"). The profile
itself refuses the same name on both lists, and releasing a globally denied entity needs a
documented `entities:<name>` exception (see governance/profile.py).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from sdgf.governance._util import Finding, GovernanceEngineError, TextScanner, sort_findings
from sdgf.governance.profile import GLOBAL_PROFILE, GovernanceProfile


@dataclass(frozen=True)
class EntityFinding(Finding):
    scanner: ClassVar[str] = "entities"


def _name_pattern(names: Iterable[str]) -> re.Pattern[str] | None:
    parts = []
    for name in names:
        words = name.split()
        if not words:
            raise GovernanceEngineError(f"entity name {name!r} is empty")
        parts.append(r"\s+".join(re.escape(w) for w in words))
    if not parts:
        return None
    # Longest first, so the alternation prefers the fuller name at the same position.
    parts.sort(key=len, reverse=True)
    return re.compile(rf"(?<!\w)(?:{'|'.join(parts)})(?!\w)", re.IGNORECASE)


def _canonical(names: Iterable[str]) -> dict[str, str]:
    return {" ".join(n.split()).casefold(): n for n in names}


class EntityScanner(TextScanner):
    engine = "list"

    def __init__(self, profile: GovernanceProfile = GLOBAL_PROFILE) -> None:
        self._deny_names = _canonical(profile.entity_deny)
        self.deny = _name_pattern(sorted(profile.entity_deny))
        self.allow = _name_pattern(sorted(profile.entity_allow))

    def scan_text(self, text: str, path: str = "") -> list[EntityFinding]:
        if self.deny is None:
            return []
        allowed = [m.span() for m in self.allow.finditer(text)] if self.allow else []
        findings: list[EntityFinding] = []
        for m in self.deny.finditer(text):
            start, end = m.span()
            if any(a <= start and end <= b for a, b in allowed):
                continue
            key = " ".join(m.group(0).split()).casefold()
            rule = self._deny_names.get(key, m.group(0))
            findings.append(EntityFinding(rule, m.group(0), start, end, path, self.engine))
        return sort_findings(findings)


def build_entity_scanner(profile: GovernanceProfile = GLOBAL_PROFILE) -> EntityScanner:
    return EntityScanner(profile)
