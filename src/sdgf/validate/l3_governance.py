"""L3: governance (FRAMEWORK_DESIGN.md §6.4, §7.5).

Runs every governance scanner over the record's text (the "_"-private keys such as
_provenance are skipped): PII, toxicity, secrets and the real-entity deny list, all built
from the task's effective governance profile. Every finding is fail_hard: the record is
dropped, never repaired, because feeding "remove the TFN" back to the generator would train
the loop to hide violations rather than avoid them.

Records whose tool trace carries a sensitive label get stricter handling on top:

    tool_data_leak          a value from a sensitive tool result appears verbatim in the
                            record (case- and whitespace-insensitive)
    sensitive_identifier    any identifier-shaped digit run (6+ digits, spaces or hyphens
                            allowed), whether or not a PII rule recognises its format

A label is sensitive unless it is absent or "public"; an unknown label counts as sensitive,
so a mislabelled tool fails closed. The trace comes from context.extra["tool_trace"] (the
M6 gateway fills it) as ToolTraceEntry values or their dicts.

Issues never repeat the matched text: drop logs are written to disk, and copying a found
TFN into drops.jsonl would leak it a second time. They carry the rule, path and offsets.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from sdgf.governance._util import Finding, TextScanner, iter_strings
from sdgf.governance.entities import build_entity_scanner
from sdgf.governance.pii import build_pii_scanner
from sdgf.governance.profile import GLOBAL_PROFILE, GovernanceProfile, profile_for
from sdgf.governance.secrets import build_secret_scanner
from sdgf.governance.toxicity import build_toxicity_scanner
from sdgf.validate.base import Layer, LayerVerdict, Record, ValidationContext, ValidationIssue

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec

TOOL_TRACE_KEY = "tool_trace"
NON_SENSITIVE: frozenset[str | None] = frozenset({None, "", "public"})
MIN_LEAK_CHARS = 6  # shorter tool values ("yes", "AUD", "12") are too common to call a leak

_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9])\d(?:[ -]?\d){5,}(?![A-Za-z0-9])")
_WS = re.compile(r"\s+")


def is_sensitive(label: str | None) -> bool:
    return label not in NON_SENSITIVE


def _entry(entry: Any) -> Mapping[str, Any]:
    if is_dataclass(entry) and not isinstance(entry, type):
        return asdict(entry)
    if isinstance(entry, Mapping):
        return entry
    raise TypeError(f"tool trace entry must be a ToolTraceEntry or dict, got {type(entry)!r}")


def sensitive_entries(trace: Iterable[Any]) -> list[Mapping[str, Any]]:
    return [e for e in map(_entry, trace) if is_sensitive(e.get("sensitivity"))]


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def _leaf_values(value: Any) -> Iterator[str]:
    """Every scalar in a tool result, as text. Booleans and None carry no data."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (str, int, float)):
        yield str(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _leaf_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaf_values(item)


def _issue(finding: Finding) -> ValidationIssue:
    scanner = finding.scanner
    code = "entity_denied" if scanner == "entities" else f"{scanner}_{finding.rule}"
    return ValidationIssue(
        code=code,
        message=(
            f"{scanner} finding ({finding.rule}) at characters {finding.start}-{finding.end}; "
            "governance failures are dropped, not repaired"
        ),
        path=finding.path or None,
        details={
            "scanner": scanner,
            "rule": finding.rule,
            "start": finding.start,
            "end": finding.end,
            "engine": finding.engine,
            "score": finding.score,
        },
    )


class GovernanceLayer(Layer):
    name = "L3"

    def __init__(
        self,
        scanners: Sequence[TextScanner],
        *,
        min_leak_chars: int = MIN_LEAK_CHARS,
    ) -> None:
        if not scanners:
            raise ValueError("L3 needs at least one governance scanner")
        self.scanners = tuple(scanners)
        self.min_leak_chars = min_leak_chars

    @classmethod
    def from_profile(
        cls,
        profile: GovernanceProfile = GLOBAL_PROFILE,
        *,
        pii_engines: Iterable[str] = ("regex",),
        toxicity_engines: Iterable[str] = ("keywords",),
        min_leak_chars: int = MIN_LEAK_CHARS,
    ) -> GovernanceLayer:
        scanners = (
            build_pii_scanner(profile, pii_engines),
            build_toxicity_scanner(profile, toxicity_engines),
            build_secret_scanner(profile),
            build_entity_scanner(profile),
        )
        return cls(scanners, min_leak_chars=min_leak_chars)

    @classmethod
    def from_spec(cls, compiled: CompiledSpec, **kwargs: Any) -> GovernanceLayer:
        return cls.from_profile(profile_for(compiled), **kwargs)

    def check(self, record: Record, context: ValidationContext) -> LayerVerdict:
        issues = [_issue(f) for s in self.scanners for f in s.scan_record(record)]
        sensitive = sensitive_entries(context.extra.get(TOOL_TRACE_KEY) or ())
        if sensitive:
            issues.extend(self._identifier_issues(record, issues))
            issues.extend(self._leak_issues(record, sensitive))
        return self.verdict(issues, repairable=False)

    def _identifier_issues(
        self, record: Record, found: list[ValidationIssue]
    ) -> Iterator[ValidationIssue]:
        """Digit runs the regular PII rules didn't already report."""
        covered = {
            (i.path, i.details["start"], i.details["end"])
            for i in found
            if i.details.get("scanner") == "pii"
        }
        for path, text in iter_strings(record):
            for m in _IDENTIFIER.finditer(text):
                if any(
                    p == (path or None) and s <= m.start() and m.end() <= e for p, s, e in covered
                ):
                    continue
                yield ValidationIssue(
                    code="sensitive_identifier",
                    message=(
                        f"identifier-shaped number at characters {m.start()}-{m.end()} in a "
                        "record built from sensitive tool data"
                    ),
                    path=path or None,
                    details={"start": m.start(), "end": m.end()},
                )

    def _leak_issues(
        self, record: Record, entries: list[Mapping[str, Any]]
    ) -> Iterator[ValidationIssue]:
        texts = [(path, _norm(text)) for path, text in iter_strings(record)]
        seen: set[tuple[str, str]] = set()
        for entry in entries:
            for value in _leaf_values(entry.get("result")):
                needle = _norm(value)
                if len(needle) < self.min_leak_chars:
                    continue
                for path, text in texts:
                    key = (path, needle)
                    if needle in text and key not in seen:
                        seen.add(key)
                        yield ValidationIssue(
                            code="tool_data_leak",
                            message=(
                                f"record repeats a value from the {entry.get('sensitivity')} "
                                f"result of tool {entry.get('tool')!r}"
                            ),
                            path=path or None,
                            details={
                                "tool": entry.get("tool"),
                                "sensitivity": entry.get("sensitivity"),
                                "length": len(needle),
                            },
                        )
