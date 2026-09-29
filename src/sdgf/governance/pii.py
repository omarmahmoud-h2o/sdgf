"""PII scanning (FRAMEWORK_DESIGN.md §7.5): regex baseline plus an optional Presidio adapter.

The regex scanner implements the global rules named in governance/profile.py (email, phone
and the Australian banking identifiers ABN, TFN, BSB, account number) and any per-task
patterns the profile carries. It is deliberately conservative: an identifier-shaped string is
a finding whether or not it is real, because L3 treats every finding as a hard drop and a
fictional-looking TFN in a released record is still a governance problem.

Digit runs are matched whole: a pattern never matches part of a longer number (a TFN inside
an ABN, a BSB at the start of a TFN), and digits glued to letters (hashes, ids) don't count.

A per-task pattern may mark the sensitive part with a named group `pii`; otherwise the whole
match is the finding.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sdgf.governance._util import GovernanceEngineError, iter_strings, lazy_import
from sdgf.governance.profile import GLOBAL_PROFILE, GovernanceProfile


@dataclass(frozen=True)
class PIIFinding:
    rule: str
    text: str
    start: int
    end: int
    path: str = ""
    engine: str = "regex"
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "path": self.path,
            "engine": self.engine,
            "score": self.score,
        }


class PIIScanner(ABC):
    """One interface for every PII engine."""

    engine: str = ""

    @abstractmethod
    def scan_text(self, text: str, path: str = "") -> list[PIIFinding]: ...

    def scan_record(
        self, record: Mapping[str, Any], *, skip_private: bool = True
    ) -> list[PIIFinding]:
        findings: list[PIIFinding] = []
        for path, text in iter_strings(record, skip_private=skip_private):
            findings.extend(self.scan_text(text, path))
        return findings


# ── regex baseline ───────────────────────────────────────────────

# Whole digit runs only: not preceded by an alphanumeric, "+" (an international prefix) or
# "<digit><sep>", and not
# followed by an alphanumeric or by "<sep><digit>".
_B = r"(?<![A-Za-z0-9+])(?<![0-9][ -])"
_E = r"(?![A-Za-z0-9])(?![ -][0-9])"
_S = r"[ -]?"

GLOBAL_PII_PATTERNS: Mapping[str, str] = {
    "email": r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
    "phone": (
        rf"(?:(?<![\w+])\+61{_S}\(?0?\)?{_S}[2-478]{_S}\d{{4}}{_S}\d{{4}}"  # +61 2 9999 9999
        rf"|(?<![\w+])\+61{_S}4\d{{2}}{_S}\d{{3}}{_S}\d{{3}}"  # +61 412 345 678
        rf"|{_B}04\d{{2}}{_S}\d{{3}}{_S}\d{{3}}"  # 0412 345 678
        rf"|(?<![\w(])\(0[2378]\){_S}\d{{4}}{_S}\d{{4}}"  # (02) 9999 9999
        rf"|{_B}0[2378]{_S}\d{{4}}{_S}\d{{4}}"  # 02 9999 9999
        rf"|{_B}1[38]00{_S}\d{{3}}{_S}\d{{3}}"  # 1300 123 456
        rf"|{_B}13[ -]\d{{2}}[ -]\d{{2}})"  # 13 12 34
        rf"{_E}"
    ),
    "abn": rf"{_B}(?:\d{{2}}[ -]\d{{3}}[ -]\d{{3}}[ -]\d{{3}}|\d{{11}}){_E}",
    "tfn": rf"{_B}(?:\d{{3}}[ -]\d{{3}}[ -]\d{{3}}|\d{{9}}){_E}",
    "bsb": (
        rf"(?:(?i:\bBSB)\s*(?:(?i:number|no\.?)\s*)?[\s:#]*(?P<pii>\d{{3}}{_S}\d{{3}}){_E}"
        rf"|{_B}\d{{3}}-\d{{3}}{_E})"
    ),
    "account_number": (
        rf"(?:{_B}\d{{3}}{_S}\d{{3}}[ -]\d{{6,10}}{_E}"  # BSB + account: 000-000 00000000
        rf"|(?i:\b(?:account|acct|a/c))\s*(?:(?i:number|num|no\.?)\s*)?[\s:#]*"
        rf"(?P<pii>\d(?:{_S}\d){{5,9}}){_E})"
    ),
}


class RegexPIIScanner(PIIScanner):
    engine = "regex"

    def __init__(self, profile: GovernanceProfile = GLOBAL_PROFILE) -> None:
        unknown = profile.pii_rules - GLOBAL_PII_PATTERNS.keys()
        if unknown:
            raise GovernanceEngineError(
                f"profile names PII rules with no regex implementation: {sorted(unknown)}"
            )
        patterns = {name: GLOBAL_PII_PATTERNS[name] for name in sorted(profile.pii_rules)}
        patterns.update(profile.pii_patterns)
        self.patterns: dict[str, re.Pattern[str]] = {
            name: re.compile(p) for name, p in patterns.items()
        }

    def scan_text(self, text: str, path: str = "") -> list[PIIFinding]:
        findings: list[PIIFinding] = []
        for rule, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                group = "pii" if "pii" in pattern.groupindex and m.group("pii") else 0
                start, end = m.span(group)
                findings.append(PIIFinding(rule, text[start:end], start, end, path, self.engine))
        return sorted(findings, key=lambda f: (f.start, f.end, f.rule))


# ── optional Presidio adapter ────────────────────────────────────

PRESIDIO_RULES: Mapping[str, str] = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "AU_ABN": "abn",
    "AU_TFN": "tfn",
    "AU_ACN": "acn",
    "AU_MEDICARE": "medicare",
    "CREDIT_CARD": "credit_card",
    "IBAN_CODE": "account_number",
    "PERSON": "person",
    "LOCATION": "location",
    "IP_ADDRESS": "ip_address",
}


class PresidioPIIScanner(PIIScanner):
    """General PII engine. presidio_analyzer is imported only when this adapter is built.

    Findings keep the regex scanner's rule names where Presidio has an equivalent entity;
    other entities are reported as their lower-cased Presidio name."""

    engine = "presidio"

    def __init__(
        self,
        analyzer: Any = None,
        *,
        language: str = "en",
        entities: Iterable[str] | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        if analyzer is None:
            module = lazy_import("presidio_analyzer", "sdgf[pii]")
            analyzer = module.AnalyzerEngine()
        self.analyzer = analyzer
        self.language = language
        self.entities = list(entities) if entities is not None else None
        self.score_threshold = score_threshold

    def scan_text(self, text: str, path: str = "") -> list[PIIFinding]:
        results = self.analyzer.analyze(text=text, language=self.language, entities=self.entities)
        findings = [
            PIIFinding(
                PRESIDIO_RULES.get(r.entity_type, r.entity_type.lower()),
                text[r.start : r.end],
                r.start,
                r.end,
                path,
                self.engine,
                float(r.score),
            )
            for r in results
            if r.score >= self.score_threshold
        ]
        return sorted(findings, key=lambda f: (f.start, f.end, f.rule))


@dataclass
class CompositePIIScanner(PIIScanner):
    """Runs several engines and reports every finding (duplicates across engines kept, since
    each engine's hit is evidence on its own)."""

    scanners: list[PIIScanner] = field(default_factory=list)
    engine: str = "composite"

    def scan_text(self, text: str, path: str = "") -> list[PIIFinding]:
        findings = [f for s in self.scanners for f in s.scan_text(text, path)]
        return sorted(findings, key=lambda f: (f.start, f.end, f.rule, f.engine))


ENGINES: tuple[str, ...] = ("regex", "presidio")


def build_pii_scanner(
    profile: GovernanceProfile = GLOBAL_PROFILE, engines: Iterable[str] = ("regex",)
) -> PIIScanner:
    """The regex baseline always runs; optional engines are added on top of it."""
    names = list(dict.fromkeys(["regex", *engines]))
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        raise GovernanceEngineError(f"unknown PII engines {unknown}; expected one of {ENGINES}")
    scanners: list[PIIScanner] = [RegexPIIScanner(profile)]
    if "presidio" in names:
        scanners.append(PresidioPIIScanner())
    return scanners[0] if len(scanners) == 1 else CompositePIIScanner(scanners)
