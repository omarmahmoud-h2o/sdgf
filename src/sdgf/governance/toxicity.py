"""Toxicity scanning (FRAMEWORK_DESIGN.md §7.5): keyword baseline plus an optional Detoxify
adapter.

The baseline is a short, conservative keyword list per global category (profanity, insult,
threat, identity_attack, sexual, self_harm). It only catches the obvious cases; a classifier
engine is the real check. Entries are regex fragments matched case-insensitively on word
boundaries. Where a bare word has an innocent domain reading (a life-insurance "suicide
exclusion", a pest-control business dealing with "vermin") the baseline uses phrases instead,
so an ordinary banking conversation isn't hard-dropped at L3. Slurs are deliberately not
spelled out here; identity attacks are caught by phrase, and a task that needs more can pass
extra keywords.

Only the profile's active categories are reported, so a documented `toxicity:<category>`
exception switches that category off in every engine.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from sdgf.governance._util import (
    Finding,
    GovernanceEngineError,
    TextScanner,
    lazy_import,
    sort_findings,
)
from sdgf.governance.profile import GLOBAL_PROFILE, GovernanceProfile


@dataclass(frozen=True)
class ToxicityFinding(Finding):
    scanner: ClassVar[str] = "toxicity"


class ToxicityScanner(TextScanner):
    """One interface for every toxicity engine."""

    @abstractmethod
    def scan_text(self, text: str, path: str = "") -> list[ToxicityFinding]: ...


# ── keyword baseline ─────────────────────────────────────────────

_YOU_ARE = r"you(?:'re| are)"
_I_WILL = r"i(?:'ll| will| am going to|'m going to)"

GLOBAL_TOXICITY_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "profanity": (
        r"fuck\w*",
        r"motherfuck\w*",
        r"shit\w*",
        r"bullshit",
        r"cunt\w*",
        r"bastards?",
        r"bitch\w*",
        r"arseholes?",
        r"assholes?",
        r"wankers?",
        r"dickheads?",
        r"piss off",
    ),
    "insult": (
        r"idiots?",
        r"moron\w*",
        r"imbeciles?",
        r"cretins?",
        r"dumbass\w*",
        rf"{_YOU_ARE} (?:useless|pathetic|worthless|brain ?dead)",
    ),
    "threat": (
        rf"{_I_WILL} (?:kill|hurt|destroy|come after|find) you",
        r"kill you",
        rf"{_YOU_ARE} (?:dead|going to pay)",
        r"watch your back",
        r"i know where you live",
        r"burn (?:it|the place|your \w+) down",
    ),
    "identity_attack": (
        r"subhuman",
        r"go back to (?:your own country|where you came from)",
        r"(?:your|their|those|these) (?:kind|people) (?:are|don't belong|do not belong)"
        r" (?:animals|vermin|parasites|filth|here)",
    ),
    "sexual": (
        r"porn\w*",
        r"nudes",
        r"sex(?:ual)? favou?rs?",
        r"blowjobs?",
        r"have sex with",
    ),
    "self_harm": (
        r"kill(?:ing)? myself",
        r"end(?:ing)? my (?:own )?life",
        r"commit(?:ting)? suicide",
        r"suicidal",
        r"self[- ]harm\w*",
        r"(?:hurt|harm|cut)(?:ing)? myself",
        r"want to die",
    ),
}


def _compile(fragments: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(f"(?:{f})" for f in fragments)
    return re.compile(rf"(?<![\w'])(?:{alternation})(?![\w])", re.IGNORECASE)


class KeywordToxicityScanner(ToxicityScanner):
    engine = "keywords"

    def __init__(
        self,
        profile: GovernanceProfile = GLOBAL_PROFILE,
        extra_keywords: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        extra = {k: tuple(v) for k, v in (extra_keywords or {}).items()}
        unknown = set(extra) - profile.toxicity_categories
        if unknown:
            raise GovernanceEngineError(
                f"extra toxicity keywords for unknown categories {sorted(unknown)}; "
                f"expected one of {sorted(profile.toxicity_categories)}"
            )
        missing = profile.toxicity_categories - GLOBAL_TOXICITY_KEYWORDS.keys()
        if missing:
            raise GovernanceEngineError(
                f"profile names toxicity categories with no keyword list: {sorted(missing)}"
            )
        self.patterns: dict[str, re.Pattern[str]] = {}
        for category in sorted(profile.active_toxicity_categories):
            fragments = GLOBAL_TOXICITY_KEYWORDS[category] + extra.get(category, ())
            try:
                self.patterns[category] = _compile(fragments)
            except re.error as e:
                raise GovernanceEngineError(f"toxicity keywords for {category!r}: {e}") from e

    def scan_text(self, text: str, path: str = "") -> list[ToxicityFinding]:
        findings = [
            ToxicityFinding(category, m.group(0), m.start(), m.end(), path, self.engine)
            for category, pattern in self.patterns.items()
            for m in pattern.finditer(text)
        ]
        return sort_findings(findings)


# ── optional Detoxify adapter ────────────────────────────────────

# Detoxify heads mapped onto profile categories. Its overall `toxicity` / `severe_toxicity`
# scores have no profile category (so they could never be excepted) and are not reported;
# the category heads cover them. self_harm has no Detoxify head and stays keyword-only.
DETOXIFY_CATEGORIES: Mapping[str, str] = {
    "obscene": "profanity",
    "insult": "insult",
    "threat": "threat",
    "identity_attack": "identity_attack",
    "identity_hate": "identity_attack",
    "sexual_explicit": "sexual",
}


class DetoxifyToxicityScanner(ToxicityScanner):
    """Classifier engine. detoxify is imported only when this adapter is built. It scores a
    whole string, so a finding spans the whole text it was given."""

    engine = "detoxify"

    def __init__(
        self,
        model: Any = None,
        *,
        profile: GovernanceProfile = GLOBAL_PROFILE,
        variant: str = "original",
        threshold: float = 0.5,
    ) -> None:
        if model is None:
            module = lazy_import("detoxify", "sdgf[toxicity]")
            model = module.Detoxify(variant)
        self.model = model
        self.active = profile.active_toxicity_categories
        self.threshold = threshold

    def scan_text(self, text: str, path: str = "") -> list[ToxicityFinding]:
        if not text.strip():
            return []
        scores = self.model.predict(text)
        best: dict[str, float] = {}
        for head, score in scores.items():
            category = DETOXIFY_CATEGORIES.get(head)
            score = float(score)
            if category in self.active and score >= self.threshold:
                best[category] = max(score, best.get(category, 0.0))
        findings = [
            ToxicityFinding(category, text, 0, len(text), path, self.engine, score)
            for category, score in best.items()
        ]
        return sort_findings(findings)


@dataclass
class CompositeToxicityScanner(ToxicityScanner):
    scanners: list[ToxicityScanner] = field(default_factory=list)
    engine: str = "composite"

    def scan_text(self, text: str, path: str = "") -> list[ToxicityFinding]:
        return sort_findings([f for s in self.scanners for f in s.scan_text(text, path)])


ENGINES: tuple[str, ...] = ("keywords", "detoxify")


def build_toxicity_scanner(
    profile: GovernanceProfile = GLOBAL_PROFILE, engines: Iterable[str] = ("keywords",)
) -> ToxicityScanner:
    """The keyword baseline always runs; optional engines are added on top of it."""
    names = list(dict.fromkeys(["keywords", *engines]))
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        raise GovernanceEngineError(
            f"unknown toxicity engines {unknown}; expected one of {ENGINES}"
        )
    scanners: list[ToxicityScanner] = [KeywordToxicityScanner(profile)]
    if "detoxify" in names:
        scanners.append(DetoxifyToxicityScanner(profile=profile))
    return scanners[0] if len(scanners) == 1 else CompositeToxicityScanner(scanners)
