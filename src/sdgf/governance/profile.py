"""Governance profile: global rules plus per-task tightening (FRAMEWORK_DESIGN.md §7.5).

The global profile applies to every task. A task's `governance` section may only add to it
(extra PII patterns, extra denied entities). Anything that would relax a global rule is
refused, except the two documented-exception kinds the design allows:

- `toxicity:<category>`: exempt a toxicity category, e.g. a toxicity-detection task that
  needs toxic examples. The category must also be listed in `toxicity_exceptions`.
- `entities:<name>`: allow an entity that the global profile denies.

PII and secrets rules can never be excepted. Each exception needs a reason, must match the
relaxation it documents, and an exception that documents nothing is an error, so the spec
stays an honest record of what was relaxed.

Rule names here are the contract with the scanners (pii.py, toxicity.py, secrets.py,
entities.py): each scanner implements the global rules it is handed by the profile.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from sdgf.spec.schema import GovernanceException, GovernanceSection

SCANNERS: tuple[str, ...] = ("pii", "toxicity", "secrets", "entities")
EXCEPTABLE_SCANNERS: tuple[str, ...] = ("toxicity", "entities")

GLOBAL_PII_RULES: tuple[str, ...] = ("email", "phone", "abn", "tfn", "bsb", "account_number")
GLOBAL_TOXICITY_CATEGORIES: tuple[str, ...] = (
    "profanity",
    "insult",
    "threat",
    "identity_attack",
    "sexual",
    "self_harm",
)
GLOBAL_SECRET_RULES: tuple[str, ...] = (
    "api_key",
    "aws_access_key",
    "private_key",
    "bearer_token",
    "jwt",
    "password_assignment",
)


class GovernanceProfileError(ValueError):
    """A task's governance section is malformed or would loosen the global profile."""


class LooseningError(GovernanceProfileError):
    """A task tried to relax a global rule without a permitted, documented exception."""


def _frozen_map(m: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return MappingProxyType(dict(m or {}))


@dataclass(frozen=True)
class GovernanceProfile:
    pii_rules: frozenset[str] = frozenset(GLOBAL_PII_RULES)
    pii_patterns: Mapping[str, str] = field(default_factory=_frozen_map)
    toxicity_categories: frozenset[str] = frozenset(GLOBAL_TOXICITY_CATEGORIES)
    toxicity_exceptions: frozenset[str] = frozenset()
    secret_rules: frozenset[str] = frozenset(GLOBAL_SECRET_RULES)
    entity_deny: frozenset[str] = frozenset()
    entity_allow: frozenset[str] = frozenset()
    exceptions: tuple[GovernanceException, ...] = ()
    max_violations: int = 0

    def __post_init__(self) -> None:
        if self.max_violations != 0:
            raise GovernanceProfileError("max_violations must be 0: governance is never traded off")
        object.__setattr__(self, "pii_patterns", _frozen_map(self.pii_patterns))
        for name, pattern in self.pii_patterns.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise GovernanceProfileError(f"pii pattern {name!r} is not a valid regex: {e}")
        stray = self.toxicity_exceptions - self.toxicity_categories
        if stray:
            raise GovernanceProfileError(
                f"toxicity exceptions for unknown categories {sorted(stray)}"
            )
        both = self.entity_allow & self.entity_deny
        if both:
            raise GovernanceProfileError(f"entities both allowed and denied: {sorted(both)}")

    @property
    def active_toxicity_categories(self) -> frozenset[str]:
        return self.toxicity_categories - self.toxicity_exceptions

    def to_dict(self) -> dict[str, Any]:
        return {
            "pii_rules": sorted(self.pii_rules),
            "pii_patterns": dict(sorted(self.pii_patterns.items())),
            "toxicity_categories": sorted(self.toxicity_categories),
            "toxicity_exceptions": sorted(self.toxicity_exceptions),
            "secret_rules": sorted(self.secret_rules),
            "entity_deny": sorted(self.entity_deny),
            "entity_allow": sorted(self.entity_allow),
            "exceptions": [e.model_dump() for e in self.exceptions],
            "max_violations": self.max_violations,
        }


GLOBAL_PROFILE = GovernanceProfile()


def _parse_rule(rule: str) -> tuple[str, str]:
    scanner, sep, name = rule.partition(":")
    scanner, name = scanner.strip(), name.strip()
    if not sep or not name:
        raise GovernanceProfileError(
            f"exception rule {rule!r} must look like '<scanner>:<name>', e.g. 'toxicity:profanity'"
        )
    if scanner not in SCANNERS:
        raise GovernanceProfileError(
            f"exception rule {rule!r} names unknown scanner {scanner!r}; expected one of {SCANNERS}"
        )
    if scanner not in EXCEPTABLE_SCANNERS:
        raise LooseningError(f"exception rule {rule!r}: {scanner} rules can never be excepted")
    return scanner, name


def merge_profile(
    section: GovernanceSection | None, base: GovernanceProfile = GLOBAL_PROFILE
) -> GovernanceProfile:
    """Apply a task's governance section to `base`, refusing anything that loosens it."""
    section = section or GovernanceSection()

    documented: dict[tuple[str, str], GovernanceException] = {}
    for exc in section.exceptions:
        key = _parse_rule(exc.rule)
        if key in documented:
            raise GovernanceProfileError(f"exception rule {exc.rule!r} declared twice")
        documented[key] = exc
    used: set[tuple[str, str]] = set()

    for name, pattern in section.extra_pii_patterns.items():
        if name in base.pii_rules:
            raise LooseningError(
                f"extra_pii_patterns.{name}: cannot redefine global PII rule {name!r}"
            )
        if name in base.pii_patterns and base.pii_patterns[name] != pattern:
            raise LooseningError(f"extra_pii_patterns.{name}: cannot redefine PII pattern {name!r}")

    for category in section.toxicity_exceptions:
        if category not in base.toxicity_categories:
            raise GovernanceProfileError(
                f"toxicity_exceptions: unknown category {category!r}; "
                f"expected one of {sorted(base.toxicity_categories)}"
            )
        if ("toxicity", category) not in documented:
            raise LooseningError(
                f"toxicity_exceptions: {category!r} relaxes a global rule and needs a documented "
                f"exception (exceptions: [{{rule: 'toxicity:{category}', reason: ...}}])"
            )
        used.add(("toxicity", category))

    task_deny = set(section.entity_deny)
    contradictory = task_deny & set(section.entity_allow)
    if contradictory:
        raise GovernanceProfileError(f"entities both allowed and denied: {sorted(contradictory)}")
    released: set[str] = set()
    for entity in section.entity_allow:
        if entity in base.entity_deny:
            if ("entities", entity) not in documented:
                raise LooseningError(
                    f"entity_allow: {entity!r} is on the global deny list and needs a documented "
                    f"exception (exceptions: [{{rule: 'entities:{entity}', reason: ...}}])"
                )
            used.add(("entities", entity))
            released.add(entity)

    unused = [documented[k].rule for k in documented if k not in used]
    if unused:
        raise GovernanceProfileError(
            f"exceptions {unused} document no relaxation in this spec; remove them or add the "
            "matching toxicity_exceptions / entity_allow entry"
        )

    merged = replace(
        base,
        pii_patterns={**base.pii_patterns, **section.extra_pii_patterns},
        toxicity_exceptions=base.toxicity_exceptions | set(section.toxicity_exceptions),
        entity_deny=(base.entity_deny - released) | task_deny,
        entity_allow=base.entity_allow | set(section.entity_allow),
        exceptions=base.exceptions + tuple(section.exceptions),
    )
    check_tightens(base, merged)
    return merged


def check_tightens(base: GovernanceProfile, profile: GovernanceProfile) -> None:
    """Raise LooseningError unless `profile` is at least as strict as `base`, apart from its
    declared exceptions. Guards profiles built by code rather than by merge_profile."""
    declared = {_parse_rule(e.rule) for e in profile.exceptions}
    problems: list[str] = []
    for label, have, need in (
        ("pii_rules", profile.pii_rules, base.pii_rules),
        ("secret_rules", profile.secret_rules, base.secret_rules),
        ("toxicity_categories", profile.toxicity_categories, base.toxicity_categories),
    ):
        if need - have:
            problems.append(f"{label} dropped {sorted(need - have)}")
    for name, pattern in base.pii_patterns.items():
        if profile.pii_patterns.get(name) != pattern:
            problems.append(f"pii pattern {name!r} removed or changed")
    for category in sorted(profile.toxicity_exceptions - base.toxicity_exceptions):
        if ("toxicity", category) not in declared:
            problems.append(f"toxicity category {category!r} exempted without an exception")
    for entity in sorted(base.entity_deny - profile.entity_deny):
        if ("entities", entity) not in declared:
            problems.append(f"denied entity {entity!r} released without an exception")
    if problems:
        raise LooseningError("profile loosens the global profile: " + "; ".join(problems))


def profile_for(compiled: Any, base: GovernanceProfile = GLOBAL_PROFILE) -> GovernanceProfile:
    """The effective profile for a CompiledSpec (or bare TaskSpec)."""
    spec = getattr(compiled, "spec", compiled)
    return merge_profile(spec.governance, base)
