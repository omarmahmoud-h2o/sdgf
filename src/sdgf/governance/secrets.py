"""Secret and credential scanning (FRAMEWORK_DESIGN.md §7.5).

Regex rules for the global secret rules named in governance/profile.py: API keys (known
provider prefixes and `api_key = ...` style assignments), AWS access key ids, PEM private
keys, bearer tokens, JWTs and password assignments. Like the PII scanner it is conservative:
a credential-shaped string is a finding whether or not it is live.

Assignment rules need a value that looks like a credential (long enough, and for passwords
containing a digit or symbol), so ordinary prose such as "reset your password: it takes a
minute" is not a hit. A named group `secret` marks the sensitive part of a match.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from sdgf.governance._util import Finding, GovernanceEngineError, TextScanner, sort_findings
from sdgf.governance.profile import GLOBAL_PROFILE, GovernanceProfile


@dataclass(frozen=True)
class SecretFinding(Finding):
    scanner: ClassVar[str] = "secrets"


_Q = r"[\"']?"

GLOBAL_SECRET_PATTERNS: Mapping[str, str] = {
    "api_key": (
        r"(?<![\w-])(?P<secret>"
        r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"  # OpenAI / Anthropic style
        r"|gh[pousr]_[A-Za-z0-9]{30,}"  # GitHub
        r"|xox[abposr]-[A-Za-z0-9-]{10,}"  # Slack
        r"|AIza[0-9A-Za-z_-]{35}"  # Google
        r"|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"  # Stripe
        r")(?![\w-])"
        rf"|(?i:\b(?:api[_ -]?key|api[_ -]?secret|access[_ -]?token|client[_ -]?secret"
        rf"|secret[_ -]?key|auth[_ -]?token))\s*[:=]\s*{_Q}(?P<secret2>[A-Za-z0-9_\-./+=]{{16,}}){_Q}"
    ),
    "aws_access_key": r"(?<![A-Za-z0-9])(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}(?![A-Za-z0-9])",
    "private_key": (
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
        r"(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----)?"
    ),
    "bearer_token": r"(?i:\bbearer)\s+(?P<secret>[A-Za-z0-9_\-.~+/]{20,}=*)",
    "jwt": (r"(?<![\w-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![\w-])"),
    "password_assignment": (
        r"(?i:\b(?:password|passwd|pwd|passcode|pass[_ -]?phrase))\s*[:=]\s*"
        rf"{_Q}(?P<secret>(?=[^\s\"']*[0-9!@#$%^&*_+=?~])[^\s\"']{{6,}}){_Q}"
    ),
}


class SecretScanner(TextScanner):
    engine = "regex"

    def __init__(self, profile: GovernanceProfile = GLOBAL_PROFILE) -> None:
        unknown = profile.secret_rules - GLOBAL_SECRET_PATTERNS.keys()
        if unknown:
            raise GovernanceEngineError(
                f"profile names secret rules with no regex implementation: {sorted(unknown)}"
            )
        self.patterns: dict[str, re.Pattern[str]] = {
            name: re.compile(GLOBAL_SECRET_PATTERNS[name]) for name in sorted(profile.secret_rules)
        }

    def scan_text(self, text: str, path: str = "") -> list[SecretFinding]:
        findings: list[SecretFinding] = []
        for rule, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                group = next(
                    (g for g in ("secret", "secret2") if g in pattern.groupindex and m.group(g)),
                    0,
                )
                start, end = m.span(group)
                findings.append(SecretFinding(rule, text[start:end], start, end, path, self.engine))
        return sort_findings(findings)


def build_secret_scanner(profile: GovernanceProfile = GLOBAL_PROFILE) -> SecretScanner:
    return SecretScanner(profile)
