"""PII scanner tests. Every identifier here is fictional (zeros or obviously made-up values)."""

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from sdgf.governance._util import GovernanceEngineError
from sdgf.governance.pii import (
    GLOBAL_PII_PATTERNS,
    CompositePIIScanner,
    PIIFinding,
    PresidioPIIScanner,
    RegexPIIScanner,
    build_pii_scanner,
)
from sdgf.governance.profile import GLOBAL_PII_RULES, GovernanceProfile, merge_profile
from sdgf.spec.schema import GovernanceSection

FAG_SEEDS = Path(__file__).resolve().parents[1] / "tasks" / "fag" / "seeds.jsonl"

SCANNER = RegexPIIScanner()


def rules(text: str) -> list[tuple[str, str]]:
    return [(f.rule, f.text) for f in SCANNER.scan_text(text)]


def test_every_global_rule_has_a_pattern():
    assert set(GLOBAL_PII_RULES) == set(GLOBAL_PII_PATTERNS)


@pytest.mark.parametrize(
    "text, rule, hit",
    [
        ("Email me at test.user@example.com please", "email", "test.user@example.com"),
        ("contact: a+b@mail.example.test", "email", "a+b@mail.example.test"),
        ("Call 0400 000 000 today", "phone", "0400 000 000"),
        ("Call 0400000000 today", "phone", "0400000000"),
        ("Ring +61 400 000 000", "phone", "+61 400 000 000"),
        ("Office (02) 0000 0000", "phone", "(02) 0000 0000"),
        ("Office 03 0000 0000", "phone", "03 0000 0000"),
        ("Ring +61 2 0000 0000", "phone", "+61 2 0000 0000"),
        ("Hotline 1300 000 000", "phone", "1300 000 000"),
        ("Hotline 1800-000-000", "phone", "1800-000-000"),
        ("Short line 13 00 00", "phone", "13 00 00"),
        ("Our ABN is 00 000 000 000.", "abn", "00 000 000 000"),
        ("ABN 00000000000", "abn", "00000000000"),
        ("My TFN is 000 000 000", "tfn", "000 000 000"),
        ("TFN: 000-000-000", "tfn", "000-000-000"),
        ("TFN 000000000", "tfn", "000000000"),
        ("BSB 000-000 for the branch", "bsb", "000-000"),
        ("BSB: 000000", "bsb", "000000"),
        ("bsb no. 000 000", "bsb", "000 000"),
        ("Pay into 000-000 00000000 now", "account_number", "000-000 00000000"),
        ("account number 00000000", "account_number", "00000000"),
        ("acct #: 000 000 00", "account_number", "000 000 00"),
    ],
)
def test_each_rule_detects_fictional_values(text, rule, hit):
    findings = SCANNER.scan_text(text)
    assert [(f.rule, f.text) for f in findings] == [(rule, hit)]
    f = findings[0]
    assert text[f.start : f.end] == hit
    assert f.engine == "regex" and f.score == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "The monthly fee is $10 and the limit is $50,000.",
        "Transfers settle in 1 to 2 business days.",
        "Rates start at 5.25% p.a. for balances over $250,000.",
        "The year 2026 and the ratio 3:1.",
        "Reference a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6 is a hash, 1234567890abcdef too.",
        "hash 0123456789abc and id x123456789",
        "Card ending 0000 and an order of 12 345 units.",
        "No email here @example or user@ or user@localhost.",
    ],
)
def test_non_pii_text_is_clean(text):
    assert SCANNER.scan_text(text) == []


def test_digit_runs_match_whole():
    # A TFN shape inside an ABN, or a BSB at the start of a TFN, is not a separate finding.
    assert rules("ABN 00 000 000 000") == [("abn", "00 000 000 000")]
    assert rules("TFN 000 000 000") == [("tfn", "000 000 000")]
    assert rules("BSB and account 000-000 00000000") == [("account_number", "000-000 00000000")]
    # Too long for any rule.
    assert rules("id 0000000000000000") == []


def test_multiple_findings_are_sorted_with_offsets():
    text = "TFN 000 000 000, email x@example.com, mobile 0400 000 000"
    findings = SCANNER.scan_text(text, path="messages[1].content")
    assert [f.rule for f in findings] == ["tfn", "email", "phone"]
    assert all(text[f.start : f.end] == f.text for f in findings)
    assert {f.path for f in findings} == {"messages[1].content"}


def test_scan_record_walks_nested_strings_with_paths():
    record = {
        "messages": [
            {"turn": 1, "role": "customer", "content": "My TFN is 000 000 000."},
            {"turn": 2, "role": "assistant", "content": "Please don't share that."},
        ],
        "notes": ["reach me on x@example.com"],
        "_provenance": {"contact": "y@example.com"},
    }
    findings = SCANNER.scan_record(record)
    assert [(f.path, f.rule) for f in findings] == [
        ("messages[0].content", "tfn"),
        ("notes[0]", "email"),
    ]
    with_private = SCANNER.scan_record(record, skip_private=False)
    assert ("_provenance.contact", "email") in [(f.path, f.rule) for f in with_private]


def test_finding_to_dict():
    f = PIIFinding("tfn", "000 000 000", 4, 15, "messages[0].content")
    assert f.to_dict() == {
        "rule": "tfn",
        "text": "000 000 000",
        "start": 4,
        "end": 15,
        "path": "messages[0].content",
        "engine": "regex",
        "score": 1.0,
    }


# ── profile-driven rules ─────────────────────────────────────────


def test_per_task_patterns_from_profile():
    profile = merge_profile(
        GovernanceSection(
            extra_pii_patterns={
                "customer_id": r"\bCUST-\d{6}\b",
                "member_no": r"member (?:no\.? )?(?P<pii>M\d{5})",
            }
        )
    )
    scanner = RegexPIIScanner(profile)
    text = "Customer CUST-000000, member no. M00000, TFN 000 000 000"
    assert [(f.rule, f.text) for f in scanner.scan_text(text)] == [
        ("customer_id", "CUST-000000"),
        ("member_no", "M00000"),
        ("tfn", "000 000 000"),
    ]


def test_profile_rule_subset_and_unknown_rule():
    only_email = GovernanceProfile(pii_rules=frozenset({"email"}))
    assert [f.rule for f in RegexPIIScanner(only_email).scan_text("x@example.com 000 000 000")] == [
        "email"
    ]
    with pytest.raises(GovernanceEngineError, match="passport"):
        RegexPIIScanner(GovernanceProfile(pii_rules=frozenset({"email", "passport"})))


def test_fag_seeds_are_clean():
    for line in FAG_SEEDS.read_text().splitlines():
        if line.strip():
            assert SCANNER.scan_record(json.loads(line)) == []


# ── Presidio adapter (fake engine; Presidio itself is never needed) ──


@dataclass
class FakeResult:
    entity_type: str
    start: int
    end: int
    score: float


class FakeAnalyzer:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def analyze(self, text, language, entities=None):
        self.calls.append((text, language, entities))
        return self.results


def test_presidio_adapter_maps_entities_and_threshold():
    text = "Jane Citizen, x@example.com"
    analyzer = FakeAnalyzer(
        [
            FakeResult("EMAIL_ADDRESS", 14, 27, 0.99),
            FakeResult("PERSON", 0, 12, 0.85),
            FakeResult("NRP", 0, 4, 0.2),
        ]
    )
    scanner = PresidioPIIScanner(analyzer, entities=["EMAIL_ADDRESS", "PERSON"])
    findings = scanner.scan_text(text, "messages[0].content")
    assert [(f.rule, f.text, f.engine, f.score) for f in findings] == [
        ("person", "Jane Citizen", "presidio", 0.85),
        ("email", "x@example.com", "presidio", 0.99),
    ]
    assert analyzer.calls == [(text, "en", ["EMAIL_ADDRESS", "PERSON"])]


def test_presidio_imported_lazily(monkeypatch):
    built = []

    class AnalyzerEngine(FakeAnalyzer):
        def __init__(self):
            super().__init__([FakeResult("AU_TFN", 0, 11, 0.9)])
            built.append(self)

    monkeypatch.setitem(
        sys.modules, "presidio_analyzer", types.SimpleNamespace(AnalyzerEngine=AnalyzerEngine)
    )
    scanner = build_pii_scanner(engines=["presidio"])
    assert isinstance(scanner, CompositePIIScanner) and len(built) == 1
    findings = scanner.scan_text("000 000 000")
    assert sorted((f.engine, f.rule) for f in findings) == [("presidio", "tfn"), ("regex", "tfn")]


def test_presidio_missing_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    with pytest.raises(GovernanceEngineError, match=r"pip install sdgf\[pii\]"):
        PresidioPIIScanner()


def test_build_pii_scanner_defaults_and_unknown_engine():
    assert isinstance(build_pii_scanner(), RegexPIIScanner)
    with pytest.raises(GovernanceEngineError, match="spacy"):
        build_pii_scanner(engines=["spacy"])
