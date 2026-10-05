"""Toxicity, secrets and entity scanner tests. Every secret here is a fictional placeholder
(zeros, EXAMPLE, repeated characters) and every entity name is made up."""

import json
import sys
import types
from pathlib import Path

import pytest

from sdgf.governance._util import GovernanceEngineError
from sdgf.governance.entities import EntityFinding, EntityScanner, build_entity_scanner
from sdgf.governance.profile import (
    GLOBAL_SECRET_RULES,
    GLOBAL_TOXICITY_CATEGORIES,
    GovernanceProfile,
    merge_profile,
)
from sdgf.governance.secrets import (
    SecretFinding,
    SecretScanner,
    build_secret_scanner,
)
from sdgf.governance.toxicity import (
    CompositeToxicityScanner,
    DetoxifyToxicityScanner,
    KeywordToxicityScanner,
    ToxicityFinding,
    build_toxicity_scanner,
)
from sdgf.spec.schema import GovernanceException, GovernanceSection

FAG_SEEDS = Path(__file__).resolve().parents[1] / "tasks" / "fag" / "seeds.jsonl"


def seeds() -> list[dict]:
    return [json.loads(line) for line in FAG_SEEDS.read_text().splitlines() if line.strip()]


# ── toxicity ─────────────────────────────────────────────────────

TOX = KeywordToxicityScanner()


def categories(text: str, scanner=TOX) -> list[tuple[str, str]]:
    return [(f.rule, f.text) for f in scanner.scan_text(text)]


@pytest.mark.parametrize(
    "text, category, hit",
    [
        ("This fee is complete bullshit", "profanity", "bullshit"),
        ("What the FUCKING hell", "profanity", "FUCKING"),
        ("Just piss off", "profanity", "piss off"),
        ("You idiot, fix my account", "insult", "idiot"),
        ("You're useless at this", "insult", "You're useless"),
        ("I'll find you and sort this out", "threat", "I'll find you"),
        ("I know where you live", "threat", "I know where you live"),
        ("Go back to where you came from", "identity_attack", "Go back to where you came from"),
        ("They are subhuman", "identity_attack", "subhuman"),
        ("Send me nudes", "sexual", "nudes"),
        ("some porn site", "sexual", "porn"),
        ("I want to kill myself", "self_harm", "kill myself"),
        ("I've been feeling suicidal", "self_harm", "suicidal"),
    ],
)
def test_keyword_categories(text, category, hit):
    assert (category, hit) in categories(text)


@pytest.mark.parametrize(
    "text",
    [
        "Can you tell me the monthly fee on the business transaction account?",
        "The policy has a suicide exclusion in the first 13 months.",  # insurance wording
        "We run a pest control business dealing with vermin.",
        "Our shiitake mushroom farm needs merchant services.",
        "My business is in Scunthorpe Road.",
        "I'd kill for a lower interest rate, honestly.",
        "That's a damn good question about overdrafts.",
    ],
)
def test_ordinary_text_is_clean(text):
    assert categories(text) == []


def test_finding_offsets_and_path():
    text = "Well, you idiot."
    [f] = TOX.scan_text(text, "messages[0].content")
    assert isinstance(f, ToxicityFinding)
    assert text[f.start : f.end] == f.text == "idiot"
    assert f.path == "messages[0].content"
    assert f.to_dict()["scanner"] == "toxicity"
    assert f.engine == "keywords"


def test_scan_record_paths():
    record = {
        "messages": [{"turn": 1, "role": "customer", "content": "This is bullshit"}],
        "_provenance": {"note": "bullshit"},
    }
    assert [(f.path, f.rule) for f in TOX.scan_record(record)] == [
        ("messages[0].content", "profanity")
    ]


def test_documented_exception_switches_category_off():
    profile = merge_profile(
        GovernanceSection(
            toxicity_exceptions=["profanity"],
            exceptions=[GovernanceException(rule="toxicity:profanity", reason="detector task")],
        )
    )
    scanner = KeywordToxicityScanner(profile)
    assert categories("This is bullshit, you idiot", scanner) == [("insult", "idiot")]


def test_extra_keywords():
    scanner = KeywordToxicityScanner(extra_keywords={"insult": [r"numpty|numpties"]})
    assert categories("You numpty", scanner) == [("insult", "numpty")]
    with pytest.raises(GovernanceEngineError, match="unknown categories"):
        KeywordToxicityScanner(extra_keywords={"rudeness": ["x"]})
    with pytest.raises(GovernanceEngineError, match="insult"):
        KeywordToxicityScanner(extra_keywords={"insult": ["(unclosed"]})


def test_profile_category_without_keywords_raises():
    profile = GovernanceProfile(toxicity_categories=frozenset({*GLOBAL_TOXICITY_CATEGORIES, "x"}))
    with pytest.raises(GovernanceEngineError, match="no keyword list"):
        KeywordToxicityScanner(profile)


def test_fag_seeds_have_no_toxicity():
    for seed in seeds():
        assert TOX.scan_record(seed) == []


class FakeDetoxify:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, text: str) -> dict[str, float]:
        return self.scores


def test_detoxify_maps_heads_and_thresholds():
    model = FakeDetoxify(
        {
            "toxicity": 0.99,
            "severe_toxicity": 0.9,
            "obscene": 0.8,
            "insult": 0.2,
            "threat": 0.6,
            "identity_attack": 0.1,
            "sexual_explicit": 0.05,
        }
    )
    scanner = DetoxifyToxicityScanner(model, threshold=0.5)
    found = scanner.scan_text("some text", "x")
    assert [(f.rule, f.score) for f in found] == [("profanity", 0.8), ("threat", 0.6)]
    assert all(f.start == 0 and f.end == len("some text") and f.engine == "detoxify" for f in found)
    assert scanner.scan_text("   ") == []


def test_detoxify_respects_exceptions():
    profile = merge_profile(
        GovernanceSection(
            toxicity_exceptions=["threat"],
            exceptions=[GovernanceException(rule="toxicity:threat", reason="detector task")],
        )
    )
    scanner = DetoxifyToxicityScanner(FakeDetoxify({"threat": 0.9}), profile=profile)
    assert scanner.scan_text("text") == []


def test_detoxify_lazy_import(monkeypatch):
    built: list[str] = []

    class Model(FakeDetoxify):
        def __init__(self, variant: str) -> None:
            built.append(variant)
            super().__init__({"insult": 0.9})

    monkeypatch.setitem(sys.modules, "detoxify", types.SimpleNamespace(Detoxify=Model))
    scanner = build_toxicity_scanner(engines=["detoxify"])
    assert isinstance(scanner, CompositeToxicityScanner)
    assert built == ["original"]
    engines = {(f.rule, f.engine) for f in scanner.scan_text("you idiot")}
    assert engines == {("insult", "keywords"), ("insult", "detoxify")}


def test_detoxify_missing_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "detoxify", None)
    with pytest.raises(GovernanceEngineError, match=r"sdgf\[toxicity\]"):
        DetoxifyToxicityScanner()


def test_build_toxicity_scanner_defaults_and_unknown():
    assert isinstance(build_toxicity_scanner(), KeywordToxicityScanner)
    with pytest.raises(GovernanceEngineError, match="unknown toxicity engines"):
        build_toxicity_scanner(engines=["perspective"])


# ── secrets ──────────────────────────────────────────────────────

SEC = SecretScanner()


def secrets_in(text: str) -> list[tuple[str, str]]:
    return [(f.rule, f.text) for f in SEC.scan_text(text)]


FAKE_JWT = "eyJ" + "A" * 20 + ".eyJ" + "B" * 20 + "." + "C" * 20

PEM = "-----BEGIN RSA PRIVATE KEY-----\nMIIEXAMPLE000000\n-----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize(
    "text, rule, hit",
    [
        ("key sk-" + "0" * 32 + " ok", "api_key", "sk-" + "0" * 32),
        ("sk-ant-" + "x" * 30, "api_key", "sk-ant-" + "x" * 30),
        ("token ghp_" + "0" * 36, "api_key", "ghp_" + "0" * 36),
        ("xoxb-0000000000-EXAMPLE", "api_key", "xoxb-0000000000-EXAMPLE"),
        ("AIza" + "0" * 35, "api_key", "AIza" + "0" * 35),
        ("sk_test_" + "0" * 24, "api_key", "sk_test_" + "0" * 24),
        ("API_KEY = 'EXAMPLE0000000000abcd'", "api_key", "EXAMPLE0000000000abcd"),
        ("client secret: EXAMPLEexampleEXAMPLE", "api_key", "EXAMPLEexampleEXAMPLE"),
        ("AKIA0000000000000000 in env", "aws_access_key", "AKIA0000000000000000"),
        (PEM, "private_key", PEM),
        ("-----BEGIN PRIVATE KEY-----", "private_key", "-----BEGIN PRIVATE KEY-----"),
        ("Authorization: Bearer " + "a" * 32, "bearer_token", "a" * 32),
        ("jwt " + FAKE_JWT, "jwt", FAKE_JWT),
        ("password: Examp1e!", "password_assignment", "Examp1e!"),
        ('pwd="000000"', "password_assignment", "000000"),
    ],
)
def test_secret_rules(text, rule, hit):
    assert (rule, hit) in secrets_in(text)


@pytest.mark.parametrize(
    "text",
    [
        "Reset your password: it takes a minute.",
        "Your password = something memorable",
        "We never ask for your API key over chat.",
        "Ask for the bearer of the cheque.",
        "The bank's sk-ated? No.",
        "Reference AKIA000 is too short",
        "The business account fee is $10 a month.",
    ],
)
def test_ordinary_text_has_no_secrets(text):
    assert secrets_in(text) == []


def test_secret_finding_shape():
    [f] = SEC.scan_text("password: Examp1e!", "messages[3].content")
    assert isinstance(f, SecretFinding)
    assert f.to_dict()["scanner"] == "secrets"
    assert (f.start, f.end, f.path) == (10, 18, "messages[3].content")


def test_secret_rule_subset_and_unknown():
    scanner = SecretScanner(GovernanceProfile(secret_rules=frozenset({"jwt"})))
    assert [f.rule for f in scanner.scan_text(f"{FAKE_JWT} AKIA0000000000000000")] == ["jwt"]
    with pytest.raises(GovernanceEngineError, match="no regex implementation"):
        SecretScanner(GovernanceProfile(secret_rules=frozenset({*GLOBAL_SECRET_RULES, "pgp"})))
    assert isinstance(build_secret_scanner(), SecretScanner)


def test_fag_seeds_have_no_secrets():
    for seed in seeds():
        assert SEC.scan_record(seed) == []


# ── entities ─────────────────────────────────────────────────────


def entity_profile(deny=(), allow=()) -> GovernanceProfile:
    return GovernanceProfile(entity_deny=frozenset(deny), entity_allow=frozenset(allow))


def test_empty_deny_list_finds_nothing():
    assert EntityScanner().scan_text("Globex Corporation and Example Bank") == []
    assert isinstance(build_entity_scanner(), EntityScanner)


def test_deny_matches_case_and_whitespace_insensitively():
    scanner = EntityScanner(entity_profile(deny=["Example Bank", "Globex"]))
    text = "I moved from EXAMPLE   bank to Globex."
    found = scanner.scan_text(text, "messages[0].content")
    assert [(f.rule, f.text) for f in found] == [
        ("Example Bank", "EXAMPLE   bank"),
        ("Globex", "Globex"),
    ]
    assert all(isinstance(f, EntityFinding) and text[f.start : f.end] == f.text for f in found)
    assert found[0].to_dict()["scanner"] == "entities"


def test_deny_respects_word_boundaries():
    scanner = EntityScanner(entity_profile(deny=["Globex"]))
    assert scanner.scan_text("Globexian Holdings and globex_test") == []


def test_allow_list_shields_longer_fictional_name():
    scanner = EntityScanner(entity_profile(deny=["Acme"], allow=["Acme Test Pty Ltd"]))
    text = "Acme Test Pty Ltd is not the same as Acme."
    assert [(f.text, f.start) for f in scanner.scan_text(text)] == [("Acme", text.rindex("Acme"))]


def test_merged_profile_drives_entity_scanner():
    base = entity_profile(deny=["Globex"])
    released = merge_profile(
        GovernanceSection(
            entity_allow=["Globex"],
            entity_deny=["Initech"],
            exceptions=[GovernanceException(rule="entities:Globex", reason="fictional here")],
        ),
        base,
    )
    scanner = EntityScanner(released)
    assert [f.rule for f in scanner.scan_text("Globex and Initech")] == ["Initech"]


def test_empty_entity_name_raises():
    with pytest.raises(GovernanceEngineError, match="empty"):
        EntityScanner(entity_profile(deny=["  "]))


def test_fag_seeds_clean_against_denied_names():
    scanner = EntityScanner(entity_profile(deny=["Example Bank", "Globex"]))
    for seed in seeds():
        assert scanner.scan_record(seed) == []
