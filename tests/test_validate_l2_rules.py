"""L2 rules layer: keyword rules, label_rule agreement and extra_validators hooks."""

from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import KeywordRule, SpecValidationError, ValidationSection
from sdgf.validate.base import ValidationContext, ValidationIssue
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CTX = ValidationContext()


def record(**extra):
    base = {
        "messages": [
            {"turn": 1, "role": "customer", "content": "Which Acme Test account suits me?"},
            {"turn": 2, "role": "assistant", "content": "The fee is $0. It is ideal for you."},
        ],
        "label": False,
        "spans": [],
        "summary": "A fictional fee question.",
    }
    return {**base, **extra}


def rule(**kw):
    return KeywordRule(**{"name": "r", "kind": "forbidden", "keywords": ["ideal"], **kw})


def codes(verdict):
    return list(verdict.codes)


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


# ── keyword rules ──────────────────────────────────────────────


def test_no_rules_or_hooks_passes():
    verdict = RulesLayer().check(record(), CTX)
    assert verdict.passed and verdict.layer == "L2"


def test_forbidden_keyword_found_is_repairable():
    verdict = RulesLayer([rule()]).check(record(), CTX)
    assert verdict.repairable and codes(verdict) == ["keyword_forbidden"]
    issue = verdict.errors[0]
    assert issue.path == "messages[1].content"
    assert issue.details == {"rule": "r", "keyword": "ideal", "found": "ideal"}


def test_forbidden_keyword_case_insensitive_by_default():
    assert not RulesLayer([rule(keywords=["IDEAL"])]).check(record(), CTX).passed
    assert RulesLayer([rule(keywords=["IDEAL"], case_sensitive=True)]).check(record(), CTX).passed


def test_word_match_respects_word_boundaries():
    assert RulesLayer([rule(keywords=["idea"])]).check(record(), CTX).passed
    assert not RulesLayer([rule(keywords=["idea"], match="substring")]).check(record(), CTX).passed


def test_regex_match():
    verdict = RulesLayer([rule(keywords=[r"\$\d+"], match="regex")]).check(record(), CTX)
    assert verdict.errors[0].details["found"] == "$0"


def test_roles_restrict_scanned_messages():
    r = rule(keywords=["suits"])
    assert not RulesLayer([r]).check(record(), CTX).passed
    assert RulesLayer([rule(keywords=["suits"], roles=["assistant"])]).check(record(), CTX).passed


def test_fields_are_scanned():
    r = rule(keywords=["fictional"], roles=[], fields=["summary"])
    verdict = RulesLayer([r]).check(record(), CTX)
    assert codes(verdict) == ["keyword_forbidden"] and verdict.errors[0].path == "summary"


def test_every_forbidden_hit_reported():
    r = rule(keywords=["ideal", "fee"])
    assert codes(RulesLayer([r]).check(record(), CTX)) == ["keyword_forbidden"] * 2


def test_required_any():
    ok = KeywordRule(name="q", kind="required", keywords=["nope", "fee"])
    bad = KeywordRule(name="q", kind="required", keywords=["nope", "never"])
    assert RulesLayer([ok]).check(record(), CTX).passed
    verdict = RulesLayer([bad]).check(record(), CTX)
    assert codes(verdict) == ["keyword_required"]
    assert verdict.errors[0].details["missing"] == ["nope", "never"]


def test_required_all_reports_each_missing_keyword():
    r = KeywordRule(name="q", kind="required", keywords=["fee", "nope", "never"], require="all")
    verdict = RulesLayer([r]).check(record(), CTX)
    assert [e.details["missing"] for e in verdict.errors] == [["nope"], ["never"]]


def test_when_restricts_rule_to_matching_records():
    r = rule(when={"label": False})
    assert not RulesLayer([r]).check(record(), CTX).passed
    assert RulesLayer([r]).check(record(label=True), CTX).passed
    r_in = rule(when={"tier": ["A", "B"]})
    assert not RulesLayer([r_in]).check(record(tier="B"), CTX).passed
    assert RulesLayer([r_in]).check(record(tier="C"), CTX).passed


def test_rule_spec_validation():
    with pytest.raises(ValueError, match="valid regex"):
        KeywordRule(name="x", kind="forbidden", keywords=["("], match="regex")
    with pytest.raises(ValueError, match="required rules only"):
        KeywordRule(name="x", kind="forbidden", keywords=["a"], require="all")
    with pytest.raises(ValueError):
        KeywordRule(name="x", kind="forbidden", keywords=[])
    with pytest.raises(ValueError, match="duplicate rule names"):
        ValidationSection(rules=[rule(), rule()])


def test_rules_parse_from_spec_section():
    from sdgf.spec.schema import parse_spec

    with pytest.raises(SpecValidationError, match="validation.rules"):
        parse_spec(
            {
                "validation": {"rules": [{"name": "x", "kind": "maybe", "keywords": ["a"]}]},
            }
        )


# ── label_rule agreement ───────────────────────────────────────


def label_rule(record):
    return record["tier"] == "personal"


def test_label_agrees():
    layer = RulesLayer(label_rule=label_rule)
    assert layer.check(record(tier="general", label=False), CTX).passed
    assert layer.check(record(tier="personal", label=True), CTX).passed


def test_label_disagrees():
    verdict = RulesLayer(label_rule=label_rule).check(record(tier="personal"), CTX)
    assert verdict.repairable and codes(verdict) == ["label_disagrees"]
    assert verdict.errors[0].details == {"expected": True, "got": False}
    assert verdict.errors[0].path == "label"


def test_label_type_mismatch_disagrees():
    verdict = RulesLayer(label_rule=label_rule).check(record(tier="personal", label=1), CTX)
    assert codes(verdict) == ["label_disagrees"]


def test_label_rule_error_on_missing_fact():
    verdict = RulesLayer(label_rule=label_rule).check(record(), CTX)
    assert verdict.repairable and codes(verdict) == ["label_rule_error"]
    assert verdict.errors[0].details["exception"] == "KeyError"


# ── extra_validators ───────────────────────────────────────────


def test_extra_validator_strings_become_issues():
    layer = RulesLayer(extra_validators=lambda r: ["span not found verbatim in turn 2"])
    verdict = layer.check(record(), CTX)
    assert verdict.repairable and codes(verdict) == ["extra_validator"]
    assert "span not found verbatim" in verdict.messages()[0]


def test_extra_validator_may_return_issues():
    issue = ValidationIssue("span_not_verbatim", "reworded", "spans[0].text")
    verdict = RulesLayer(extra_validators=lambda r: [issue]).check(record(), CTX)
    assert verdict.errors == (issue,)


def test_extra_validator_empty_passes():
    assert RulesLayer(extra_validators=lambda r: []).check(record(), CTX).passed


def test_extra_validator_exception_is_record_error():
    verdict = RulesLayer(extra_validators=lambda r: r["missing"]).check(record(), CTX)
    assert codes(verdict) == ["extra_validator_error"]


def test_extra_validator_bad_return_type_is_a_bug():
    with pytest.raises(TypeError, match="expected str or ValidationIssue"):
        RulesLayer(extra_validators=lambda r: [42]).check(record(), CTX)


def test_all_checks_reported_together():
    layer = RulesLayer([rule()], label_rule=label_rule, extra_validators=lambda r: ["bad span"])
    verdict = layer.check(record(tier="personal"), CTX)
    assert codes(verdict) == ["keyword_forbidden", "label_disagrees", "extra_validator"]


# ── FAG ────────────────────────────────────────────────────────


def test_from_spec_uses_fag_label_rule(fag):
    layer = RulesLayer.from_spec(fag)
    assert layer.label_rule is fag.hooks.label_rule
    assert layer.rules == ()


def test_fag_seeds_pass_l2(fag):
    layer = RulesLayer.from_spec(fag)
    for seed in fag.seeds:
        assert layer.check(dict(seed), CTX).passed, seed.get("conversation_id")


def test_fag_label_disagreeing_with_tier_and_scope_fails_at_l2(fag):
    seed = next(s for s in fag.seeds if s["advice_tier"] == "PERSONAL_ADVICE")
    bad = {**seed, "label": False}
    cascade = Cascade([SchemaLayer.from_spec(fag), RulesLayer.from_spec(fag)])
    result = cascade.run(bad, CTX)
    assert result.failed_layer == "L2" and result.repairable
    assert [e.code for e in result.errors] == ["label_disagrees"]
