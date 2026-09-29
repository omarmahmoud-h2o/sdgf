"""FAG extra_validators: one test per rule ported from scripts/utils.validate_conversation.

Each case mutates a valid seed so that exactly one rule fires, then checks the original
validator rejects the same record too (field names mapped back to scripts/).
"""

import copy
import sys
from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer

SDGF_DIR = Path(__file__).resolve().parents[1]
FAG_DIR = SDGF_DIR / "tasks" / "fag"
SCRIPTS_DIR = SDGF_DIR.parent / "scripts"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def validate(fag):
    return fag.hooks.extra_validators


@pytest.fixture(scope="module")
def seeds(fag):
    return {s["conversation_id"][-1]: s for s in fag.seeds}


@pytest.fixture(scope="module")
def orig_utils():
    # scripts/ is read-only here: import without writing bytecode into it.
    sys.path.insert(0, str(SCRIPTS_DIR))
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import utils
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.dont_write_bytecode = dont_write
    return utils


def seed(seeds, n):
    return copy.deepcopy(seeds[str(n)])


def codes(validate, record):
    return [i.code for i in validate(record)]


def orig_rejects(orig_utils, record):
    rec = {k: v for k, v in record.items() if k not in ("label", "spans")}
    rec["financial_advice_breach"] = record["label"]
    rec["problematic_spans"] = record["spans"]
    ok, _ = orig_utils.validate_conversation(rec)
    return not ok


def test_hook_is_present(fag):
    assert "extra_validators" in fag.hooks.present()


def test_every_seed_passes(validate, seeds):
    for s in seeds.values():
        assert validate(s) == [], s["conversation_id"]


def test_seed_without_policy_categories_passes(validate, seeds):
    # generated records reach L2 before post_process derives these
    r = seed(seeds, 1)
    del r["policy_categories"]
    assert validate(r) == []


def test_span_not_verbatim(validate, seeds, orig_utils):
    r = seed(seeds, 1)
    r["spans"][0]["text"] = r["spans"][0]["text"].replace("fits", "suits")
    issues = validate(r)
    assert [i.code for i in issues] == ["span_not_verbatim"]
    assert issues[0].path == "spans[0].text"
    assert orig_rejects(orig_utils, r)


def test_span_not_assistant(validate, seeds, orig_utils):
    r = seed(seeds, 1)
    customer = r["messages"][2]["content"]
    r["spans"][0] = {**r["spans"][0], "turn": 3, "text": customer[:20]}
    assert codes(validate, r) == ["span_not_assistant"]
    assert orig_rejects(orig_utils, r)


def test_span_category_not_in_signals(validate, seeds, orig_utils):
    r = seed(seeds, 1)
    r["spans"].append({**r["spans"][1], "category": "SUBJECTIVE_DESCRIPTOR"})
    issues = validate(r)
    assert [i.code for i in issues] == ["span_category_not_signal"]
    assert issues[0].path == "spans[2].category"
    assert orig_rejects(orig_utils, r)


def test_declared_signal_without_span(validate, seeds, orig_utils):
    r = seed(seeds, 1)
    r["spans"] = r["spans"][:1]
    issues = validate(r)
    assert [i.code for i in issues] == ["signal_without_span"]
    assert issues[0].details == {"signal": "PRODUCT_RECOMMENDATION"}
    assert orig_rejects(orig_utils, r)


def test_breach_without_any_span(validate, seeds, orig_utils):
    r = seed(seeds, 2)
    r["spans"] = []
    got = codes(validate, r)
    assert "breach_without_span" in got
    # its one signal is then also unevidenced
    assert set(got) == {"breach_without_span", "signal_without_span"}
    assert orig_rejects(orig_utils, r)


def test_non_breach_with_spans(validate, seeds, orig_utils):
    r = seed(seeds, 6)
    text = r["messages"][1]["content"][:30]
    r["spans"] = [{"turn": 2, "text": text, "category": "SUBJECTIVE_DESCRIPTOR"}]
    assert codes(validate, r) == ["spans_on_non_breach"]
    assert orig_rejects(orig_utils, r)


def test_non_breach_with_problematic_turns(validate, seeds, orig_utils):
    r = seed(seeds, 4)
    r["problematic_turns"] = [2]
    assert codes(validate, r) == ["spans_on_non_breach"]
    assert orig_rejects(orig_utils, r)


def test_hard_negative_signals_without_spans_pass(validate, seeds):
    r = seed(seeds, 6)
    assert r["signal_categories"] and not r["spans"] and r["label"] is False
    assert validate(r) == []


def test_breach_needs_severity(validate, seeds, orig_utils):
    r = seed(seeds, 1)
    r["severity"] = None
    assert codes(validate, r) == ["severity_missing"]
    assert orig_rejects(orig_utils, r)


def test_non_breach_must_not_have_severity(validate, seeds, orig_utils):
    r = seed(seeds, 4)
    r["severity"] = "HIGH"
    assert codes(validate, r) == ["severity_on_non_breach"]
    assert orig_rejects(orig_utils, r)


def test_corps_question_requires_corps_scope(validate, seeds, orig_utils):
    r = seed(seeds, 6)
    r["is_corps_question"] = True
    r["policy_categories"] = fag_derive(validate, r)
    assert codes(validate, r) == ["corps_question_scope"]
    assert orig_rejects(orig_utils, r)


def test_unknown_signal(validate, seeds, orig_utils):
    r = seed(seeds, 6)
    r["signal_categories"] = ["SUBJECTIVE_DESCRIPTOR", "VIBES"]
    assert codes(validate, r) == ["unknown_signal"]
    assert orig_rejects(orig_utils, r)


def test_policy_categories_mismatch(validate, seeds, orig_utils):
    r = seed(seeds, 3)
    r["policy_categories"] = {**r["policy_categories"], "tax_advice": False}
    assert codes(validate, r) == ["policy_categories_mismatch"]
    assert orig_rejects(orig_utils, r)


def test_unexplained_breach(validate, seeds, orig_utils):
    # personal advice off Corps products whose only signal is a general one:
    # no production policy category explains the breach
    r = seed(seeds, 3)
    del r["policy_categories"]
    r["signal_categories"] = ["SUBJECTIVE_DESCRIPTOR"]
    r["spans"] = [{**r["spans"][0], "category": "SUBJECTIVE_DESCRIPTOR"}]
    issues = validate(r)
    assert [i.code for i in issues] == ["unexplained_breach"]
    assert issues[0].path == "label"
    assert orig_rejects(orig_utils, r)


def test_issues_reach_the_cascade_as_repairable_l2(fag, seeds):
    r = seed(seeds, 1)
    r["spans"][0]["text"] = "I think Business Flex is fine."
    cascade = Cascade([SchemaLayer.from_spec(fag), RulesLayer.from_spec(fag)])
    result = cascade.run(r, ValidationContext())
    assert result.failed_layer == "L2"
    assert result.repairable
    assert [e.code for e in result.errors] == ["span_not_verbatim"]


def fag_derive(validate, record):
    return validate.__globals__["derive_policy_categories"](record)
