"""The FAG policy at L2: each rule pinned through RulesLayer.from_spec(fag).check.

Every case starts from a seed in tasks/fag/seeds.jsonl (looked up by conversation_id)
and changes one thing, so the candidate is sent back (fail_repairable) with exactly the
expected codes. The expected codes and the label truth table are literals from the FAG
policy in tasks/fag/hooks.py's docstring:

    FACTUAL_INFORMATION           -> no breach
    GENERAL_ADVICE on corps_act   -> breach (Tier 2 prohibited on Corps products)
    GENERAL_ADVICE on non_corps   -> no breach (permitted, monitored)
    PERSONAL_ADVICE               -> breach, always
"""

import copy
from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CTX = ValidationContext()


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def layer(fag):
    return RulesLayer.from_spec(fag)


@pytest.fixture(scope="module")
def seeds(fag):
    return {s["conversation_id"]: s for s in fag.seeds}


def seed(seeds, n):
    return copy.deepcopy(seeds[f"SEED-FAG-{n:06d}"])


def codes(verdict):
    return list(verdict.codes)


# ── the label set by code ──────────────────────────────────────


@pytest.mark.parametrize(
    "tier, scope, breach",
    [
        ("FACTUAL_INFORMATION", "corps_act", False),
        ("FACTUAL_INFORMATION", "non_corps_act", False),
        ("GENERAL_ADVICE", "corps_act", True),
        ("GENERAL_ADVICE", "non_corps_act", False),
        ("PERSONAL_ADVICE", "corps_act", True),
        ("PERSONAL_ADVICE", "non_corps_act", True),
    ],
)
def test_label_rule_follows_the_tier_and_scope_truth_table(layer, seeds, tier, scope, breach):
    # Seed 6 is labelled "no breach"; with other fixed facts the label must follow the table.
    r = seed(seeds, 6)
    r["advice_tier"], r["product_scope"] = tier, scope
    verdict = layer.check(r, CTX)
    disagreements = [e.details for e in verdict.errors if e.code == "label_disagrees"]
    assert disagreements == ([{"expected": True, "got": False}] if breach else [])


def test_a_flipped_label_is_sent_back_as_label_disagrees(layer, seeds):
    r = seed(seeds, 3)  # PERSONAL_ADVICE on non_corps_act: always a breach
    r["label"] = False
    verdict = layer.check(r, CTX)
    assert verdict.repairable
    assert codes(verdict)[0] == "label_disagrees"
    assert verdict.errors[0].details == {"expected": True, "got": False}


# ── spans ──────────────────────────────────────────────────────


def test_every_seed_passes(layer, seeds):
    for s in seeds.values():
        assert layer.check(copy.deepcopy(s), CTX).passed, s["conversation_id"]


def test_candidate_without_policy_categories_passes(layer, seeds):
    # generated candidates reach L2 before post_process derives these
    r = seed(seeds, 1)
    del r["policy_categories"]
    assert layer.check(r, CTX).passed


def test_reworded_span_is_sent_back_as_span_not_verbatim(layer, seeds):
    r = seed(seeds, 1)
    r["spans"][0]["text"] = r["spans"][0]["text"].replace("fits", "suits")
    verdict = layer.check(r, CTX)
    assert verdict.repairable
    assert codes(verdict) == ["span_not_verbatim"]
    assert verdict.errors[0].path == "spans[0].text"


def test_span_on_a_customer_turn_is_span_not_assistant(layer, seeds):
    r = seed(seeds, 1)
    r["spans"][0] = {**r["spans"][0], "turn": 3, "text": r["messages"][2]["content"][:20]}
    assert codes(layer.check(r, CTX)) == ["span_not_assistant"]


def test_span_category_outside_the_signals_is_span_category_not_signal(layer, seeds):
    r = seed(seeds, 1)
    r["spans"].append({**r["spans"][1], "category": "SUBJECTIVE_DESCRIPTOR"})
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["span_category_not_signal"]
    assert verdict.errors[0].path == "spans[2].category"


def test_declared_signal_without_a_span_is_signal_without_span(layer, seeds):
    r = seed(seeds, 1)
    r["spans"] = r["spans"][:1]
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["signal_without_span"]
    assert verdict.errors[0].details == {"signal": "PRODUCT_RECOMMENDATION"}


def test_breach_without_any_span(layer, seeds):
    r = seed(seeds, 2)
    r["spans"] = []
    # its one signal is then also unevidenced
    assert codes(layer.check(r, CTX)) == ["breach_without_span", "signal_without_span"]


# ── the hard negative ──────────────────────────────────────────


def test_hard_negative_keeps_its_signals_with_no_spans_and_passes(layer, seeds):
    r = seed(seeds, 6)  # GENERAL_ADVICE on non_corps_act, advisory signals, "no breach"
    assert r["signal_categories"] == ["SUBJECTIVE_DESCRIPTOR", "PRODUCT_RECOMMENDATION"]
    assert r["spans"] == [] and r["label"] is False
    assert layer.check(r, CTX).passed


def test_hard_negative_with_a_span_is_spans_on_non_breach(layer, seeds):
    r = seed(seeds, 6)
    text = "the tap-to-pay app is often the most convenient option"
    r["spans"] = [{"turn": 2, "text": text, "category": "SUBJECTIVE_DESCRIPTOR"}]
    assert codes(layer.check(r, CTX)) == ["spans_on_non_breach"]


def test_non_breach_with_problematic_turns_is_spans_on_non_breach(layer, seeds):
    r = seed(seeds, 4)
    r["problematic_turns"] = [2]
    assert codes(layer.check(r, CTX)) == ["spans_on_non_breach"]


# ── severity and the other fixed facts ─────────────────────────


def test_non_breach_with_a_severity_is_severity_on_non_breach(layer, seeds):
    r = seed(seeds, 4)
    r["severity"] = "HIGH"
    assert codes(layer.check(r, CTX)) == ["severity_on_non_breach"]


def test_breach_without_a_severity_is_severity_missing(layer, seeds):
    r = seed(seeds, 1)
    r["severity"] = None
    assert codes(layer.check(r, CTX)) == ["severity_missing"]


def test_corps_question_off_corps_products_is_corps_question_scope(layer, seeds):
    r = seed(seeds, 6)  # non_corps_act
    r["is_corps_question"] = True
    assert codes(layer.check(r, CTX)) == ["corps_question_scope"]


def test_unknown_signal(layer, seeds):
    r = seed(seeds, 6)
    r["signal_categories"] = ["SUBJECTIVE_DESCRIPTOR", "VIBES"]
    assert codes(layer.check(r, CTX)) == ["unknown_signal"]


# ── policy categories ──────────────────────────────────────────


def test_policy_categories_that_disagree_with_the_signals_are_a_mismatch(layer, seeds):
    r = seed(seeds, 3)  # carries TAX_ADVICE, so tax_advice is true
    r["policy_categories"] = {**r["policy_categories"], "tax_advice": False}
    assert codes(layer.check(r, CTX)) == ["policy_categories_mismatch"]


def test_breach_no_policy_category_explains_is_unexplained_breach(layer, seeds):
    # personal advice off Corps products whose only signal is a general one:
    # no production policy category explains the breach
    r = seed(seeds, 3)
    del r["policy_categories"]
    r["signal_categories"] = ["SUBJECTIVE_DESCRIPTOR"]
    r["spans"] = [{**r["spans"][0], "category": "SUBJECTIVE_DESCRIPTOR"}]
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["unexplained_breach"]
    assert verdict.errors[0].path == "label"


# ── through the checks ─────────────────────────────────────────


def test_rule_failures_reach_the_checks_as_sent_back_at_l2(fag, seeds):
    r = seed(seeds, 1)
    r["spans"][0]["text"] = "I think Business Flex is fine."
    cascade = Cascade([SchemaLayer.from_spec(fag), RulesLayer.from_spec(fag)])
    result = cascade.run(r, CTX)
    assert result.failed_layer == "L2"
    assert result.repairable
    assert [e.code for e in result.errors] == ["span_not_verbatim"]
