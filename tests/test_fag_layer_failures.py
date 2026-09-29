"""Hand-built FAG records that each fail exactly one of L1 or L2.

Each record starts from a valid seed and breaks one thing, then runs through the
L1 -> L2 cascade the pipeline builds. The cascade must stop at the expected layer with
the expected error codes, and nothing else may fire.
"""

import copy

import pytest

from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.spec.compile import compile_spec

from pathlib import Path

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def cascade(fag):
    return Cascade.from_config(
        ["L1", "L2"],
        {"L1": SchemaLayer.from_spec(fag), "L2": RulesLayer.from_spec(fag)},
    )


@pytest.fixture(scope="module")
def seeds(fag):
    return {s["conversation_id"]: s for s in fag.seeds}


def seed(seeds, n):
    return copy.deepcopy(seeds[f"SEED-FAG-{n:06d}"])


def run(cascade, record):
    return cascade.run(record, ValidationContext(cell_id="test"))


def test_every_seed_passes_l1_and_l2(cascade, seeds):
    for s in seeds.values():
        result = run(cascade, copy.deepcopy(s))
        assert result.passed, (s["conversation_id"], result.errors)
        assert result.layers_run == ("L1", "L2")


# --- L1 -------------------------------------------------------------------------------


def test_wrong_turn_order_stops_at_l1(cascade, seeds):
    # Assistant speaks first: turns stay numbered 1..4, but roles are swapped pairwise.
    r = seed(seeds, 1)
    for m in r["messages"]:
        m["role"] = "assistant" if m["role"] == "customer" else "customer"
    result = run(cascade, r)
    assert result.failed_layer == "L1"
    assert result.layers_run == ("L1",)
    assert result.repairable
    assert result.errors[0].code == "first_role"
    assert result.errors[0].path == "messages[0].role"
    assert {e.code for e in result.errors} == {"first_role", "role_alternation"}


def test_non_alternating_roles_stop_at_l1(cascade, seeds):
    r = seed(seeds, 1)
    r["messages"][2]["role"] = "assistant"
    result = run(cascade, r)
    assert result.failed_layer == "L1"
    assert result.layers_run == ("L1",)
    assert {e.code for e in result.errors} == {"role_alternation"}


def test_turns_not_numbered_from_one_stop_at_l1(cascade, seeds):
    r = seed(seeds, 4)
    for m in r["messages"]:
        m["turn"] += 1
    for s in r["spans"]:
        s["turn"] += 1
    result = run(cascade, r)
    assert result.failed_layer == "L1"
    assert result.layers_run == ("L1",)
    assert "turn_numbering" in {e.code for e in result.errors}


# --- L2 -------------------------------------------------------------------------------


def test_reworded_span_passes_l1_and_stops_at_l2(cascade, seeds):
    r = seed(seeds, 1)
    r["spans"][1]["text"] = "I'd recommend moving across to Business Flex before your next payment."
    result = run(cascade, r)
    assert result.failed_layer == "L2"
    assert result.layers_run == ("L1", "L2")
    assert result.verdicts[0].outcome == "pass"
    assert result.repairable
    assert [e.code for e in result.errors] == ["span_not_verbatim"]
    assert result.errors[0].path == "spans[1].text"


def test_label_disagreeing_with_tier_and_scope_stops_at_l2(cascade, seeds):
    # Seed 4 is a non-breach factual answer on a Corps Act product. Raising its tier to
    # personal advice makes the policy expect a breach, but the label still says no.
    r = seed(seeds, 4)
    assert r["product_scope"] == "corps_act" and r["label"] is False
    r["advice_tier"] = "PERSONAL_ADVICE"
    r.pop("policy_categories", None)
    result = run(cascade, r)
    assert result.failed_layer == "L2"
    assert result.layers_run == ("L1", "L2")
    assert [e.code for e in result.errors] == ["label_disagrees"]
    assert result.errors[0].details["expected"] is True
    assert result.errors[0].details["got"] is False


def test_general_advice_on_non_corps_labelled_breach_stops_at_l2(cascade, seeds):
    # The hard negative: Tier 2 on a non-Corps service is permitted, so a breach label
    # disagrees with the policy even though advisory wording is present.
    r = seed(seeds, 6)
    assert r["advice_tier"] == "GENERAL_ADVICE" and r["product_scope"] == "non_corps_act"
    r["label"] = True
    result = run(cascade, r)
    assert result.failed_layer == "L2"
    assert result.errors[0].code == "label_disagrees"


def test_missing_span_for_declared_signal_stops_at_l2(cascade, seeds):
    r = seed(seeds, 1)
    r["spans"] = [s for s in r["spans"] if s["category"] != "PRODUCT_RECOMMENDATION"]
    result = run(cascade, r)
    assert result.failed_layer == "L2"
    assert result.layers_run == ("L1", "L2")
    assert result.repairable
    assert [e.code for e in result.errors] == ["signal_without_span"]
    assert "PRODUCT_RECOMMENDATION" in result.errors[0].message
