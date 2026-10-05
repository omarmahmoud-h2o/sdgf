"""The groundness policy at L2, through RulesLayer.from_spec(groundness).check.

Each case starts from a seed in tasks/groundness/seeds.jsonl, picked by its defect_type,
and changes one thing. Expected codes are literals from the disposition table in the
task's hooks.py docstring and the category minimums in its task.yaml:

    essential_fact       minimum "fully supported"
    interpretive         minimum "partially supported"
    caveats_limitations  minimum "no grounding required" (always passes)

A message is blocked (label true) if any claim in its final assistant turn is below its
category's minimum.
"""

import copy
from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.validate.base import ValidationContext
from sdgf.validate.l2_rules import RulesLayer

TASK = Path(__file__).resolve().parents[1] / "tasks" / "groundness" / "task.yaml"
CTX = ValidationContext()

pytestmark = pytest.mark.skipif(not TASK.exists(), reason="tasks/groundness not present")


@pytest.fixture(scope="module")
def groundness():
    return compile_spec(TASK)


@pytest.fixture(scope="module")
def layer(groundness):
    return RulesLayer.from_spec(groundness)


@pytest.fixture(scope="module")
def seeds(groundness):
    return {s["defect_type"]: s for s in groundness.seeds}


def seed(seeds, defect_type):
    """invented_feature: final turn 4, spans [fact fully, TARGET fact unsupported, caveat fully].
    fabricated_figure: final turn 6, three fully supported facts then the unsupported target."""
    return copy.deepcopy(seeds[defect_type])


def codes(verdict):
    return list(verdict.codes)


def test_every_seed_passes(layer, seeds):
    for name, s in seeds.items():
        assert layer.check(copy.deepcopy(s), CTX).passed, name


def test_a_flipped_label_is_sent_back_as_label_disagrees(layer, seeds):
    r = seed(seeds, "invented_feature")
    r["label"] = False
    verdict = layer.check(r, CTX)
    assert verdict.repairable
    assert codes(verdict) == ["label_disagrees"]
    assert verdict.errors[0].details == {"expected": True, "got": False}


def test_a_message_whose_claims_all_meet_their_minimum_is_not_blocked(layer, seeds):
    r = seed(seeds, "invented_feature")
    r["spans"][1]["support_level"] = r["target_support_level"] = "fully supported"
    assert layer.check({**r, "label": False}, CTX).passed
    verdict = layer.check({**r, "label": True}, CTX)
    assert codes(verdict) == ["label_disagrees"]
    assert verdict.errors[0].details == {"expected": False, "got": True}


def test_an_interpretive_claim_passes_when_partially_supported(layer, seeds):
    r = seed(seeds, "invented_feature")
    r["spans"][1]["category"] = r["target_category"] = "interpretive"
    r["spans"][1]["support_level"] = r["target_support_level"] = "partially supported"
    r["label"] = False
    assert layer.check(r, CTX).passed


def test_an_unsupported_caveat_is_a_hard_negative_that_passes(layer, seeds):
    r = seed(seeds, "invented_feature")
    assert r["spans"][2]["category"] == "caveats_limitations"
    r["spans"][2]["support_level"] = "unsupported"
    assert layer.check(r, CTX).passed


def test_a_caveat_cannot_contradict_the_evidence(layer, seeds):
    r = seed(seeds, "invented_feature")
    r["spans"][2]["support_level"] = "contradict evidence"
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["invalid_support_level"]
    assert verdict.errors[0].path == "spans[2].support_level"


def test_a_blocking_claim_that_is_not_the_target_is_non_target_blocks(layer, seeds):
    r = seed(seeds, "fabricated_figure")
    r["spans"][0]["support_level"] = "unsupported"
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["non_target_blocks"]
    assert verdict.errors[0].path == "spans[0].support_level"


def test_two_targets_is_target_count(layer, seeds):
    r = seed(seeds, "invented_feature")
    r["spans"][0]["is_target"] = True
    assert codes(layer.check(r, CTX)) == ["target_count"]


def test_a_reworded_claim_is_span_not_verbatim(layer, seeds):
    r = seed(seeds, "fabricated_figure")
    r["spans"][0]["text"] = r["spans"][0]["text"].replace("linked to", "attached to")
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["span_not_verbatim"]
    assert verdict.errors[0].path == "spans[0].text"


def test_a_claim_on_an_earlier_turn_is_claim_not_final_turn(layer, seeds):
    r = seed(seeds, "invented_feature")
    r["spans"][0]["turn"] = 2
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["claim_not_final_turn"]
    assert verdict.errors[0].path == "spans[0].turn"


def test_a_grounding_required_target_without_tool_calls_is_no_sources(layer, seeds):
    r = seed(seeds, "invented_feature")
    for msg in r["messages"]:
        msg.pop("tool_calls", None)
    assert codes(layer.check(r, CTX)) == ["no_sources"]


def test_naming_the_architecture_in_the_assistant_text_is_forbidden(layer, seeds):
    r = seed(seeds, "invented_feature")
    r["messages"][-1]["content"] += " I checked this with a tool call."
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["keyword_forbidden"]
    assert verdict.errors[0].path == "messages[3].content"
    assert verdict.errors[0].details == {
        "rule": "no_architecture_leak",
        "keyword": "tool call",
        "found": "tool call",
    }


def test_compliance_tags_are_forbidden_in_the_assistant_text_but_not_in_tool_outputs(layer, seeds):
    r = seed(seeds, "invented_feature")
    # the seed's loan_repayments_flow output already carries the tags, and passes
    assert "<compliance_language>" in r["messages"][-1]["tool_calls"][0]["outputs"]
    r["messages"][-1]["content"] += "\n<compliance_language>"
    verdict = layer.check(r, CTX)
    assert codes(verdict) == ["keyword_forbidden"]
    assert verdict.errors[0].details["rule"] == "no_compliance_tags"
