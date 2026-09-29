"""Hand-built FAG records that each fail exactly one of L1, L2, L3, L4 or L5.

Each record starts from a valid seed and breaks one thing, then runs through the
L1 -> L2 -> L3 -> L4 cascade the pipeline builds. The cascade must stop at the expected
layer with the expected error codes, and nothing else may fire. L1 and L2 failures are
repairable; L3 and L4 failures are hard drops.

A seed is itself a copy of a seed, so unmodified seeds pass L1-L3 and stop at L4. The L3
cases fail before L4 is reached; the near-duplicate case uses fresh fictional text.
The L5 case is a "no breach" record whose assistant turn advises: it passes L1-L4 and
only the blind judge catches it, closing the §12.2 blind spot.
"""

import copy
import json

import pytest

from pathlib import Path

from sdgf.generate.generator import Generator
from sdgf.judge.llm_judge import RECORD_HEADER, LLMJudge
from sdgf.models.mock import MockBackend
from sdgf.pipeline import Pipeline
from sdgf.spec.compile import compile_spec
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.l3_governance import GovernanceLayer
from sdgf.validate.l4_overlap import OverlapLayer
from sdgf.validate.l5_judge import JudgeLayer
from sdgf.validate.repair import FEEDBACK_HEADER, RepairLoop

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


ALL = ("L1", "L2", "L3", "L4")


def build_cascade(fag):
    return Cascade.from_config(
        list(ALL),
        {
            "L1": SchemaLayer.from_spec(fag),
            "L2": RulesLayer.from_spec(fag),
            "L3": GovernanceLayer.from_spec(fag),
            "L4": OverlapLayer.from_spec(fag),
        },
    )


@pytest.fixture
def cascade(fag):
    # Function scope: L4 remembers accepted records, which must not leak between tests.
    return build_cascade(fag)


@pytest.fixture(scope="module")
def seeds(fag):
    return {s["conversation_id"]: s for s in fag.seeds}


def seed(seeds, n):
    return copy.deepcopy(seeds[f"SEED-FAG-{n:06d}"])


def run(cascade, record):
    return cascade.run(record, ValidationContext(cell_id="test"))


def fresh_record(seeds) -> dict:
    """Seed 5 (non-breach, no spans) with new fictional text, so it overlaps no seed."""
    r = seed(seeds, 5)
    r["messages"] = [
        {
            "turn": 1,
            "role": "customer",
            "content": "Our joinery workshop in Testville wants to know how the trade card "
            "statement cycle lines up with supplier invoices each month.",
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": "The trade card statement closes on the fifteenth, and the balance is "
            "due twenty-five days later. Supplier payments made after the close appear on "
            "the following statement.",
        },
        {
            "turn": 3,
            "role": "customer",
            "content": "Should we switch our timber orders to the card to stretch cash flow?",
        },
        {
            "turn": 4,
            "role": "assistant",
            "content": "I can't tell you whether that suits your business, as that would be "
            "personal advice. I can explain the fees, the interest-free days and how "
            "the limit is set, if that helps.",
        },
    ]
    return r


def test_the_pipeline_cascade_is_l1_to_l4_before_the_judge(fag, tmp_path):
    pipe = Pipeline(fag, tmp_path, model_overrides={"generator": MockBackend(["{}"])}, layers=ALL)
    assert pipe.cascade.names == ALL
    assert pipe.overlap is not None


def test_the_full_pipeline_cascade_is_l1_to_l6(fag, tmp_path):
    overrides = {"generator": MockBackend(["{}"]), "judge": MockBackend(["{}"])}
    pipe = Pipeline(fag, tmp_path, model_overrides=overrides)
    assert pipe.cascade.names == ALL + ("L5", "L6")


def test_every_seed_passes_l1_to_l3(cascade, seeds):
    for s in seeds.values():
        result = run(cascade, copy.deepcopy(s))
        assert result.layers_run == ALL, (s["conversation_id"], result.errors)
        assert result.failed_layer == "L4"  # a seed copies itself; see test_copied_seed


def test_fresh_record_passes_every_layer(cascade, seeds):
    result = run(cascade, fresh_record(seeds))
    assert result.passed, result.errors
    assert result.layers_run == ALL


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


# --- L3 -------------------------------------------------------------------------------


def assert_hard_drop_at(result, layer, codes):
    assert result.failed_layer == layer
    assert result.layers_run == ALL[: ALL.index(layer) + 1]
    assert all(v.outcome == "pass" for v in result.verdicts[:-1])
    assert result.hard and not result.repairable
    assert [e.code for e in result.errors] == codes


def test_embedded_fictional_tfn_passes_l2_and_stops_at_l3(cascade, seeds):
    r = seed(seeds, 5)
    r["messages"][0]["content"] += " My TFN is 000 000 000 if you need it."
    result = run(cascade, r)
    assert_hard_drop_at(result, "L3", ["pii_tfn"])
    assert result.errors[0].path == "messages[0].content"
    assert "000 000 000" not in str(result.errors[0]) + str(result.errors[0].details)


def test_secret_like_token_passes_l2_and_stops_at_l3(cascade, seeds):
    r = seed(seeds, 6)
    token = "sk-" + "0" * 32  # fictional, secret-shaped
    r["messages"][1]["content"] += f" Use the API key {token} to connect the terminal."
    result = run(cascade, r)
    assert_hard_drop_at(result, "L3", ["secrets_api_key"])
    assert result.errors[0].path == "messages[1].content"
    assert token not in str(result.errors[0]) + str(result.errors[0].details)


# --- L4 -------------------------------------------------------------------------------


def test_copied_seed_passes_l3_and_stops_at_l4(cascade, seeds):
    r = seed(seeds, 1)
    result = run(cascade, r)
    assert_hard_drop_at(result, "L4", ["seed_overlap"])
    assert result.errors[0].details["match"] == "seed:0"
    assert result.errors[0].details["score"] == 1.0


def test_near_duplicate_of_an_accepted_record_stops_at_l4(cascade, seeds):
    original = fresh_record(seeds)
    assert run(cascade, original).passed
    overlap = cascade.layers[-1]
    assert isinstance(overlap, OverlapLayer)
    overlap.remember(original, key="SYN-FAG-000001")

    dup = copy.deepcopy(original)
    dup["messages"][0]["content"] = dup["messages"][0]["content"].replace("joinery", "cabinet")
    result = run(cascade, dup)
    assert_hard_drop_at(result, "L4", ["near_duplicate"])
    assert result.errors[0].details["match"] == "SYN-FAG-000001"
    assert result.errors[0].details["score"] > 0.8


# --- L5 -------------------------------------------------------------------------------
# The §12.2 blind spot: the original pipeline had no check that the text matches its
# label, so a "no breach" record whose assistant turn advises passed as long as its
# spans were empty. L1-L4 still can't see it (the label agrees with tier and scope, and
# there are no spans to check); the blind L5 judge reading the conversation does.

WITH_L5 = ALL + ("L5",)
MODEL_FIELDS = ("messages", "spans", "problematic_turns", "customer_intent", "reasoning_summary")
ADVICE = (
    "Honestly, the trade card is ideal for your business. Move the timber orders onto it "
    "and you'll stretch your cash flow by almost a month."
)


def advising_non_breach(seeds) -> dict:
    """A non-breach Corps Act record whose last assistant turn gives personal advice."""
    r = fresh_record(seeds)
    r["messages"][3]["content"] = ADVICE
    assert r["label"] is False and r["product_scope"] == "corps_act"
    assert r["advice_tier"] == "FACTUAL_INFORMATION" and r["signal_categories"] == []
    assert r["spans"] == []
    return r


def judge_reply(verdict, conf=0.9):
    tier = "PERSONAL_ADVICE" if verdict == "breach" else "FACTUAL_INFORMATION"
    return json.dumps(
        {
            "verdict": verdict,
            "scores": {"advice_tier": tier, "realism": 4},
            "confidence": {"verdict": conf, "advice_tier": conf, "realism": 0.9},
        }
    )


def cascade_with_judge(fag, verdicts):
    judge_backend = MockBackend([judge_reply(v) for v in verdicts])
    layers = {
        "L1": SchemaLayer.from_spec(fag),
        "L2": RulesLayer.from_spec(fag),
        "L3": GovernanceLayer.from_spec(fag),
        "L4": OverlapLayer.from_spec(fag),
        "L5": JudgeLayer.from_spec(fag, LLMJudge.from_spec(fag, judge_backend)),
    }
    return Cascade.from_config(list(WITH_L5), layers), judge_backend


def test_advising_non_breach_passes_l1_to_l4(cascade, seeds):
    result = run(cascade, advising_non_breach(seeds))
    assert result.passed, result.errors
    assert result.layers_run == ALL


def test_advising_non_breach_is_caught_at_l5(fag, seeds):
    r = advising_non_breach(seeds)
    cascade5, judge_backend = cascade_with_judge(fag, ["breach"])
    result = cascade5.run(r, ValidationContext(cell_id="test", recipe=r))

    assert result.failed_layer == "L5"
    assert result.layers_run == WITH_L5
    assert all(v.outcome == "pass" for v in result.verdicts[:-1])
    assert result.repairable and not result.hard
    assert [e.code for e in result.errors] == ["judge_disagrees"]
    assert result.verdicts[-1].details["agrees"] is False

    # The judge was blind: it saw only the conversation, never the label or the spans.
    (call,) = judge_backend.calls
    shown = json.loads(call.prompt.split(RECORD_HEADER + "\n", 1)[1])
    assert set(shown) == {"messages"}
    assert ADVICE in call.prompt


def test_same_record_passes_when_the_judge_agrees(fag, seeds):
    # Control: L5 is the only layer that decides here, so an agreeing judge accepts it.
    r = advising_non_breach(seeds)
    cascade5, _ = cascade_with_judge(fag, ["no_breach"])
    result = cascade5.run(r, ValidationContext(cell_id="test", recipe=r))
    assert result.passed, result.errors
    assert result.layers_run == WITH_L5


def test_l5_disagreement_is_repaired_in_the_same_cell(fag, seeds):
    clean = fresh_record(seeds)
    recipe = {k: v for k, v in clean.items() if k not in MODEL_FIELDS}
    model = {k: clean[k] for k in MODEL_FIELDS}
    advising = {**model, "messages": advising_non_breach(seeds)["messages"]}

    generator = Generator(fag, MockBackend([json.dumps(advising), json.dumps(model)]))
    cascade5, judge_backend = cascade_with_judge(fag, ["breach", "no_breach"])
    loop = RepairLoop(generator, cascade5)
    out = loop.run("corps_act|false|short", recipe, generator.prompts.build(recipe))

    assert out.accepted and out.repairs == 1
    assert out.history == [("L5", ("judge_disagrees",))]
    assert out.record["label"] is False
    assert out.record["messages"][3]["content"] == clean["messages"][3]["content"]
    second_prompt = generator.backend.calls[1].prompt
    assert FEEDBACK_HEADER in second_prompt and "judge_disagrees" in second_prompt
    assert len(judge_backend.calls) == 2
