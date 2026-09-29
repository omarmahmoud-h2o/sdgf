import json
from pathlib import Path

import pytest

from sdgf.judge.interface import Judge, JudgeError, JudgeParseError, JudgeResult, compile_rubric
from sdgf.judge.jev import JevNotImplementedError
from sdgf.judge.llm_judge import RECORD_HEADER, LLMJudge
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import EscalationRules, RubricSection, SpecValidationError, Verdict
from sdgf.validate.base import ValidationContext
from sdgf.validate.l5_judge import JudgeLayer, ListReviewSink

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def seeds():
    return [json.loads(line) for line in (FAG_DIR / "seeds.jsonl").read_text().splitlines()]


def schema(reason="never"):
    return compile_rubric(
        RubricSection(
            verdict={"values": ["yes", "no", "unclear"]},
            criteria=[{"name": "realism", "min": 1, "max": 5}],
            reason_required=reason,
        )
    )


class FakeJudge(Judge):
    """Returns a fixed verdict and confidence, and records every record it was shown."""

    name = "fake"

    def __init__(self, verdict="yes", conf=0.9, reason="because", writes=True, fail=False):
        super().__init__(schema())
        self.verdict, self.conf, self.reason, self.fail = verdict, conf, reason, fail
        self.writes_reasons = writes
        self.seen: list[dict] = []
        self.explained: list[dict] = []

    def judge(self, record):
        self.seen.append(record)
        if self.fail:
            raise JudgeParseError(["<root>: reply was not a parseable JSON object"])
        return JudgeResult(self.verdict, {"realism": 4}, {"verdict": self.conf, "realism": 0.8})

    def explain(self, record, result):
        if not self.writes_reasons:
            return super().explain(record, result)
        self.explained.append(record)
        return self.reason


RECORD = {"question": "Q?", "answer": "A.", "label": "yes", "_provenance": {"x": 1}}
CTX = ValidationContext(cell_id="c1", recipe={"label": "yes"})


def layer(judge=None, **kw):
    kw.setdefault("fields", ("question", "answer"))
    return JudgeLayer(judge or FakeJudge(), **kw)


# ── fidelity ─────────────────────────────────────────────────────


def test_agreeing_confident_verdict_passes_with_details():
    v = layer().check(RECORD, CTX)
    assert v.passed and v.errors == ()
    assert v.details["agrees"] is True
    assert v.details["low_confidence"] is False
    assert v.details["escalate"] is False
    assert v.details["judge"]["verdict"] == "yes"
    assert "reason" not in v.details


def test_disagreement_is_repairable_with_expected_verdict():
    v = layer(FakeJudge(verdict="no")).check(RECORD, CTX)
    assert v.repairable and v.codes == ("judge_disagrees",)
    issue = v.errors[0]
    assert issue.path == "verdict"
    assert issue.details["verdict"] == "no" and issue.details["expected"] == ["yes"]
    assert "'no'" in issue.message and "'yes'" in issue.message
    assert v.details["agrees"] is False


def test_verdict_outside_the_label_map_never_agrees():
    v = layer(FakeJudge(verdict="unclear"), labels={"yes": "yes", "no": "no"}).check(RECORD, CTX)
    assert v.codes == ("judge_disagrees",)


def test_label_comes_from_the_recipe_over_the_record():
    record = {**RECORD, "label": "no"}  # the recipe owns the label
    assert layer().check(record, CTX).passed


def test_label_map_maps_verdicts_onto_bool_labels():
    lay = layer(FakeJudge(verdict="yes"), labels={"yes": True, "no": False})
    assert lay.check(RECORD, ValidationContext(recipe={"label": True})).passed
    assert lay.check(RECORD, ValidationContext(recipe={"label": False})).repairable
    # 1 == True in Python, but an int label doesn't mean a bool verdict
    assert lay.check(RECORD, ValidationContext(recipe={"label": 1})).repairable
    assert lay.verdicts_for(False) == ["no"]


def test_label_map_naming_unknown_verdict_raises():
    with pytest.raises(JudgeError, match="unknown verdict values"):
        layer(labels={"maybe": True})


def test_spec_rejects_label_map_with_unknown_verdict():
    with pytest.raises(ValueError, match="unknown verdict values"):
        Verdict(values=["a", "b"], labels={"c": True})


# ── blindness ────────────────────────────────────────────────────


def test_judge_sees_only_the_view_never_label_or_private_keys():
    judge = FakeJudge()
    layer(judge, fields=("question", "answer", "label")).check(RECORD, CTX)
    assert judge.seen == [{"question": "Q?", "answer": "A."}]


def test_layer_needs_a_visible_field():
    with pytest.raises(JudgeError, match="at least one"):
        layer(fields=("label",))


# ── low confidence and review ────────────────────────────────────


def test_low_confidence_goes_to_review_when_enabled():
    sink = ListReviewSink()
    v = layer(FakeJudge(conf=0.4), review=sink).check(RECORD, CTX)
    assert v.hard and v.codes == ("sent_to_review",)
    assert v.details["low_confidence"] is True
    [item] = sink.items
    assert item.code == "low_confidence" and item.layer == "L5"
    assert item.cell_id == "c1" and item.intended_label == "yes"
    assert "_provenance" not in item.record
    assert item.judge["verdict"] == "yes"
    assert item.to_dict()["record"]["question"] == "Q?"


def test_low_confidence_disagreement_also_goes_to_review():
    sink = ListReviewSink()
    v = layer(FakeJudge(verdict="no", conf=0.2), review=sink).check(RECORD, CTX)
    assert v.codes == ("sent_to_review",) and v.errors[0].details["agrees"] is False
    assert len(sink.items) == 1


def test_low_confidence_without_review_is_judged_and_escalated():
    v = layer(FakeJudge(conf=0.4)).check(RECORD, CTX)
    assert v.passed and v.details["low_confidence"] and v.details["escalate"]
    v = layer(FakeJudge(verdict="no", conf=0.4)).check(RECORD, CTX)
    assert v.codes == ("judge_disagrees",) and v.details["escalate"]


def test_threshold_is_the_spec_escalation_value():
    rules = EscalationRules(low_confidence=0.95)
    sink = ListReviewSink()
    assert layer(escalation=rules, review=sink).check(RECORD, CTX).hard
    assert layer(escalation=EscalationRules(low_confidence=0.5)).check(RECORD, CTX).passed


def test_confident_records_never_reach_review():
    sink = ListReviewSink()
    layer(review=sink).check(RECORD, CTX)
    layer(FakeJudge(verdict="no"), review=sink).check(RECORD, CTX)
    assert sink.items == []


@pytest.mark.parametrize(
    "facts, rules, escalate",
    [
        ({"difficulty": "HARD"}, EscalationRules(), True),
        ({"difficulty": "HARD"}, EscalationRules(on_hard_cells=False), False),
        ({"contestable": True}, EscalationRules(), True),
        ({"contestable": True}, EscalationRules(on_contestable=False), False),
        ({"difficulty": "EASY", "contestable": False}, EscalationRules(), False),
    ],
)
def test_escalation_on_hard_and_contestable(facts, rules, escalate):
    ctx = ValidationContext(recipe={"label": "yes", **facts})
    assert layer(escalation=rules).check(RECORD, ctx).details["escalate"] is escalate


# ── judge failures ───────────────────────────────────────────────


def test_unusable_judge_output_is_a_hard_drop():
    v = layer(FakeJudge(fail=True)).check(RECORD, CTX)
    assert v.hard and v.codes == ("judge_error",)
    assert v.details["judge"] is None


def test_unusable_judge_output_goes_to_review_when_enabled():
    sink = ListReviewSink()
    v = layer(FakeJudge(fail=True), review=sink).check(RECORD, CTX)
    assert v.codes == ("sent_to_review",)
    assert sink.items[0].code == "judge_error" and sink.items[0].judge is None


# ── reasons ──────────────────────────────────────────────────────


def with_reasons(policy, verdict="yes", conf=0.9, fallback=None, writes=True):
    judge = FakeJudge(verdict=verdict, conf=conf, writes=writes)
    judge.schema = schema(policy)
    return judge, layer(judge, fallback_judge=fallback)


def test_no_reason_when_rubric_says_never():
    judge = FakeJudge(verdict="no")
    layer(judge).check(RECORD, CTX)
    assert judge.explained == []


def test_flagged_reason_only_on_flagged_records():
    judge, lay = with_reasons("flagged")
    assert "reason" not in lay.check(RECORD, CTX).details
    assert judge.explained == []
    judge, lay = with_reasons("flagged", verdict="no")
    v = lay.check(RECORD, CTX)
    assert v.details["reason"] == "because" and "because" in v.errors[0].message
    judge, lay = with_reasons("flagged", conf=0.3)
    assert lay.check(RECORD, CTX).details["reason"] == "because"


def test_always_reason_on_every_record_and_blind():
    judge, lay = with_reasons("always")
    assert lay.check(RECORD, CTX).details["reason"] == "because"
    assert judge.explained == [{"question": "Q?", "answer": "A."}]


def test_reason_comes_from_the_fallback_judge():
    fallback = FakeJudge(reason="fallback says so")
    judge, lay = with_reasons("always", writes=False, fallback=fallback)
    assert lay.check(RECORD, CTX).details["reason"] == "fallback says so"
    assert fallback.seen == [] and len(fallback.explained) == 1


def test_reasons_need_a_judge_that_writes_them():
    with pytest.raises(JudgeError, match="fallback judge"):
        with_reasons("always", writes=False)


def test_reason_goes_into_the_review_item():
    judge = FakeJudge(conf=0.2)
    judge.schema = schema("flagged")
    sink = ListReviewSink()
    layer(judge, review=sink).check(RECORD, CTX)
    assert sink.items[0].reason == "because"


# ── FAG ──────────────────────────────────────────────────────────


def fag_reply(verdict, conf=0.9):
    return json.dumps(
        {
            "verdict": verdict,
            "scores": {"advice_tier": "FACTUAL_INFORMATION", "realism": 4},
            "confidence": {"verdict": conf, "advice_tier": 0.9, "realism": 0.9},
        }
    )


def fag_layer(fag, responses, review=None):
    backend = MockBackend(responses)
    return JudgeLayer.from_spec(fag, LLMJudge.from_spec(fag, backend), review=review), backend


def test_fag_spec_maps_breach_verdicts_to_bool_labels(fag):
    assert fag.spec.rubric.verdict.labels == {"breach": True, "no_breach": False}
    lay, _ = fag_layer(fag, [fag_reply("breach")])
    assert lay.fields == ("messages",)
    assert lay.verdicts_for(True) == ["breach"] and lay.verdicts_for(False) == ["no_breach"]


def test_fag_seeds_pass_when_the_judge_agrees(fag, seeds):
    replies = [fag_reply("breach" if s["label"] else "no_breach") for s in seeds]
    lay, _ = fag_layer(fag, replies)
    for s in seeds:
        v = lay.check(s, ValidationContext(recipe=s))
        assert v.passed, v.errors
        assert v.details["escalate"] is (s["difficulty"] == "HARD" or s["contestable"])


def test_fag_prompt_is_blind_to_the_label(fag, seeds):
    seed = seeds[0]
    lay, backend = fag_layer(fag, [fag_reply("breach"), fag_reply("breach")])
    lay.check(seed, ValidationContext(recipe=seed))
    flipped = {**seed, "label": not seed["label"]}
    lay.check(flipped, ValidationContext(recipe=flipped))
    first, second = (c.prompt for c in backend.calls)
    assert first == second
    record = json.loads(first.split(RECORD_HEADER + "\n", 1)[1])
    assert set(record) == {"messages"}


def test_fag_review_sink_used_only_when_hitl_enables_it(fag, seeds):
    sink = ListReviewSink()
    lay, _ = fag_layer(fag, [fag_reply("breach", conf=0.3)], review=sink)
    assert fag.spec.hitl.review_flagged is False and lay.review is None
    seed = seeds[0]
    assert lay.check(seed, ValidationContext(recipe=seed)).passed
    assert sink.items == []


def test_selecting_jev_as_judge_fails_clearly(fag):
    from sdgf.judge.jev import JevJudge

    with pytest.raises(JevNotImplementedError):
        JudgeLayer.from_spec(fag, JevJudge(compile_rubric(fag.spec.rubric), None))


def test_fag_spec_rejects_label_map_typo(fag):
    from sdgf.spec.schema import parse_spec

    data = fag.spec.model_dump()
    data["rubric"]["verdict"]["labels"] = {"breech": True}
    with pytest.raises(SpecValidationError, match="rubric.verdict"):
        parse_spec(data)
