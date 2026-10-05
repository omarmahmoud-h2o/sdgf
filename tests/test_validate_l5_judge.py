import json
from pathlib import Path

import pytest

from sdgf.judge.interface import Judge, JudgeError, JudgeParseError, JudgeResult, compile_rubric
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

    def __init__(
        self, verdict="yes", conf=0.9, reason="because", writes=True, fail=False, policy="never"
    ):
        super().__init__(schema(policy))
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


def test_unusable_judge_output_goes_to_review_when_enabled():
    sink = ListReviewSink()
    v = layer(FakeJudge(fail=True), review=sink).check(RECORD, CTX)
    assert v.codes == ("sent_to_review",)
    assert sink.items[0].code == "judge_error" and sink.items[0].judge is None


# ── reasons ──────────────────────────────────────────────────────


def with_reasons(policy, verdict="yes", conf=0.9, fallback=None, writes=True):
    judge = FakeJudge(verdict=verdict, conf=conf, writes=writes, policy=policy)
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
    judge = FakeJudge(conf=0.2, policy="flagged")
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


def jev_answers(verdict, conf=0.9):
    return {
        "verdict": {"type": "choice", "choice": verdict, "confidence": conf},
        "advice_tier": {"type": "choice", "choice": "FACTUAL_INFORMATION", "confidence": 0.8},
        "realism": {"type": "score", "probabilities": {"3": 1.0}, "confidence": 0.8},
    }


@pytest.mark.parametrize(
    "reply, outcome, code",
    [
        ((200, {"answers": jev_answers("no_breach")}), "pass", None),
        ((200, {"answers": jev_answers("breach")}), "fail_repairable", "judge_disagrees"),
        ((422, {"error": {"message": "state too large"}}), "fail_hard", "judge_error"),
    ],
)
def test_l5_with_the_jev_judge(fag, monkeypatch, reply, outcome, code):
    # The same L5 runs on Jev's typed answers as on an LLM judge's JSON.
    from sdgf.judge.jev import JevBackend
    from sdgf.judge.select import judge_from_spec

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key-000")
    backend = JevBackend("jev-1.13", transport=lambda *a: reply, sleep=lambda s: None)
    lay = JudgeLayer.from_spec(fag, judge_from_spec(fag, backend))
    seed = next(s for s in fag.seeds if s["label"] is False)
    v = lay.check(seed, ValidationContext(recipe=seed))
    assert v.outcome == outcome
    assert (v.codes[0] if v.codes else None) == code


def test_fag_spec_rejects_label_map_typo(fag):
    from sdgf.spec.schema import parse_spec

    data = fag.spec.model_dump()
    data["rubric"]["verdict"]["labels"] = {"breech": True}
    with pytest.raises(SpecValidationError, match="rubric.verdict"):
        parse_spec(data)


# ── answer_emergent ──────────────────────────────────────────────


def emergent_layer(verdict):
    judge = FakeJudge(verdict)
    return judge, layer(judge, fields=("question",), label_field="answer", answer_emergent=True)


def test_answer_emergent_fidelity_compares_the_judges_answer_with_the_records():
    judge, lay = emergent_layer("yes")
    record = {"question": "Q?", "response": "Answer: yes", "answer": "yes"}
    v = lay.check(record, ValidationContext(cell_id="c1", recipe={"keyword": "k"}))
    assert v.passed and v.details["agrees"]
    assert judge.seen == [{"question": "Q?"}]


def test_answer_emergent_disagreement_names_the_records_answer_not_a_fixed_label():
    _, lay = emergent_layer("no")
    record = {"question": "Q?", "response": "Answer: yes", "answer": "yes"}
    v = lay.check(record, ValidationContext(cell_id="c1", recipe={}))
    assert v.repairable and v.codes == ("judge_disagrees",)
    message = v.errors[0].message
    assert "answered the question 'no'" in message and "answer is 'yes'" in message
    assert "fixed label" not in message


def test_from_spec_takes_the_label_field_from_the_task_type():
    cfa = compile_spec(Path(__file__).resolve().parents[1] / "tasks" / "cfa")
    lay = JudgeLayer.from_spec(cfa, FakeJudge())
    assert (lay.label_field, lay.answer_emergent, lay.fields) == ("answer", True, ("question",))
    fag = compile_spec(FAG_DIR)
    fag_judge = LLMJudge.from_spec(fag, MockBackend(["{}"], cycle=True))
    fag_layer = JudgeLayer.from_spec(fag, fag_judge)
    assert (fag_layer.label_field, fag_layer.answer_emergent) == ("label", False)


def test_escalation_always_escalates_every_record():
    rules = EscalationRules(always=True, on_hard_cells=False, on_contestable=False)
    lay = layer(
        FakeJudge("yes"),
        fields=("question",),
        label_field="answer",
        answer_emergent=True,
        escalation=rules,
    )
    record = {"question": "Q?", "response": "Answer: yes", "answer": "yes"}
    v = lay.check(record, ValidationContext(cell_id="c1", recipe={}))
    assert v.passed and v.details["escalate"] and not v.details["low_confidence"]


# ── the outcome table at the seam, on a FAG-shaped rubric ────────

FAG_RUBRIC = RubricSection(
    verdict={"values": ["breach", "no_breach"]},
    criteria=[
        {"name": "advice_tier", "values": ["FACTUAL_INFORMATION", "GENERAL_ADVICE"]},
        {"name": "realism", "min": 1, "max": 5},
    ],
)
FAG_LABELS = {"breach": True, "no_breach": False}
CONVERSATION = [
    {"role": "customer", "content": "Should I put our surplus into a term deposit?"},
    {"role": "assistant", "content": "For your business, the 12-month term deposit suits you."},
]
FAG_RECORD = {
    "messages": CONVERSATION,
    "label": True,
    "spans": [{"turn": 2, "text": "suits you"}],
    "_provenance": {"attempt": 1},
}


class ScriptedJudge(Judge):
    """A judge whose answer is scripted: a verdict and confidences, or a parse error.
    It records the views it was asked to explain."""

    name = "scripted"
    writes_reasons = True

    def __init__(
        self,
        verdict="breach",
        conf=0.9,
        *,
        tier_conf=0.9,
        error=None,
        reason_required="never",
        reason="the assistant recommends a product",
    ):
        rubric = FAG_RUBRIC.model_copy(update={"reason_required": reason_required})
        super().__init__(compile_rubric(rubric))
        self.verdict, self.conf, self.tier_conf = verdict, conf, tier_conf
        self.error, self.reason = error, reason
        self.explained: list[dict] = []

    def judge(self, record):
        if self.error is not None:
            raise self.error
        return JudgeResult(
            self.verdict,
            {"advice_tier": "GENERAL_ADVICE", "realism": 4},
            {"verdict": self.conf, "advice_tier": self.tier_conf, "realism": 0.9},
        )

    def explain(self, record, result):
        self.explained.append(record)
        return self.reason


def fag_check(judge, label=True, *, escalation=None, review=None):
    lay = JudgeLayer(
        judge, fields=("messages",), labels=FAG_LABELS, escalation=escalation, review=review
    )
    return lay.check(FAG_RECORD, ValidationContext(cell_id="c7", recipe={"label": label}))


def test_a_confident_agreeing_verdict_passes_the_record_without_escalating():
    v = fag_check(ScriptedJudge("breach", 0.9), label=True)
    assert v.outcome == "pass" and v.errors == ()
    assert v.details["agrees"] is True
    assert v.details["escalate"] is False
    assert v.details["judge"]["verdict"] == "breach"
    assert "reason" not in v.details


def test_a_confident_disagreeing_verdict_is_sent_back_naming_both_verdicts():
    v = fag_check(ScriptedJudge("no_breach", 0.9), label=True)
    assert v.outcome == "fail_repairable" and v.codes == ("judge_disagrees",)
    issue = v.errors[0]
    assert issue.path == "verdict"
    assert issue.details["verdict"] == "no_breach"
    assert issue.details["expected"] == ["breach"]
    assert "'no_breach'" in issue.message and "'breach'" in issue.message


@pytest.mark.parametrize("conf, low", [(0.7, False), (0.69, True)])
def test_confidence_at_the_low_confidence_threshold_is_not_low(conf, low):
    rules = EscalationRules(low_confidence=0.7)
    v = fag_check(ScriptedJudge("breach", conf), label=True, escalation=rules)
    assert v.details["low_confidence"] is low


def test_low_confidence_with_review_on_drops_the_record_into_the_review_queue():
    sink = ListReviewSink()
    v = fag_check(ScriptedJudge("breach", 0.4), label=True, review=sink)
    assert v.outcome == "fail_hard" and v.codes == ("sent_to_review",)
    [item] = sink.items
    assert item.code == "low_confidence" and item.intended_label is True
    assert item.layer == "L5" and item.cell_id == "c7"
    assert item.judge["verdict"] == "breach"
    assert "_provenance" not in item.record
    assert item.record["messages"] == CONVERSATION


def test_low_confidence_with_review_off_passes_and_escalates_to_extra_votes():
    v = fag_check(ScriptedJudge("breach", 0.4), label=True, review=None)
    assert v.outcome == "pass"
    assert v.details["escalate"] is True


def test_a_low_criterion_confidence_does_not_make_the_verdict_low_confidence():
    v = fag_check(ScriptedJudge("breach", 0.9, tier_conf=0.1), label=True)
    assert v.details["low_confidence"] is False


def test_an_unparseable_judge_answer_with_review_off_drops_the_record_as_judge_error():
    judge = ScriptedJudge(error=JudgeParseError(["<root>: bad"]))
    v = fag_check(judge, label=True, review=None)
    assert v.outcome == "fail_hard" and v.codes == ("judge_error",)
    assert v.details["judge"] is None
    assert "<root>: bad" in v.errors[0].message


def test_an_int_fixed_label_is_not_met_by_a_verdict_that_means_true():
    v = fag_check(ScriptedJudge("breach", 0.9), label=1)
    assert v.codes == ("judge_disagrees",)


def test_a_flagged_record_gets_a_blind_reason_appended_to_the_sent_back_message():
    judge = ScriptedJudge("no_breach", 0.9, reason_required="flagged", reason="no product named")
    v = fag_check(judge, label=True)
    assert judge.explained == [{"messages": CONVERSATION}]
    assert v.errors[0].message.endswith("Judge's reason: no product named")
