import json
from pathlib import Path

import pytest

from sdgf.judge.interface import Judge, JudgeParseError, JudgeResult, compile_rubric
from sdgf.judge.llm_judge import LLMJudge
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import EscalationRules, RubricSection
from sdgf.validate.base import LayerVerdict, ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l5_judge import JudgeLayer
from sdgf.validate.l6_consistency import (
    BackendAnswerer,
    ConsistencyError,
    ConsistencyLayer,
    majority,
)

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"


def schema():
    return compile_rubric(
        RubricSection(
            verdict={"values": ["yes", "no", "unclear"]},
            criteria=[{"name": "realism", "min": 1, "max": 5}],
        )
    )


class VoteJudge(Judge):
    """Returns scripted verdicts in turn; None raises JudgeParseError (an abstention)."""

    name = "votes"

    def __init__(self, verdicts, conf=0.9):
        super().__init__(schema())
        self.verdicts = list(verdicts)
        self.conf = conf
        self.seen: list[dict] = []

    def judge(self, record):
        self.seen.append(record)
        v = self.verdicts.pop(0)
        if v is None:
            raise JudgeParseError(["<root>: reply was not a parseable JSON object"])
        return JudgeResult(v, {"realism": 4}, {"verdict": self.conf, "realism": 0.8})


RECORD = {"question": "Q?", "answer": "A.", "label": "yes", "_provenance": {"x": 1}}
FIELDS = ("question", "answer")


def l5_verdict(escalate=True, agrees=True, conf=0.9, outcome="pass"):
    judge = {"verdict": "yes", "scores": {}, "confidence": {"verdict": conf}}
    details = {"judge": judge, "agrees": agrees, "low_confidence": conf < 0.7}
    details["escalate"] = escalate
    return LayerVerdict("L5", outcome, (), details)


def ctx(*previous, recipe=None):
    return ValidationContext(cell_id="c1", recipe=recipe or {"label": "yes"}, previous=previous)


def layer(judge, **kw):
    kw.setdefault("fields", FIELDS)
    kw.setdefault("k", 5)
    return ConsistencyLayer(mode="label_first", judge=judge, **kw)


# ── escalation gate ──────────────────────────────────────────────


def test_not_escalated_passes_without_votes():
    judge = VoteJudge([])
    v = layer(judge).check(RECORD, ctx(l5_verdict(escalate=False)))
    assert v.passed and v.details == {"escalated": False, "method": "skipped"}
    assert judge.seen == []


def test_escalation_from_recipe_when_no_l5_verdict():
    judge = VoteJudge(["yes"] * 5)
    assert layer(judge).check(RECORD, ctx()).details["method"] == "skipped"
    for recipe in ({"label": "yes", "difficulty": "HARD"}, {"label": "yes", "contestable": True}):
        judge = VoteJudge(["yes"] * 5)
        v = layer(judge).check(RECORD, ctx(recipe=recipe))
        assert v.passed and v.details["method"] == "votes" and len(judge.seen) == 5


def test_escalation_rules_can_turn_off_hard_and_contestable():
    rules = EscalationRules(on_hard_cells=False, on_contestable=False)
    recipe = {"label": "yes", "difficulty": "hard", "contestable": True}
    v = layer(VoteJudge([]), escalation=rules).check(RECORD, ctx(recipe=recipe))
    assert v.details["method"] == "skipped"


def test_escalated_only_false_votes_on_every_record():
    judge = VoteJudge(["yes"] * 3)
    v = layer(judge, k=3, escalated_only=False).check(RECORD, ctx(l5_verdict(escalate=False)))
    assert v.details["method"] == "votes" and len(judge.seen) == 3


# ── label_first K votes ──────────────────────────────────────────


def test_majority_matching_label_passes():
    v = layer(VoteJudge(["yes", "no", "yes", "yes", "unclear"])).check(RECORD, ctx(l5_verdict()))
    assert v.passed
    assert v.details["agree"] == 3 and v.details["cast"] == 5 and v.details["majority"] == "yes"


def test_majority_against_label_is_repairable():
    v = layer(VoteJudge(["no", "no", "yes", "no", "yes"])).check(RECORD, ctx(l5_verdict()))
    assert v.repairable and v.codes == ("consistency_disagrees",)
    assert v.errors[0].details == {"agree": 2, "cast": 5, "majority": "no"}
    assert "fixed label" in v.errors[0].message


def test_tie_is_not_a_majority():
    v = layer(VoteJudge(["yes", "no", "yes", "no"]), k=4).check(RECORD, ctx(l5_verdict()))
    assert v.repairable


def test_split_non_label_votes_still_fail_without_label_majority():
    # "yes" is the most common single verdict but not more than half of the votes.
    v = layer(VoteJudge(["yes", "yes", "no", "unclear", "no"])).check(RECORD, ctx(l5_verdict()))
    assert v.repairable


def test_unparseable_votes_are_abstentions_not_in_denominator():
    # §12.1: 2 agreeing of 2 cast passes; DS²-Instruct would score it 2/5 and drop it.
    judge = VoteJudge(["yes", None, None, "yes", None])
    v = layer(judge).check(RECORD, ctx(l5_verdict()))
    assert v.passed
    assert v.details["cast"] == 2 and v.details["abstained"] == 3
    assert v.details["votes"] == ["yes", None, None, "yes", None]


def test_abstentions_do_not_rescue_a_disagreement():
    v = layer(VoteJudge(["no", None, "yes", "no", None])).check(RECORD, ctx(l5_verdict()))
    assert v.repairable and v.errors[0].details["cast"] == 3


def test_all_votes_unparseable_is_a_hard_drop():
    v = layer(VoteJudge([None] * 5)).check(RECORD, ctx(l5_verdict()))
    assert v.hard and v.codes == ("consistency_no_votes",)


def test_votes_are_blind_to_label_and_private_keys():
    judge = VoteJudge(["yes"] * 5)
    layer(judge).check(RECORD, ctx(l5_verdict()))
    assert judge.seen == [{"question": "Q?", "answer": "A."}] * 5


def test_label_map_and_recipe_label_win():
    labels = {"yes": True, "no": False}
    record = dict(RECORD, label=False)  # the recipe owns the label
    lay = layer(VoteJudge(["yes"] * 3), k=3, labels=labels)
    assert lay.check(record, ctx(l5_verdict(), recipe={"label": True})).passed
    # A bool mapping only matches a bool label: 1 is not True.
    lay = layer(VoteJudge(["yes"] * 3), k=3, labels=labels)
    assert lay.check(record, ctx(l5_verdict(), recipe={"label": 1})).repairable


# ── trusted judge: confidence replaces votes ─────────────────────


def test_trusted_confident_agreeing_judge_skips_votes():
    judge = VoteJudge([])
    v = layer(judge, trusted=True).check(RECORD, ctx(l5_verdict(conf=0.95)))
    assert v.passed and v.details == {"escalated": True, "method": "confidence", "confidence": 0.95}
    assert judge.seen == []


def test_trusted_judge_with_low_confidence_still_votes():
    judge = VoteJudge(["yes"] * 5)
    v = layer(judge, trusted=True).check(RECORD, ctx(l5_verdict(conf=0.5)))
    assert v.details["method"] == "votes" and len(judge.seen) == 5


def test_trusted_judge_without_l5_verdict_votes():
    judge = VoteJudge(["yes"] * 5)
    v = layer(judge, trusted=True).check(RECORD, ctx(recipe={"label": "yes", "contestable": True}))
    assert v.details["method"] == "votes"


def test_untrusted_judge_votes_even_when_confident():
    judge = VoteJudge(["yes"] * 5)
    v = layer(judge).check(RECORD, ctx(l5_verdict(conf=0.99)))
    assert v.details["method"] == "votes"


# ── answer_emergent ──────────────────────────────────────────────


def letter(text):
    return text.strip()[-1] if text and text.strip()[-1] in "ABCD" else None


def answer_layer(replies, k=5, **kw):
    replies = list(replies)
    return ConsistencyLayer(
        mode="answer_emergent",
        k=k,
        fields=("question",),
        answerer=lambda view, i: replies[i],
        extractor=letter,
        **kw,
    )


def test_answer_emergent_majority_becomes_the_answer():
    lay = answer_layer(["Answer: B", "Answer: B", "Answer: C", "so B", "Answer: A"])
    v = lay.check(RECORD, ctx(l5_verdict()))
    assert v.passed and v.details["answer"] == "B" and v.details["response"] == "Answer: B"
    assert v.details["votes"] == ["B", "B", "C", "B", "A"]


def test_answer_emergent_ignores_trust_and_always_votes():
    lay = answer_layer(["B"] * 5, trusted=True)
    assert lay.check(RECORD, ctx(l5_verdict(conf=0.99))).details["method"] == "votes"


def test_answer_emergent_unreadable_answers_abstain():
    # 2 of 2 readable votes, where the §12.1 bug would score 2/5.
    lay = answer_layer(["B", "no idea", None, "B", "hmm"])
    v = lay.check(RECORD, ctx(l5_verdict()))
    assert v.passed and v.details["answer"] == "B"
    assert v.details["cast"] == 2 and v.details["abstained"] == 3


def test_answer_emergent_without_majority_is_repairable():
    v = answer_layer(["A", "B", "C", "A", "B"]).check(RECORD, ctx(l5_verdict()))
    assert v.repairable and v.codes == ("consistency_no_majority",)


def test_answer_emergent_no_readable_answers_is_hard():
    v = answer_layer(["?", None, "x", "", "y"]).check(RECORD, ctx(l5_verdict()))
    assert v.hard and v.codes == ("consistency_no_votes",)


def test_answer_emergent_requires_extractor_and_answerer():
    with pytest.raises(ConsistencyError, match="extractor"):
        ConsistencyLayer(mode="answer_emergent", k=3, fields=("q",), answerer=lambda v, i: "A")
    with pytest.raises(ConsistencyError, match="answerer"):
        ConsistencyLayer(mode="answer_emergent", k=3, fields=("q",), extractor=letter)


def test_backend_answerer_cycles_temperatures():
    backend = MockBackend(["Answer: B"] * 4)
    answerer = BackendAnswerer(backend, suffix="Answer with a letter.", temperatures=(0.7, 0.9))
    lay = ConsistencyLayer(
        mode="answer_emergent", k=4, fields=("question",), answerer=answerer, extractor=letter
    )
    v = lay.check(RECORD, ctx(l5_verdict()))
    assert v.details["answer"] == "B"
    assert [c.temperature for c in backend.calls] == [0.7, 0.9, 0.7, 0.9]
    assert backend.calls[0].prompt == "Q?\n\nAnswer with a letter."


# ── construction and helpers ─────────────────────────────────────


def test_bad_configuration_raises():
    with pytest.raises(ConsistencyError, match="judge"):
        ConsistencyLayer(mode="label_first", k=3, fields=FIELDS)
    with pytest.raises(ConsistencyError, match="k >= 1"):
        layer(VoteJudge([]), k=0)
    with pytest.raises(ConsistencyError, match="field"):
        layer(VoteJudge([]), fields=("label",))


def test_majority_helper():
    assert majority([None, None]) == (None, 0, 0)
    assert majority(["a", None, "b", "a"]) == ("a", 2, 3)
    assert majority(["b", "a"])[0] == "b"  # ties go to the first seen


# ── cascade integration and FAG ──────────────────────────────────


def fake_l5(details):
    class L5:
        name = "L5"

        def check(self, record, context):
            return LayerVerdict("L5", "pass", (), details)

    return L5()


def test_cascade_hands_l6_the_l5_verdict():
    judge = VoteJudge(["no"] * 5)
    cascade = Cascade([fake_l5({"escalate": True, "agrees": True}), layer(judge)])
    result = cascade.run(RECORD, ValidationContext(recipe={"label": "yes"}))
    assert result.failed_layer == "L6" and result.repairable
    cascade = Cascade([fake_l5({"escalate": False}), layer(VoteJudge([]))])
    assert cascade.run(RECORD, ValidationContext(recipe={"label": "yes"})).passed


def judge_reply(verdict, conf):
    return json.dumps(
        {
            "verdict": verdict,
            "scores": {"advice_tier": "GENERAL_ADVICE", "realism": 4},
            "confidence": {"verdict": conf, "advice_tier": 0.8, "realism": 0.8},
        }
    )


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def seeds():
    return [json.loads(line) for line in (FAG_DIR / "seeds.jsonl").read_text().splitlines()]


def test_fag_contestable_seed_escalates_from_l5_and_votes(fag, seeds):
    seed = next(s for s in seeds if s.get("contestable") is True)
    assert fag.spec.validation.consistency_k >= 3
    k = fag.spec.validation.consistency_k
    l5_backend = MockBackend([judge_reply("breach", 0.95)])
    votes = ["not json"] + [judge_reply("breach", 0.6)] * (k - 1)
    l6_backend = MockBackend(votes)
    l5 = JudgeLayer.from_spec(fag, LLMJudge.from_spec(fag, l5_backend))
    l6 = ConsistencyLayer.from_spec(fag, judge=LLMJudge.from_spec(fag, l6_backend, parse_retries=0))
    result = Cascade([l5, l6]).run(seed, ValidationContext(recipe=seed))
    assert result.passed
    l6_verdict = result.verdicts[-1]
    assert l6_verdict.details["method"] == "votes"
    assert l6_verdict.details["abstained"] == 1 and l6_verdict.details["cast"] == k - 1
    # The judge never sees the label: every vote prompt is the same blind view.
    assert len({c.prompt for c in l6_backend.calls}) == 1
    assert '"label"' not in l6_backend.calls[0].prompt


def test_fag_contestable_seed_fails_when_votes_disagree(fag, seeds):
    seed = next(s for s in seeds if s.get("contestable") is True)
    k = fag.spec.validation.consistency_k
    l5 = JudgeLayer.from_spec(
        fag, LLMJudge.from_spec(fag, MockBackend([judge_reply("breach", 0.95)]))
    )
    l6_backend = MockBackend([judge_reply("no_breach", 0.9)] * k)
    l6 = ConsistencyLayer.from_spec(fag, judge=LLMJudge.from_spec(fag, l6_backend), trusted=False)
    result = Cascade([l5, l6]).run(seed, ValidationContext(recipe=seed))
    assert result.failed_layer == "L6" and result.errors[0].code == "consistency_disagrees"

    unused = MockBackend(lambda call: pytest.fail("a trusted, confident judge must not vote"))
    trusted = ConsistencyLayer.from_spec(fag, judge=LLMJudge.from_spec(fag, unused), trusted=True)
    l5 = JudgeLayer.from_spec(
        fag, LLMJudge.from_spec(fag, MockBackend([judge_reply("breach", 0.95)]))
    )
    result = Cascade([l5, trusted]).run(seed, ValidationContext(recipe=seed))
    assert result.passed and result.verdicts[-1].details["method"] == "confidence"
