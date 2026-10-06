import dataclasses
import json
from pathlib import Path

import pytest

from sdgf.models.base import ModelBackendError
from sdgf.store.provenance import ProvenanceBuilder

from sdgf.judge.interface import Judge, JudgeParseError, JudgeResult, compile_rubric
from sdgf.judge.llm_judge import LLMJudge
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.pipeline import Pipeline
from sdgf.spec.schema import (
    ConsistencyRules,
    EscalationRules,
    RubricSection,
    SpecValidationError,
    parse_spec,
)
from sdgf.tasktypes.sft_qa import MULTIPLE_CHOICE
from sdgf.validate.base import ValidationContext
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


FLAGGED = {"label": "yes", "contestable": True}  # fixed facts that flag for extra votes


def judged(recipe=FLAGGED, verdict="yes", conf=0.9, labels=None, record=RECORD):
    """A context whose L5 verdict comes from running the real blind judge layer (L5)."""
    context = ValidationContext(cell_id="c1", recipe=recipe)
    l5 = JudgeLayer(VoteJudge([verdict], conf=conf), fields=FIELDS, labels=labels)
    return dataclasses.replace(context, previous=(l5.check(record, context),))


def unflagged():
    """L5 confident and agreeing, on a cell neither hard nor contestable."""
    return judged(recipe={"label": "yes"})


def ctx(recipe=None):
    return ValidationContext(cell_id="c1", recipe=recipe or {"label": "yes"})


def layer(judge, **kw):
    kw.setdefault("fields", FIELDS)
    kw.setdefault("k", 5)
    return ConsistencyLayer(mode="label_first", judge=judge, **kw)


# ── escalation gate ──────────────────────────────────────────────


def test_not_escalated_passes_without_votes():
    judge = VoteJudge([])
    v = layer(judge).check(RECORD, unflagged())
    assert v.passed and v.details == {"escalated": False, "method": "skipped"}


def test_escalation_from_recipe_when_no_l5_verdict():
    judge = VoteJudge(["yes"] * 5)
    assert layer(judge).check(RECORD, ctx()).details["method"] == "skipped"
    for recipe in ({"label": "yes", "difficulty": "HARD"}, {"label": "yes", "contestable": True}):
        judge = VoteJudge(["yes"] * 5)
        v = layer(judge).check(RECORD, ctx(recipe=recipe))
        assert v.passed and v.details["method"] == "votes" and v.details["cast"] == 5


def test_escalation_rules_can_turn_off_hard_and_contestable():
    rules = EscalationRules(on_hard_cells=False, on_contestable=False)
    recipe = {"label": "yes", "difficulty": "hard", "contestable": True}
    v = layer(VoteJudge([]), escalation=rules).check(RECORD, ctx(recipe=recipe))
    assert v.details["method"] == "skipped"


def test_escalated_only_false_votes_on_every_record():
    judge = VoteJudge(["yes"] * 3)
    v = layer(judge, k=3, escalated_only=False).check(RECORD, unflagged())
    assert v.details["method"] == "votes" and v.details["cast"] == 3


# ── label_first K votes ──────────────────────────────────────────


def test_majority_matching_label_passes():
    v = layer(VoteJudge(["yes", "no", "yes", "yes", "unclear"])).check(RECORD, judged())
    assert v.passed
    assert v.details["agree"] == 3 and v.details["cast"] == 5 and v.details["majority"] == "yes"


def test_majority_against_label_is_sent_back():
    v = layer(VoteJudge(["no", "no", "yes", "no", "yes"])).check(RECORD, judged())
    assert v.repairable and v.codes == ("consistency_disagrees",)
    assert v.errors[0].details == {"agree": 2, "cast": 5, "majority": "no"}
    assert "fixed label" in v.errors[0].message


def test_two_of_four_tie_is_sent_back():
    v = layer(VoteJudge(["yes", "no", "yes", "no"]), k=4).check(RECORD, judged())
    assert v.repairable and v.codes == ("consistency_disagrees",)
    assert v.errors[0].details == {"agree": 2, "cast": 4, "majority": "yes"}


def test_split_non_label_votes_still_fail_without_label_majority():
    # "yes" is the most common single verdict but not more than half of the votes.
    v = layer(VoteJudge(["yes", "yes", "no", "unclear", "no"])).check(RECORD, judged())
    assert v.repairable


def test_unparseable_votes_are_abstentions_not_in_denominator():
    # §12.1: 2 agreeing of 2 cast passes; DS²-Instruct would score it 2/5 and drop it.
    judge = VoteJudge(["yes", None, None, "yes", None])
    v = layer(judge).check(RECORD, judged())
    assert v.passed
    assert v.details["cast"] == 2 and v.details["abstained"] == 3
    assert v.details["votes"] == ["yes", None, None, "yes", None]


class FlakyVoteJudge(VoteJudge):
    """Like VoteJudge, but "down" raises a backend error (the voting model failed)."""

    def judge(self, record):
        if self.verdicts[0] == "down":
            self.verdicts.pop(0)
            raise ModelBackendError("voting model API down")
        return super().judge(record)


def test_a_vote_whose_model_fails_is_an_abstention_not_a_crash():
    v = layer(FlakyVoteJudge(["yes", "down", "yes", "no", "yes"])).check(RECORD, judged())
    assert v.passed
    assert v.details["votes"] == ["yes", None, "yes", "no", "yes"]
    assert v.details["cast"] == 4 and v.details["abstained"] == 1
    assert v.details["ballots"][1]["error"] == "ModelBackendError: voting model API down"


def test_a_failed_vote_reaches_provenance_with_its_error():
    v = layer(FlakyVoteJudge(["yes", "down", "yes", "no", "yes"])).check(RECORD, judged())
    prov = ProvenanceBuilder(
        "sha256:" + "0" * 64,
        "c1",
        0,
        [{"stage": "generator", "backend": "mock", "model": "m", "hosting": "local"}],
    )
    prov.set_prompt("prompt")
    prov.add_layer_result("L6", v.outcome, v.messages(), ballots=v.details["ballots"])
    ballots = prov.build().layer_results[0].ballots
    assert ballots[1].vote is None
    assert ballots[1].error == "ModelBackendError: voting model API down"
    assert ballots[0].error is None


def test_abstentions_do_not_rescue_a_disagreement():
    v = layer(VoteJudge(["no", None, "yes", "no", None])).check(RECORD, judged())
    assert v.repairable and v.errors[0].details["cast"] == 3


def test_all_votes_unparseable_is_dropped():
    v = layer(VoteJudge([None] * 5)).check(RECORD, judged())
    assert v.hard and v.codes == ("consistency_no_votes",)
    assert v.details["cast"] == 0 and v.details["abstained"] == 5


def test_votes_are_blind_to_label_and_private_keys():
    judge = VoteJudge(["yes"] * 5)
    layer(judge).check(RECORD, judged())
    assert judge.seen == [{"question": "Q?", "answer": "A."}] * 5


def test_label_map_and_recipe_label_win():
    labels = {"yes": True, "no": False}
    record = dict(RECORD, label=False)  # the fixed facts own the label
    fixed = {"label": True, "contestable": True}
    lay = layer(VoteJudge(["yes"] * 3), k=3, labels=labels)
    assert lay.check(record, judged(fixed, labels=labels, record=record)).passed
    # A bool mapping only matches a bool label: 1 is not True.
    fixed = {"label": 1, "contestable": True}
    lay = layer(VoteJudge(["yes"] * 3), k=3, labels=labels)
    assert lay.check(record, judged(fixed, labels=labels, record=record)).repairable


def test_majority_is_read_by_meaning_through_the_labels_map():
    # No raw verdict wins (1 of 3 each), but "yes" and "unclear" both mean True:
    # 2 of 3 votes agree with the fixed label by meaning.
    labels = {"yes": True, "unclear": True, "no": False}
    fixed = {"label": True, "contestable": True}
    lay = layer(VoteJudge(["yes", "unclear", "no"]), k=3, labels=labels)
    v = lay.check(RECORD, judged(fixed, labels=labels))
    assert v.passed and v.details["agree"] == 2


# ── trusted judge: confidence replaces votes ─────────────────────


def test_trusted_judge_confident_at_the_threshold_passes_without_votes():
    # L5 agreed at confidence 0.70, exactly escalation.low_confidence: not low.
    unused = VoteJudge([])  # any vote would pop an empty script and raise
    v = layer(unused, trusted=True).check(RECORD, judged(conf=0.70))
    assert v.passed and v.details == {"escalated": True, "method": "confidence", "confidence": 0.7}


def test_trusted_judge_just_below_the_threshold_takes_votes():
    v = layer(VoteJudge(["yes"] * 5), trusted=True).check(RECORD, judged(conf=0.69))
    assert v.passed and v.details["method"] == "votes" and v.details["cast"] == 5


def test_trusted_judge_without_l5_verdict_votes():
    judge = VoteJudge(["yes"] * 5)
    v = layer(judge, trusted=True).check(RECORD, ctx(recipe={"label": "yes", "contestable": True}))
    assert v.details["method"] == "votes"


def test_untrusted_judge_votes_even_when_confident():
    judge = VoteJudge(["yes"] * 5)
    v = layer(judge).check(RECORD, judged(conf=0.99))
    assert v.details["method"] == "votes"


# ── answer_emergent ──────────────────────────────────────────────


letter = MULTIPLE_CHOICE.extractor  # the real sft_qa multiple_choice extractor


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
    v = lay.check(RECORD, judged())
    assert v.passed and v.details["answer"] == "B" and v.details["response"] == "Answer: B"
    assert v.details["votes"] == ["B", "B", "C", "B", "A"]


def test_answer_emergent_ignores_trust_and_always_votes():
    lay = answer_layer(["B"] * 5, trusted=True)
    assert lay.check(RECORD, judged(conf=0.99)).details["method"] == "votes"


def test_answer_emergent_unreadable_answers_abstain():
    # 2 of 2 readable votes, where the §12.1 bug would score 2/5.
    lay = answer_layer(
        ["Answer: B", "I am not sure.", None, "Answer: B", ""], answer_field="answer"
    )
    record = {"question": "Q?", "response": "Answer: B", "answer": "B"}
    v = lay.check(record, judged())
    assert v.passed and v.details["answer"] == "B"
    assert v.details["cast"] == 2 and v.details["abstained"] == 3


def test_answer_emergent_vote_whose_model_fails_is_an_abstention():
    def answerer(view, i):
        if i == 1:
            raise ModelBackendError("answering model API down")
        return "Answer: B"

    lay = ConsistencyLayer(
        mode="answer_emergent",
        k=3,
        fields=("question",),
        answerer=answerer,
        extractor=letter,
        answer_field="answer",
    )
    record = {"question": "Q?", "response": "Answer: B", "answer": "B"}
    v = lay.check(record, judged())
    assert v.passed and v.details["cast"] == 2 and v.details["abstained"] == 1


def test_answer_emergent_without_majority_is_repairable():
    v = answer_layer(["A", "B", "C", "A", "B"]).check(RECORD, judged())
    assert v.repairable and v.codes == ("consistency_no_majority",)


def test_answer_emergent_no_readable_answers_is_hard():
    v = answer_layer(["?", None, "x", "", "y"]).check(RECORD, judged())
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
    v = lay.check(RECORD, judged())
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


def test_cascade_hands_l6_the_l5_verdict():
    def l5():
        return JudgeLayer(VoteJudge(["yes"]), fields=FIELDS)

    judge = VoteJudge(["no"] * 5)
    cascade = Cascade([l5(), layer(judge)])
    result = cascade.run(RECORD, ValidationContext(recipe=FLAGGED))
    assert result.failed_layer == "L6" and result.repairable
    cascade = Cascade([l5(), layer(VoteJudge([]))])
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


# ── answer_emergent: the majority is the record's answer ─────────


def test_answer_emergent_record_answer_differing_from_the_majority_is_sent_back():
    replies = ["Answer: B", "Answer: B", "Answer: C", "Answer: B", "Answer: A"]
    lay = answer_layer(replies, answer_field="answer")
    record = {"question": "Q?", "response": "Answer: C", "answer": "C"}
    v = lay.check(record, judged())
    assert v.repairable and v.codes == ("consistency_answer_mismatch",)
    assert v.errors[0].path == "answer"
    assert v.errors[0].details == {"agree": 3, "cast": 5, "majority": "B", "answer": "C"}
    assert v.details["majority"] == "B"


def test_answer_emergent_record_answer_matching_the_majority_passes():
    lay = answer_layer(["B", "B", "B", "C", "A"], answer_field="answer")
    record = {"question": "Q?", "response": "Answer: B", "answer": "B"}
    v = lay.check(record, judged())
    assert v.passed and v.details["answer"] == "B"


def test_answer_field_is_hidden_from_the_voters():
    seen = []
    lay = ConsistencyLayer(
        mode="answer_emergent",
        k=1,
        fields=("question", "answer"),
        answerer=lambda view, i: seen.append(view) or "B",
        extractor=letter,
        answer_field="answer",
    )
    lay.check({"question": "Q?", "answer": "B"}, judged())
    assert seen == [{"question": "Q?"}]


def test_from_spec_checks_the_task_types_answer_under_answer_emergent():
    cfa = compile_spec(Path(__file__).resolve().parents[1] / "tasks" / "cfa")
    lay = ConsistencyLayer.from_spec(cfa, answerer=lambda view, i: "Answer: A")
    # escalation.always: voted with no L5 verdict and no hard or contestable facts
    record = {"question": "Q?\nA) x\nB) y", "response": "Answer: B", "answer": "B"}
    v = lay.check(record, ValidationContext(cell_id="c1", recipe={}))
    assert v.repairable and v.codes == ("consistency_answer_mismatch",)


# ── per-vote diversity (validation.consistency, models.consistency_judge) ──


def by_temperature(table):
    """A judge backend whose verdict depends only on the sampling temperature."""
    return MockBackend(lambda call: judge_reply(table[call.temperature], 0.6))


def with_validation(compiled, **update):
    validation = compiled.spec.validation.model_copy(update=update)
    return dataclasses.replace(
        compiled, spec=compiled.spec.model_copy(update={"validation": validation})
    )


def escalated(seed):
    return ValidationContext(recipe={**seed, "contestable": True})


def test_label_first_votes_cycle_temperatures_and_differ(fag, seeds):
    seed = next(s for s in seeds if s["label"] is True)
    spec = with_validation(
        fag, consistency_k=5, consistency=ConsistencyRules(temperatures=[0.7, 0.8, 0.9])
    )
    backend = by_temperature({0.7: "breach", 0.8: "no_breach", 0.9: "breach"})
    v = ConsistencyLayer.from_spec(spec, backend=backend).check(seed, escalated(seed))
    assert [(b["temperature"], b["vote"]) for b in v.details["ballots"]] == [
        (0.7, "breach"),
        (0.8, "no_breach"),
        (0.9, "breach"),
        (0.7, "breach"),
        (0.8, "no_breach"),
    ]
    # The majority is computed over the differing votes, not one repeated verdict.
    assert v.details["agree"] == 3 and v.details["majority"] == "breach"
    assert v.passed
    # Every vote is still blind: same view, no label.
    assert '"label"' not in backend.calls[0].prompt


def test_votes_differing_against_the_label_fail(fag, seeds):
    seed = next(s for s in seeds if s["label"] is True)
    backend = by_temperature({0.7: "no_breach", 0.8: "breach", 0.9: "no_breach"})
    v = ConsistencyLayer.from_spec(fag, backend=backend).check(seed, escalated(seed))
    assert set(v.details["votes"]) == {"breach", "no_breach"}
    assert v.details["majority"] == "no_breach"
    assert v.codes == ("consistency_disagrees",)


def test_configured_temperatures_are_cycled(fag, seeds):
    seed = next(s for s in seeds if s["label"] is True)
    spec = with_validation(
        fag, consistency_k=4, consistency=ConsistencyRules(temperatures=[0.2, 1.1])
    )
    backend = by_temperature({0.2: "breach", 1.1: "no_breach"})
    v = ConsistencyLayer.from_spec(spec, backend=backend).check(seed, escalated(seed))
    assert [c.temperature for c in backend.calls] == [0.2, 1.1, 0.2, 1.1]
    assert v.details["votes"] == ["breach", "no_breach", "breach", "no_breach"]
    assert v.codes == ("consistency_disagrees",)  # 2 of 4 is not a majority


def test_l6_does_not_reuse_the_l5_judge(fag, seeds):
    seed = next(s for s in seeds if s.get("contestable") is True)
    l5_backend = MockBackend([judge_reply("breach", 0.95)])
    votes = by_temperature({0.7: "breach", 0.8: "breach", 0.9: "breach"})
    l5 = JudgeLayer.from_spec(fag, LLMJudge.from_spec(fag, l5_backend))
    l6 = ConsistencyLayer.from_spec(fag, backend=votes)
    result = Cascade([l5, l6]).run(seed, ValidationContext(recipe=seed))
    assert result.passed and result.verdicts[-1].details["method"] == "votes"
    assert [c.temperature for c in l5_backend.calls] == [fag.spec.models.judge.temperature]
    assert len(votes.calls) == fag.spec.validation.consistency_k


def test_explicit_voters_are_cycled_and_exclusive_with_judge():
    a, b = VoteJudge(["yes"] * 3), VoteJudge(["no"] * 2)
    lay = ConsistencyLayer(mode="label_first", k=5, voters=(a, b), fields=FIELDS)
    v = lay.check(RECORD, judged())
    assert v.details["votes"] == ["yes", "no", "yes", "no", "yes"] and v.passed
    with pytest.raises(ConsistencyError, match="not both"):
        ConsistencyLayer(mode="label_first", k=3, judge=a, voters=(b,), fields=FIELDS)


def test_consistency_temperatures_are_validated(fag):
    data = fag.spec.model_dump(mode="json")
    for bad in ([], [2.5], [-0.1]):
        data["validation"]["consistency"] = {"temperatures": bad}
        with pytest.raises(SpecValidationError, match="temperatures"):
            parse_spec(data)
    assert fag.spec.validation.consistency.temperatures == [0.7, 0.8, 0.9]


def test_answer_emergent_answerer_cycles_the_configured_temperatures():
    cfa = compile_spec(Path(__file__).resolve().parents[1] / "tasks" / "cfa")
    spec = with_validation(
        cfa,
        consistency_k=5,
        consistency=ConsistencyRules(temperatures=[0.3, 0.6]),
        escalation=EscalationRules(always=True),
    )
    lay = ConsistencyLayer.from_spec(spec, backend=MockBackend(["Answer: A"], cycle=True))
    record = {"question": "Q?\nA) x\nB) y", "response": "Answer: A", "answer": "A"}
    v = lay.check(record, ValidationContext(cell_id="c1", recipe={}))
    assert v.passed
    assert [(b["stage"], b["temperature"]) for b in v.details["ballots"]] == [
        ("judge", 0.3),
        ("judge", 0.6),
        ("judge", 0.3),
        ("judge", 0.6),
        ("judge", 0.3),
    ]


def test_pipeline_votes_on_the_consistency_judge_stage(fag, seeds, tmp_path):
    models = fag.spec.models
    voter = models.judge.model_copy(update={"model": "voter"})
    spec = dataclasses.replace(
        fag,
        spec=fag.spec.model_copy(
            update={"models": models.model_copy(update={"consistency_judge": voter})}
        ),
    )
    l5_backend = MockBackend([judge_reply("breach", 0.95)], cycle=True)
    votes = by_temperature({0.7: "breach", 0.8: "no_breach", 0.9: "breach"})
    pipe = Pipeline(
        spec,
        tmp_path / "store",
        model_overrides={
            "generator": MockBackend(["{}"]),
            "judge": l5_backend,
            "consistency_judge": votes,
        },
        layers=["L5", "L6"],
    )
    assert pipe.used_stages == ("generator", "judge", "consistency_judge")
    seed = next(s for s in seeds if s.get("contestable") is True)
    result = pipe.cascade.run(seed, ValidationContext(recipe=seed))
    assert result.verdicts[-1].details["method"] == "votes"
    assert len(l5_backend.calls) == 1
    assert len(votes.calls) == spec.spec.validation.consistency_k
    assert {"stage": "consistency_judge", "backend": "mock", "model": "mock"} == {
        k: v for k, v in pipe.models.endpoints()[-1].items() if k != "hosting"
    }

    # With L5 off and a consistency_judge set, the L5 judge model gets no data.
    alone = Pipeline(
        spec,
        tmp_path / "store2",
        model_overrides={"generator": MockBackend(["{}"]), "consistency_judge": votes},
        layers=["L6"],
    )
    assert alone.used_stages == ("generator", "consistency_judge")


# ── per-vote provenance (details["ballots"], layer_results[].ballots) ──


def test_label_first_ballots_name_each_votes_stage_model_and_temperature(fag, seeds):
    seed = next(s for s in seeds if s["label"] is True)
    spec = with_validation(fag, consistency_k=4)
    backend = by_temperature({0.7: "breach", 0.8: "no_breach", 0.9: "breach"})
    v = ConsistencyLayer.from_spec(spec, backend=backend).check(seed, escalated(seed))
    assert v.details["ballots"] == [
        {"stage": "judge", "model": "mock", "temperature": t, "vote": vote}
        for t, vote in [(0.7, "breach"), (0.8, "no_breach"), (0.9, "breach"), (0.7, "breach")]
    ]
    assert [b["vote"] for b in v.details["ballots"]] == v.details["votes"]


def test_ballots_follow_explicit_voters_and_record_abstentions():
    a = LLMJudge(
        schema(),
        MockBackend(["not json"], cycle=True, model="voter-a"),
        fields=FIELDS,
        temperature=0.4,
        parse_retries=0,
        stage="consistency_judge",
    )
    b = VoteJudge(["yes", "yes"])  # a hand-built judge reports no stage, model or temperature
    v = ConsistencyLayer(mode="label_first", k=3, voters=(a, b), fields=FIELDS).check(
        RECORD, judged()
    )
    assert v.details["ballots"] == [
        {"stage": "consistency_judge", "model": "voter-a", "temperature": 0.4, "vote": None},
        {"stage": None, "model": None, "temperature": None, "vote": "yes"},
        {"stage": "consistency_judge", "model": "voter-a", "temperature": 0.4, "vote": None},
    ]


def test_answer_emergent_ballots_cycle_the_answerers_temperatures():
    backend = MockBackend(["Answer: B", "Answer: B", "no idea"], model="answerer")
    answerer = BackendAnswerer(backend, temperatures=(0.3, 0.6), stage="consistency_judge")
    lay = ConsistencyLayer(
        mode="answer_emergent", k=3, fields=("question",), answerer=answerer, extractor=letter
    )
    v = lay.check(RECORD, judged())
    assert v.details["ballots"] == [
        {"stage": "consistency_judge", "model": "answerer", "temperature": 0.3, "vote": "B"},
        {"stage": "consistency_judge", "model": "answerer", "temperature": 0.6, "vote": "B"},
        {"stage": "consistency_judge", "model": "answerer", "temperature": 0.3, "vote": None},
    ]
    # A bare answerer callable records only the votes.
    plain = answer_layer(["Answer: B"] * 5).check(RECORD, judged())
    assert plain.details["ballots"][0] == {
        "stage": None,
        "model": None,
        "temperature": None,
        "vote": "B",
    }


def test_cascade_writes_ballots_into_provenance(fag, seeds):
    from sdgf.store.provenance import Ballot, Provenance, ProvenanceBuilder

    seed = next(s for s in seeds if s["label"] is True)
    spec = with_validation(fag, consistency_k=3)
    backend = by_temperature({0.7: "breach", 0.8: "no_breach", 0.9: "breach"})
    lay = ConsistencyLayer.from_spec(spec, backend=backend)
    prov = ProvenanceBuilder(
        "v1", "c1", 0, [{"stage": "generator", "backend": "mock", "model": "g", "hosting": "local"}]
    )
    prov.set_prompt("p")
    assert Cascade([lay]).run(seed, escalated(seed), prov).passed
    (l6,) = prov.build().layer_results
    assert l6.ballots == (
        Ballot("breach", "judge", "mock", 0.7),
        Ballot("no_breach", "judge", "mock", 0.8),
        Ballot("breach", "judge", "mock", 0.9),
    )
    # The ballots survive the JSON round trip of the accepted stream.
    stored = json.loads(json.dumps(prov.build().to_dict()))
    assert stored["layer_results"][0]["ballots"][1] == {
        "vote": "no_breach",
        "stage": "judge",
        "model": "mock",
        "temperature": 0.8,
    }
    assert Provenance.from_dict(stored).layer_results[0].ballots == l6.ballots


def test_pipeline_ballots_name_the_consistency_judge_model(fag, seeds, tmp_path):
    models = fag.spec.models
    voter = models.judge.model_copy(update={"model": "voter"})
    spec = dataclasses.replace(
        fag,
        spec=fag.spec.model_copy(
            update={"models": models.model_copy(update={"consistency_judge": voter})}
        ),
    )
    votes = by_temperature({0.7: "breach", 0.8: "no_breach", 0.9: "breach"})
    pipe = Pipeline(
        spec,
        tmp_path / "store",
        model_overrides={"generator": MockBackend(["{}"]), "consistency_judge": votes},
        layers=["L6"],
    )
    seed = next(s for s in seeds if s.get("contestable") is True)
    result = pipe.cascade.run(seed, ValidationContext(recipe=seed))
    ballots = result.verdicts[-1].details["ballots"]
    assert {b["stage"] for b in ballots} == {"consistency_judge"}
    assert [b["temperature"] for b in ballots[:3]] == [0.7, 0.8, 0.9]
