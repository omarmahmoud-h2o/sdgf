import json
from pathlib import Path

import pytest

from sdgf.judge.interface import JudgeError, JudgeParseError, JudgeResult, compile_rubric
from sdgf.judge.llm_judge import RECORD_HEADER, LLMJudge, judge_view
from sdgf.models.base import ModelResponse
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import RubricSection
from sdgf.tasktypes.base import TaskType
from sdgf.tasktypes.registry import get_task_type

FAG_TASK = Path(__file__).resolve().parents[1] / "tasks" / "fag" / "task.yaml"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_TASK)


def schema(reason="never"):
    return compile_rubric(
        RubricSection(
            verdict={"values": ["yes", "no"], "description": "is it a yes?"},
            criteria=[
                {"name": "tier", "values": ["A", "B"], "description": "which tier"},
                {"name": "realism", "min": 1, "max": 5},
            ],
            reason_required=reason,
        )
    )


def reply(verdict="yes", tier="A", realism=4, conf=None):
    c = {"verdict": 0.9, "tier": 0.8, "realism": 0.7, **(conf or {})}
    return json.dumps(
        {"verdict": verdict, "scores": {"tier": tier, "realism": realism}, "confidence": c}
    )


RECORD = {"question": "Q?", "answer": "A.", "label": "yes", "_provenance": {"x": 1}}


def make(responses, **kw):
    backend = MockBackend(responses)
    kw.setdefault("fields", ("question", "answer"))
    return LLMJudge(schema(), backend, **kw), backend


def record_section(prompt):
    return json.loads(prompt.split(RECORD_HEADER + "\n", 1)[1].split("\n\n", 1)[0])


# ── view and prompt ──────────────────────────────────────────────


def test_judge_view_keeps_only_listed_public_fields():
    assert judge_view(RECORD, ("question", "label", "_provenance", "missing")) == {
        "question": "Q?",
        "label": "yes",
    }
    assert judge_view(RECORD, ("question", "answer")) == {"question": "Q?", "answer": "A."}


def test_default_judge_fields_drop_the_label():
    class QA(TaskType):
        name = "qa_toy"
        generation_modes = ("label_first",)

        def base_schema(self):
            return {"type": "object", "properties": {"question": {}, "label": {}, "answer": {}}}

    assert QA().judge_fields() == ("question", "answer")
    assert get_task_type("classification_spans").judge_fields() == ("messages",)


def test_prompt_has_rubric_schema_and_record_last():
    judge, _ = make([reply()], context="Some policy context.")
    p = judge.judge_prompt(RECORD)
    assert p.startswith(judge.static_prefix)
    assert "Some policy context." in p and "is it a yes?" in p and "which tier" in p
    assert "integer 1..5" in p and '"confidence"' in p
    assert '"reason"' not in p  # decision schema only
    assert p.index("## Output format") < p.index(RECORD_HEADER)
    assert record_section(p) == {"question": "Q?", "answer": "A."}


def test_static_prefix_shared_across_records():
    judge, _ = make([reply()])
    a = judge.judge_prompt({"question": "one", "answer": "x"})
    b = judge.judge_prompt({"question": "two", "answer": "y"})
    assert a.split(RECORD_HEADER)[0] == b.split(RECORD_HEADER)[0]


def test_prompt_never_contains_the_label():
    judge, backend = make([reply()] * 2)
    judge.judge({**RECORD, "label": "SECRET-LABEL-yes"})
    judge.judge({**RECORD, "label": "SECRET-LABEL-no"})
    assert all("SECRET-LABEL" not in c.prompt for c in backend.calls)
    assert backend.calls[0].prompt == backend.calls[1].prompt
    assert all("_provenance" not in c.prompt for c in backend.calls)


def test_empty_fields_rejected():
    with pytest.raises(JudgeError, match="at least one"):
        LLMJudge(schema(), MockBackend([reply()]), fields=())
    with pytest.raises(JudgeError, match="parse_retries"):
        LLMJudge(schema(), MockBackend([reply()]), fields=("q",), parse_retries=-1)


# ── parsing ──────────────────────────────────────────────────────


def test_parses_verdict_scores_and_confidence():
    judge, backend = make([reply("no", "B", 2, {"verdict": 0.55})], max_tokens=77, temperature=0.1)
    r = judge.judge(RECORD)
    assert r == JudgeResult("no", {"tier": "B", "realism": 2}, r.confidence)
    assert r.verdict_confidence == 0.55 and r.confidence["realism"] == 0.7
    assert r.reason is None
    assert (backend.calls[0].max_tokens, backend.calls[0].temperature) == (77, 0.1)


def test_tolerates_fences_and_chatter():
    judge, _ = make([f"Sure!\n```json\n{reply()}\n```"])
    assert judge.judge(RECORD).verdict == "yes"


def test_retries_with_errors_then_succeeds():
    bad = reply(tier="Z")
    judge, backend = make(["not json", bad, reply()], parse_retries=2)
    assert judge.judge(RECORD).verdict == "yes"
    assert len(backend.calls) == 3
    first, second, third = (c.prompt for c in backend.calls)
    assert "previous reply was rejected" not in first
    assert "not a parseable JSON object" in second
    assert "scores.tier" in third
    assert third.startswith(first)  # feedback appended; cached prefix unchanged
    assert "SECRET" not in third


def test_raises_after_retries_exhausted():
    judge, backend = make(["nope", reply(verdict="maybe")], parse_retries=1)
    with pytest.raises(JudgeParseError) as e:
        judge.judge(RECORD)
    assert any("verdict" in err for err in e.value.errors)
    assert len(backend.calls) == 2


def test_no_retries_and_no_text():
    judge, backend = make([ModelResponse(text=None)], parse_retries=0)
    with pytest.raises(JudgeParseError):
        judge.judge(RECORD)
    assert len(backend.calls) == 1


def test_reason_in_decision_reply_is_rejected():
    extra = json.loads(reply())
    extra["reason"] = "because"
    judge, _ = make([json.dumps(extra)], parse_retries=0)
    with pytest.raises(JudgeParseError, match="reason"):
        judge.judge(RECORD)


# ── reason-writing fallback ──────────────────────────────────────


def test_explain_writes_a_reason_as_fallback_judge():
    judge, backend = make([reply(), "  The answer says yes plainly.  "])
    assert judge.writes_reasons
    r = judge.judge(RECORD)
    reason = judge.explain(RECORD, r)
    assert reason == "The answer says yes plainly."
    assert r.with_reason(reason).to_dict()["reason"] == reason
    p = backend.calls[1].prompt
    assert '"verdict": "yes"' in p and "## Decision" in p
    assert record_section(p) == {"question": "Q?", "answer": "A."}


def test_explain_empty_reply_raises():
    judge, _ = make(["   "])
    with pytest.raises(JudgeError, match="empty reason"):
        judge.explain(RECORD, JudgeResult("yes", {"tier": "A", "realism": 3}, {}))


# ── FAG ──────────────────────────────────────────────────────────


def fag_reply(verdict="breach"):
    return json.dumps(
        {
            "verdict": verdict,
            "scores": {"advice_tier": "PERSONAL_ADVICE", "realism": 4},
            "confidence": {"verdict": 0.9, "advice_tier": 0.8, "realism": 0.6},
        }
    )


def test_fag_from_spec_uses_judge_config_and_messages_only(fag):
    backend = MockBackend([fag_reply()])
    judge = LLMJudge.from_spec(fag, backend)
    cfg = fag.spec.models.judge
    assert judge.fields == ("messages",)
    assert (judge.max_tokens, judge.temperature) == (cfg.max_tokens, cfg.temperature)
    seed = fag.seeds[0]
    r = judge.judge(seed)
    assert r.verdict == "breach" and r.scores["advice_tier"] == "PERSONAL_ADVICE"
    assert record_section(backend.calls[0].prompt) == {"messages": seed["messages"]}


def test_fag_prompt_blind_to_label_and_label_encoding_fields(fag):
    """Flipping the label and every field that encodes it leaves the prompt unchanged."""
    backend = MockBackend([fag_reply()], cycle=True)
    judge = LLMJudge.from_spec(fag, backend)
    for seed in fag.seeds:
        judge.judge(seed)
        flipped = {
            **seed,
            "label": not seed["label"],
            "advice_tier": "FACTUAL_INFORMATION",
            "severity": None if seed["label"] else "HIGH",
            "signal_categories": [],
            "problematic_spans": [],
            "policy_categories": {},
        }
        judge.judge(flipped)
        a, b = backend.calls[-2].prompt, backend.calls[-1].prompt
        assert a == b
        section = json.dumps(record_section(a))
        for key in ("label", "advice_tier", "severity", "signal_categories", "problematic"):
            assert f'"{key}' not in section
