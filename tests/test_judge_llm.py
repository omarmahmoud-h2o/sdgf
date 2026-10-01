import dataclasses
import json
from pathlib import Path

import pytest

from sdgf.judge.interface import JudgeError, JudgeParseError, JudgeResult, compile_rubric
from sdgf.judge.llm_judge import RECORD_HEADER, LLMJudge, judge_view
from sdgf.models.base import ModelResponse
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import RubricExample, RubricSection
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


# ── judge_context ────────────────────────────────────────────────

JUDGE_CONTEXT = "Judge-only definitions: a breach is advice about the customer's own position."


def with_judge_context(fag, text):
    spec = fag.spec.model_copy(
        update={"rubric": fag.spec.rubric.model_copy(update={"judge_context": text})}
    )
    return dataclasses.replace(fag, spec=spec)


def test_judge_context_unset_falls_back_to_task_description(fag):
    assert fag.spec.rubric.judge_context is None
    judge = LLMJudge.from_spec(fag, MockBackend([fag_reply()]))
    assert judge.context == fag.spec.task.description.strip()
    assert "## Context\n" + fag.spec.task.description.strip() in judge.static_prefix


def test_judge_context_replaces_task_description_in_judge_prompt_only(fag):
    from sdgf.generate.prompts import build_static_prefix
    from sdgf.tasktypes.registry import REGISTRY

    compiled = with_judge_context(fag, JUDGE_CONTEXT)
    backend = MockBackend([fag_reply()], cycle=True)
    judge = LLMJudge.from_spec(compiled, backend)
    judge.judge(fag.seeds[0])
    prompt = backend.calls[0].prompt
    description = fag.spec.task.description.strip()
    assert "## Context\n" + JUDGE_CONTEXT in prompt
    assert description not in prompt

    generation = build_static_prefix(compiled, REGISTRY.resolve(compiled.spec.task), [])
    assert description in generation
    assert JUDGE_CONTEXT not in generation


def test_judge_context_prompt_still_blind_to_the_label(fag):
    compiled = with_judge_context(fag, JUDGE_CONTEXT)
    backend = MockBackend([fag_reply()], cycle=True)
    judge = LLMJudge.from_spec(compiled, backend)
    for seed in fag.seeds:
        judge.judge(seed)
        judge.judge({**seed, "label": not seed["label"]})
        assert backend.calls[-2].prompt == backend.calls[-1].prompt
        assert '"label"' not in json.dumps(record_section(backend.calls[-1].prompt))


def test_judge_context_from_yaml(tmp_path):
    import shutil

    import yaml

    task_dir = tmp_path / "fag"
    shutil.copytree(FAG_TASK.parent, task_dir)
    path = task_dir / "task.yaml"
    data = yaml.safe_load(path.read_text())
    data["rubric"]["judge_context"] = JUDGE_CONTEXT
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    compiled = compile_spec(path)
    assert compiled.spec.rubric.judge_context == JUDGE_CONTEXT
    assert LLMJudge.from_spec(compiled, MockBackend([fag_reply()])).context == JUDGE_CONTEXT


# ── rubric.examples ──────────────────────────────────────────────

EXAMPLES = [
    RubricExample(
        record={"question": "Is 2 even?", "answer": "Yes."},
        verdict="yes",
        scores={"tier": "A", "realism": 5},
        note="A correct answer.",
    ),
    RubricExample(record={"question": "Is 3 even?"}, verdict="no"),
]


def test_examples_render_in_static_prefix_before_the_record():
    judge, backend = make([reply()], examples=EXAMPLES)
    judge.judge(RECORD)
    prompt = backend.calls[0].prompt
    assert prompt.startswith(judge.static_prefix)
    section = judge.static_prefix.split("## Worked examples\n", 1)[1]
    assert judge.static_prefix.index("## Rubric") < judge.static_prefix.index("## Worked")
    assert '### Example 1\nRecord: {"answer": "Yes.", "question": "Is 2 even?"}' in section
    assert 'Expected: {"scores": {"realism": 5, "tier": "A"}, "verdict": "yes"}' in section
    assert "Why: A correct answer." in section
    assert '### Example 2\nRecord: {"question": "Is 3 even?"}\nExpected: {"verdict": "no"}' in (
        section
    )
    assert prompt.index("## Worked examples") < prompt.index(RECORD_HEADER)
    assert record_section(prompt) == {"question": "Q?", "answer": "A."}


def test_no_examples_no_section():
    judge, _ = make([reply()])
    assert "## Worked examples" not in judge.static_prefix


def test_examples_may_show_only_judge_fields_and_known_verdicts():
    with pytest.raises(JudgeError, match="may not see"):
        make([reply()], examples=[RubricExample(record={"label": "yes"}, verdict="yes")])
    with pytest.raises(JudgeError, match="unknown verdict"):
        make([reply()], examples=[RubricExample(record={"question": "Q"}, verdict="maybe")])


def with_examples(fag, examples):
    spec = fag.spec.model_copy(
        update={"rubric": fag.spec.rubric.model_copy(update={"examples": examples})}
    )
    return dataclasses.replace(fag, spec=spec)


def fag_examples(fag):
    seed = fag.seeds[0]
    verdict = fag.spec.rubric.verdict.values[0]
    return [RubricExample(record={"messages": seed["messages"]}, verdict=verdict, note="N.")]


def test_examples_in_judge_prompt_only_and_label_still_blind(fag):
    from sdgf.generate.prompts import build_static_prefix
    from sdgf.tasktypes.registry import REGISTRY

    compiled = with_examples(fag, fag_examples(fag))
    backend = MockBackend([fag_reply()], cycle=True)
    judge = LLMJudge.from_spec(compiled, backend)
    assert judge.examples == tuple(compiled.spec.rubric.examples)
    for seed in fag.seeds:
        judge.judge(seed)
        judge.judge({**seed, "label": not seed["label"]})
        assert backend.calls[-2].prompt == backend.calls[-1].prompt
        assert "## Worked examples" in backend.calls[-1].prompt
        assert '"label"' not in json.dumps(record_section(backend.calls[-1].prompt))
    generation = build_static_prefix(compiled, REGISTRY.resolve(compiled.spec.task), [])
    assert "Worked examples" not in generation and "Why: N." not in generation


def test_examples_from_yaml(tmp_path):
    import shutil

    import yaml

    task_dir = tmp_path / "fag"
    shutil.copytree(FAG_TASK.parent, task_dir)
    path = task_dir / "task.yaml"
    data = yaml.safe_load(path.read_text())
    verdict = data["rubric"]["verdict"]["values"][0]
    data["rubric"]["examples"] = [
        {"record": {"messages": [{"role": "customer", "content": "Hi"}]}, "verdict": verdict}
    ]
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    compiled = compile_spec(path)
    judge = LLMJudge.from_spec(compiled, MockBackend([fag_reply()]))
    assert '### Example 1\nRecord: {"messages": [{"content": "Hi", "role": "customer"}]}' in (
        judge.static_prefix
    )
