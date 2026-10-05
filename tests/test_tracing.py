"""Tracing a run (README "Tracing with LangSmith"): one trace per candidate, through the
Pipeline seam with RecordingTracer, plus the LangSmith adapter against a fake client.
No network and no API key."""

import json
import logging
from pathlib import Path

import pytest

from sdgf.cli import main
from sdgf.models.mock import MockBackend
from sdgf.pipeline import Pipeline
from sdgf.spec.compile import compile_spec
from sdgf.tracing import TRACE_SINKS_STAGE, RecordingTracer, TraceNode, layer_outputs
from sdgf.validate.base import ValidationContext
from sdgf.validate.l4_overlap import OverlapLayer
from test_m4_checkpoint import REPAIR, World
from test_pipeline import NO_JUDGE, fag_reply, filler, recipe_from_prompt

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
SINK = {
    "sink": "recorder",
    "endpoint": "memory://",
    "project": "p",
    "content": "full",
    "hosting": "local",
}


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def walk(node: TraceNode):
    yield node
    for c in node.children:
        yield from walk(c)


def reword_first_try(call) -> str:
    """Breach candidates get a paraphrased span on their first try (L2 sends them back)."""
    recipe = recipe_from_prompt(call.prompt)
    first = REPAIR not in call.prompt
    return json.dumps(fag_reply(recipe, reword=first and recipe["label"]))


def no_judge_run(fag, root, generator, tracer=None, **kw):
    pipe = Pipeline(
        fag,
        root,
        model_overrides={"generator": MockBackend(generator)},
        target_size=kw.pop("target_size", 8),
        layers=NO_JUDGE,
        tracer=tracer,
        **kw,
    )
    return pipe.run("t")


def artefacts(result) -> tuple[bytes, bytes]:
    run_dir = result.run.path
    return (run_dir / "accepted.jsonl").read_bytes(), (run_dir / "drops.jsonl").read_bytes()


# ── the span tree ────────────────────────────────────────────────


def test_a_candidate_sent_back_at_l2_then_accepted_has_one_attempt_span_per_try(fag, tmp_path):
    tracer = RecordingTracer()
    no_judge_run(fag, tmp_path / "s", reword_first_try, tracer)
    repaired = [r for r in tracer.roots if r.names() == ["attempt 0", "attempt 1"]]
    assert repaired, "no breach candidate was sent back at L2"
    root = repaired[0]

    first, second = root.child("attempt 0"), root.child("attempt 1")
    assert first.names() == ["generate", "L1 schema", "L2 rules"]
    assert first.child("generate").names() == ["generator:mock"]
    l2 = first.child("L2 rules").outputs
    assert l2["outcome"] == "fail_repairable"
    assert l2["codes"] == ["span_not_verbatim"]
    assert second.inputs["repair_feedback"] is not None
    assert second.names() == [
        "generate",
        "L1 schema",
        "L2 rules",
        "L3 safety scan",
        "L4 copy check",
    ]
    assert root.outputs == {"outcome": "accepted", "layer": None, "codes": [], "attempts": 2}
    assert root.feedback["accepted"] == 1
    assert root.feedback["attempts"] == 2


def test_a_model_call_span_carries_the_prompt_reply_stage_and_model(fag, tmp_path):
    tracer = RecordingTracer()
    no_judge_run(fag, tmp_path / "s", reword_first_try, tracer)
    call = tracer.roots[0].child("attempt 0").child("generate").child("generator:mock")
    assert call.run_type == "llm"
    assert "## " in call.inputs["prompt"]  # the full generation prompt is sent
    assert json.loads(call.outputs["text"])["messages"]
    assert call.metadata["stage"] == "generator"
    assert call.metadata["backend"] == "mock"


def test_the_root_names_the_run_cell_and_fixed_facts(fag, tmp_path):
    tracer = RecordingTracer()
    no_judge_run(fag, tmp_path / "s", reword_first_try, tracer)
    root = tracer.roots[0]
    assert root.metadata["run_id"] == "t"
    assert root.metadata["spec_version"] == fag.spec_version
    assert "label" in root.inputs["fixed_facts"]
    assert "task:fag" in root.tags and "run:t" in root.tags


def test_judge_verdict_confidence_and_votes_appear_on_the_l5_and_l6_spans(fag, tmp_path):
    tracer = RecordingTracer()
    world = World()
    Pipeline(
        fag, tmp_path / "s", model_overrides=world.backends(), target_size=20, tracer=tracer
    ).run("t")
    accepted = [r for r in tracer.roots if r.outputs["outcome"] == "accepted"]
    last = accepted[0].children[-1]
    l5 = last.child("L5 judge")
    assert l5.outputs["details"]["agrees"] is True
    assert l5.outputs["details"]["judge"]["confidence"]["verdict"] == 0.9
    assert l5.names() == ["judge:mock"]
    assert accepted[0].feedback["l5_agrees"] is True
    assert accepted[0].feedback["l5_confidence"] == 0.9

    voted = [
        n
        for r in tracer.roots
        for n in walk(r)
        if n.name == "L6 extra votes" and n.outputs["details"].get("method") == "votes"
    ]
    assert voted, "no escalated record took L6 votes"
    assert len(voted[0].children) == voted[0].outputs["details"]["k"] == 5
    assert len(voted[0].outputs["details"]["ballots"]) == 5


def test_a_record_dropped_by_settle_ends_dropped_at_l4_though_its_checks_passed(fag, tmp_path):
    def same_text(call) -> str:
        # valid replies whose per-recipe filler is replaced by fixed words, so candidates of
        # one cell read alike and a wave's later ones are near-copies of its first
        recipe = recipe_from_prompt(call.prompt)
        reply = fag_reply(recipe)
        for turn, message in enumerate(reply["messages"], start=1):
            message["content"] = message["content"].replace(
                filler(recipe, turn), "the same fixed words in every candidate"
            )
        return json.dumps(reply)

    tracer = RecordingTracer()
    no_judge_run(fag, tmp_path / "s", same_text, tracer, target_size=20, max_attempts_per_cell=3)
    overturned = [
        r
        for r in tracer.roots
        if r.outputs["outcome"] == "dropped" and r.children[-1].outputs["outcome"] == "pass"
    ]
    assert overturned, "no accepted candidate was dropped as a near-duplicate at settle"
    assert overturned[0].outputs["layer"] == "L4"
    assert overturned[0].outputs["codes"] == ["near_duplicate"]
    assert overturned[0].feedback["accepted"] == 0


# ── tracing never changes a run ──────────────────────────────────


def test_tracing_on_or_off_writes_the_same_artefacts(fag, tmp_path):
    off = no_judge_run(fag, tmp_path / "off", reword_first_try)
    on = no_judge_run(fag, tmp_path / "on", reword_first_try, RecordingTracer())
    assert artefacts(on) == artefacts(off)


class BrokenTracer(RecordingTracer):
    def _start(self, *args, **kwargs):
        raise RuntimeError("tracing service down")


def test_a_failing_tracer_is_logged_and_the_run_is_unchanged(fag, tmp_path, caplog):
    off = no_judge_run(fag, tmp_path / "off", reword_first_try)
    with caplog.at_level(logging.WARNING, logger="sdgf.tracing"):
        on = no_judge_run(fag, tmp_path / "on", reword_first_try, BrokenTracer())
    assert artefacts(on) == artefacts(off)
    assert "tracing service down" in caplog.text


# ── governance ───────────────────────────────────────────────────


def test_a_traced_run_records_the_trace_sink_as_a_data_destination(fag, tmp_path):
    result = no_judge_run(fag, tmp_path / "s", reword_first_try, RecordingTracer(sink=SINK))
    assert result.run.read_stage(TRACE_SINKS_STAGE) == [SINK]


def test_an_untraced_run_records_no_trace_sink(fag, tmp_path):
    result = no_judge_run(fag, tmp_path / "s", reword_first_try)
    assert not result.run.has_stage(TRACE_SINKS_STAGE)


def test_held_out_matches_never_reach_a_span(fag, tmp_path):
    seed = dict(fag.seeds[0])
    held = tmp_path / "eval_secret.jsonl"
    held.write_text(json.dumps(seed) + "\n", encoding="utf-8")
    layer = OverlapLayer.from_spec(fag, held_out_paths=[held])
    verdict = layer.check(seed, ValidationContext())
    assert "held_out_overlap" in verdict.codes

    payload = json.dumps(layer_outputs(verdict))
    assert "eval_secret" not in payload
    assert "held_out_overlap" in payload


# ── the LangSmith adapter ────────────────────────────────────────


class FakeLangSmithClient:
    def __init__(self):
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.feedback: list[tuple] = []

    def create_run(self, **kwargs):
        self.created.append(kwargs)

    def update_run(self, run_id, **kwargs):
        self.updated.append({"id": run_id, **kwargs})

    def create_feedback(self, run_id, key, **kwargs):
        self.feedback.append((run_id, key, kwargs))

    def flush(self):
        pass


def test_langsmith_tracer_posts_a_nested_tree_and_feedback(fag, tmp_path):
    pytest.importorskip("langsmith")
    from sdgf.tracing import LangSmithTracer

    client = FakeLangSmithClient()
    tracer = LangSmithTracer("sdgf-test", client=client, endpoint="https://example.invalid")
    no_judge_run(fag, tmp_path / "s", reword_first_try, tracer, target_size=2)

    by_id = {str(r["id"]): r for r in client.created}
    roots = [r for r in client.created if r.get("parent_run_id") is None]
    assert roots and all(r["name"].startswith("candidate ") for r in roots)
    llm = [r for r in client.created if r["run_type"] == "llm"]
    assert llm and all(str(r["parent_run_id"]) in by_id for r in llm)
    assert by_id[str(llm[0]["parent_run_id"])]["name"] == "generate"
    keys = {key for _, key, _ in client.feedback}
    assert {"accepted", "attempts"} <= keys
    assert tracer.sink["endpoint"] == "https://example.invalid"
    assert tracer.sink["content"] == "full"


def test_cli_trace_without_an_api_key_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    code = main(["run", str(FAG_DIR), "--store", str(tmp_path / "s"), "--trace", "langsmith"])
    assert code == 2
    err = capsys.readouterr().err
    assert "LANGSMITH_API_KEY" in err or "pip install sdgf[tracing]" in err
