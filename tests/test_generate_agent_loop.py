"""The generator's agent loop (FRAMEWORK_DESIGN.md §6.3, §7.4): tool calls run through the
gateway and are fed back until the model returns a record or the budget runs out."""

import json
import random
import shutil
from pathlib import Path

import pytest
from test_pipeline import NO_JUDGE, fag_reply, recipe_from_prompt

from sdgf.generate.generator import TOOL_RESULTS_HEADER, Generator, GeneratorError
from sdgf.models.base import ModelResponse, ToolCall
from sdgf.models.mock import MockBackend
from sdgf.pipeline import ACCEPTED_STREAM, Pipeline
from sdgf.spec.compile import Stage0Error, compile_spec
from sdgf.spec.schema import ToolUse
from sdgf.store.provenance import ModelRef, ProvenanceBuilder, split
from sdgf.tools.cache import ToolCache
from sdgf.tools.gateway import CALL_BUDGET_EXHAUSTED, NOT_ALLOWED, ToolGateway
from sdgf.tools.registry import ToolDefinition, ToolRegistry
from sdgf.validate.cascade import Cascade
from sdgf.validate.l3_governance import GovernanceLayer
from sdgf.validate.repair import RepairLoop

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CELL = {"product_scope": "corps_act", "label": True, "conversation_length": "single_turn"}
PRODUCT = {"product_code": "TST-001", "name": "Zebra Test Everyday Account"}
GENERATOR_REF = ModelRef(stage="generator", backend="mock", model="mock", hosting="local")
LOOKUP = ToolCall("catalogue_lookup", {"product_code": "TST-001"}, id="c1")


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def make_registry(sensitivity="public", handler=None):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="catalogue_lookup",
            description="Look up a fictional product by code.",
            input_schema={
                "type": "object",
                "properties": {"product_code": {"type": "string", "minLength": 1}},
                "required": ["product_code"],
                "additionalProperties": False,
            },
            sensitivity=sensitivity,
            handler=handler or (lambda a: dict(PRODUCT, product_code=a["product_code"])),
        )
    )
    return registry


def make_gateway(max_calls=5, cache=None, **kw):
    uses = [ToolUse(name="catalogue_lookup", max_calls_per_record=max_calls)]
    return ToolGateway(make_registry(**kw).for_task(uses), cache)


def final_reply(call):
    return json.dumps(fag_reply(recipe_from_prompt(call.prompt)))


# ── the loop ──────────────────────────────────────────────────────


def test_tool_call_is_run_and_fed_back_until_a_record(fag):
    recipe = Generator(fag, MockBackend(["x"])).recipe(CELL, random.Random(0))
    backend = MockBackend([LOOKUP, json.dumps(fag_reply(recipe))])
    gen = Generator(fag, backend, gateway=make_gateway())
    prompt = gen.prompts.build(recipe)
    result = gen.complete("c", recipe, prompt)

    assert result.ok, result.detail
    assert (result.tool_rounds, result.calls) == (1, 2)
    assert [r.tool for r in result.tool_results] == ["catalogue_lookup"]
    first, second = backend.calls
    assert first.prompt == prompt.text
    assert first.tools and first.tools[0]["name"] == "catalogue_lookup"
    # the tool round is appended after the prompt, so the static prefix is unchanged
    assert second.prompt.startswith(prompt.text)
    assert f"{TOOL_RESULTS_HEADER} (round 1)" in second.prompt
    assert '### catalogue_lookup {"product_code": "TST-001"}' in second.prompt
    assert "Zebra Test Everyday Account" in second.prompt
    # labels still come from the cell, never from the model
    assert result.record["label"] is True


def test_no_gateway_means_no_tools_offered_and_tool_calls_fail(fag):
    backend = MockBackend([LOOKUP])
    result = Generator(fag, backend).generate(CELL, random.Random(0))
    assert result.error == "tool_calls" and "allows none" in result.detail
    assert backend.calls[0].tools is None


def test_gateway_with_no_tools_is_treated_as_none(fag):
    gateway = ToolGateway(make_registry().for_task([]))
    gen = Generator(fag, MockBackend([LOOKUP]), gateway=gateway)
    assert gen.gateway is None and gen.session() is None
    assert gen.generate(CELL, random.Random(0)).error == "tool_calls"


def test_denied_call_is_fed_back_as_an_error_and_traced(fag):
    stray = ToolCall("delete_everything", {})
    replies = iter([stray])

    def reply(call):
        return next(replies, None) or final_reply(call)

    backend = MockBackend(reply)
    gen = Generator(fag, backend, gateway=make_gateway())
    session = gen.session()
    recipe = gen.recipe(CELL, random.Random(0))
    result = gen.complete(None, recipe, gen.prompts.build(recipe), session=session)
    assert result.ok
    assert result.tool_results[0].error == NOT_ALLOWED
    assert f"error ({NOT_ALLOWED})" in backend.calls[1].prompt
    assert session.trace[0].tool == "delete_everything" and session.trace[0].error


def test_exhausted_budget_stops_offering_tools_and_asks_for_the_record(fag):
    replies = iter([LOOKUP])

    def reply(call):
        return next(replies, None) or final_reply(call)

    backend = MockBackend(reply)
    gen = Generator(fag, backend, gateway=make_gateway(max_calls=1))
    result = gen.generate(CELL, random.Random(0))
    assert result.ok
    assert backend.calls[0].tools is not None
    assert backend.calls[1].tools is None
    assert "No tool budget is left" in backend.calls[1].prompt


def test_calls_past_the_budget_are_denied(fag):
    replies = iter([[LOOKUP, LOOKUP]])

    def reply(call):
        return next(replies, None) or final_reply(call)

    gen = Generator(fag, MockBackend(reply), gateway=make_gateway(max_calls=1))
    result = gen.generate(CELL, random.Random(0))
    assert result.ok and result.tool_rounds == 1
    assert [r.error for r in result.tool_results] == [None, CALL_BUDGET_EXHAUSTED]


def test_a_model_that_never_stops_calling_tools_hits_the_round_limit(fag):
    backend = MockBackend(lambda call: ToolCall("catalogue_lookup", {"product_code": ""}))
    gen = Generator(fag, backend, gateway=make_gateway(), max_tool_rounds=2)
    result = gen.generate(CELL, random.Random(0))
    assert result.error == "tool_rounds_exhausted"
    assert (result.tool_rounds, result.calls) == (2, 3)
    assert all(r.error == "bad_arguments" for r in result.tool_results)


def test_default_round_limit_covers_every_call_budget(fag):
    assert (
        Generator(fag, MockBackend(["x"]), gateway=make_gateway(max_calls=3)).max_tool_rounds == 4
    )
    assert Generator(fag, MockBackend(["x"])).max_tool_rounds == 1
    with pytest.raises(GeneratorError):
        Generator(fag, MockBackend(["x"]), gateway=make_gateway(), max_tool_rounds=-1)


def test_session_without_gateway_is_refused(fag):
    gen = Generator(fag, MockBackend(["x"]))
    recipe = gen.recipe(CELL, random.Random(0))
    with pytest.raises(GeneratorError):
        gen.complete(None, recipe, gen.prompts.build(recipe), session=make_gateway().session())


def test_text_alongside_tool_calls_still_runs_the_tools(fag):
    replies = iter([ModelResponse(text="let me check", tool_calls=(LOOKUP,))])

    def reply(call):
        return next(replies, None) or final_reply(call)

    result = Generator(fag, MockBackend(reply), gateway=make_gateway()).generate(
        CELL, random.Random(0)
    )
    assert result.ok and result.tool_rounds == 1


def test_mock_scripts_tool_calls():
    backend = MockBackend([LOOKUP, [LOOKUP, LOOKUP], "done"])
    assert backend.call("p", 1, 0.0).tool_calls == (LOOKUP,)
    assert backend.call("p", 1, 0.0).tool_calls == (LOOKUP, LOOKUP)
    assert backend.call("p", 1, 0.0) == ModelResponse(text="done")


# ── repair loop: shared session, L3 and provenance ────────────────


def tool_spec_dir(tmp_path, max_calls=1):
    task = tmp_path / "fag_tools"
    shutil.copytree(FAG_DIR, task, ignore=shutil.ignore_patterns("__pycache__"))
    with (task / "task.yaml").open("a") as f:
        f.write(f"\ntools:\n  - name: catalogue_lookup\n    max_calls_per_record: {max_calls}\n")
    return task


def test_sensitive_tool_data_reaches_l3_and_provenance(tmp_path):
    compiled = compile_spec(
        tool_spec_dir(tmp_path), tool_registry=make_registry(sensitivity="internal")
    )

    def reply(call):
        if TOOL_RESULTS_HEADER not in call.prompt:
            return LOOKUP
        body = fag_reply(recipe_from_prompt(call.prompt))
        body["customer_intent"] = f"Asks about the {PRODUCT['name']}."  # copies tool data
        return json.dumps(body)

    backend = MockBackend(reply)
    gateway = ToolGateway(
        make_registry(sensitivity="internal").for_task(compiled.spec), ToolCache()
    )
    gen = Generator(compiled, backend, gateway=gateway)
    cascade = Cascade([GovernanceLayer.from_spec(compiled)])
    loop = RepairLoop(gen, cascade, repair_tries=1)
    recipe = gen.recipe(CELL, random.Random(0))
    prov = ProvenanceBuilder(compiled.spec_version, "c", 0, [GENERATOR_REF])
    outcome = loop.run("c", recipe, gen.prompts.build(recipe), prov)
    # L3 is hard-fail, so the leak drops the slot rather than repairing it
    assert not outcome.accepted
    assert outcome.drop.layer == "L3" and "tool_data_leak" in outcome.drop.codes
    assert [e.tool for e in prov.tool_trace] == ["catalogue_lookup"]
    assert prov.tool_trace[0].sensitivity == "internal"


# ── pipeline: traces in provenance, shared cache ──────────────────


def test_pipeline_runs_the_agent_and_records_traces(tmp_path):
    task = tool_spec_dir(tmp_path)
    registry = make_registry()

    def reply(call):
        if TOOL_RESULTS_HEADER not in call.prompt:
            return LOOKUP
        return final_reply(call)

    pipe = Pipeline(
        task,
        tmp_path / "store",
        model_overrides={"generator": MockBackend(reply)},
        target_size=8,
        layers=NO_JUDGE,
        tool_registry=registry,
    )
    result = pipe.run("r1")
    assert result.complete and len(result.accepted) == 8
    for rec in pipe.store.open_run(pipe.compiled.spec_version, "r1").read_jsonl(ACCEPTED_STREAM):
        _, prov = split(rec)
        assert [t.tool for t in prov.tool_trace] == ["catalogue_lookup"]
        assert prov.tool_trace[0].result == PRODUCT
    cached = [split(r)[1].tool_trace[0].cached for r in result.accepted]
    assert cached.count(False) == 1  # one fresh call, then the shared cache answers
    cache_file = pipe.store.shared_jsonl_path(pipe.compiled.spec_version, "tool_cache")
    assert len(ToolCache(cache_file)) == 1


def test_pipeline_needs_listed_tools_in_the_registry(tmp_path):
    with pytest.raises(Stage0Error):
        Pipeline(
            tool_spec_dir(tmp_path),
            tmp_path / "store",
            model_overrides={"generator": MockBackend(["x"])},
            tool_registry=ToolRegistry(),
        )


def test_repair_tries_share_the_record_tool_budget(tmp_path):
    compiled = compile_spec(tool_spec_dir(tmp_path), tool_registry=make_registry())
    first_try = {"on": True}

    def reply(call):
        if call.tools is not None:
            return LOOKUP
        if first_try.pop("on", False):
            return "not json"  # a repairable generation failure
        return final_reply(call)

    backend = MockBackend(reply)
    gateway = ToolGateway(make_registry().for_task(compiled.spec), ToolCache())
    gen = Generator(compiled, backend, gateway=gateway)
    loop = RepairLoop(gen, Cascade([GovernanceLayer.from_spec(compiled)]), repair_tries=1)
    recipe = gen.recipe(CELL, random.Random(0))
    prov = ProvenanceBuilder(compiled.spec_version, "c", 0, [GENERATOR_REF])
    outcome = loop.run("c", recipe, gen.prompts.build(recipe), prov)
    assert outcome.accepted and outcome.attempts == 2
    # the one call the record may make was spent on the first try; the repair gets no tools
    assert [c.tools is not None for c in backend.calls] == [True, False, False]
    assert len(prov.tool_trace) == 1
