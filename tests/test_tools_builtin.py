"""Built-in tools (FRAMEWORK_DESIGN.md §7.4): a fictional product catalogue and a
calculator, both read-only and public, and exact replay of a record from its tool trace."""

import dataclasses
import json
import random
import shutil
from pathlib import Path

import pytest
from test_pipeline import NO_JUDGE, fag_reply, recipe_from_prompt

from sdgf.generate.generator import TOOL_RESULTS_HEADER, Generator
from sdgf.governance.pii import RegexPIIScanner
from sdgf.models.base import ToolCall
from sdgf.models.mock import MockBackend
from sdgf.pipeline import Pipeline
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import ToolUse
from sdgf.store.provenance import split
from sdgf.tools.builtin import (
    CALCULATOR,
    CATALOGUE_LOOKUP,
    DEFAULT_CATALOGUE,
    builtin_tools,
    calculate,
    load_catalogue,
    register_builtin_tools,
)
from sdgf.tools.cache import ToolCache
from sdgf.tools.gateway import NO_HANDLER, TOOL_FAILED, ToolGateway
from sdgf.tools.registry import REGISTRY, ToolRegistry

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CELL = {"product_scope": "corps_act", "label": True, "conversation_length": "single_turn"}
LOOKUP = ToolCall(CATALOGUE_LOOKUP, {"product_code": "TST-TD-001"}, id="c1")
CALC = ToolCall(CALCULATOR, {"expression": "5000 * 4.1 / 100"}, id="c2")
USES = [ToolUse(name=CATALOGUE_LOOKUP), ToolUse(name=CALCULATOR)]


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def registry(**kw):
    reg = ToolRegistry()
    register_builtin_tools(reg, **kw)
    return reg


def call(reg, tool_call):
    return ToolGateway(reg.for_task(USES)).session().call(tool_call)


# ── registration ─────────────────────────────────────────────────


def test_builtins_are_registered_read_only_and_public():
    for name in (CATALOGUE_LOOKUP, CALCULATOR):
        tool = REGISTRY.get(name)
        assert tool.read_only and tool.sensitivity == "public" and tool.handler is not None
    assert [t.name for t in builtin_tools()] == [CATALOGUE_LOOKUP, CALCULATOR]


def test_register_twice_needs_replace():
    reg = registry()
    with pytest.raises(Exception, match="already registered"):
        register_builtin_tools(reg)
    register_builtin_tools(reg, replace=True)


# ── catalogue lookup ─────────────────────────────────────────────


def test_lookup_returns_the_product():
    result = call(registry(), LOOKUP)
    assert result.ok and result.sensitivity == "public"
    assert result.result["found"] is True
    assert result.result["product"]["name"] == "Zebra Test Business Term Deposit"


def test_lookup_normalises_the_code():
    result = call(registry(), ToolCall(CATALOGUE_LOOKUP, {"product_code": " tst-acc-001 "}))
    assert result.result["product"]["product_code"] == "TST-ACC-001"


def test_unknown_code_is_a_result_not_a_failure():
    result = call(registry(), ToolCall(CATALOGUE_LOOKUP, {"product_code": "NOPE"}))
    assert result.ok and result.result["found"] is False
    assert "TST-ACC-001" in result.result["known_codes"]


def test_lookup_rejects_bad_arguments():
    result = call(registry(), ToolCall(CATALOGUE_LOOKUP, {"code": "TST-ACC-001"}))
    assert result.error == "bad_arguments"


def test_catalogue_from_another_fixture(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"products": [{"product_code": "TST-X", "name": "Test X"}]}))
    result = call(
        registry(catalogue_path=path), ToolCall(CATALOGUE_LOOKUP, {"product_code": "tst-x"})
    )
    assert result.result == {"found": True, "product": {"product_code": "TST-X", "name": "Test X"}}


@pytest.mark.parametrize(
    "content, match",
    [
        ({"items": []}, "'products' list"),
        ({"products": [{"name": "no code"}]}, "no product_code"),
        ({"products": [{"product_code": "A"}, {"product_code": "A"}]}, "duplicate"),
    ],
)
def test_bad_catalogue_fixtures_raise(tmp_path, content, match):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=match):
        load_catalogue(path)


def test_default_fixture_is_fictional_and_pii_free():
    catalogue = load_catalogue()
    assert catalogue and all("Test" in p["name"] for p in catalogue.values())
    assert RegexPIIScanner().scan_text(DEFAULT_CATALOGUE.read_text()) == []


# ── calculator ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expression, value",
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("10 / 4", 2.5),
        ("10 / 5", 2),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("-2 ** 2", -4),
        ("2 ** -1", 0.5),
        ("1200 * 0.045 / 12", 4.5),
    ],
)
def test_calculator_arithmetic(expression, value):
    assert calculate(expression) == value


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "abs(-1)",
        "x + 1",
        "(1).real",
        "'a' * 3",
        "True + 1",
        "[1, 2]",
        "1 if 1 else 2",
        "1 +",
        "2 ** 1000",
        "10 ** 30",
        "1 / 0",
        "(-1) ** 0.5",
        "1 + " * 60 + "1",
    ],
)
def test_calculator_refuses_anything_but_bounded_arithmetic(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_calculator_through_the_gateway():
    reg = registry()
    assert call(reg, CALC).result == {"expression": "5000 * 4.1 / 100", "result": 205}
    failed = call(reg, ToolCall(CALCULATOR, {"expression": "open('x')"}))
    assert failed.error == TOOL_FAILED and "unsupported" in failed.message


# ── replay from the tool trace ───────────────────────────────────


def agent_reply(call):
    if TOOL_RESULTS_HEADER not in call.prompt:
        return [LOOKUP, CALC]
    return json.dumps(fag_reply(recipe_from_prompt(call.prompt)))


def run_once(fag, reg, cache):
    backend = MockBackend(agent_reply)
    gen = Generator(fag, backend, gateway=ToolGateway(reg.for_task(USES), cache))
    session = gen.session()
    recipe = gen.recipe(CELL, random.Random(0))
    result = gen.complete("c", recipe, gen.prompts.build(recipe), session=session)
    assert result.ok, result.detail
    return result, session, [c.prompt for c in backend.calls]


def test_record_replays_exactly_from_its_cached_tool_trace(fag, tmp_path):
    original, session, prompts = run_once(fag, registry(), ToolCache(tmp_path / "cache.jsonl"))
    assert [e.cached for e in session.trace] == [False, False]
    assert all(e.error is None for e in session.trace)

    # No handlers: every answer has to come from the trace, or the call fails with no_handler.
    offline = ToolRegistry()
    for tool in builtin_tools():
        offline.register(dataclasses.replace(tool, handler=None))
    replay, replay_session, replay_prompts = run_once(
        fag, offline, ToolCache.from_trace(session.trace)
    )

    assert replay.record == original.record
    assert replay_prompts == prompts  # the model saw byte-identical tool results
    assert all(r.error != NO_HANDLER for r in replay.tool_results)
    strip = [(e.tool, e.arguments, e.result, e.sensitivity) for e in session.trace]
    assert [(e.tool, e.arguments, e.result, e.sensitivity) for e in replay_session.trace] == strip
    assert all(e.cached for e in replay_session.trace)


def test_replay_also_works_from_the_persisted_cache_and_trace_dicts(fag, tmp_path):
    path = tmp_path / "cache.jsonl"
    original, session, _ = run_once(fag, registry(), ToolCache(path))
    offline = ToolRegistry()
    for tool in builtin_tools():
        offline.register(dataclasses.replace(tool, handler=None))
    for cache in (ToolCache(path), ToolCache.from_trace(session.trace_dicts())):
        assert run_once(fag, offline, cache)[0].record == original.record


def test_from_trace_skips_denied_calls():
    trace = [
        {"tool": CALCULATOR, "arguments": {"expression": "1"}, "result": None, "error": "x: y"},
        {"tool": CALCULATOR, "arguments": {"expression": "2"}, "result": {"result": 2}},
    ]
    cache = ToolCache.from_trace(trace)
    assert len(cache) == 1
    assert cache.get(CALCULATOR, {"expression": "2"}) == (True, {"result": 2})


def test_a_spec_can_list_the_builtins_from_the_default_registry(tmp_path):
    task = tmp_path / "fag_tools"
    shutil.copytree(FAG_DIR, task, ignore=shutil.ignore_patterns("__pycache__"))
    with (task / "task.yaml").open("a") as f:
        f.write(f"\ntools:\n  - name: {CATALOGUE_LOOKUP}\n  - name: {CALCULATOR}\n")
    pipe = Pipeline(
        task,
        tmp_path / "store",
        model_overrides={"generator": MockBackend(agent_reply)},
        target_size=4,
        layers=NO_JUDGE,
    )
    result = pipe.run("r1")
    assert result.complete and len(result.accepted) == 4
    for rec in result.accepted:
        trace = split(rec)[1].tool_trace
        assert [t.tool for t in trace] == [CATALOGUE_LOOKUP, CALCULATOR]
        assert trace[1].result == {"expression": "5000 * 4.1 / 100", "result": 205}
