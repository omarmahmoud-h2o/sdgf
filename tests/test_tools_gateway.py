import json
from dataclasses import asdict

import pytest

from sdgf.models.base import ToolCall
from sdgf.spec.schema import ToolUse
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import Provenance, ProvenanceBuilder, ToolTraceEntry
from sdgf.tools.cache import ToolCache, ToolCacheError, cache_key, canonical_arguments
from sdgf.tools.gateway import (
    BAD_ARGUMENTS,
    CALL_BUDGET_EXHAUSTED,
    NO_HANDLER,
    NOT_ALLOWED,
    TOKEN_BUDGET_EXHAUSTED,
    TOOL_FAILED,
    ToolGateway,
    estimate_tokens,
)
from sdgf.tools.registry import ToolDefinition, ToolRegistry
from sdgf.validate.base import ValidationContext
from sdgf.validate.l3_governance import GovernanceLayer, sensitive_entries

LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {"product_code": {"type": "string", "minLength": 1}},
    "required": ["product_code"],
    "additionalProperties": False,
}


class Counter:
    """A handler that counts how often it actually runs."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, args):
        self.calls += 1
        return self.fn(args)


def make_registry(lookup_handler=None, calc_handler=None):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="catalogue_lookup",
            description="Look up a fictional product by code.",
            input_schema=LOOKUP_SCHEMA,
            sensitivity="internal",
            handler=lookup_handler
            or (lambda a: {"product_code": a["product_code"], "name": "Test Everyday Account"}),
        )
    )
    registry.register(
        ToolDefinition(
            name="calculator",
            description="Evaluate simple arithmetic.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            sensitivity="public",
            handler=calc_handler or (lambda a: {"value": 4}),
        )
    )
    registry.register(
        ToolDefinition(
            name="not_for_this_task",
            description="A registered tool the task does not list.",
            input_schema={"type": "object"},
            sensitivity="public",
            handler=lambda a: "should never run",
        )
    )
    return registry


def make_gateway(uses=None, cache=None, registry=None, **kwargs):
    registry = registry or make_registry()
    uses = uses or [ToolUse(name="catalogue_lookup"), ToolUse(name="calculator")]
    return ToolGateway(registry.for_task(uses), cache=cache, **kwargs)


def lookup_call(code="TST-001", call_id=None):
    return ToolCall(name="catalogue_lookup", arguments={"product_code": code}, id=call_id)


# ── cache ─────────────────────────────────────────────────────────


def test_cache_key_is_canonical_over_argument_order():
    assert canonical_arguments({"b": 2, "a": 1}) == canonical_arguments({"a": 1, "b": 2})
    assert cache_key("t", {"b": 2, "a": 1}) == cache_key("t", {"a": 1, "b": 2})
    assert cache_key("t", {"a": 1}) != cache_key("u", {"a": 1})
    assert cache_key("t", {"a": 1}) != cache_key("t", {"a": "1"})


def test_cache_get_put_and_copy_semantics():
    cache = ToolCache()
    assert cache.get("t", {"a": 1}) == (False, None)
    stored = cache.put("t", {"a": 1}, {"items": ("x", "y")})
    assert stored == {"items": ["x", "y"]}  # JSON form
    hit, value = cache.get("t", {"a": 1})
    assert hit and value == {"items": ["x", "y"]}
    value["items"].append("mutated")
    assert cache.get("t", {"a": 1})[1] == {"items": ["x", "y"]}
    assert len(cache) == 1


def test_cache_rejects_unserialisable_results():
    with pytest.raises(ToolCacheError, match="not JSON-serialisable"):
        ToolCache().put("t", {}, {"obj": object()})


def test_cache_persists_and_reloads(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = ToolCache(path)
    cache.put("t", {"a": 1}, {"v": 1})
    cache.put("t", {"a": 1}, {"v": 999})  # already cached: not overwritten or re-appended
    cache.close()
    assert len(path.read_text().splitlines()) == 1
    reloaded = ToolCache(path)
    assert reloaded.get("t", {"a": 1}) == (True, {"v": 1})


def test_cache_detects_tampered_entries(tmp_path):
    path = tmp_path / "cache.jsonl"
    entry = {"key": cache_key("t", {"a": 1}), "tool": "t", "arguments": {"a": 2}, "result": 1}
    path.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ToolCacheError, match="does not match"):
        ToolCache(path)
    path.write_text(json.dumps({"key": "x"}) + "\n")
    with pytest.raises(ToolCacheError, match="malformed"):
        ToolCache(path)


def test_cache_for_store_lives_in_shared_area(tmp_path):
    store = ArtefactStore(tmp_path)
    cache = ToolCache.for_store(store, "v1")
    cache.put("t", {}, "ok")
    cache.close()
    assert cache.path == tmp_path / "v1" / "shared" / "tool_cache.jsonl"
    assert ToolCache.for_store(store, "v1").get("t", {}) == (True, "ok")
    assert ToolCache.for_store(store, "v2").get("t", {}) == (False, None)


# ── allowlist and argument denial ─────────────────────────────────


def test_tool_not_on_allowlist_is_denied_and_never_runs():
    ran = Counter(lambda a: "x")
    registry = make_registry()
    registry.register(
        ToolDefinition(
            name="not_for_this_task",
            description="d",
            input_schema={"type": "object"},
            sensitivity="public",
            handler=ran,
        ),
        replace=True,
    )
    session = make_gateway(registry=registry).session()
    for name in ("not_for_this_task", "never_registered"):
        result = session.call(ToolCall(name=name, arguments={}))
        assert not result.ok and result.error == NOT_ALLOWED
        assert result.result is None and "catalogue_lookup" in result.message
    assert ran.calls == 0
    assert [e.tool for e in session.trace] == ["not_for_this_task", "never_registered"]
    assert all(e.error.startswith(NOT_ALLOWED) for e in session.trace)
    assert session.calls == {}


def test_bad_arguments_are_denied_without_spending_budget():
    session = make_gateway([ToolUse(name="catalogue_lookup", max_calls_per_record=1)]).session()
    result = session.call(ToolCall(name="catalogue_lookup", arguments={"product_code": ""}))
    assert result.error == BAD_ARGUMENTS and "product_code" in result.message
    assert session.call(lookup_call()).ok


def test_tool_without_handler_and_failing_tool():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="declared_only",
            description="d",
            input_schema={"type": "object"},
            sensitivity="public",
        )
    )

    def boom(args):
        raise RuntimeError("catalogue offline")

    registry.register(
        ToolDefinition(
            name="flaky",
            description="d",
            input_schema={"type": "object"},
            sensitivity="public",
            handler=boom,
        )
    )
    gateway = ToolGateway(registry.for_task([ToolUse(name="declared_only"), ToolUse(name="flaky")]))
    session = gateway.session()
    assert session.call(ToolCall(name="declared_only")).error == NO_HANDLER
    failed = session.call(ToolCall(name="flaky"))
    assert failed.error == TOOL_FAILED and "catalogue offline" in failed.message
    assert "catalogue offline" in failed.content()
    assert len(gateway.cache) == 0  # failures are not cached
    assert session.calls == {"declared_only": 1, "flaky": 1}


# ── budgets ───────────────────────────────────────────────────────


def test_call_budget_exhaustion_is_per_tool_and_per_record():
    gateway = make_gateway(
        [
            ToolUse(name="catalogue_lookup", max_calls_per_record=2),
            ToolUse(name="calculator", max_calls_per_record=1),
        ]
    )
    session = gateway.session()
    assert session.call(lookup_call("A")).ok
    assert session.call(lookup_call("B")).ok
    third = session.call(lookup_call("C"))
    assert third.error == CALL_BUDGET_EXHAUSTED and "2 of 2" in third.message
    assert not session.exhausted()  # calculator still has budget
    assert session.call(ToolCall(name="calculator", arguments={"expression": "2+2"})).ok
    assert session.exhausted()
    # a new record starts with fresh budgets
    assert gateway.session().call(lookup_call("C")).ok


def test_zero_call_budget_denies_immediately():
    session = make_gateway([ToolUse(name="catalogue_lookup", max_calls_per_record=0)]).session()
    assert session.exhausted()
    assert session.call(lookup_call()).error == CALL_BUDGET_EXHAUSTED


def test_token_budget_withholds_result_that_would_exceed_it():
    handler = Counter(lambda a: "x" * 40)  # 10 tokens by the default estimate
    registry = make_registry(lookup_handler=handler)
    uses = [ToolUse(name="catalogue_lookup", max_tokens_per_record=25)]
    session = make_gateway(uses, registry=registry).session()
    assert estimate_tokens("x" * 40) == 10
    first = session.call(lookup_call("A"))
    second = session.call(lookup_call("B"))
    assert first.ok and first.tokens == 10 and second.ok
    assert session.tokens["catalogue_lookup"] == 20
    over = session.call(lookup_call("C"))
    assert over.error == TOKEN_BUDGET_EXHAUSTED and over.result is None
    assert session.trace[-1].result is None  # the withheld result never reaches the trace
    assert session.exhausted()
    after = session.call(lookup_call("D"))
    assert after.error == TOKEN_BUDGET_EXHAUSTED
    assert handler.calls == 3  # D is refused before the tool runs


def test_custom_token_counter():
    uses = [ToolUse(name="catalogue_lookup", max_tokens_per_record=5)]
    session = make_gateway(uses, token_counter=lambda result: 5).session()
    assert session.call(lookup_call("A")).ok
    assert session.call(lookup_call("B")).error == TOKEN_BUDGET_EXHAUSTED


# ── cache through the gateway ─────────────────────────────────────


def test_cache_hit_skips_handler_and_is_marked_cached():
    handler = Counter(lambda a: {"product_code": a["product_code"], "rate": 1.5})
    gateway = make_gateway(registry=make_registry(lookup_handler=handler))
    first = gateway.session().call(lookup_call())
    second = gateway.session().call(lookup_call())
    assert handler.calls == 1
    assert not first.cached and second.cached
    assert first.result == second.result == {"product_code": "TST-001", "rate": 1.5}


def test_cache_hits_still_count_against_budgets():
    gateway = make_gateway([ToolUse(name="catalogue_lookup", max_calls_per_record=1)])
    session = gateway.session()
    assert session.call(lookup_call()).ok
    repeat = session.call(lookup_call())
    assert repeat.error == CALL_BUDGET_EXHAUSTED


def test_persistent_cache_serves_a_new_gateway(tmp_path):
    handler = Counter(lambda a: {"name": "Test Everyday Account"})
    cache = ToolCache(tmp_path / "c.jsonl")
    make_gateway(cache=cache, registry=make_registry(lookup_handler=handler)).session().call(
        lookup_call()
    )
    cache.close()
    handler2 = Counter(lambda a: {"name": "changed"})
    gateway = make_gateway(
        cache=ToolCache(tmp_path / "c.jsonl"), registry=make_registry(lookup_handler=handler2)
    )
    result = gateway.session().call(lookup_call())
    assert result.cached and result.result == {"name": "Test Everyday Account"}
    assert handler2.calls == 0


# ── sensitivity and trace ─────────────────────────────────────────


def test_result_carries_tool_sensitivity_label():
    session = make_gateway().session()
    assert session.call(lookup_call()).sensitivity == "internal"
    calc = session.call(ToolCall(name="calculator", arguments={"expression": "2+2"}))
    assert calc.sensitivity == "public"


def test_trace_records_every_call_in_order_with_content():
    gateway = make_gateway(
        [ToolUse(name="catalogue_lookup", max_calls_per_record=2), ToolUse(name="calculator")]
    )
    session = gateway.session()
    session.call(lookup_call("TST-001", call_id="c1"))
    session.call(lookup_call("TST-001", call_id="c2"))
    session.call(lookup_call("TST-002"))
    session.call(ToolCall(name="calculator", arguments={"expression": "2+2"}))
    session.call(ToolCall(name="nope"))
    assert [e.tool for e in session.trace] == [
        "catalogue_lookup",
        "catalogue_lookup",
        "catalogue_lookup",
        "calculator",
        "nope",
    ]
    first, cached, denied, calc, unknown = session.trace
    assert first == ToolTraceEntry(
        tool="catalogue_lookup",
        arguments={"product_code": "TST-001"},
        result={"product_code": "TST-001", "name": "Test Everyday Account"},
        sensitivity="internal",
        cached=False,
    )
    assert cached.cached and cached.result == first.result and cached.error is None
    assert denied.result is None and denied.error.startswith(CALL_BUDGET_EXHAUSTED)
    assert calc.result == {"value": 4} and calc.sensitivity == "public"
    assert unknown.error.startswith(NOT_ALLOWED) and unknown.sensitivity is None
    assert session.trace_dicts() == [asdict(e) for e in session.trace]


def test_trace_goes_into_provenance_and_round_trips():
    session = make_gateway().session()
    session.call(lookup_call())
    session.call(ToolCall(name="nope"))
    builder = ProvenanceBuilder(
        "v1",
        "cell-a",
        7,
        [{"stage": "generator", "backend": "mock", "model": "mock", "hosting": "local"}],
    )
    builder.set_prompt("prompt")
    for entry in session.trace:
        builder.add_tool_call(entry)
    provenance = builder.build()
    assert Provenance.from_dict(provenance.to_dict()).tool_trace == tuple(session.trace)


def test_sensitive_trace_reaches_l3_and_flags_a_leak():
    session = make_gateway().session()
    session.call(lookup_call())
    session.call(ToolCall(name="calculator", arguments={"expression": "2+2"}))
    trace = session.trace_dicts()
    assert [e["tool"] for e in sensitive_entries(trace)] == ["catalogue_lookup"]
    record = {
        "messages": [
            {"turn": 1, "role": "customer", "content": "What is it called?"},
            {"turn": 2, "role": "assistant", "content": "It is the Test Everyday Account."},
        ]
    }
    verdict = GovernanceLayer.from_profile().check(
        record, ValidationContext(extra={"tool_trace": trace})
    )
    assert verdict.outcome == "fail_hard"
    assert any(i.code == "tool_data_leak" for i in verdict.errors)


def test_content_for_model():
    session = make_gateway().session()
    ok = session.call(lookup_call())
    assert json.loads(ok.content()) == ok.result
    denied = session.call(ToolCall(name="nope"))
    assert denied.content().startswith("error (not_allowed):")


def test_arguments_are_copied_not_shared():
    args = {"product_code": "TST-001"}
    session = make_gateway().session()
    session.call(ToolCall(name="catalogue_lookup", arguments=args))
    args["product_code"] = "changed"
    assert session.trace[0].arguments == {"product_code": "TST-001"}
