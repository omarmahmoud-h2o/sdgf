import pytest

from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import ToolUse
from sdgf.tools import registry as tool_registry_module
from sdgf.tools.registry import (
    SENSITIVITY_LEVELS,
    ToolArgumentError,
    ToolDefinition,
    ToolError,
    ToolRegistry,
    UnknownToolError,
)
from sdgf.validate.l3_governance import is_sensitive
from test_spec_stage0 import TASK_YAML, write_task

LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {"product_code": {"type": "string", "minLength": 1}},
    "required": ["product_code"],
    "additionalProperties": False,
}


def lookup(**overrides):
    fields = {
        "name": "catalogue_lookup",
        "description": "Look up a fictional product by code.",
        "input_schema": LOOKUP_SCHEMA,
        "sensitivity": "internal",
        "handler": lambda args: {"product_code": args["product_code"], "name": "Test Account"},
    }
    return ToolDefinition(**{**fields, **overrides})


def calculator(**overrides):
    fields = {
        "name": "calculator",
        "description": "Evaluate simple arithmetic.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
        "sensitivity": "public",
    }
    return ToolDefinition(**{**fields, **overrides})


def registry_with(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ── definitions ──────────────────────────────────────────────────


def test_read_only_defaults_true():
    assert lookup().read_only is True


def test_definition_fields():
    t = lookup()
    assert (t.name, t.sensitivity) == ("catalogue_lookup", "internal")
    assert t.input_schema == LOOKUP_SCHEMA
    assert t.handler({"product_code": "TEST-01"})["name"] == "Test Account"


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_rejected(name):
    with pytest.raises(ToolError, match="name must be non-empty"):
        lookup(name=name)


def test_empty_description_rejected():
    with pytest.raises(ToolError, match="description"):
        lookup(description=" ")


def test_unknown_sensitivity_rejected():
    with pytest.raises(ToolError, match="sensitivity 'secret-ish'"):
        lookup(sensitivity="secret-ish")


def test_every_level_but_public_is_sensitive_to_l3():
    assert [lvl for lvl in SENSITIVITY_LEVELS if not is_sensitive(lvl)] == ["public"]


def test_input_schema_must_be_object():
    with pytest.raises(ToolError, match="type 'object'"):
        lookup(input_schema={"type": "string"})


def test_invalid_json_schema_rejected():
    with pytest.raises(ToolError, match="invalid input_schema"):
        lookup(input_schema={"type": "object", "properties": {"x": {"type": 5}}})


def test_non_callable_handler_rejected():
    with pytest.raises(ToolError, match="handler must be callable"):
        lookup(handler="not a function")


def test_valid_arguments_pass():
    lookup().check_arguments({"product_code": "TEST-01"})


def test_argument_errors_name_the_path():
    errors = lookup().argument_errors({"product_code": "", "extra": 1})
    assert any(e.startswith("product_code:") for e in errors)
    assert any(e.startswith("<root>:") and "extra" in e for e in errors)


def test_missing_argument_raises():
    with pytest.raises(ToolArgumentError, match="'product_code' is a required property"):
        lookup().check_arguments({})


def test_tool_spec_shape_for_backends():
    assert lookup().tool_spec() == {
        "name": "catalogue_lookup",
        "description": "Look up a fictional product by code.",
        "input_schema": LOOKUP_SCHEMA,
    }


# ── registry ─────────────────────────────────────────────────────


def test_register_and_get():
    reg = registry_with(lookup(), calculator())
    assert reg.get("calculator").sensitivity == "public"
    assert reg.names() == ["calculator", "catalogue_lookup"]
    assert "calculator" in reg and "missing" not in reg


def test_duplicate_registration_rejected():
    reg = registry_with(lookup())
    with pytest.raises(ToolError, match="already registered"):
        reg.register(lookup())
    reg.register(lookup(sensitivity="confidential"), replace=True)
    assert reg.get("catalogue_lookup").sensitivity == "confidential"


def test_unknown_tool_lists_known_ones():
    reg = registry_with(calculator())
    with pytest.raises(UnknownToolError) as exc:
        reg.get("missing")
    assert str(exc.value) == "unknown tool 'missing'; registered tools: calculator"


def test_module_registry_helpers(monkeypatch):
    monkeypatch.setattr(tool_registry_module, "REGISTRY", ToolRegistry())
    tool_registry_module.register_tool(calculator())
    assert tool_registry_module.get_tool("calculator").name == "calculator"


# ── per-task allowlist ───────────────────────────────────────────


def test_for_task_resolves_listed_tools_with_budgets():
    reg = registry_with(lookup(), calculator())
    tools = reg.for_task(
        [
            ToolUse(name="calculator"),
            ToolUse(name="catalogue_lookup", max_calls_per_record=2, max_tokens_per_record=500),
        ]
    )
    assert list(tools) == ["calculator", "catalogue_lookup"]
    assert tools["calculator"].max_calls_per_record == 5
    assert tools["calculator"].max_tokens_per_record is None
    assert tools["catalogue_lookup"].max_calls_per_record == 2
    assert tools["catalogue_lookup"].max_tokens_per_record == 500
    assert tools["catalogue_lookup"].definition is reg.get("catalogue_lookup")


def test_task_allows_only_its_listed_tools():
    reg = registry_with(lookup(), calculator())
    tools = reg.for_task([ToolUse(name="calculator")])
    assert tools.allows("calculator")
    assert not tools.allows("catalogue_lookup")
    assert [s["name"] for s in tools.tool_specs()] == ["calculator"]


def test_task_with_no_tools_allows_nothing():
    tools = registry_with(lookup()).for_task([])
    assert len(tools) == 0 and tools.tool_specs() == []


def test_for_task_unknown_tool_raises():
    with pytest.raises(UnknownToolError, match="unknown tool 'missing'"):
        registry_with(lookup()).for_task([ToolUse(name="missing")])


def test_side_effect_tool_needs_explicit_opt_in():
    reg = registry_with(lookup(name="send_email", read_only=False))
    with pytest.raises(ToolError, match="has side effects"):
        reg.for_task([ToolUse(name="send_email")])
    tools = reg.for_task([ToolUse(name="send_email")], allow_side_effects=True)
    assert tools["send_email"].definition.read_only is False


def test_for_task_accepts_a_spec(tmp_path):
    yaml_text = TASK_YAML + "tools:\n  - name: catalogue_lookup\n    max_calls_per_record: 1\n"
    reg = registry_with(lookup(), calculator())
    compiled = compile_spec(write_task(tmp_path, yaml_text=yaml_text), tool_registry=reg)
    tools = reg.for_task(compiled.spec)
    assert list(tools) == ["catalogue_lookup"]
    assert tools["catalogue_lookup"].max_calls_per_record == 1


def test_registry_serves_the_stage0_tool_gate(tmp_path):
    yaml_text = TASK_YAML + "tools:\n  - name: calculator\n"
    task_dir = write_task(tmp_path, yaml_text=yaml_text)
    compile_spec(task_dir, tool_registry=registry_with(calculator()))
    with pytest.raises(Exception, match="'calculator' is not in the tool registry"):
        compile_spec(task_dir, tool_registry=registry_with(lookup()))
