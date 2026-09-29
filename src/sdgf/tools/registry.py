"""Tool registry: every tool declared once (FRAMEWORK_DESIGN.md §7.4, D4).

A ToolDefinition carries the tool's name, description, JSON input schema, sensitivity
level and read_only flag (default True), plus the handler the gateway runs. Each task
lists the tools it may use in spec.tools; for_task() resolves that list against the
registry into TaskTools, the task's allowlist with its per-record budgets.

Sensitivity is one of SENSITIVITY_LEVELS. Everything but "public" is sensitive to L3
(validate/l3_governance.py), so a tool is only "public" when its results may appear
verbatim in a record.

A tool with side effects (read_only=False) is refused for a task unless the caller
passes allow_side_effects=True, so a spec can't opt into writes by listing a name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping

import jsonschema

from sdgf.models.base import ToolSpec
from sdgf.spec.schema import TaskSpec, ToolUse

SENSITIVITY_LEVELS: tuple[str, ...] = ("public", "internal", "confidential", "restricted")

ToolHandler = Callable[[dict[str, Any]], Any]


class ToolError(Exception):
    """Base error for tool declaration, lookup and argument checks."""


class UnknownToolError(ToolError, KeyError):
    """No tool is registered under the requested name."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


class ToolArgumentError(ToolError, ValueError):
    """A tool call's arguments do not match the tool's input schema."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    sensitivity: str
    read_only: bool = True
    handler: ToolHandler | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ToolError("tool name must be non-empty")
        if not self.description.strip():
            raise ToolError(f"tool {self.name!r}: description must be non-empty")
        if self.sensitivity not in SENSITIVITY_LEVELS:
            raise ToolError(
                f"tool {self.name!r}: sensitivity {self.sensitivity!r} is not one of "
                f"{', '.join(SENSITIVITY_LEVELS)}"
            )
        if self.input_schema.get("type") != "object":
            raise ToolError(f"tool {self.name!r}: input_schema must have type 'object'")
        try:
            jsonschema.Draft202012Validator.check_schema(self.input_schema)
        except jsonschema.SchemaError as e:
            raise ToolError(f"tool {self.name!r}: invalid input_schema: {e.message}") from e
        if self.handler is not None and not callable(self.handler):
            raise ToolError(f"tool {self.name!r}: handler must be callable")

    def argument_errors(self, arguments: Any) -> list[str]:
        """Every way `arguments` fails the input schema, as path-prefixed messages."""
        validator = jsonschema.Draft202012Validator(self.input_schema)
        errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path))
        return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors]

    def check_arguments(self, arguments: Any) -> None:
        errors = self.argument_errors(arguments)
        if errors:
            raise ToolArgumentError(f"tool {self.name!r}: bad arguments: {'; '.join(errors)}")

    def tool_spec(self) -> ToolSpec:
        """The shape model backends take in call(..., tools=[...])."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class AllowedTool:
    definition: ToolDefinition
    max_calls_per_record: int
    max_tokens_per_record: int | None


class TaskTools(Mapping[str, AllowedTool]):
    """The tools one task may use, in spec order, each with its per-record budget."""

    def __init__(self, tools: list[AllowedTool]):
        self._tools = {t.definition.name: t for t in tools}

    def __getitem__(self, name: str) -> AllowedTool:
        return self._tools[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def allows(self, name: str) -> bool:
        return name in self._tools

    def tool_specs(self) -> list[ToolSpec]:
        return [t.definition.tool_spec() for t in self._tools.values()]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition, *, replace: bool = False) -> ToolDefinition:
        if tool.name in self._tools and not replace:
            raise ToolError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "none"
            raise UnknownToolError(f"unknown tool {name!r}; registered tools: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def for_task(
        self, spec: TaskSpec | list[ToolUse], *, allow_side_effects: bool = False
    ) -> TaskTools:
        """Resolve a task's tool list into its allowlist; unknown tools raise."""
        uses = spec.tools if isinstance(spec, TaskSpec) else spec
        allowed = []
        for use in uses:
            tool = self.get(use.name)
            if not tool.read_only and not allow_side_effects:
                raise ToolError(
                    f"tool {tool.name!r} has side effects (read_only=False); "
                    "pass allow_side_effects=True to allow it"
                )
            allowed.append(
                AllowedTool(
                    definition=tool,
                    max_calls_per_record=use.max_calls_per_record,
                    max_tokens_per_record=use.max_tokens_per_record,
                )
            )
        return TaskTools(allowed)


REGISTRY = ToolRegistry()


def register_tool(tool: ToolDefinition, *, replace: bool = False) -> ToolDefinition:
    return REGISTRY.register(tool, replace=replace)


def get_tool(name: str) -> ToolDefinition:
    return REGISTRY.get(name)
