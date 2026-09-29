"""Tool gateway: the only way an agent's tool call reaches a tool (FRAMEWORK_DESIGN.md §7.4, D4).

A ToolGateway is built once per task from its TaskTools allowlist and a ToolCache.
Each record gets its own RecordToolSession (gateway.session()), which holds that
record's budgets and trace. For every call the session:

  1. checks the tool is on the task's allowlist            -> not_allowed
  2. checks the tool's per-record call budget               -> call_budget_exhausted
  3. checks the arguments against the tool's input schema   -> bad_arguments
  4. answers from the cache, or runs the handler and caches the result
                                                            -> no_handler, tool_failed
  5. charges the result's tokens against the tool's per-record token budget; a result
     that would go over is withheld                         -> token_budget_exhausted
  6. labels the result with the tool's sensitivity, and appends the call, denied or not,
     to the record's trace

Denials come back as a ToolResult with `error` set rather than as exceptions, so the
agent loop can hand the message to the model; nothing from a denied call reaches it.
Cache hits count against both budgets like fresh calls, because the model sees the
same result either way, which also keeps a replay from the cache identical to the
original run. Tokens are estimated from the result's JSON (about four characters per
token) unless a token_counter is given.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from sdgf.models.base import ToolCall, ToolSpec
from sdgf.store.provenance import ToolTraceEntry
from sdgf.tools.cache import ToolCache, ToolCacheError
from sdgf.tools.registry import TaskTools

TokenCounter = Callable[[Any], int]

NOT_ALLOWED = "not_allowed"
CALL_BUDGET_EXHAUSTED = "call_budget_exhausted"
TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
BAD_ARGUMENTS = "bad_arguments"
NO_HANDLER = "no_handler"
TOOL_FAILED = "tool_failed"
BUDGET_ERRORS = frozenset({CALL_BUDGET_EXHAUSTED, TOKEN_BUDGET_EXHAUSTED})


def estimate_tokens(result: Any) -> int:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class ToolResult:
    tool: str
    arguments: dict[str, Any]
    result: Any = None
    sensitivity: str | None = None
    cached: bool = False
    tokens: int = 0
    error: str | None = None  # one of the codes above
    message: str | None = None
    call_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def content(self) -> str:
        """What the agent loop feeds back to the model."""
        if self.ok:
            return (
                self.result
                if isinstance(self.result, str)
                else json.dumps(self.result, ensure_ascii=False, sort_keys=True)
            )
        return f"error ({self.error}): {self.message}"


class ToolGateway:
    def __init__(
        self,
        tools: TaskTools,
        cache: ToolCache | None = None,
        token_counter: TokenCounter = estimate_tokens,
    ):
        self.tools = tools
        self.cache = cache if cache is not None else ToolCache()
        self.token_counter = token_counter

    def tool_specs(self) -> list[ToolSpec]:
        return self.tools.tool_specs()

    def session(self) -> RecordToolSession:
        return RecordToolSession(self)


class RecordToolSession:
    """One record's tool use: per-tool call and token counts plus the trace."""

    def __init__(self, gateway: ToolGateway):
        self.gateway = gateway
        self.calls: dict[str, int] = {}
        self.tokens: dict[str, int] = {}
        self.trace: list[ToolTraceEntry] = []

    def call(self, call: ToolCall) -> ToolResult:
        result = self._run(call)
        self.trace.append(
            ToolTraceEntry(
                tool=result.tool,
                arguments=result.arguments,
                result=result.result,
                sensitivity=result.sensitivity,
                cached=result.cached,
                error=None if result.ok else f"{result.error}: {result.message}",
            )
        )
        return result

    def _run(self, call: ToolCall) -> ToolResult:
        name, arguments = call.name, dict(call.arguments)

        def deny(code: str, message: str, sensitivity: str | None = None) -> ToolResult:
            return ToolResult(
                tool=name,
                arguments=arguments,
                sensitivity=sensitivity,
                error=code,
                message=message,
                call_id=call.id,
            )

        tools = self.gateway.tools
        if not tools.allows(name):
            allowed = ", ".join(tools) or "none"
            return deny(NOT_ALLOWED, f"tool {name!r} is not allowed; allowed tools: {allowed}")
        allowed_tool = tools[name]
        tool = allowed_tool.definition
        used = self.calls.get(name, 0)
        if used >= allowed_tool.max_calls_per_record:
            return deny(
                CALL_BUDGET_EXHAUSTED,
                f"tool {name!r} used {used} of {allowed_tool.max_calls_per_record} calls "
                "for this record",
            )
        token_budget = allowed_tool.max_tokens_per_record
        spent = self.tokens.get(name, 0)
        if token_budget is not None and spent >= token_budget:
            return deny(
                TOKEN_BUDGET_EXHAUSTED,
                f"tool {name!r} used {spent} of {token_budget} tokens for this record",
            )
        errors = tool.argument_errors(arguments)
        if errors:
            return deny(BAD_ARGUMENTS, "; ".join(errors))

        # Every check passed: this call now counts against the record's call budget.
        self.calls[name] = used + 1
        cache = self.gateway.cache
        try:
            cached, value = cache.get(name, arguments)
        except ToolCacheError as e:
            return deny(BAD_ARGUMENTS, str(e))
        if not cached:
            if tool.handler is None:
                return deny(NO_HANDLER, f"tool {name!r} has no handler")
            try:
                value = cache.put(name, arguments, tool.handler(dict(arguments)))
            except Exception as e:  # a tool failure goes back to the model, not up the stack
                return deny(TOOL_FAILED, f"tool {name!r} failed: {type(e).__name__}: {e}")

        tokens = self.gateway.token_counter(value)
        if token_budget is not None and spent + tokens > token_budget:
            self.tokens[name] = token_budget  # nothing more fits; later calls are refused
            return deny(
                TOKEN_BUDGET_EXHAUSTED,
                f"tool {name!r} result of {tokens} tokens would exceed the {token_budget}-token "
                f"budget for this record ({spent} used)",
            )
        self.tokens[name] = spent + tokens
        return ToolResult(
            tool=name,
            arguments=arguments,
            result=value,
            sensitivity=tool.sensitivity,
            cached=cached,
            tokens=tokens,
            call_id=call.id,
        )

    def exhausted(self) -> bool:
        """True once no allowed tool has call or token budget left for this record."""
        for name, allowed in self.gateway.tools.items():
            calls_left = self.calls.get(name, 0) < allowed.max_calls_per_record
            budget = allowed.max_tokens_per_record
            tokens_left = budget is None or self.tokens.get(name, 0) < budget
            if calls_left and tokens_left:
                return False
        return True

    def trace_dicts(self) -> list[dict[str, Any]]:
        """The trace as plain dicts, the form L3 reads from context.extra["tool_trace"]."""
        return [asdict(e) for e in self.trace]
