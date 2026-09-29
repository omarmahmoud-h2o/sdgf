"""Candidate generation for one cell (FRAMEWORK_DESIGN.md §6.3, §11).

One candidate is:

    cell params ─► sampler_constraints(cell, rng) ─► recipe
    recipe ─► PromptBuilder ─► generator backend ─► extract_json ─► merge ─► candidate
                                    ▲        │ tool calls
                                    └ gateway┘ (agent loop, §6.3)

The recipe is code-owned. merge() lays every recipe field over the model's output, so
under label_first the label and every context fact come from the cell, never from the
model; the model contributes only the fields the recipe doesn't fix (the prose, the
messages and the spans). Keys starting with "_" are pipeline-private and are dropped
from model output.

The agent loop (§6.3, D4): when the task lists tools, the generator is given a
ToolGateway and the backend is offered the task's tool specs. A reply with tool calls is
run through the record's RecordToolSession (allowlist, budgets, cache, trace), and the
results, denials included, are appended to the prompt in a "## Tool results" section so
the static prefix stays cached; the backend is then called again. This repeats until it
returns a record without tool calls or max_tool_rounds rounds of tool calls have run.
Once the session has no tool budget left the backend is no longer offered tools and is
told to return the final record. The tool session is per record: repair tries of the same
slot share it (and so its budgets and trace) when the caller passes it in.

post_process runs after validation (§4.3), so it is not applied here. Failures are
returned, not raised, with a machine-readable error so the scheduler can requeue the
same cell and repair can re-prompt:

    invalid_cell          sampler_constraints rejected the cell's combination
    no_text               the model returned no text
    tool_calls            the model asked for tools but the task allows none
    tool_rounds_exhausted the model was still calling tools after max_tool_rounds rounds
    no_json               no JSON object could be parsed from the reply
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping

from sdgf.generate.prompts import Prompt, PromptBuilder
from sdgf.generate.scheduler import Cell
from sdgf.models.base import ModelBackend, ModelResponse
from sdgf.spec.compile import CompiledSpec
from sdgf.tasktypes.base import TaskType
from sdgf.tools.gateway import RecordToolSession, ToolGateway, ToolResult

Record = dict[str, Any]

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")

TOOL_RESULTS_HEADER = "## Tool results"


class GeneratorError(ValueError):
    """The generator is misconfigured for its spec (not a per-candidate failure)."""


def extract_json(response: str | None) -> dict[str, Any] | None:
    """Pull a JSON object out of a raw LLM reply, tolerating markdown fences.

    Ported from scripts/utils.extract_json: strip fences, then parse the text between the
    first "{" and the last "}".
    """
    if not response:
        return None
    text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", response.strip()))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def render_tool_round(round_no: int, results: list[ToolResult], *, exhausted: bool) -> str:
    """One round of tool results, as appended to the prompt for the next backend call."""
    lines = [f"{TOOL_RESULTS_HEADER} (round {round_no})"]
    for r in results:
        args = json.dumps(r.arguments, ensure_ascii=False, sort_keys=True)
        lines += [f"### {r.tool} {args}", r.content()]
    if exhausted:
        lines.append(
            "No tool budget is left for this record. Return the final record now as one "
            "JSON object."
        )
    else:
        lines.append(
            "Call another tool if you need one, or return the final record as one JSON object."
        )
    return "\n".join(lines)


def merge(recipe: Mapping[str, Any], model_fields: Mapping[str, Any]) -> Record:
    """Model output with every recipe field laid over it; the recipe always wins."""
    record = {k: v for k, v in model_fields.items() if not k.startswith("_")}
    record.update(recipe)
    return record


@dataclass(frozen=True)
class GenerationResult:
    cell_id: str | None
    recipe: Record | None
    prompt: Prompt | None = None
    response: ModelResponse | None = None
    record: Record | None = None
    error: str | None = None
    detail: str = ""
    tool_results: tuple[ToolResult, ...] = ()  # this attempt's tool calls, in order
    tool_rounds: int = 0
    calls: int = 0  # backend calls this attempt made

    @property
    def ok(self) -> bool:
        return self.record is not None


class Generator:
    """Generates candidates for cells of one compiled spec with one generator backend."""

    def __init__(
        self,
        compiled: CompiledSpec,
        backend: ModelBackend,
        *,
        task_type: TaskType | None = None,
        prompts: PromptBuilder | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        gateway: ToolGateway | None = None,
        max_tool_rounds: int | None = None,
    ):
        self.compiled = compiled
        self.backend = backend
        self.gateway = gateway if gateway is not None and len(gateway.tools) else None
        if max_tool_rounds is None:
            # Enough rounds to spend every call budget one call at a time, plus one;
            # calls with bad arguments cost no budget, so this also bounds those.
            tools = self.gateway.tools.values() if self.gateway is not None else ()
            max_tool_rounds = sum(t.max_calls_per_record for t in tools) + 1
        if max_tool_rounds < 0:
            raise GeneratorError(f"max_tool_rounds must be >= 0, got {max_tool_rounds}")
        self.max_tool_rounds = max_tool_rounds
        self.prompts = prompts or PromptBuilder(compiled, task_type)
        cfg = compiled.spec.models.generator
        self.max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.label_first = compiled.spec.task.generation_mode == "label_first"

    def session(self) -> RecordToolSession | None:
        """A fresh per-record tool session, or None when the task has no tools."""
        return self.gateway.session() if self.gateway is not None else None

    def recipe(self, params: Mapping[str, Any], rng: random.Random) -> Record | None:
        """The cell's fixed parameters after sampler_constraints; None if invalid."""
        hook = self.compiled.hooks.sampler_constraints
        recipe = dict(params) if hook is None else hook(dict(params), rng)
        if recipe is None:
            return None
        if self.label_first and "label" not in recipe:
            raise GeneratorError(
                "label_first task: the cell recipe has no 'label' after sampler_constraints; "
                "the label must come from the cell, never from the model"
            )
        return dict(recipe)

    def generate(self, cell: Cell | Mapping[str, Any], rng: random.Random) -> GenerationResult:
        cell_id = cell.id if isinstance(cell, Cell) else None
        params = cell.params if isinstance(cell, Cell) else cell
        recipe = self.recipe(params, rng)
        if recipe is None:
            return GenerationResult(
                cell_id, None, error="invalid_cell", detail="sampler_constraints returned None"
            )
        return self.complete(cell_id, recipe, self.prompts.build(recipe))

    def complete(
        self,
        cell_id: str | None,
        recipe: Mapping[str, Any],
        prompt: Prompt,
        extra: str = "",
        session: RecordToolSession | None = None,
    ) -> GenerationResult:
        """Run the agent loop on a built prompt and merge the final reply with the recipe.

        `extra` is appended after the prompt (e.g. validator errors for a repair try) so
        the cached static prefix is unchanged. `session` is the record's tool session;
        a new one is opened when the task has tools and none is given.
        """
        if session is None:
            session = self.session()
        elif self.gateway is None:
            raise GeneratorError("a tool session was given but the generator has no gateway")
        text = prompt.text if not extra else f"{prompt.text}\n\n{extra}"
        specs = self.gateway.tool_specs() if self.gateway is not None else None
        recipe = dict(recipe)
        results: list[ToolResult] = []
        rounds = calls = 0

        while True:
            offer = specs if session is not None and not session.exhausted() else None
            response = self.backend.call(text, self.max_tokens, self.temperature, tools=offer)
            calls += 1

            def fail(error: str, detail: str) -> GenerationResult:
                return GenerationResult(
                    cell_id,
                    recipe,
                    prompt,
                    response,
                    error=error,
                    detail=detail,
                    tool_results=tuple(results),
                    tool_rounds=rounds,
                    calls=calls,
                )

            if response.tool_calls:
                names = ", ".join(c.name for c in response.tool_calls)
                if session is None:
                    return fail(
                        "tool_calls", f"model requested tools ({names}); the task allows none"
                    )
                if rounds >= self.max_tool_rounds:
                    return fail(
                        "tool_rounds_exhausted",
                        f"model still requested tools ({names}) after {rounds} round(s) of "
                        "tool calls",
                    )
                rounds += 1
                round_results = [session.call(c) for c in response.tool_calls]
                results += round_results
                text += "\n\n" + render_tool_round(
                    rounds, round_results, exhausted=session.exhausted()
                )
                continue
            if not response.text:
                return fail("no_text", "model returned no text")
            parsed = extract_json(response.text)
            if parsed is None:
                return fail("no_json", "model response was not valid/parseable JSON")
            return GenerationResult(
                cell_id,
                recipe,
                prompt,
                response,
                record=merge(recipe, parsed),
                tool_results=tuple(results),
                tool_rounds=rounds,
                calls=calls,
            )
