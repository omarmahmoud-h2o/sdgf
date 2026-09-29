"""Candidate generation for one cell (FRAMEWORK_DESIGN.md §6.3, §11).

One candidate is:

    cell params ─► sampler_constraints(cell, rng) ─► recipe
    recipe ─► PromptBuilder ─► generator backend ─► extract_json ─► merge ─► candidate

The recipe is code-owned. merge() lays every recipe field over the model's output, so
under label_first the label and every context fact come from the cell, never from the
model; the model contributes only the fields the recipe doesn't fix (the prose, the
messages and the spans). Keys starting with "_" are pipeline-private and are dropped
from model output.

post_process runs after validation (§4.3), so it is not applied here. Failures are
returned, not raised, with a machine-readable error so the scheduler can requeue the
same cell and repair can re-prompt:

    invalid_cell   sampler_constraints rejected the cell's combination
    no_text        the model returned no text
    tool_calls     the model asked for tools; the agent loop arrives in M6
    no_json        no JSON object could be parsed from the reply
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

Record = dict[str, Any]

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")


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
    ):
        self.compiled = compiled
        self.backend = backend
        self.prompts = prompts or PromptBuilder(compiled, task_type)
        cfg = compiled.spec.models.generator
        self.max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.label_first = compiled.spec.task.generation_mode == "label_first"

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
        self, cell_id: str | None, recipe: Mapping[str, Any], prompt: Prompt, extra: str = ""
    ) -> GenerationResult:
        """Call the backend on a built prompt and merge the reply with the recipe.

        `extra` is appended after the prompt (e.g. validator errors for a repair try) so
        the cached static prefix is unchanged.
        """
        text = prompt.text if not extra else f"{prompt.text}\n\n{extra}"
        response = self.backend.call(text, self.max_tokens, self.temperature)
        recipe = dict(recipe)

        def fail(error: str, detail: str) -> GenerationResult:
            return GenerationResult(cell_id, recipe, prompt, response, error=error, detail=detail)

        if response.tool_calls:
            names = ", ".join(c.name for c in response.tool_calls)
            return fail("tool_calls", f"model requested tools ({names}); no agent loop yet")
        if not response.text:
            return fail("no_text", "model returned no text")
        parsed = extract_json(response.text)
        if parsed is None:
            return fail("no_json", "model response was not valid/parseable JSON")
        return GenerationResult(cell_id, recipe, prompt, response, record=merge(recipe, parsed))
