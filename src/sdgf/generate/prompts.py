"""Generation prompts laid out for prefix caching (FRAMEWORK_DESIGN.md §9.3).

A prompt is two parts joined in a fixed order:

    static prefix   task description, rubric, output schema, turn/span rules, few-shot
                    seeds and the output-format instructions; identical for every cell
    cell section    the cell's fixed parameters, which vary per candidate

Everything static comes first so vLLM prefix caching and Anthropic prompt caching can
reuse it across the whole run. prefix_hash identifies the static part: two cells of the
same compiled spec share it, and any spec, rubric or seed change alters it.

A task type with an answer suffix (sft_qa) gets a response-format section in the prefix,
so the response ends in a form the answer extractor reads. A cell value on a Bloom axis
is followed in the cell section by that level's instruction (coverage/axes.py), as
DS²-Instruct put the query type's description into its generation prompt.

Few-shot seeds are picked once per spec (not per cell) so they stay in the prefix. When
seeds carry a `label`, the pick interleaves label values so the examples aren't all one
class (FAG's seed file lists its three breaches first).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sdgf.generate.scheduler import Cell
from sdgf.coverage.axes import bloom_description
from sdgf.spec.compile import CompiledSpec
from sdgf.spec.schema import RubricSection, TurnStructure
from sdgf.store.provenance import prompt_hash
from sdgf.tasktypes.base import TaskType
from sdgf.tasktypes.registry import REGISTRY

CELL_HEADER = "## Fixed parameters for this record"


@dataclass(frozen=True)
class Prompt:
    static: str
    cell: str
    prefix_hash: str

    @property
    def text(self) -> str:
        return f"{self.static}\n\n{self.cell}"

    @property
    def hash(self) -> str:
        return prompt_hash(self.text)


def _hash_prefix(static: str) -> str:
    return "sha256:" + hashlib.sha256(static.encode("utf-8")).hexdigest()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def select_few_shot(seeds: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    """Up to `count` seeds, interleaving label values in first-seen order."""
    if count <= 0 or not seeds:
        return []
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for seed in seeds:
        groups.setdefault(_dumps(seed.get("label")), []).append(seed)
    picked: list[dict[str, Any]] = []
    depth = 0
    while len(picked) < count and any(depth < len(g) for g in groups.values()):
        for group in groups.values():
            if depth < len(group) and len(picked) < count:
                picked.append(dict(group[depth]))
        depth += 1
    return picked


def _strip_private(record: Mapping[str, Any]) -> dict[str, Any]:
    """Drop pipeline-private keys (e.g. _provenance) from a seed shown to the model."""
    return {k: v for k, v in record.items() if not k.startswith("_")}


def _rubric_section(rubric: RubricSection) -> str:
    lines = ["## Rubric", f"Verdict values: {', '.join(rubric.verdict.values)}."]
    if rubric.verdict.description:
        lines.append(rubric.verdict.description.strip())
    for c in rubric.criteria:
        scale = ", ".join(c.values) if c.values is not None else f"integer {c.min}..{c.max}"
        desc = f" {c.description.strip()}" if c.description else ""
        lines.append(f"- {c.name} ({scale}):{desc}")
    return "\n".join(lines)


def _structure_section(turns: TurnStructure | None, spans: bool) -> str | None:
    rules = []
    if turns is not None:
        order = " / ".join(turns.roles)
        how = "alternate" if turns.alternating else "use the roles"
        rules.append(
            f"- messages {how} {order}, starting with {turns.first_role}, "
            f"with turns numbered consecutively from {turns.numbered_from}."
        )
    if spans:
        rules.append(
            "- Each span's text is copied exactly from the content of the turn it cites: "
            "no paraphrase, no ellipses. Quote only the words that justify its category."
        )
    return "## Structure rules\n" + "\n".join(rules) if rules else None


def build_static_prefix(
    compiled: CompiledSpec,
    task_type: TaskType,
    few_shot: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    spec = compiled.spec
    if few_shot is None:
        few_shot = (
            select_few_shot(compiled.seeds, spec.seeds.few_shot_count)
            if "few_shot" in spec.seeds.uses
            else []
        )
    schema = task_type.output_schema(spec.output_schema)

    parts = [
        f"# Task: {spec.task.name}",
        spec.task.description.strip(),
        _rubric_section(spec.rubric),
    ]
    parts.append("## Output schema\n" + json.dumps(schema, indent=2, ensure_ascii=False))
    structure = _structure_section(spec.output_schema.turns, spec.output_schema.spans)
    if structure:
        parts.append(structure)
    suffix = task_type.answer_suffix()
    if suffix:
        parts.append(
            "## Response format\n"
            "Write the response as a solver would answer the question, following:\n" + suffix
        )
    if few_shot:
        examples = "\n".join(_dumps(_strip_private(s)) for s in few_shot)
        parts.append(f"## Examples\n{examples}")
    fixed = (
        "The fixed parameters below are decided by the pipeline, including the label. "
        "Write the record so it genuinely matches them; any value you return for a fixed "
        "parameter is overwritten."
        if spec.task.generation_mode == "label_first"
        else "The fixed parameters below are decided by the pipeline; work out the answer yourself."
    )
    parts.append(
        "## Output format\n"
        f"{fixed}\n"
        "Return ONLY one valid JSON object matching the output schema: no markdown code "
        "fences and no text before or after it."
    )
    return "\n\n".join(parts)


def build_cell_section(params: Mapping[str, Any], bloom_axes: Sequence[str] = ()) -> str:
    lines = [CELL_HEADER]
    lines += [f"- {key}: {_dumps(value)}" for key, value in params.items()]
    guidance = [
        f"- {key} {params[key]}: {bloom_description(params[key])}"
        for key in bloom_axes
        if key in params
    ]
    if guidance:
        lines += ["", "Guidance for these parameters:", *guidance]
    return "\n".join(lines)


class PromptBuilder:
    """Builds generation prompts for one compiled spec; the static prefix is built once."""

    def __init__(
        self,
        compiled: CompiledSpec,
        task_type: TaskType | None = None,
        few_shot: Sequence[Mapping[str, Any]] | None = None,
    ):
        self.compiled = compiled
        self.task_type = task_type or REGISTRY.resolve(compiled.spec.task)
        self.static_prefix = build_static_prefix(compiled, self.task_type, few_shot)
        self.prefix_hash = _hash_prefix(self.static_prefix)
        self.bloom_axes = tuple(a.name for a in compiled.spec.coverage.axes if a.source == "bloom")

    def build(self, cell: Cell | Mapping[str, Any]) -> Prompt:
        params = cell.params if isinstance(cell, Cell) else cell
        section = build_cell_section(params, self.bloom_axes)
        return Prompt(self.static_prefix, section, self.prefix_hash)
