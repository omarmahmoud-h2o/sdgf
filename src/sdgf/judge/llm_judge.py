"""A judge backed by any ModelBackend (FRAMEWORK_DESIGN.md §6.4, §7.3).

The model is shown the rubric, the compiled output schema and a *view* of the record:
only the fields the task type lets a judge see (judge_fields), never the label, the
spans that justify it, or pipeline-private "_" keys. So the prompt for a record is the
same whatever its intended label is, and L5 fidelity is a real check rather than an
echo of the recipe.

The prompt keeps static content first (instructions, context, rubric, schema) and the
record last, like the generation prompt, so prefix caching works across records.

A generative LLMJudge can also write text, so the same class serves as the fallback
reason-writing judge (explain) behind a decision model such as Jev that can't.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from sdgf.generate.generator import extract_json
from sdgf.judge.interface import (
    Judge,
    JudgeError,
    JudgeParseError,
    JudgeResult,
    JudgeSchema,
    Record,
    compile_rubric,
)
from sdgf.models.base import ModelBackend
from sdgf.spec.compile import CompiledSpec
from sdgf.tasktypes.base import TaskType
from sdgf.tasktypes.registry import REGISTRY

RECORD_HEADER = "## Record to judge"


def judge_view(record: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """The part of a record a blind judge may see: the listed public fields only."""
    return {k: record[k] for k in fields if k in record and not k.startswith("_")}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _field_lines(schema: JudgeSchema) -> list[str]:
    lines = [f"- verdict (one of: {', '.join(schema.verdict_values)})"]
    if schema.verdict.description:
        lines[0] += f": {schema.verdict.description.strip()}"
    for c in schema.criteria:
        scale = (
            f"one of: {', '.join(c.choices)}" if c.kind == "enum" else f"integer {c.min}..{c.max}"
        )
        desc = f": {c.description.strip()}" if c.description else ""
        lines.append(f"- scores.{c.name} ({scale}){desc}")
    return lines


class LLMJudge(Judge):
    name = "llm"
    writes_reasons = True

    def __init__(
        self,
        schema: JudgeSchema,
        backend: ModelBackend,
        *,
        fields: Iterable[str],
        context: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        parse_retries: int = 1,
        stage: str | None = None,
    ):
        super().__init__(schema)
        if parse_retries < 0:
            raise JudgeError("parse_retries must be >= 0")
        self.backend = backend
        self.fields = tuple(fields)
        if not self.fields:
            raise JudgeError("a judge needs at least one record field to look at")
        self.context = context.strip()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.parse_retries = parse_retries
        self.stage = stage  # the models stage this judge calls, for provenance
        self.static_prefix = self._static_prefix()

    @classmethod
    def from_spec(
        cls,
        compiled: CompiledSpec,
        backend: ModelBackend,
        *,
        stage: str = "judge",
        task_type: TaskType | None = None,
        **kwargs: Any,
    ) -> LLMJudge:
        """A judge for a compiled spec, using models.<stage> for tokens and temperature.

        Its context is rubric.judge_context when the spec sets one, else task.description.
        """
        spec = compiled.spec
        task_type = task_type or REGISTRY.resolve(spec.task)
        config = getattr(spec.models, stage, None)
        if config is not None:
            kwargs.setdefault("max_tokens", config.max_tokens)
            kwargs.setdefault("temperature", config.temperature)
        kwargs.setdefault("fields", task_type.judge_fields())
        kwargs.setdefault("context", spec.rubric.judge_context or spec.task.description)
        kwargs.setdefault("stage", stage)
        return cls(compile_rubric(spec.rubric), backend, **kwargs)

    # ── prompts ──────────────────────────────────────────────────

    def _static_prefix(self) -> str:
        parts = [
            "# Judge\n"
            "You are an independent judge. Assess the record below against the rubric. "
            "Judge only what the record itself shows.",
        ]
        if self.context:
            parts.append("## Context\n" + self.context)
        parts.append("## Rubric\n" + "\n".join(_field_lines(self.schema)))
        return "\n\n".join(parts)

    def _record_section(self, record: Mapping[str, Any]) -> str:
        return f"{RECORD_HEADER}\n{_dumps(judge_view(record, self.fields))}"

    def judge_prompt(self, record: Mapping[str, Any]) -> str:
        schema = json.dumps(self.schema.json_schema(with_reason=False), indent=2)
        output = (
            "## Output format\n"
            "Return ONLY one JSON object matching this schema, with no markdown fences and "
            "no text before or after it. confidence.<field> is your probability, from 0 "
            "to 1, that the value you gave for that field is correct.\n" + schema
        )
        return "\n\n".join([self.static_prefix, output, self._record_section(record)])

    def reason_prompt(self, record: Mapping[str, Any], result: JudgeResult) -> str:
        decision = _dumps({"verdict": result.verdict, "scores": dict(result.scores)})
        output = (
            "## Output format\n"
            "A decision has already been made on the record below. In two to four plain "
            "sentences, explain the evidence in the record for that decision, quoting "
            "the words that matter. Return only the explanation."
        )
        return "\n\n".join(
            [self.static_prefix, output, self._record_section(record), f"## Decision\n{decision}"]
        )

    # ── calls ────────────────────────────────────────────────────

    def _call(self, prompt: str) -> str | None:
        return self.backend.call(prompt, self.max_tokens, self.temperature).text

    def judge(self, record: Record) -> JudgeResult:
        prompt = self.judge_prompt(record)
        attempt_prompt = prompt
        errors: list[str] = []
        for _ in range(self.parse_retries + 1):
            data = extract_json(self._call(attempt_prompt))
            if data is None:
                errors = ["<root>: reply was not a parseable JSON object"]
            else:
                try:
                    return self.schema.parse(data, with_reason=False)
                except JudgeParseError as e:
                    errors = e.errors
            attempt_prompt = (
                prompt
                + "\n\n## Your previous reply was rejected\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nReturn a corrected JSON object only."
            )
        raise JudgeParseError(errors)

    def explain(self, record: Record, result: JudgeResult) -> str:
        text = (self._call(self.reason_prompt(record, result)) or "").strip()
        if not text:
            raise JudgeError(f"judge {self.name!r} returned an empty reason")
        return text
