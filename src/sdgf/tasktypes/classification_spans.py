"""Label-first classification with span annotations (FRAMEWORK_DESIGN.md §7.1).

A record is a conversation plus a code-owned label and the spans that justify it:

    messages  [{turn, role, content}, ...]   turns numbered consecutively
    label     boolean, string or integer     fixed by the cell, never by the model
    spans     [{turn, text, category}, ...]  each span points at an existing turn

Turn numbering, role names and alternation belong to the spec (output_schema.turns,
which stage 0 requires for this task type) and are checked by L1 alone, so the base
schema only requires an integer turn and a non-empty role string. The one default
validator checks that every span cites a turn that exists. Whether a span's text is
verbatim, and which categories or labels need spans, is task policy and lives in the
task's hooks.
"""

from __future__ import annotations

from typing import Any

from sdgf.tasktypes.base import Record, TaskType, Validator
from sdgf.tasktypes.registry import REGISTRY

_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn": {"type": "integer"},
        "role": {"type": "string", "minLength": 1},
        "content": {"type": "string", "minLength": 1},
    },
    "required": ["turn", "role", "content"],
}

_SPAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn": {"type": "integer"},
        "text": {"type": "string", "minLength": 1},
        "category": {"type": "string", "minLength": 1},
    },
    "required": ["turn", "text", "category"],
}


def _messages(record: Record) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def span_turn_errors(record: Record) -> list[str]:
    turns = {m.get("turn") for m in _messages(record)}
    spans = record.get("spans")
    if not isinstance(spans, list):
        return []
    errors = []
    for i, span in enumerate(spans):
        if isinstance(span, dict) and span.get("turn") not in turns:
            errors.append(f"spans[{i}] cites turn={span.get('turn')!r}, which has no message")
    return errors


class ClassificationSpans(TaskType):
    name = "classification_spans"
    generation_modes = ("label_first",)

    def base_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "messages": {"type": "array", "minItems": 1, "items": _MESSAGE_SCHEMA},
                "label": {"type": ["boolean", "string", "integer"]},
                "spans": {"type": "array", "items": _SPAN_SCHEMA},
            },
            "required": ["messages", "label", "spans"],
        }

    def default_validators(self) -> list[Validator]:
        return [span_turn_errors]

    def judge_fields(self) -> tuple[str, ...]:
        # Spans justify the label (non-empty spans mean a positive), so the judge
        # sees only the conversation.
        return ("messages",)


CLASSIFICATION_SPANS = REGISTRY.register(ClassificationSpans())
