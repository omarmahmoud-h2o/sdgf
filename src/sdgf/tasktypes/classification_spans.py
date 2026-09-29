"""Label-first classification with span annotations (FRAMEWORK_DESIGN.md §7.1).

A record is a conversation plus a code-owned label and the spans that justify it:

    messages  [{turn, role, content}, ...]   turns numbered consecutively from 1
    label     boolean, string or integer     fixed by the cell, never by the model
    spans     [{turn, text, category}, ...]  each span points at an existing turn

Role names and alternation belong to the spec (output_schema.turns), so the base
schema only requires a non-empty role string; turn_structure_errors() checks a record
against a TurnStructure and is what L1 calls. Whether a span's text is verbatim, and
which categories or labels need spans, is task policy and lives in the task's hooks.
"""

from __future__ import annotations

from typing import Any

from sdgf.spec.schema import TurnStructure
from sdgf.tasktypes.base import Record, TaskType, Validator
from sdgf.tasktypes.registry import REGISTRY

_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn": {"type": "integer", "minimum": 1},
        "role": {"type": "string", "minLength": 1},
        "content": {"type": "string", "minLength": 1},
    },
    "required": ["turn", "role", "content"],
}

_SPAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn": {"type": "integer", "minimum": 1},
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


def turn_numbering_errors(record: Record) -> list[str]:
    errors = []
    for i, msg in enumerate(_messages(record), start=1):
        if msg.get("turn") != i:
            errors.append(f"messages[{i - 1}] has turn={msg.get('turn')!r}, expected {i}")
    return errors


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


def turn_structure_errors(record: Record, structure: TurnStructure) -> list[str]:
    """Check numbering, role names and alternation against the spec's turn structure."""
    errors = []
    roles = structure.roles
    for i, msg in enumerate(_messages(record)):
        expected_turn = structure.numbered_from + i
        if msg.get("turn") != expected_turn:
            errors.append(f"messages[{i}] has turn={msg.get('turn')!r}, expected {expected_turn}")
        role = msg.get("role")
        if role not in roles:
            errors.append(f"messages[{i}] has unknown role {role!r}; allowed: {roles}")
        elif structure.alternating:
            expected_role = roles[(roles.index(structure.first_role) + i) % len(roles)]
            if role != expected_role:
                errors.append(f"messages[{i}] has role={role!r}, expected {expected_role!r}")
        elif i == 0 and role != structure.first_role:
            errors.append(f"messages[0] has role={role!r}, expected {structure.first_role!r}")
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
        return [turn_numbering_errors, span_turn_errors]


CLASSIFICATION_SPANS = REGISTRY.register(ClassificationSpans())
