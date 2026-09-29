"""The TaskType interface (FRAMEWORK_DESIGN.md §7.1).

A task type is the *kind* of dataset. It supplies what every task of that kind shares:

    base_schema()        JSON schema of the record fields the type owns
    generation_modes     label_first and/or answer_emergent (§11)
    default_axes         coverage axes added unless the spec declares the same name
    default_validators   structural checks run for every task of this type;
                         each takes a record and returns a list of errors (empty = pass)
    answer_extractor     text -> answer or None; required for answer_emergent, since
                         K-vote consistency has nothing to vote on without one (§12.1)
    configure(task)      the instance for a spec, for types with options (answer_format)
    judge_fields         the record fields a judge may see; never the label or any
                         field that encodes it, since L5 judges blind (§6.4)
    label_field          the field L5 fidelity compares the verdict with: the fixed label
                         (label_first) or the answer the model gave (answer_emergent)
    answer_suffix        the answer form a response must end in, for extractable answers
    derive_fields        fills fields readable from the model's output (e.g. the answer
                         from the response) before validation

The spec's output_schema.fields are layered on top by output_schema(); a spec field may
not redefine a field the type owns.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from sdgf.spec.schema import Axis, FieldSpec, GenerationMode, OutputSchemaSection, TaskSection

Record = dict[str, Any]
Validator = Callable[[Record], list[str]]
AnswerExtractor = Callable[[str], Any | None]


class TaskTypeError(ValueError):
    """A task type is misdefined, or a spec is incompatible with its task type."""


class TaskType(ABC):
    name: ClassVar[str]
    generation_modes: ClassVar[tuple[GenerationMode, ...]]

    @abstractmethod
    def base_schema(self) -> dict[str, Any]:
        """JSON schema (type: object) for the fields this task type owns."""

    def default_axes(self) -> list[Axis]:
        return []

    def default_validators(self) -> list[Validator]:
        return []

    def answer_extractor(self) -> AnswerExtractor | None:
        return None

    def configure(self, task: TaskSection) -> TaskType:
        """The instance for a spec's task section; types with options override this."""
        if task.answer_format is not None:
            raise TaskTypeError(
                f"task type {self.name!r} takes no answer_format, got {task.answer_format!r}"
            )
        return self

    def judge_fields(self) -> tuple[str, ...]:
        """Fields a blind judge sees: every type-owned field except the label."""
        return tuple(k for k in self.base_schema()["properties"] if k != self.label_field())

    def label_field(self) -> str:
        return "label"

    def answer_suffix(self) -> str:
        return ""

    def derive_fields(self, record: Record) -> Record:
        return record

    def check_definition(self) -> None:
        """Raise TaskTypeError if the type itself is inconsistent; run at registration."""
        name = getattr(self, "name", "")
        if not isinstance(name, str) or not name:
            raise TaskTypeError(f"{type(self).__name__} must set a non-empty name")
        modes = getattr(self, "generation_modes", ())
        if not modes or any(m not in ("label_first", "answer_emergent") for m in modes):
            raise TaskTypeError(
                f"task type {name!r}: generation_modes must be a non-empty subset of "
                f"('label_first', 'answer_emergent'), got {modes!r}"
            )
        if "answer_emergent" in modes and self.answer_extractor() is None:
            raise TaskTypeError(
                f"task type {name!r} supports answer_emergent but has no answer extractor"
            )
        schema = self.base_schema()
        if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
            raise TaskTypeError(f"task type {name!r}: base_schema must be an object schema")

    def check_mode(self, mode: str) -> None:
        if mode not in self.generation_modes:
            raise TaskTypeError(
                f"task type {self.name!r} does not support generation_mode {mode!r}; "
                f"supported: {list(self.generation_modes)}"
            )

    def output_schema(self, section: OutputSchemaSection | None = None) -> dict[str, Any]:
        """The full record schema: base_schema() plus the spec's extra fields."""
        schema = copy.deepcopy(self.base_schema())
        if section is None or not section.fields:
            return schema
        props = schema["properties"]
        required = list(schema.get("required", []))
        clashes = sorted(set(section.fields) & set(props))
        if clashes:
            raise TaskTypeError(
                f"output_schema.fields {clashes} redefine fields owned by task type {self.name!r}"
            )
        for field_name, spec in section.fields.items():
            props[field_name] = _field_schema(spec)
            if spec.required:
                required.append(field_name)
        schema["required"] = required
        return schema

    def axes(self, spec_axes: list[Axis]) -> list[Axis]:
        """Spec axes, then default axes the spec doesn't override by name."""
        declared = {a.name for a in spec_axes}
        return list(spec_axes) + [a for a in self.default_axes() if a.name not in declared]


def _field_schema(spec: FieldSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"type": [spec.type, "null"] if spec.nullable else spec.type}
    if spec.description:
        out["description"] = spec.description
    if spec.enum is not None:
        out["enum"] = list(spec.enum) + ([None] if spec.nullable and None not in spec.enum else [])
    return out
