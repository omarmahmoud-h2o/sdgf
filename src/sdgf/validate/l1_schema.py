"""L1: schema and layout (FRAMEWORK_DESIGN.md §6.4).

Three stages, each run only if the previous one is clean, since a record with the
wrong shape makes the later checks report noise rather than the real problem:

    1. schema     the task type's JSON schema plus the spec's extra fields
                  (required fields, field types, enums)
    2. turns      numbering from numbered_from, known roles, alternation starting
                  from first_role; only when the spec declares output_schema.turns
    3. task type  the task type's default_validators (e.g. spans cite existing turns)

Every failure is repairable: the errors go back to the generator as feedback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from sdgf.spec.schema import TurnStructure
from sdgf.tasktypes.base import TaskType, Validator
from sdgf.tasktypes.registry import REGISTRY
from sdgf.validate.base import Layer, LayerVerdict, Record, ValidationContext, ValidationIssue

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec


def format_path(parts: Iterable[Any]) -> str | None:
    """["messages", 3, "content"] -> "messages[3].content"; the root is None."""
    out = ""
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or None


def schema_issues(validator: Draft202012Validator, record: Any) -> list[ValidationIssue]:
    errors = sorted(
        validator.iter_errors(record), key=lambda e: (list(map(str, e.path)), e.message)
    )
    return [
        ValidationIssue(
            code=f"schema_{e.validator}",
            message=e.message,
            path=format_path(e.absolute_path),
            details={"schema_path": format_path(e.absolute_schema_path)},
        )
        for e in errors
    ]


def turn_issues(record: Record, structure: TurnStructure) -> list[ValidationIssue]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    roles = structure.roles
    start = roles.index(structure.first_role)
    issues = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        turn, role = msg.get("turn"), msg.get("role")
        expected_turn = structure.numbered_from + i
        if turn != expected_turn:
            issues.append(
                ValidationIssue(
                    "turn_numbering",
                    f"messages[{i}] has turn={turn!r}, expected {expected_turn}",
                    f"messages[{i}].turn",
                    {"expected": expected_turn, "got": turn},
                )
            )
        if role not in roles:
            issues.append(
                ValidationIssue(
                    "unknown_role",
                    f"messages[{i}] has unknown role {role!r}; allowed: {roles}",
                    f"messages[{i}].role",
                    {"allowed": list(roles), "got": role},
                )
            )
            continue
        if structure.alternating:
            expected_role = roles[(start + i) % len(roles)]
        elif i == 0:
            expected_role = structure.first_role
        else:
            continue
        if role != expected_role:
            code = "first_role" if i == 0 else "role_alternation"
            issues.append(
                ValidationIssue(
                    code,
                    f"messages[{i}] has role={role!r}, expected {expected_role!r}",
                    f"messages[{i}].role",
                    {"expected": expected_role, "got": role},
                )
            )
    return issues


def validator_code(validator: Validator) -> str:
    name = getattr(validator, "__name__", type(validator).__name__)
    return name.removesuffix("_errors")


def task_type_issues(record: Record, validators: Sequence[Validator]) -> list[ValidationIssue]:
    issues = []
    for validator in validators:
        code = validator_code(validator)
        issues += [
            ValidationIssue(code, message, details={"validator": code})
            for message in validator(record)
        ]
    return issues


class SchemaLayer(Layer):
    name = "L1"

    def __init__(
        self,
        schema: Mapping[str, Any],
        turns: TurnStructure | None = None,
        validators: Sequence[Validator] = (),
    ):
        Draft202012Validator.check_schema(schema)
        self.schema = schema
        self.turns = turns
        self.validators = tuple(validators)
        self._validator = Draft202012Validator(schema)

    @classmethod
    def from_spec(cls, compiled: CompiledSpec, task_type: TaskType | None = None) -> SchemaLayer:
        spec = compiled.spec
        task_type = task_type or REGISTRY.resolve(spec.task)
        return cls(
            task_type.output_schema(spec.output_schema),
            spec.output_schema.turns,
            task_type.default_validators(),
        )

    def check(self, record: Record, context: ValidationContext) -> LayerVerdict:
        issues = schema_issues(self._validator, record)
        if not issues and self.turns is not None:
            issues = turn_issues(record, self.turns)
        if not issues:
            issues = task_type_issues(record, self.validators)
        return self.verdict(issues, repairable=True)
