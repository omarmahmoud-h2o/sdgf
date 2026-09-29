"""Typed judge interface (FRAMEWORK_DESIGN.md §7.3, D9, D11).

The rubric in task.yaml compiles into a JudgeSchema: a verdict enum, one field per
rubric criterion (an enum, or a bounded int, at most 255 values each) and a confidence
in [0, 1] for every one of those fields. A free-text reason is part of the output
only when the rubric requires one (reason_required flagged or always), and then only
from a judge that can write text: decision models such as Jev can't, so the reason
comes from the fallback generative judge.

Judge output looks like:

    {"verdict": "breach",
     "scores": {"advice_tier": "PERSONAL_ADVICE", "realism": 4},
     "confidence": {"verdict": 0.93, "advice_tier": 0.88, "realism": 0.61},
     "reason": "..."}                      # only when the rubric requires it
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import jsonschema

from sdgf.spec.schema import MAX_CHOICES, RubricSection

Record = dict[str, Any]
ReasonPolicy = Literal["never", "flagged", "always"]
VERDICT = "verdict"


class JudgeError(RuntimeError):
    """A judge could not be built or called."""


class RubricCompileError(ValueError):
    """The rubric can't be expressed as a typed judge schema."""


class JudgeParseError(ValueError):
    """Judge output does not match the compiled schema. `errors` lists the problems."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("judge output does not match the schema: " + "; ".join(errors))


@dataclass(frozen=True)
class JudgeField:
    """One typed output field: an enum of strings or a bounded int range."""

    name: str
    kind: Literal["enum", "int"]
    choices: tuple[str, ...] = ()
    min: int | None = None
    max: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise RubricCompileError("a judge field needs a name")
        if self.kind == "enum":
            if len(set(self.choices)) != len(self.choices):
                raise RubricCompileError(f"{self.name}: choices repeat")
            if not 2 <= len(self.choices) <= MAX_CHOICES:
                raise RubricCompileError(f"{self.name}: needs 2..{MAX_CHOICES} choices")
        elif self.kind == "int":
            if self.min is None or self.max is None or self.max <= self.min:
                raise RubricCompileError(f"{self.name}: an int field needs min < max")
            if self.size > MAX_CHOICES:
                raise RubricCompileError(f"{self.name}: more than {MAX_CHOICES} values")
        else:
            raise RubricCompileError(f"{self.name}: unknown kind {self.kind!r}")

    @property
    def size(self) -> int:
        if self.kind == "enum":
            return len(self.choices)
        return self.max - self.min + 1

    def values(self) -> tuple[Any, ...]:
        if self.kind == "enum":
            return self.choices
        return tuple(range(self.min, self.max + 1))

    def accepts(self, value: Any) -> bool:
        if self.kind == "enum":
            return isinstance(value, str) and value in self.choices
        # bool is an int subclass; a judge answering True for a 1..5 score is wrong.
        return (
            isinstance(value, int) and not isinstance(value, bool) and self.min <= value <= self.max
        )

    def json_schema(self) -> dict[str, Any]:
        if self.kind == "enum":
            s: dict[str, Any] = {"type": "string", "enum": list(self.choices)}
        else:
            s = {"type": "integer", "minimum": self.min, "maximum": self.max}
        if self.description:
            s["description"] = self.description
        return s


@dataclass(frozen=True)
class JudgeSchema:
    """The judge's output type, compiled from a rubric."""

    verdict: JudgeField
    criteria: tuple[JudgeField, ...] = ()
    reason_required: ReasonPolicy = "never"

    def __post_init__(self) -> None:
        if self.verdict.kind != "enum":
            raise RubricCompileError("the verdict must be an enum")
        names = [c.name for c in self.criteria]
        if VERDICT in names:
            raise RubricCompileError(f"a criterion can't be named {VERDICT!r}")
        if len(set(names)) != len(names):
            raise RubricCompileError(f"criterion names repeat: {names}")
        if self.reason_required not in ("never", "flagged", "always"):
            raise RubricCompileError(f"unknown reason policy {self.reason_required!r}")

    @property
    def fields(self) -> tuple[JudgeField, ...]:
        """Every field that carries a confidence: the verdict first, then criteria."""
        return (self.verdict, *self.criteria)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def verdict_values(self) -> tuple[str, ...]:
        return self.verdict.choices

    def criterion(self, name: str) -> JudgeField:
        for c in self.criteria:
            if c.name == name:
                return c
        raise KeyError(name)

    def needs_reason(self, flagged: bool = False) -> bool:
        """Whether this record needs a reason: always, or only when it was flagged."""
        return self.reason_required == "always" or (self.reason_required == "flagged" and flagged)

    def json_schema(self, with_reason: bool | None = None) -> dict[str, Any]:
        """JSON schema of the judge output.

        with_reason=None includes the reason property only when the rubric may require
        one (optional for flagged, required for always); False gives the pure decision
        schema a typed decision model answers; True forces it in (required) for a
        reason-writing call.
        """
        if with_reason is None:
            with_reason = self.reason_required != "never"
            reason_mandatory = self.reason_required == "always"
        else:
            reason_mandatory = with_reason
        confidence = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        props: dict[str, Any] = {
            VERDICT: self.verdict.json_schema(),
            "scores": {
                "type": "object",
                "properties": {c.name: c.json_schema() for c in self.criteria},
                "required": [c.name for c in self.criteria],
                "additionalProperties": False,
            },
            "confidence": {
                "type": "object",
                "properties": {n: dict(confidence) for n in self.field_names},
                "required": list(self.field_names),
                "additionalProperties": False,
            },
        }
        required = [VERDICT, "scores", "confidence"]
        if with_reason:
            props["reason"] = {"type": "string", "minLength": 1}
            if reason_mandatory:
                required.append("reason")
        return {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }

    def parse(self, data: Any, with_reason: bool | None = None) -> JudgeResult:
        """Check judge output against the schema and return a typed JudgeResult."""
        validator = jsonschema.Draft202012Validator(self.json_schema(with_reason))
        errors = []
        for e in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
            where = ".".join(str(p) for p in e.absolute_path) or "<root>"
            errors.append(f"{where}: {e.message}")
        if not errors:
            # jsonschema treats True as an integer and 1.0 as one too; be strict.
            for c in self.criteria:
                if not c.accepts(data["scores"][c.name]):
                    errors.append(f"scores.{c.name}: {data['scores'][c.name]!r} not allowed")
            for n, v in data["confidence"].items():
                if isinstance(v, bool):
                    errors.append(f"confidence.{n}: must be a number, not a bool")
        if errors:
            raise JudgeParseError(errors)
        return JudgeResult(
            verdict=data[VERDICT],
            scores=dict(data["scores"]),
            confidence={k: float(v) for k, v in data["confidence"].items()},
            reason=data.get("reason"),
        )


def compile_rubric(rubric: RubricSection) -> JudgeSchema:
    """Compile a spec rubric into the typed judge output schema."""
    criteria = []
    for c in rubric.criteria:
        if c.values is not None:
            f = JudgeField(c.name, "enum", choices=tuple(c.values), description=c.description)
        else:
            f = JudgeField(c.name, "int", min=c.min, max=c.max, description=c.description)
        criteria.append(f)
    verdict = JudgeField(
        VERDICT,
        "enum",
        choices=tuple(rubric.verdict.values),
        description=rubric.verdict.description,
    )
    return JudgeSchema(verdict, tuple(criteria), rubric.reason_required)


@dataclass(frozen=True)
class JudgeResult:
    verdict: str
    scores: Mapping[str, Any] = field(default_factory=dict)
    confidence: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None

    @property
    def verdict_confidence(self) -> float:
        return self.confidence[VERDICT]

    def min_confidence(self) -> float:
        return min(self.confidence.values())

    def with_reason(self, reason: str) -> JudgeResult:
        return JudgeResult(self.verdict, self.scores, self.confidence, reason)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            VERDICT: self.verdict,
            "scores": dict(self.scores),
            "confidence": dict(self.confidence),
        }
        if self.reason is not None:
            d["reason"] = self.reason
        return d


def verdict_means(labels: Mapping[str, Any], verdict: str, label: Any) -> bool:
    """Whether a judge verdict means the intended label under rubric.verdict.labels."""
    if verdict not in labels:
        return False
    meant = labels[verdict]
    # True == 1 in Python; a bool label only agrees with a bool, and vice versa.
    if isinstance(meant, bool) or isinstance(label, bool):
        return type(meant) is type(label) and meant == label
    return meant == label


class Judge(ABC):
    """A judge scores one record against the compiled rubric, blind to its label.

    Callers pass the record with the intended label already removed; a judge never
    sees it. `writes_reasons` marks judges that can also serve as the fallback
    reason-writing judge.
    """

    name: str = "judge"
    writes_reasons: bool = False

    def __init__(self, schema: JudgeSchema):
        self.schema = schema

    @abstractmethod
    def judge(self, record: Record) -> JudgeResult:
        """Return verdict, rubric scores and per-field confidence (no reason)."""

    def explain(self, record: Record, result: JudgeResult) -> str:
        """Write a reason for a verdict. Only judges with writes_reasons implement it."""
        raise JudgeError(f"judge {self.name!r} can't write reasons; use a fallback judge")
