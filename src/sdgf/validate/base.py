"""Validation layer interface (FRAMEWORK_DESIGN.md §6.4).

A Layer checks one candidate record and returns a LayerVerdict with one of three
outcomes:

    pass              the record is fine at this layer
    fail_repairable   the generator can be re-prompted with the errors (L1, L2, L5, L6)
    fail_hard         drop the record, never repair (L3 governance, L4 overlap)

Errors are machine-readable ValidationIssue values: a stable `code` that drop logs and
error-rate metrics group by, a human `message` that repair feeds back to the
generator, and optional `path` / `details` locating the problem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from sdgf.spec.schema import ALL_LAYERS, LayerName
from sdgf.store.provenance import LayerOutcome

Record = dict[str, Any]

OUTCOMES: tuple[str, ...] = ("pass", "fail_repairable", "fail_hard")


class ValidationError(ValueError):
    """A layer or verdict is malformed (a pipeline bug, not a bad record)."""


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None  # e.g. "messages[3].content" or "spans[0].text"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValidationError("a validation issue needs a code")

    def __str__(self) -> str:
        where = f" at {self.path}" if self.path else ""
        return f"{self.code}{where}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            d["path"] = self.path
        if self.details:
            d["details"] = dict(self.details)
        return d


@dataclass(frozen=True)
class ValidationContext:
    """What a layer may know about a candidate besides the record itself."""

    cell_id: str | None = None
    recipe: Mapping[str, Any] = field(default_factory=dict)
    attempt: int = 0  # 0 is the first generation, n is the n-th repair
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerVerdict:
    layer: LayerName
    outcome: LayerOutcome
    errors: tuple[ValidationIssue, ...] = ()
    # What the layer found beyond pass/fail, e.g. L5's judge result and whether the
    # record should escalate to L6. Allowed on a pass too.
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer not in ALL_LAYERS:
            raise ValidationError(f"unknown layer {self.layer!r}")
        if self.outcome not in OUTCOMES:
            raise ValidationError(f"layer {self.layer}: unknown outcome {self.outcome!r}")
        if (self.outcome == "pass") != (not self.errors):
            raise ValidationError(
                f"layer {self.layer}: a pass carries no errors and a failure at least one"
            )

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"

    @property
    def repairable(self) -> bool:
        return self.outcome == "fail_repairable"

    @property
    def hard(self) -> bool:
        return self.outcome == "fail_hard"

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(e.code for e in self.errors)

    def messages(self) -> tuple[str, ...]:
        return tuple(str(e) for e in self.errors)


class Layer(ABC):
    """One step of the cascade. Subclasses set `name` and implement check()."""

    name: LayerName

    @abstractmethod
    def check(self, record: Record, context: ValidationContext) -> LayerVerdict: ...

    def verdict(
        self, errors: Iterable[ValidationIssue] = (), *, repairable: bool = True
    ) -> LayerVerdict:
        """Pass if there are no errors, else fail_repairable or fail_hard."""
        errors = tuple(errors)
        if not errors:
            return LayerVerdict(self.name, "pass")
        return LayerVerdict(self.name, "fail_repairable" if repairable else "fail_hard", errors)
