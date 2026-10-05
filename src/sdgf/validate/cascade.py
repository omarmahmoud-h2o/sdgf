"""The validation cascade (FRAMEWORK_DESIGN.md §6.4).

Layers run in the order configured in spec.validation.layers (cheapest first, L1..L6)
and the cascade stops at the first layer that does not pass, so the paid judge layers
never see a record a free layer already rejected. The result says which layer failed
and whether the failure is repairable, which is all repair and the drop log need.

Each layer sees the verdicts of the layers before it in context.previous.

A layer that raises is a pipeline bug, not a bad record, so the exception propagates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from sdgf.spec.schema import ALL_LAYERS, LayerName, ValidationSection
from sdgf.store.provenance import LayerOutcome, ProvenanceBuilder
from sdgf.tracing import NullTracer, Tracer, layer_outputs
from sdgf.validate.base import (
    Layer,
    LayerVerdict,
    Record,
    ValidationContext,
    ValidationError,
    ValidationIssue,
)


class CascadeError(ValidationError):
    """The cascade is misconfigured: a layer is missing, repeated or out of order."""


@dataclass(frozen=True)
class CascadeResult:
    verdicts: tuple[LayerVerdict, ...]

    @property
    def outcome(self) -> LayerOutcome:
        return self.verdicts[-1].outcome if self.verdicts else "pass"

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
    def failed_layer(self) -> LayerName | None:
        return None if self.passed else self.verdicts[-1].layer

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return () if self.passed else self.verdicts[-1].errors

    @property
    def layers_run(self) -> tuple[LayerName, ...]:
        return tuple(v.layer for v in self.verdicts)


# Span names for the check layers (README "Tracing with LangSmith").
LAYER_SPAN_NAMES: dict[str, str] = {
    "L1": "L1 schema",
    "L2": "L2 rules",
    "L3": "L3 safety scan",
    "L4": "L4 copy check",
    "L5": "L5 judge",
    "L6": "L6 extra votes",
}


class Cascade:
    def __init__(self, layers: Sequence[Layer], *, tracer: Tracer | None = None):
        names = [layer.name for layer in layers]
        unknown = [n for n in names if n not in ALL_LAYERS]
        if unknown:
            raise CascadeError(f"unknown layer name(s) {unknown}; expected {list(ALL_LAYERS)}")
        if len(set(names)) != len(names):
            raise CascadeError(f"layers must not repeat: {names}")
        if names != sorted(names, key=ALL_LAYERS.index):
            raise CascadeError(f"layers must run cheapest first, in order L1..L6: {names}")
        self.layers: tuple[Layer, ...] = tuple(layers)
        self.tracer: Tracer = tracer if tracer is not None else NullTracer()

    @classmethod
    def from_config(
        cls,
        config: ValidationSection | Sequence[str],
        available: Mapping[str, Layer] | Iterable[Layer],
        *,
        tracer: Tracer | None = None,
    ) -> Cascade:
        """Pick the configured layers, in configured order, from the implementations given.

        Implementations for layers the spec doesn't enable are ignored; an enabled layer
        with no implementation is an error rather than a silently skipped check.
        """
        order = config.layers if isinstance(config, ValidationSection) else list(config)
        if not isinstance(available, Mapping):
            available = {layer.name: layer for layer in available}
        missing = [name for name in order if name not in available]
        if missing:
            raise CascadeError(
                f"validation.layers enables {missing} but no implementation was given "
                f"(have {sorted(available)})"
            )
        for name, layer in available.items():
            if layer.name != name:
                raise CascadeError(f"layer registered as {name!r} reports name {layer.name!r}")
        return cls([available[name] for name in order], tracer=tracer)

    @property
    def names(self) -> tuple[LayerName, ...]:
        return tuple(layer.name for layer in self.layers)

    def run(
        self,
        record: Record,
        context: ValidationContext | None = None,
        provenance: ProvenanceBuilder | None = None,
    ) -> CascadeResult:
        context = context or ValidationContext()
        verdicts: list[LayerVerdict] = []
        for layer in self.layers:
            name = LAYER_SPAN_NAMES.get(layer.name, layer.name)
            with self.tracer.span(name, "chain", {"attempt": context.attempt}) as span:
                verdict = layer.check(record, replace(context, previous=tuple(verdicts)))
                if isinstance(verdict, LayerVerdict):
                    span.end(outputs=layer_outputs(verdict))
            if not isinstance(verdict, LayerVerdict):
                raise CascadeError(f"layer {layer.name} returned {type(verdict).__name__}")
            if verdict.layer != layer.name:
                raise CascadeError(f"layer {layer.name} returned a verdict for {verdict.layer}")
            verdicts.append(verdict)
            if provenance is not None:
                provenance.add_layer_result(
                    verdict.layer,
                    verdict.outcome,
                    verdict.messages(),
                    ballots=verdict.details.get("ballots", ()),
                )
            if not verdict.passed:
                break
        return CascadeResult(tuple(verdicts))
