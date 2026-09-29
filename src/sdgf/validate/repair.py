"""Repair with error feedback (FRAMEWORK_DESIGN.md §6.4).

On a repairable failure the generator is re-prompted with the *original* prompt plus the
specific validator errors (e.g. "span_not_verbatim at spans[0].text: ..."), up to
validation.repair_tries times, in the same cell with the same code-owned recipe. This
replaces the blind retry of the inherited codebases.

    attempt 0 ─► cascade ─► pass ─────────────► accepted
                        ├► fail_hard ─────────► drop (never repaired: L3, L4)
                        └► fail_repairable ─► attempt 1 with errors ─► ... ─► drop

A generation failure (no parseable JSON, no text) is fed back the same way; its drop
entry has layer "generate". When the tries are exhausted the candidate is dropped with a
logged reason keyed by cell and layer, which the drop log counts for error-rate metrics.
The scheduler then requeues the slot in the same cell (scheduler.reject).

When the task has tools, one tool session is opened per slot and shared by every try,
so the per-record call and token budgets cover repairs too. Its trace so far is handed
to the cascade as context.extra["tool_trace"] (which L3 reads), and every call is
appended to the record's provenance.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from sdgf.generate.generator import GenerationResult, Generator
from sdgf.generate.prompts import Prompt
from sdgf.store.artefacts import JsonlWriter
from sdgf.store.provenance import ProvenanceBuilder
from sdgf.validate.base import Record, ValidationContext, ValidationIssue
from sdgf.validate.cascade import Cascade, CascadeResult
from sdgf.validate.l3_governance import TOOL_TRACE_KEY

log = logging.getLogger(__name__)

GENERATE_STAGE = "generate"  # drop "layer" for candidates that never reached the cascade

FEEDBACK_HEADER = "## Your previous attempt was rejected"


def repair_feedback(errors: Iterable[ValidationIssue | str]) -> str:
    """The text appended to the original prompt for a repair try."""
    lines = [
        FEEDBACK_HEADER,
        "Fix every error below and return the whole record again as one JSON object. "
        "Keep the fixed parameters unchanged.",
    ]
    lines += [f"- {e}" for e in errors]
    return "\n".join(lines)


def _generation_issue(result: GenerationResult) -> ValidationIssue:
    return ValidationIssue(result.error or "generation_failed", result.detail or "no record")


@dataclass(frozen=True)
class Drop:
    cell_id: str | None
    layer: str  # L1..L6, or "generate"
    codes: tuple[str, ...]
    reason: str
    attempts: int  # candidates generated for this slot, including repairs
    hard: bool = False
    errors: tuple[Mapping[str, Any], ...] = ()
    # (layer, codes) of every failed try, the last one included, for per-layer error rates
    history: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "layer": self.layer,
            "codes": list(self.codes),
            "reason": self.reason,
            "attempts": self.attempts,
            "hard": self.hard,
            "errors": [dict(e) for e in self.errors],
            "history": [[layer, list(codes)] for layer, codes in self.history],
        }


class DropLog:
    """Every dropped candidate with its reason; optionally streamed to a JSONL file."""

    def __init__(self, writer: JsonlWriter | None = None):
        self.writer = writer
        self.drops: list[Drop] = []

    def add(self, drop: Drop) -> None:
        self.drops.append(drop)
        if self.writer is not None:
            self.writer.write(drop.to_dict())
        log.info(
            "dropped candidate in cell %s at %s after %d attempt(s): %s",
            drop.cell_id,
            drop.layer,
            drop.attempts,
            drop.reason,
        )

    def __len__(self) -> int:
        return len(self.drops)

    def by_cell_layer(self) -> dict[tuple[str | None, str], int]:
        return dict(Counter((d.cell_id, d.layer) for d in self.drops))

    def by_layer(self) -> dict[str, int]:
        return dict(Counter(d.layer for d in self.drops))

    def by_code(self) -> dict[str, int]:
        return dict(Counter(c for d in self.drops for c in d.codes))


@dataclass
class RepairOutcome:
    cell_id: str | None
    record: Record | None  # the accepted record, None if dropped
    attempts: int
    result: CascadeResult | None = None  # the last cascade run, None if never validated
    drop: Drop | None = None
    history: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)  # (layer, codes)

    @property
    def accepted(self) -> bool:
        return self.record is not None

    @property
    def repairs(self) -> int:
        return self.attempts - 1


class RepairLoop:
    """Generate, validate and repair one candidate slot of one cell."""

    def __init__(
        self,
        generator: Generator,
        cascade: Cascade,
        *,
        repair_tries: int | None = None,
        drop_log: DropLog | None = None,
    ):
        self.generator = generator
        self.cascade = cascade
        tries = repair_tries
        if tries is None:
            tries = generator.compiled.spec.validation.repair_tries
        if tries < 0:
            raise ValueError(f"repair_tries must be >= 0, got {tries}")
        self.repair_tries = tries
        self.drop_log = drop_log if drop_log is not None else DropLog()

    def run(
        self,
        cell_id: str | None,
        recipe: Mapping[str, Any],
        prompt: Prompt,
        provenance: ProvenanceBuilder | None = None,
        extra_context: Mapping[str, Any] | None = None,
    ) -> RepairOutcome:
        if provenance is not None:
            provenance.set_prompt(prompt.text)
        outcome = RepairOutcome(cell_id, None, 0)
        feedback = ""
        session = self.generator.session()
        for attempt in range(self.repair_tries + 1):
            if attempt and provenance is not None:
                provenance.start_repair()
            outcome.attempts = attempt + 1
            traced = len(session.trace) if session is not None else 0
            gen = self.generator.complete(cell_id, recipe, prompt, extra=feedback, session=session)
            if session is not None and provenance is not None:
                for entry in session.trace[traced:]:
                    provenance.add_tool_call(entry)
            if not gen.ok:
                issue = _generation_issue(gen)
                outcome.history.append((GENERATE_STAGE, (issue.code,)))
                if attempt == self.repair_tries:
                    return self._drop(outcome, GENERATE_STAGE, (issue,), hard=False)
                feedback = repair_feedback([issue])
                continue

            extra = dict(extra_context or {})
            if session is not None:
                extra[TOOL_TRACE_KEY] = session.trace_dicts()
            context = ValidationContext(
                cell_id=cell_id, recipe=dict(recipe), attempt=attempt, extra=extra
            )
            result = self.cascade.run(gen.record, context, provenance)
            outcome.result = result
            if result.passed:
                outcome.record = gen.record
                return outcome
            layer = result.failed_layer
            assert layer is not None
            outcome.history.append((layer, tuple(e.code for e in result.errors)))
            if result.hard or attempt == self.repair_tries:
                return self._drop(outcome, layer, result.errors, hard=result.hard)
            feedback = repair_feedback(result.errors)
        raise AssertionError("unreachable")  # pragma: no cover

    def _drop(
        self,
        outcome: RepairOutcome,
        layer: str,
        errors: tuple[ValidationIssue, ...],
        *,
        hard: bool,
    ) -> RepairOutcome:
        codes = tuple(e.code for e in errors)
        why = "hard failure, not repaired" if hard else f"{self.repair_tries} repair(s) exhausted"
        drop = Drop(
            cell_id=outcome.cell_id,
            layer=layer,
            codes=codes,
            reason=f"{layer} {', '.join(codes)} ({why})",
            attempts=outcome.attempts,
            hard=hard,
            errors=tuple(e.to_dict() for e in errors),
            history=tuple(outcome.history),
        )
        self.drop_log.add(drop)
        outcome.drop = drop
        return outcome
