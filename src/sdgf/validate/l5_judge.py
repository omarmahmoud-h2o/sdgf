"""L5: decision judge (FRAMEWORK_DESIGN.md §6.4, §7.3).

A separate judge scores the record **blind to its intended label**: it is handed only
the fields the task type lets a judge see (judge_fields; for classification_spans just
the conversation), never the label, the spans that justify it or "_" keys. Fidelity is
the judge's verdict meaning the intended label, via rubric.verdict.labels (e.g. FAG
breach -> true), or the verdict value itself when no mapping is given.

    agrees, confident            pass
    disagrees, confident         fail_repairable  judge_disagrees (fed back to the generator)
    low confidence, review on    fail_hard        sent_to_review  (queued for a person)
    low confidence, review off   judged as above, and marked for escalation to L6
    judge output unusable        sent_to_review when review is on, else fail_hard judge_error

Low confidence is the verdict confidence below validation.escalation.low_confidence.
A record queued for review leaves the automatic path (fail_hard, so it is neither
repaired nor accepted here); the review queue resolves it. A judge that can't produce a
schema-valid answer is dropped rather than repaired, since regenerating the record
doesn't fix the judge; the drop log counts it so a broken judge shows up in metrics.

Every verdict carries details for L6 and provenance: the judge result, agreement,
low_confidence, and escalate (low confidence, a hard cell or a contestable record, per
validation.escalation). When the rubric requires a reason (always, or flagged and the
record was flagged by disagreement or low confidence), the fallback judge writes it and
it goes into details, the review item and the disagreement message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol

from sdgf.judge.interface import Judge, JudgeError, JudgeParseError, JudgeResult
from sdgf.judge.llm_judge import judge_view
from sdgf.spec.schema import EscalationRules
from sdgf.validate.base import Layer, LayerVerdict, Record, ValidationContext, ValidationIssue

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec

LABEL_FIELD = "label"
HARD_DIFFICULTY = "hard"


@dataclass(frozen=True)
class ReviewItem:
    """A record routed to people instead of being accepted, repaired or dropped."""

    layer: str
    code: str  # low_confidence or judge_error
    cell_id: str | None
    attempt: int
    record: Record
    intended_label: Any
    judge: Mapping[str, Any] | None = None  # JudgeResult.to_dict(), None on judge_error
    reason: str | None = None
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "code": self.code,
            "cell_id": self.cell_id,
            "attempt": self.attempt,
            "record": self.record,
            "intended_label": self.intended_label,
            "judge": None if self.judge is None else dict(self.judge),
            "reason": self.reason,
            "errors": list(self.errors),
        }


class ReviewSink(Protocol):
    """Where L5 sends records for human review (the M8 HITL queue implements it)."""

    def submit(self, item: ReviewItem) -> None: ...


@dataclass
class ListReviewSink:
    """An in-memory review sink, for tests and callers without a queue yet."""

    items: list[ReviewItem] = field(default_factory=list)

    def submit(self, item: ReviewItem) -> None:
        self.items.append(item)


class JudgeLayer(Layer):
    name = "L5"

    def __init__(
        self,
        judge: Judge,
        *,
        fields: Iterable[str],
        labels: Mapping[str, Any] | None = None,
        escalation: EscalationRules | None = None,
        review: ReviewSink | None = None,
        fallback_judge: Judge | None = None,
    ):
        self.judge = judge
        self.schema = judge.schema
        self.fields = tuple(f for f in fields if f != LABEL_FIELD)
        if not self.fields:
            raise JudgeError("L5 needs at least one record field the judge may see")
        values = self.schema.verdict_values
        self.labels = dict(labels) if labels is not None else {v: v for v in values}
        unknown = sorted(set(self.labels) - set(values))
        if unknown:
            raise JudgeError(f"verdict labels name unknown verdict values {unknown}")
        self.escalation = escalation or EscalationRules()
        self.review = review
        if self.schema.reason_required != "never":
            reasoner = fallback_judge or judge
            if not reasoner.writes_reasons:
                raise JudgeError(
                    f"the rubric requires reasons but judge {reasoner.name!r} can't write "
                    "them; give a fallback judge"
                )
            self.reasoner: Judge | None = reasoner
        else:
            self.reasoner = None

    @classmethod
    def from_spec(
        cls,
        compiled: CompiledSpec,
        judge: Judge,
        *,
        review: ReviewSink | None = None,
        fallback_judge: Judge | None = None,
        fields: Iterable[str] | None = None,
    ) -> JudgeLayer:
        """L5 for a compiled spec. The review sink is used only when hitl.review_flagged."""
        from sdgf.tasktypes.registry import REGISTRY

        spec = compiled.spec
        if fields is None:
            fields = REGISTRY.resolve(spec.task).judge_fields()
        return cls(
            judge,
            fields=fields,
            labels=spec.rubric.verdict.labels,
            escalation=spec.validation.escalation,
            review=review if spec.hitl.review_flagged else None,
            fallback_judge=fallback_judge,
        )

    # ── helpers ──────────────────────────────────────────────────

    def agrees(self, verdict: str, label: Any) -> bool:
        if verdict not in self.labels:
            return False
        meant = self.labels[verdict]
        # True == 1 in Python; a bool label only agrees with a bool, and vice versa.
        if isinstance(meant, bool) or isinstance(label, bool):
            return type(meant) is type(label) and meant == label
        return meant == label

    def verdicts_for(self, label: Any) -> list[str]:
        return [v for v in self.schema.verdict_values if self.agrees(v, label)]

    def _escalate(self, record: Record, context: ValidationContext, low: bool) -> bool:
        def fact(name: str) -> Any:
            return context.recipe.get(name, record.get(name))

        hard = str(fact("difficulty") or "").casefold() == HARD_DIFFICULTY
        contestable = fact("contestable") is True
        return (
            low
            or (self.escalation.on_hard_cells and hard)
            or (self.escalation.on_contestable and contestable)
        )

    def _reason(self, view: Record, result: JudgeResult, flagged: bool) -> str | None:
        if self.reasoner is None or not self.schema.needs_reason(flagged):
            return None
        return self.reasoner.explain(view, result)

    def _send_to_review(
        self,
        code: str,
        record: Record,
        label: Any,
        context: ValidationContext,
        result: JudgeResult | None = None,
        reason: str | None = None,
        errors: tuple[str, ...] = (),
    ) -> None:
        self.review.submit(
            ReviewItem(
                layer=self.name,
                code=code,
                cell_id=context.cell_id,
                attempt=context.attempt,
                record={k: v for k, v in record.items() if not k.startswith("_")},
                intended_label=label,
                judge=None if result is None else result.to_dict(),
                reason=reason,
                errors=errors,
            )
        )

    # ── check ────────────────────────────────────────────────────

    def check(self, record: Record, context: ValidationContext) -> LayerVerdict:
        label = context.recipe.get(LABEL_FIELD, record.get(LABEL_FIELD))
        view = judge_view(record, self.fields)
        try:
            result = self.judge.judge(view)
        except JudgeParseError as e:
            return self._judge_failed(record, label, context, tuple(e.errors))

        agrees = self.agrees(result.verdict, label)
        low = result.verdict_confidence < self.escalation.low_confidence
        reason = self._reason(view, result, flagged=low or not agrees)
        details: dict[str, Any] = {
            "judge": result.to_dict(),
            "agrees": agrees,
            "low_confidence": low,
            "escalate": self._escalate(record, context, low),
        }
        if reason is not None:
            details["reason"] = reason
        conf = round(result.verdict_confidence, 3)

        if low and self.review is not None:
            self._send_to_review("low_confidence", record, label, context, result, reason)
            issue = ValidationIssue(
                "sent_to_review",
                f"judge confidence {conf} is below {self.escalation.low_confidence}; "
                "queued for human review",
                path="verdict",
                details={"verdict": result.verdict, "confidence": conf, "agrees": agrees},
            )
            return LayerVerdict(self.name, "fail_hard", (issue,), details)

        if not agrees:
            expected = self.verdicts_for(label)
            message = (
                f"an independent judge read this record as {result.verdict!r}, but the "
                f"fixed label calls for {' or '.join(map(repr, expected)) or 'another verdict'}; "
                "rewrite the content so it clearly matches the fixed label"
            )
            if reason:
                message += f". Judge's reason: {reason}"
            issue = ValidationIssue(
                "judge_disagrees",
                message,
                path="verdict",
                details={
                    "verdict": result.verdict,
                    "expected": expected,
                    "confidence": conf,
                    "scores": dict(result.scores),
                },
            )
            return LayerVerdict(self.name, "fail_repairable", (issue,), details)

        return LayerVerdict(self.name, "pass", (), details)

    def _judge_failed(
        self, record: Record, label: Any, context: ValidationContext, errors: tuple[str, ...]
    ) -> LayerVerdict:
        details = {"judge": None, "judge_errors": list(errors)}
        if self.review is not None:
            self._send_to_review("judge_error", record, label, context, errors=errors)
            code, message = "sent_to_review", "judge output unusable; queued for human review"
        else:
            code, message = "judge_error", "judge output unusable: " + "; ".join(errors)
        issue = ValidationIssue(code, message, details={"judge_errors": list(errors)})
        return LayerVerdict(self.name, "fail_hard", (issue,), details)
