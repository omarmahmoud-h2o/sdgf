"""L6: adaptive consistency (FRAMEWORK_DESIGN.md §6.4, §7.3, §11).

L6 runs only on escalated records: L5 marks a record for escalation when its verdict
confidence is low, the cell is hard or the record is contestable (validation.escalation).
Without an L5 verdict the hard / contestable facts are read from the recipe directly.
Every other record passes untouched, so the paid K votes are spent only where needed.

    label_first, trusted judge, L5 confident and agreeing   pass on confidence, no votes
    label_first, otherwise                                  K judge votes; the majority
                                                            must mean the fixed label
    answer_emergent                                         K answers; the majority
                                                            becomes the answer

A judge is trusted only once its calibration passed (judge/calibration.py); until then
its confidence isn't evidence and K votes are taken. answer_emergent always votes, since
there the votes produce the answer rather than check it.

An unparseable vote (a JudgeParseError, or an answer the extractor can't read) is an
abstention: it counts in neither the numerator nor the denominator. DS²-Instruct divided
by K including None answers (§12.1), so abstentions silently pulled good records below
the threshold. A majority means more than half of the votes cast. With no votes cast at
all the judge is broken, not the record, so the record is dropped (fail_hard) rather than
repaired.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from sdgf.judge.calibration import CalibrationResult, trust_for
from sdgf.judge.interface import Judge, JudgeError, JudgeParseError
from sdgf.judge.llm_judge import judge_view
from sdgf.models.base import ModelBackend
from sdgf.spec.schema import EscalationRules, GenerationMode
from sdgf.tasktypes.base import AnswerExtractor
from sdgf.validate.base import Layer, LayerVerdict, Record, ValidationContext, ValidationIssue
from sdgf.validate.l5_judge import LABEL_FIELD, escalates, verdict_means

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec

# (blind record view, vote index) -> response text, or None when the model gave none
Answerer = Callable[[Record, int], str | None]


class ConsistencyError(ValueError):
    """L6 is misconfigured for its generation mode."""


class BackendAnswerer:
    """Answers a question K times on one backend, cycling temperatures (as DS²-Instruct)."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        question_field: str = "question",
        suffix: str = "",
        max_tokens: int = 1024,
        temperatures: Sequence[float] = (0.7, 0.8, 0.9),
    ):
        if not temperatures:
            raise ConsistencyError("BackendAnswerer needs at least one temperature")
        self.backend = backend
        self.question_field = question_field
        self.suffix = suffix
        self.max_tokens = max_tokens
        self.temperatures = tuple(temperatures)

    def __call__(self, view: Record, index: int) -> str | None:
        prompt = str(view[self.question_field])
        if self.suffix:
            prompt += "\n\n" + self.suffix
        temperature = self.temperatures[index % len(self.temperatures)]
        return self.backend.call(prompt, self.max_tokens, temperature).text


def majority(votes: Iterable[Any]) -> tuple[Any, int, int]:
    """(most common vote, its count, votes cast), with None votes as abstentions.

    Ties go to the value seen first, so the result is deterministic for a given order.
    """
    cast = [v for v in votes if v is not None]
    if not cast:
        return None, 0, 0
    value, count = Counter(cast).most_common(1)[0]
    return value, count, len(cast)


class ConsistencyLayer(Layer):
    name = "L6"

    def __init__(
        self,
        *,
        mode: GenerationMode,
        k: int,
        judge: Judge | None = None,
        fields: Iterable[str] = (),
        labels: Mapping[str, Any] | None = None,
        escalation: EscalationRules | None = None,
        trusted: bool = False,
        answerer: Answerer | None = None,
        extractor: AnswerExtractor | None = None,
        escalated_only: bool = True,
    ):
        if k < 1:
            raise ConsistencyError("consistency needs k >= 1")
        self.mode = mode
        self.k = k
        self.fields = tuple(f for f in fields if f != LABEL_FIELD)
        if not self.fields:
            raise ConsistencyError("L6 needs at least one record field the voters may see")
        self.escalation = escalation or EscalationRules()
        self.trusted = trusted
        self.escalated_only = escalated_only
        self.judge = judge
        self.answerer = answerer
        self.extractor = extractor
        if mode == "label_first":
            if judge is None:
                raise ConsistencyError("label_first consistency votes with a judge; give one")
            values = judge.schema.verdict_values
            self.labels = dict(labels) if labels is not None else {v: v for v in values}
            unknown = sorted(set(self.labels) - set(values))
            if unknown:
                raise JudgeError(f"verdict labels name unknown verdict values {unknown}")
        elif mode == "answer_emergent":
            if answerer is None:
                raise ConsistencyError("answer_emergent consistency needs an answerer")
            if extractor is None:
                # §12.1: DS²-Instruct skipped filtering for tasks with no extractor.
                raise ConsistencyError(
                    "answer_emergent consistency needs an answer extractor; without one "
                    "there is nothing to vote on"
                )
            self.labels = {}
        else:
            raise ConsistencyError(f"unknown generation mode {mode!r}")

    @classmethod
    def from_spec(
        cls,
        compiled: CompiledSpec,
        *,
        judge: Judge | None = None,
        trusted: bool = False,
        answerer: Answerer | None = None,
        fields: Iterable[str] | None = None,
        calibration: CalibrationResult | None = None,
        judge_name: str | None = None,
    ) -> ConsistencyLayer:
        """L6 for a compiled spec, with K from validation.consistency_k.

        The judge is trusted when `calibration` (judge/calibration.py) passed for this
        spec_version and judge model (judge_name, default the spec's models.judge), or
        when the caller says so with `trusted`. The extractor is the task type's.
        """
        from sdgf.tasktypes.registry import REGISTRY

        spec = compiled.spec
        task_type = REGISTRY.resolve(spec.task)
        return cls(
            mode=spec.task.generation_mode,
            k=spec.validation.consistency_k,
            judge=judge,
            fields=task_type.judge_fields() if fields is None else fields,
            labels=spec.rubric.verdict.labels,
            escalation=spec.validation.escalation,
            trusted=trusted or trust_for(compiled, calibration, judge_name),
            answerer=answerer,
            extractor=task_type.answer_extractor(),
        )

    # ── check ────────────────────────────────────────────────────

    def _escalated(self, record: Record, context: ValidationContext) -> bool:
        l5 = context.verdict_of("L5")
        if l5 is not None and "escalate" in l5.details:
            return bool(l5.details["escalate"])
        return escalates(self.escalation, record, context, low=False)

    def check(self, record: Record, context: ValidationContext) -> LayerVerdict:
        if self.escalated_only and not self._escalated(record, context):
            return LayerVerdict(self.name, "pass", (), {"escalated": False, "method": "skipped"})
        view = judge_view(record, self.fields)
        if self.mode == "answer_emergent":
            return self._answer_votes(view)
        if self.trusted:
            verdict = self._by_confidence(context)
            if verdict is not None:
                return verdict
        label = context.recipe.get(LABEL_FIELD, record.get(LABEL_FIELD))
        return self._judge_votes(view, label)

    def _by_confidence(self, context: ValidationContext) -> LayerVerdict | None:
        """Pass on a trusted judge's confident, agreeing L5 verdict; None means vote."""
        l5 = context.verdict_of("L5")
        if l5 is None or not l5.passed or not l5.details.get("judge"):
            return None
        confidence = l5.details["judge"]["confidence"]["verdict"]
        if not l5.details.get("agrees") or confidence < self.escalation.low_confidence:
            return None
        details = {"escalated": True, "method": "confidence", "confidence": confidence}
        return LayerVerdict(self.name, "pass", (), details)

    def _judge_votes(self, view: Record, label: Any) -> LayerVerdict:
        verdicts: list[str | None] = []
        for _ in range(self.k):
            try:
                verdicts.append(self.judge.judge(view).verdict)
            except JudgeParseError:
                verdicts.append(None)
        cast = [v for v in verdicts if v is not None]
        agree = sum(verdict_means(self.labels, v, label) for v in cast)
        top, _, _ = majority(verdicts)
        details = {
            "escalated": True,
            "method": "votes",
            "k": self.k,
            "votes": verdicts,
            "cast": len(cast),
            "abstained": self.k - len(cast),
            "agree": agree,
            "majority": top,
        }
        if not cast:
            return self._no_votes(details)
        if 2 * agree > len(cast):
            return LayerVerdict(self.name, "pass", (), details)
        issue = ValidationIssue(
            "consistency_disagrees",
            f"only {agree} of {len(cast)} independent judge votes read this record as "
            f"matching the fixed label (most voted {top!r}); rewrite the content so it "
            "clearly and unambiguously matches the fixed label",
            path="verdict",
            details={"agree": agree, "cast": len(cast), "majority": top},
        )
        return LayerVerdict(self.name, "fail_repairable", (issue,), details)

    def _answer_votes(self, view: Record) -> LayerVerdict:
        responses = [self.answerer(view, i) for i in range(self.k)]
        answers = [None if r is None else self.extractor(r) for r in responses]
        top, count, cast = majority(answers)
        details: dict[str, Any] = {
            "escalated": True,
            "method": "votes",
            "k": self.k,
            "votes": answers,
            "cast": cast,
            "abstained": self.k - cast,
            "agree": count,
            "majority": top,
        }
        if not cast:
            return self._no_votes(details)
        if 2 * count > cast:
            details["answer"] = top
            details["response"] = next(r for r, a in zip(responses, answers) if a == top)
            return LayerVerdict(self.name, "pass", (), details)
        issue = ValidationIssue(
            "consistency_no_majority",
            f"no answer won a majority of the {cast} readable votes (most common "
            f"{top!r} with {count}); make the question precise enough to have one answer",
            details={"agree": count, "cast": cast, "majority": top},
        )
        return LayerVerdict(self.name, "fail_repairable", (issue,), details)

    def _no_votes(self, details: dict[str, Any]) -> LayerVerdict:
        issue = ValidationIssue(
            "consistency_no_votes",
            f"all {self.k} consistency votes were unparseable",
            details={"k": self.k},
        )
        return LayerVerdict(self.name, "fail_hard", (issue,), details)
