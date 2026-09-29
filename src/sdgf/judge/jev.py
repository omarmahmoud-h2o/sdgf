"""Jev (TypeSafe AI, System One) judge: an interface-only stub (FRAMEWORK_DESIGN.md §7.3).

Jev is the primary candidate decision model for L5 (D11): typed verdict + rubric
scores + per-field confidence, no free text. Its API isn't documented to us yet
(§16 Q1: how a per-task rubric is supplied, data retention, rate limits), so nothing
here guesses it. A spec can select it with models.judge.backend: jev, and both the
model registry and build_judge fail with a clear error pointing at those sections
instead of silently falling back to another judge.

When the API is documented, implement JevJudge.judge() against the compiled
JudgeSchema (json_schema(with_reason=False) is the decision schema Jev answers) and
replace the factory below. Reasons stay with the fallback judge: writes_reasons is False.
"""

from __future__ import annotations

from typing import Any, NoReturn

from sdgf.judge.interface import Judge, JudgeResult, JudgeSchema, Record
from sdgf.models.base import ModelBackend, ModelBackendError
from sdgf.spec.schema import ModelConfig

JEV_BACKEND = "jev"

NOT_IMPLEMENTED = (
    "the Jev judge (TypeSafe AI System One) is an interface stub: its API isn't "
    "documented yet, so sdgf doesn't implement it. See FRAMEWORK_DESIGN.md §7.3 (judge "
    "vision) and §16 Q1 (Jev in practice). Until then select another judge backend in "
    "models.judge, e.g. openai_compat or anthropic behind the LLM judge."
)


class JevNotImplementedError(NotImplementedError, ModelBackendError):
    """Jev was selected but has no implementation yet.

    Also a ModelBackendError, so the model registry names the failing models.<stage>.
    """


def not_implemented() -> NoReturn:
    raise JevNotImplementedError(NOT_IMPLEMENTED)


class JevJudge(Judge):
    """The typed decision judge Jev would provide. Constructing it fails clearly."""

    name = JEV_BACKEND
    writes_reasons = False

    def __init__(self, schema: JudgeSchema, config: ModelConfig | None = None, **_: Any):
        not_implemented()

    def judge(self, record: Record) -> JudgeResult:  # pragma: no cover - unreachable
        not_implemented()


def factory(config: ModelConfig) -> ModelBackend:
    """Model-registry factory for backend: jev. Jev returns decisions, not text, and
    has no implementation yet, so selecting it fails at build time."""
    not_implemented()
