"""Pick the judge for a stage's backend (judge, fallback_judge or consistency_judge).

A decision backend (backend: jev) gets JevJudge, which asks the rubric as typed
questions; any other backend is a text model behind LLMJudge. Both take the same
context, worked examples and blind judge view from the spec. An explicit backend passed
by the caller (a test, or a hand-built backend) wins over the spec, as it does for
ModelRegistry overrides.
"""

from __future__ import annotations

from typing import Any

from sdgf.judge.interface import Judge, JudgeError
from sdgf.judge.jev import JEV_BACKEND, JevJudge
from sdgf.judge.llm_judge import LLMJudge
from sdgf.models.base import ModelBackend, ModelBackendError
from sdgf.models.registry import REGISTRY
from sdgf.spec.compile import CompiledSpec


def judge_from_spec(
    compiled: CompiledSpec, backend: ModelBackend, *, stage: str = "judge", **kwargs: Any
) -> Judge:
    """The judge for this backend: JevJudge for jev, LLMJudge for a text model."""
    if backend.name == JEV_BACKEND:
        return JevJudge.from_spec(compiled, backend, stage=stage, **kwargs)
    return LLMJudge.from_spec(compiled, backend, stage=stage, **kwargs)


def build_judge(
    compiled: CompiledSpec,
    backend: ModelBackend | None = None,
    *,
    stage: str = "judge",
    **kwargs: Any,
) -> Judge:
    """Build models.<stage>'s backend from the registry (unless one is given) and its judge."""
    if backend is None:
        config = getattr(compiled.spec.models, stage, None)
        if config is None:
            raise JudgeError(f"no judge configured: models.{stage} is not set")
        try:
            backend = REGISTRY.create(config)
        except ModelBackendError as e:
            raise type(e)(f"models.{stage}: {e}") from e
        backend.setup()
    return judge_from_spec(compiled, backend, stage=stage, **kwargs)
