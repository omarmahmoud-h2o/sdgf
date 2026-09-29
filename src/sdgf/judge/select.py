"""Pick the judge a spec selects for a stage (judge or fallback_judge).

models.<stage>.backend: jev selects the Jev decision judge (a stub until its API is
documented, §7.3); any other backend is a text model, built from the model registry,
behind LLMJudge. An explicit backend passed by the caller (a test, or a hand-built
backend) wins over the spec, as it does for ModelRegistry overrides.
"""

from __future__ import annotations

from typing import Any

from sdgf.judge.interface import Judge, JudgeError, compile_rubric
from sdgf.judge.jev import JEV_BACKEND, JevJudge
from sdgf.judge.llm_judge import LLMJudge
from sdgf.models.base import ModelBackend, ModelBackendError
from sdgf.models.registry import REGISTRY
from sdgf.spec.compile import CompiledSpec


def build_judge(
    compiled: CompiledSpec,
    backend: ModelBackend | None = None,
    *,
    stage: str = "judge",
    **kwargs: Any,
) -> Judge:
    config = getattr(compiled.spec.models, stage, None)
    if backend is not None:
        return LLMJudge.from_spec(compiled, backend, stage=stage, **kwargs)
    if config is None:
        raise JudgeError(f"no judge configured: models.{stage} is not set")
    try:
        if config.backend == JEV_BACKEND:
            return JevJudge(compile_rubric(compiled.spec.rubric), config)
        backend = REGISTRY.create(config)
    except ModelBackendError as e:
        raise type(e)(f"models.{stage}: {e}") from e
    backend.setup()
    return LLMJudge.from_spec(compiled, backend, stage=stage, **kwargs)
