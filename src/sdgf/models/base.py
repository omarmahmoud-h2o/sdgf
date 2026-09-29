"""The ModelBackend interface (FRAMEWORK_DESIGN.md §7.3).

Every backend — local vLLM, MLX, OpenAI-compatible, Anthropic, mock — exposes one
call(prompt, max_tokens, temperature, tools=None) returning a ModelResponse: the text
(None if the model produced none) plus any tool calls the model asked for (§6.3, D4).

Each backend also knows its hosting ("local" or "provider_api"). Under D12 a task's
model choice *is* its data-destination decision, so hosting is recorded per stage by
the model registry for provenance and the governance report.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from sdgf.spec.schema import Hosting

ToolSpec = dict[str, Any]  # name, description, JSON input schema — as the tool registry declares it


class ModelBackendError(RuntimeError):
    """A backend could not be built, set up, or called."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelBackend(ABC):
    name: ClassVar[str]  # backend kind, matches ModelConfig.backend
    default_hosting: ClassVar[Hosting | None] = None  # None: the spec must declare it

    def __init__(self, model: str, hosting: Hosting | None = None):
        self.model = model
        resolved = hosting or self.default_hosting
        if resolved is None:
            raise ModelBackendError(
                f"backend {self.name!r} has no default hosting; set hosting "
                "('local' or 'provider_api') in the model config"
            )
        self.hosting: Hosting = resolved

    def setup(self) -> None:
        """Load weights or open clients. Called once by the registry before first use."""

    @abstractmethod
    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse: ...


class BoundedBackend(ModelBackend):
    """Wraps a backend so at most `limit` calls run at once: models.<stage>.concurrency
    under the pipeline's thread pool. Name, model and hosting are the wrapped backend's."""

    def __init__(self, inner: ModelBackend, limit: int):
        if limit < 1:
            raise ModelBackendError(f"concurrency limit must be >= 1, got {limit}")
        self.inner = inner
        self.name = inner.name  # type: ignore[misc]
        self.model = inner.model
        self.hosting = inner.hosting
        self.limit = limit
        self._slots = threading.BoundedSemaphore(limit)

    def setup(self) -> None:
        self.inner.setup()

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        with self._slots:
            return self.inner.call(prompt, max_tokens, temperature, tools)
