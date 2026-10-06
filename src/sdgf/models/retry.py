"""Retrying transient model errors, for every backend (README "Backends").

A rate limit (429), an overloaded or failing service (5xx, 529) or a dropped connection
is retried with backoff (1s, 2s, 4s ... by default). Any other error is raised at once,
and a transient error that outlasts the retries becomes a ModelBackendError naming it,
so the run stops with a clear message and can be resumed.

The pipeline wraps every stage's backend with this, except Jev, whose backend already
retries the same statuses itself. A retried call is metered once: a refused request
costs no tokens.
"""

from __future__ import annotations

import time
import urllib.error
from typing import Any, Callable

from sdgf.models.base import (
    DecisionResponse,
    ModelBackend,
    ModelBackendError,
    ModelResponse,
    ToolSpec,
)

RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 1.0


def transient(error: BaseException) -> bool:
    """A rate limit, a 5xx, or a connection that dropped or timed out."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_STATUSES
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    status = getattr(error, "status_code", None)  # provider SDK errors (anthropic, openai)
    return isinstance(status, int) and status in RETRYABLE_STATUSES


def describe(error: BaseException) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}"
    status = getattr(error, "status_code", None)
    return f"{type(error).__name__}" + (f" (status {status})" if status else f": {error}")


class RetryingBackend(ModelBackend):
    def __init__(
        self,
        inner: ModelBackend,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_retries < 0 or backoff < 0:
            raise ValueError("max_retries and backoff must be >= 0")
        self.inner = inner
        self.name = inner.name  # type: ignore[misc]
        self.model = inner.model
        self.hosting = inner.hosting
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep

    def setup(self) -> None:
        self.inner.setup()

    def _retry(self, fn: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if not transient(e):
                    raise
                if attempt == self.max_retries:
                    raise ModelBackendError(
                        f"{self.name} {self.model}: {describe(e)} after {self.max_retries} "
                        "retries; resume the run when the service recovers"
                    ) from e
                self.sleep(self.backoff * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        return self._retry(lambda: self.inner.call(prompt, max_tokens, temperature, tools))

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> DecisionResponse:
        return self._retry(lambda: self.inner.decide(state, questions))
