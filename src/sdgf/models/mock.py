"""MockBackend: a scripted or callable-driven backend for tests. Never touches a network.

    MockBackend(["first reply", "second reply"])        # replies in order
    MockBackend(["same reply"], cycle=True)             # repeats the script
    MockBackend(lambda call: f"echo {call.prompt}")     # computed per call

    MockBackend([ToolCall("lookup", {"q": "x"}), "final reply"])  # a tool round, then text

Script entries and callable results may be a str (text only), a ToolCall or a list of
ToolCalls (one agent round of tool calls, no text), or a ModelResponse (for text with
tool calls, or token counts). Every call, with the tools it was offered, is recorded
in .calls.

Calls are thread-safe. Under concurrent generation a script is consumed in whatever order
the calls arrive, so tests that compare concurrent runs use a callable computed from the
prompt, which answers the same whatever the order.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Sequence, Union

from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse, ToolCall, ToolSpec
from sdgf.spec.schema import Hosting

Reply = Union[str, ModelResponse, ToolCall, Sequence[ToolCall]]


@dataclass(frozen=True)
class MockCall:
    prompt: str
    max_tokens: int
    temperature: float
    tools: tuple[ToolSpec, ...] | None


class MockExhaustedError(ModelBackendError):
    """A non-cycling script ran out of replies."""


class MockBackend(ModelBackend):
    name = "mock"
    default_hosting = "local"

    def __init__(
        self,
        responses: Sequence[Reply] | Callable[[MockCall], Reply],
        *,
        model: str = "mock",
        hosting: Hosting | None = None,
        cycle: bool = False,
    ):
        super().__init__(model, hosting)
        if callable(responses):
            self._fn: Callable[[MockCall], Reply] | None = responses
            self._script: list[Reply] = []
        else:
            self._fn = None
            self._script = list(responses)
            if not self._script:
                raise ModelBackendError("MockBackend needs at least one scripted response")
        self.cycle = cycle
        self.calls: list[MockCall] = []
        self._lock = threading.Lock()

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        call = MockCall(
            prompt, max_tokens, temperature, tuple(tools) if tools is not None else None
        )
        with self._lock:
            index = len(self.calls)
            self.calls.append(call)
        if self._fn is not None:
            reply = self._fn(call)
        elif index < len(self._script) or self.cycle:
            reply = self._script[index % len(self._script)]
        else:
            raise MockExhaustedError(
                f"MockBackend script exhausted after {len(self._script)} response(s)"
            )
        if isinstance(reply, ModelResponse):
            return reply
        if isinstance(reply, str) or reply is None:
            return ModelResponse(text=reply)
        if isinstance(reply, ToolCall):
            return ModelResponse(text=None, tool_calls=(reply,))
        if (
            isinstance(reply, (list, tuple))
            and reply
            and all(isinstance(c, ToolCall) for c in reply)
        ):
            return ModelResponse(text=None, tool_calls=tuple(reply))
        raise ModelBackendError(
            f"MockBackend reply must be str or ModelResponse, got {type(reply).__name__}"
        )
