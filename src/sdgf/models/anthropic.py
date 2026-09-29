"""Anthropic Messages API backend. Ported from scripts/model_backends.py AnthropicBackend.

The SDK is imported at setup(), not at module import. Hosting defaults to provider_api.

params: api_key_env (else the SDK reads ANTHROPIC_API_KEY itself), thinking_headroom
(extra output tokens added to max_tokens, default 4096 as in the original),
max_tokens_cap (default 16000), top_p.

Models that reject temperature get one retry without it, remembered for later calls.
"""

from __future__ import annotations

from typing import Any

from sdgf.models._util import api_key_from_env, lazy_import
from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse, ToolCall, ToolSpec
from sdgf.spec.schema import Hosting, ModelConfig


class AnthropicBackend(ModelBackend):
    name = "anthropic"
    default_hosting = "provider_api"

    def __init__(self, model: str, hosting: Hosting | None = None, params: dict | None = None):
        super().__init__(model, hosting)
        params = dict(params or {})
        self.api_key = api_key_from_env(params, self.name)
        self.thinking_headroom = int(params.get("thinking_headroom", 4096))
        self.max_tokens_cap = int(params.get("max_tokens_cap", 16000))
        self.top_p = params.get("top_p")
        self.supports_temperature = True
        self._sdk: Any = None
        self.client: Any = None

    def setup(self) -> None:
        self._sdk = lazy_import("anthropic", "anthropic")
        self.client = (
            self._sdk.Anthropic(api_key=self.api_key) if self.api_key else self._sdk.Anthropic()
        )

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        if self.client is None:
            raise ModelBackendError("AnthropicBackend.setup() must be called before call()")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": min(self.max_tokens_cap, max_tokens + self.thinking_headroom),
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", {"type": "object"}),
                }
                for t in tools
            ]
        if self.supports_temperature:
            kwargs["temperature"] = temperature

        try:
            resp = self.client.messages.create(**kwargs)
        except self._sdk.BadRequestError as e:
            if not (self.supports_temperature and "temperature" in str(e)):
                raise
            self.supports_temperature = False
            kwargs.pop("temperature")
            resp = self.client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text").strip() or None
        tool_calls = tuple(
            ToolCall(name=b.name, arguments=dict(b.input or {}), id=getattr(b, "id", None))
            for b in resp.content
            if b.type == "tool_use"
        )
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


def factory(config: ModelConfig) -> AnthropicBackend:
    return AnthropicBackend(config.model, config.hosting, config.params)
