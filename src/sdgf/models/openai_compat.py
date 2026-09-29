"""OpenAI-compatible chat-completions backend (a vLLM server, or any provider API).

Ported from scripts/model_backends.py APIBackend. Uses only the standard library.
Hosting has no default: the same client talks to a local vLLM server or to a
provider, so the spec must say which (D12).

params: api_base (required), api_key_env, top_p, stop, timeout, check_model.
Unlike the original, setup() never swaps in a different model when the configured
one is missing — the spec is the only source of the model choice.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from sdgf.models._util import api_key_from_env
from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse, ToolCall, ToolSpec
from sdgf.spec.schema import Hosting, ModelConfig


def _http_json(
    method: str, url: str, headers: dict[str, str], payload: Any | None, timeout: float
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class OpenAICompatBackend(ModelBackend):
    name = "openai_compat"
    default_hosting = None

    def __init__(self, model: str, hosting: Hosting | None = None, params: dict | None = None):
        super().__init__(model, hosting)
        params = dict(params or {})
        if not params.get("api_base"):
            raise ModelBackendError("openai_compat backend needs params.api_base")
        self.api_base = str(params["api_base"]).rstrip("/")
        self.api_key = api_key_from_env(params, self.name)
        self.top_p = params.get("top_p")
        self.stop = params.get("stop")
        self.timeout = float(params.get("timeout", 120))
        self.check_model = bool(params.get("check_model", True))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def setup(self) -> None:
        if not self.check_model:
            return
        listing = _http_json("GET", f"{self.api_base}/models", self._headers(), None, self.timeout)
        served = [m["id"] for m in listing.get("data", [])]
        if self.model not in served:
            raise ModelBackendError(
                f"model {self.model!r} is not served at {self.api_base}; "
                f"available: {', '.join(served) or 'none'}"
            )

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.stop:
            body["stop"] = self.stop
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object"}),
                    },
                }
                for t in tools
            ]
        data = _http_json(
            "POST", f"{self.api_base}/chat/completions", self._headers(), body, self.timeout
        )
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ModelBackendError(f"malformed chat-completions response: {data!r}") from e

        text = (message.get("content") or "").strip() or None
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError as e:
                raise ModelBackendError(
                    f"tool call {fn.get('name')!r} has non-JSON arguments: {raw!r}"
                ) from e
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args, id=tc.get("id")))
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            tool_calls=tuple(tool_calls),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def factory(config: ModelConfig) -> OpenAICompatBackend:
    return OpenAICompatBackend(config.model, config.hosting, config.params)
