"""In-process vLLM backend. Ported from scripts/model_backends.py VLLMDirectBackend.

vllm is imported at setup(). Hosting defaults to local. `model` is the model path or
hub id loaded into the engine.

params: engine (kwargs for vllm.LLM; defaults below match the original), top_p, stop.
"""

from __future__ import annotations

from typing import Any

from sdgf.models._util import lazy_import, reject_tools
from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse, ToolSpec
from sdgf.spec.schema import Hosting, ModelConfig

ENGINE_DEFAULTS: dict[str, Any] = {
    "max_model_len": 4096,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.8,
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "enforce_eager": False,
}


class VLLMBackend(ModelBackend):
    name = "vllm"
    default_hosting = "local"

    def __init__(self, model: str, hosting: Hosting | None = None, params: dict | None = None):
        super().__init__(model, hosting)
        params = dict(params or {})
        self.engine_args = {**ENGINE_DEFAULTS, **dict(params.get("engine") or {})}
        self.top_p = params.get("top_p", 1.0)
        self.stop = params.get("stop")
        self._sdk: Any = None
        self.llm: Any = None

    def setup(self) -> None:
        self._sdk = lazy_import("vllm", "vllm")
        self.llm = self._sdk.LLM(model=self.model, **self.engine_args)

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        reject_tools(self.name, tools)
        if self.llm is None:
            raise ModelBackendError("VLLMBackend.setup() must be called before call()")
        sampling = self._sdk.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=self.top_p, stop=self.stop
        )
        result = self.llm.generate([prompt], sampling)[0]
        completion = result.outputs[0]
        prompt_ids = getattr(result, "prompt_token_ids", None)
        output_ids = getattr(completion, "token_ids", None)
        return ModelResponse(
            text=completion.text.strip() or None,
            input_tokens=len(prompt_ids) if prompt_ids is not None else None,
            output_tokens=len(output_ids) if output_ids is not None else None,
        )


def factory(config: ModelConfig) -> VLLMBackend:
    return VLLMBackend(config.model, config.hosting, config.params)
