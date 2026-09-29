"""Apple Silicon backend via mlx-lm. Ported from scripts/model_backends.py MLXBackend.

mlx_lm is imported at setup(). Hosting defaults to local.

Fixes §12.2: the original ignored temperature. mlx_lm.generate takes sampling through
a sampler, so every call builds one with make_sampler(temp=temperature, top_p=...).

params: top_p, enable_thinking (passed to the chat template, default False).
"""

from __future__ import annotations

from typing import Any

from sdgf.models._util import lazy_import, reject_tools
from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse, ToolSpec
from sdgf.spec.schema import Hosting, ModelConfig


class MLXBackend(ModelBackend):
    name = "mlx"
    default_hosting = "local"

    def __init__(self, model: str, hosting: Hosting | None = None, params: dict | None = None):
        super().__init__(model, hosting)
        params = dict(params or {})
        self.top_p = float(params.get("top_p", 0.0))  # 0.0 disables top-p in make_sampler
        self.enable_thinking = bool(params.get("enable_thinking", False))
        self._mlx_lm: Any = None
        self._make_sampler: Any = None
        self.weights: Any = None
        self.tokenizer: Any = None

    def setup(self) -> None:
        self._mlx_lm = lazy_import("mlx_lm", "mlx mlx-lm")
        self._make_sampler = lazy_import("mlx_lm.sample_utils", "mlx mlx-lm").make_sampler
        self.weights, self.tokenizer = self._mlx_lm.load(self.model)

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        reject_tools(self.name, tools)
        if self.weights is None:
            raise ModelBackendError("MLXBackend.setup() must be called before call()")
        templated = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        sampler = self._make_sampler(temp=temperature, top_p=self.top_p)
        text = self._mlx_lm.generate(
            self.weights,
            self.tokenizer,
            prompt=templated,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return ModelResponse(text=(text or "").strip() or None)


def factory(config: ModelConfig) -> MLXBackend:
    return MLXBackend(config.model, config.hosting, config.params)
