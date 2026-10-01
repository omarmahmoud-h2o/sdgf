"""Model registry: builds one backend per stage from spec.models (FRAMEWORK_DESIGN.md §7.3).

Backend kinds register a factory under the name a spec uses in models.<stage>.backend.
build() returns StageModels, which carries each stage's backend and its resolved
hosting; endpoints() is the D12 record of where the task's data goes, for provenance
and the governance report.

Tests (and callers that need a hand-built backend) pass overrides={stage: backend};
the override is still recorded with its own model id and hosting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

from sdgf.models.base import ModelBackend, ModelBackendError
from sdgf.models.mock import MockBackend
from sdgf.spec.schema import Hosting, ModelConfig, ModelsSection

STAGES: tuple[str, ...] = (
    "generator",
    "judge",
    "fallback_judge",
    "consistency_judge",
    "expansion",
)

BackendFactory = Callable[[ModelConfig], ModelBackend]


class UnknownBackendError(ModelBackendError, KeyError):
    """No backend kind is registered under the requested name."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True)
class StageModel:
    stage: str
    backend: ModelBackend
    config: ModelConfig | None  # None for an override with no spec entry

    @property
    def hosting(self) -> Hosting:
        return self.backend.hosting

    def endpoint(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "backend": self.backend.name,
            "model": self.backend.model,
            "hosting": self.backend.hosting,
        }


class StageModels(Mapping[str, StageModel]):
    """The built backends, keyed by stage. Only configured stages are present."""

    def __init__(self, stages: dict[str, StageModel]):
        self._stages = dict(stages)

    def __getitem__(self, stage: str) -> StageModel:
        try:
            return self._stages[stage]
        except KeyError:
            configured = ", ".join(self._stages) or "none"
            raise KeyError(
                f"no model configured for stage {stage!r}; configured stages: {configured}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    def backend(self, stage: str) -> ModelBackend:
        return self[stage].backend

    def endpoints(self) -> list[dict[str, Any]]:
        return [m.endpoint() for m in self._stages.values()]

    def external_endpoints(self) -> list[dict[str, Any]]:
        """Stages whose data leaves the organisation (hosting provider_api)."""
        return [e for e in self.endpoints() if e["hosting"] == "provider_api"]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory, *, replace: bool = False) -> None:
        if name in self._factories and not replace:
            raise ModelBackendError(f"backend {name!r} is already registered")
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def create(self, config: ModelConfig) -> ModelBackend:
        try:
            factory = self._factories[config.backend]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise UnknownBackendError(
                f"unknown model backend {config.backend!r}; registered backends: {known}"
            ) from None
        return factory(config)

    def build(
        self,
        models: ModelsSection,
        overrides: Mapping[str, ModelBackend] | None = None,
    ) -> StageModels:
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(STAGES)
        if unknown:
            raise ModelBackendError(
                f"overrides for unknown stage(s) {sorted(unknown)}; stages are {list(STAGES)}"
            )
        built: dict[str, StageModel] = {}
        for stage in STAGES:
            config: ModelConfig | None = getattr(models, stage)
            if stage in overrides:
                backend = overrides[stage]
            elif config is not None:
                try:
                    backend = self.create(config)
                except ModelBackendError as e:
                    raise type(e)(f"models.{stage}: {e}") from e
            else:
                continue
            backend.setup()
            built[stage] = StageModel(stage=stage, backend=backend, config=config)
        return StageModels(built)


def _mock_factory(config: ModelConfig) -> ModelBackend:
    # A spec can script the mock via params.responses; tests usually pass overrides instead.
    responses = config.params.get("responses", [""])
    return MockBackend(
        responses,
        model=config.model,
        hosting=config.hosting,
        cycle=bool(config.params.get("cycle", True)),
    )


def _register_builtins(registry: ModelRegistry) -> None:
    # The real backends import their SDKs only at setup(), so registering them is free.
    # There is deliberately no default backend: models.<stage>.backend picks one (§12.2).
    from sdgf.judge import jev
    from sdgf.models import anthropic, mlx, openai_compat, vllm

    registry.register("mock", _mock_factory)
    registry.register(jev.JEV_BACKEND, jev.factory)  # interface stub: fails clearly (§7.3)
    registry.register(openai_compat.OpenAICompatBackend.name, openai_compat.factory)
    registry.register(anthropic.AnthropicBackend.name, anthropic.factory)
    registry.register(vllm.VLLMBackend.name, vllm.factory)
    registry.register(mlx.MLXBackend.name, mlx.factory)


REGISTRY = ModelRegistry()
_register_builtins(REGISTRY)


def build_models(
    models: ModelsSection, overrides: Mapping[str, ModelBackend] | None = None
) -> StageModels:
    return REGISTRY.build(models, overrides)
