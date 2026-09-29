import pytest

from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse, ToolCall
from sdgf.models.mock import MockBackend, MockExhaustedError
from sdgf.models.registry import (
    REGISTRY,
    ModelRegistry,
    UnknownBackendError,
    build_models,
)
from sdgf.spec.schema import ModelConfig, ModelsSection


class ApiBackend(ModelBackend):
    name = "fake_api"
    default_hosting = "provider_api"

    def __init__(self, model, hosting=None):
        super().__init__(model, hosting)
        self.setup_calls = 0

    def setup(self):
        self.setup_calls += 1

    def call(self, prompt, max_tokens, temperature, tools=None):
        return ModelResponse(text="ok")


class NoDefaultHosting(ModelBackend):
    name = "no_default"

    def call(self, prompt, max_tokens, temperature, tools=None):
        return ModelResponse(text="ok")


def registry_with_fakes():
    reg = ModelRegistry()
    reg.register("fake_api", lambda c: ApiBackend(c.model, c.hosting))
    reg.register("no_default", lambda c: NoDefaultHosting(c.model, c.hosting))
    reg.register("mock", lambda c: MockBackend(["x"], model=c.model, hosting=c.hosting))
    return reg


# ── MockBackend ──────────────────────────────────────────────────


def test_mock_scripted_replies_in_order_and_records_calls():
    m = MockBackend(["one", "two"])
    assert m.call("p1", 10, 0.1).text == "one"
    assert m.call("p2", 20, 0.2, tools=[{"name": "calc"}]).text == "two"
    assert [c.prompt for c in m.calls] == ["p1", "p2"]
    assert m.calls[1].max_tokens == 20
    assert m.calls[1].temperature == 0.2
    assert m.calls[1].tools == ({"name": "calc"},)
    assert m.calls[0].tools is None


def test_mock_exhausted_raises():
    m = MockBackend(["only"])
    m.call("p", 1, 0.0)
    with pytest.raises(MockExhaustedError, match="exhausted after 1"):
        m.call("p", 1, 0.0)


def test_mock_cycle_repeats():
    m = MockBackend(["a", "b"], cycle=True)
    assert [m.call("p", 1, 0.0).text for _ in range(5)] == ["a", "b", "a", "b", "a"]


def test_mock_callable_sees_call():
    m = MockBackend(lambda call: f"echo:{call.prompt}:{call.temperature}")
    assert m.call("hi", 5, 0.3).text == "echo:hi:0.3"


def test_mock_returns_tool_calls_and_usage():
    scripted = ModelResponse(
        text=None,
        tool_calls=(ToolCall(name="catalogue_lookup", arguments={"sku": "TEST-1"}, id="c1"),),
        input_tokens=12,
        output_tokens=3,
    )
    m = MockBackend([scripted, "final"])
    first = m.call("p", 1, 0.0, tools=[{"name": "catalogue_lookup"}])
    assert first.text is None
    assert first.tool_calls[0].name == "catalogue_lookup"
    assert first.tool_calls[0].arguments == {"sku": "TEST-1"}
    assert first.input_tokens == 12
    assert m.call("p", 1, 0.0).tool_calls == ()


def test_mock_rejects_empty_script_and_bad_reply():
    with pytest.raises(ModelBackendError, match="at least one"):
        MockBackend([])
    with pytest.raises(ModelBackendError, match="str or ModelResponse"):
        MockBackend(lambda call: 42).call("p", 1, 0.0)


def test_mock_hosting_defaults_local_and_can_be_overridden():
    assert MockBackend(["x"]).hosting == "local"
    assert MockBackend(["x"], hosting="provider_api").hosting == "provider_api"


def test_backend_without_default_hosting_requires_one():
    with pytest.raises(ModelBackendError, match="no default hosting"):
        NoDefaultHosting("m")
    assert NoDefaultHosting("m", "local").hosting == "local"


# ── Registry ─────────────────────────────────────────────────────


def cfg(backend, model="m-1", **kw):
    return ModelConfig(backend=backend, model=model, **kw)


def test_build_per_stage_records_hosting_and_endpoints():
    reg = registry_with_fakes()
    models = ModelsSection(
        generator=cfg("mock", "gen-1"),
        judge=cfg("fake_api", "judge-1"),
        expansion=cfg("no_default", "exp-1", hosting="provider_api"),
    )
    built = reg.build(models)
    assert list(built) == ["generator", "judge", "expansion"]
    assert "fallback_judge" not in built
    assert built["generator"].hosting == "local"
    assert built["judge"].hosting == "provider_api"
    assert built["judge"].backend.setup_calls == 1
    assert built["judge"].config.model == "judge-1"
    assert built.endpoints() == [
        {"stage": "generator", "backend": "mock", "model": "gen-1", "hosting": "local"},
        {"stage": "judge", "backend": "fake_api", "model": "judge-1", "hosting": "provider_api"},
        {
            "stage": "expansion",
            "backend": "no_default",
            "model": "exp-1",
            "hosting": "provider_api",
        },
    ]
    assert [e["stage"] for e in built.external_endpoints()] == ["judge", "expansion"]


def test_spec_hosting_overrides_backend_default():
    reg = registry_with_fakes()
    built = reg.build(ModelsSection(generator=cfg("fake_api", hosting="local")))
    assert built["generator"].hosting == "local"


def test_missing_hosting_error_names_stage():
    reg = registry_with_fakes()
    with pytest.raises(ModelBackendError, match=r"models\.judge: .*no default hosting"):
        reg.build(ModelsSection(generator=cfg("mock"), judge=cfg("no_default")))


def test_unknown_backend_is_clear():
    reg = registry_with_fakes()
    with pytest.raises(UnknownBackendError) as e:
        reg.build(ModelsSection(generator=cfg("nope")))
    msg = str(e.value)
    assert "models.generator" in msg
    assert "'nope'" in msg
    assert "fake_api, mock, no_default" in msg


def test_overrides_replace_spec_backend_and_are_recorded():
    reg = registry_with_fakes()
    mock = MockBackend(["scripted"], model="override-1", hosting="provider_api")
    built = reg.build(ModelsSection(generator=cfg("nope")), overrides={"generator": mock})
    assert built.backend("generator") is mock
    assert built.endpoints()[0] == {
        "stage": "generator",
        "backend": "mock",
        "model": "override-1",
        "hosting": "provider_api",
    }


def test_override_for_unconfigured_stage_is_included():
    reg = registry_with_fakes()
    judge = MockBackend(["breach"], model="judge-mock")
    built = reg.build(ModelsSection(generator=cfg("mock")), overrides={"judge": judge})
    assert built["judge"].config is None
    assert built.backend("judge") is judge


def test_override_for_unknown_stage_raises():
    with pytest.raises(ModelBackendError, match="unknown stage"):
        registry_with_fakes().build(
            ModelsSection(generator=cfg("mock")), overrides={"critic": MockBackend(["x"])}
        )


def test_missing_stage_lookup_is_clear():
    built = registry_with_fakes().build(ModelsSection(generator=cfg("mock")))
    with pytest.raises(KeyError, match="no model configured for stage 'judge'"):
        built["judge"]


def test_duplicate_registration_raises_unless_replace():
    reg = ModelRegistry()
    reg.register("a", lambda c: MockBackend(["x"]))
    with pytest.raises(ModelBackendError, match="already registered"):
        reg.register("a", lambda c: MockBackend(["y"]))
    reg.register("a", lambda c: MockBackend(["y"]), replace=True)
    assert reg.create(cfg("a")).call("p", 1, 0.0).text == "y"


def test_global_registry_builds_mock_from_spec_params():
    assert "mock" in REGISTRY
    models = ModelsSection(
        generator=cfg("mock", "mock-1", params={"responses": ["r1", "r2"], "cycle": False})
    )
    gen = build_models(models).backend("generator")
    assert gen.model == "mock-1"
    assert [gen.call("p", 1, 0.0).text for _ in range(2)] == ["r1", "r2"]
    with pytest.raises(MockExhaustedError):
        gen.call("p", 1, 0.0)
