"""Real backends, exercised only against monkeypatched clients — no network, no SDKs needed."""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from sdgf.models import openai_compat
from sdgf.models.anthropic import AnthropicBackend
from sdgf.models.base import ModelBackendError, ToolCall
from sdgf.models.mlx import MLXBackend
from sdgf.models.openai_compat import OpenAICompatBackend
from sdgf.models.registry import REGISTRY, build_models
from sdgf.models.vllm import VLLMBackend
from sdgf.spec.schema import ModelConfig, ModelsSection

TOOL = {
    "name": "calc",
    "description": "adds numbers",
    "input_schema": {"type": "object", "properties": {"a": {"type": "number"}}},
}


# ── registry ─────────────────────────────────────────────────────


def test_all_real_backends_registered():
    assert {"mock", "openai_compat", "anthropic", "vllm", "mlx"} <= set(REGISTRY.names())


def test_registry_builds_backend_named_in_spec_without_touching_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)  # would fail if imported
    backend = REGISTRY.create(ModelConfig(backend="vllm", model="some/model"))
    assert isinstance(backend, VLLMBackend)
    assert backend.hosting == "local"


def test_default_hostings():
    assert AnthropicBackend("m").hosting == "provider_api"
    assert MLXBackend("m").hosting == "local"
    with pytest.raises(ModelBackendError, match="hosting"):
        OpenAICompatBackend("m", params={"api_base": "http://x"})


def test_missing_sdk_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ModelBackendError, match="pip install vllm"):
        VLLMBackend("m").setup()


def test_literal_api_key_rejected_and_env_key_required(monkeypatch):
    with pytest.raises(ModelBackendError, match="api_key_env"):
        AnthropicBackend("m", params={"api_key": "not-a-real-key"})
    monkeypatch.delenv("SDGF_TEST_KEY", raising=False)
    with pytest.raises(ModelBackendError, match="SDGF_TEST_KEY"):
        AnthropicBackend("m", params={"api_key_env": "SDGF_TEST_KEY"})


# ── openai_compat ────────────────────────────────────────────────


class FakeHTTP:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, method, url, headers, payload, timeout):
        self.requests.append(
            SimpleNamespace(method=method, url=url, headers=headers, payload=payload)
        )
        return self.replies.pop(0)


def make_openai(monkeypatch, replies, **params):
    http = FakeHTTP(replies)
    monkeypatch.setattr(openai_compat, "_http_json", http)
    backend = OpenAICompatBackend(
        "test-model", "local", {"api_base": "http://localhost:0/v1/", **params}
    )
    return backend, http


def test_openai_setup_checks_model_and_never_swaps(monkeypatch):
    backend, http = make_openai(monkeypatch, [{"data": [{"id": "other-model"}]}])
    with pytest.raises(ModelBackendError, match="test-model.*other-model"):
        backend.setup()
    assert http.requests[0].url == "http://localhost:0/v1/models"
    assert backend.model == "test-model"


def test_openai_call_sends_request_and_parses_text_and_usage(monkeypatch):
    monkeypatch.setenv("SDGF_TEST_KEY", "fake-key-000")
    reply = {
        "choices": [{"message": {"content": "  hello  "}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    backend, http = make_openai(
        monkeypatch,
        [{"data": [{"id": "test-model"}]}, reply],
        api_key_env="SDGF_TEST_KEY",
        top_p=0.9,
        stop=["<|im_end|>"],
    )
    backend.setup()
    r = backend.call("prompt", 50, 0.3)
    assert (r.text, r.input_tokens, r.output_tokens) == ("hello", 7, 3)
    req = http.requests[1]
    assert req.url == "http://localhost:0/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer fake-key-000"
    assert req.payload["temperature"] == 0.3
    assert req.payload["max_tokens"] == 50
    assert req.payload["top_p"] == 0.9
    assert req.payload["stop"] == ["<|im_end|>"]
    assert req.payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert "tools" not in req.payload


def test_openai_tools_round_trip(monkeypatch):
    reply = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "calc", "arguments": '{"a": 1}'}}
                    ],
                }
            }
        ]
    }
    backend, http = make_openai(monkeypatch, [reply], check_model=False)
    backend.setup()
    r = backend.call("p", 10, 0.0, tools=[TOOL])
    assert r.text is None
    assert r.tool_calls == (ToolCall("calc", {"a": 1}, "c1"),)
    fn = http.requests[0].payload["tools"][0]["function"]
    assert fn["name"] == "calc" and fn["parameters"] == TOOL["input_schema"]
    assert "Authorization" not in http.requests[0].headers


def test_openai_malformed_response(monkeypatch):
    backend, _ = make_openai(monkeypatch, [{"error": "boom"}], check_model=False)
    with pytest.raises(ModelBackendError, match="malformed"):
        backend.call("p", 10, 0.0)


def test_openai_needs_api_base():
    with pytest.raises(ModelBackendError, match="api_base"):
        OpenAICompatBackend("m", "local", {})


def test_http_json_uses_urlopen(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, method=req.get_method(), body=req.data, timeout=timeout)
        return Resp()

    monkeypatch.setattr(openai_compat.urllib.request, "urlopen", fake_urlopen)
    out = openai_compat._http_json("POST", "http://localhost:0/x", {}, {"a": 1}, 5)
    assert out == {"ok": True}
    assert seen == {
        "url": "http://localhost:0/x",
        "method": "POST",
        "body": json.dumps({"a": 1}).encode(),
        "timeout": 5,
    }


# ── anthropic ────────────────────────────────────────────────────


class FakeBadRequest(Exception):
    pass


def fake_anthropic(monkeypatch, create):
    sdk = types.ModuleType("anthropic")
    sdk.BadRequestError = FakeBadRequest
    sdk.inits = []

    class Anthropic:
        def __init__(self, **kw):
            sdk.inits.append(kw)
            self.messages = SimpleNamespace(create=create)

    sdk.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    return sdk


def block(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def test_anthropic_call_text_tools_and_usage(monkeypatch):
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(
            content=[
                block("thinking", thinking="..."),
                block("text", text=" hi "),
                block("tool_use", id="t1", name="calc", input={"a": 2}),
            ],
            usage=SimpleNamespace(input_tokens=11, output_tokens=4),
        )

    sdk = fake_anthropic(monkeypatch, create)
    monkeypatch.setenv("SDGF_TEST_KEY", "fake-key-000")
    b = AnthropicBackend("claude-test", params={"api_key_env": "SDGF_TEST_KEY"})
    b.setup()
    assert sdk.inits == [{"api_key": "fake-key-000"}]
    r = b.call("p", 100, 0.5, tools=[TOOL])
    assert r.text == "hi"
    assert r.tool_calls == (ToolCall("calc", {"a": 2}, "t1"),)
    assert (r.input_tokens, r.output_tokens) == (11, 4)
    assert calls[0]["max_tokens"] == 100 + 4096
    assert calls[0]["temperature"] == 0.5
    assert calls[0]["tools"] == [TOOL]


def test_anthropic_max_tokens_capped(monkeypatch):
    calls = []
    fake_anthropic(monkeypatch, lambda **kw: calls.append(kw) or SimpleNamespace(content=[]))
    b = AnthropicBackend("claude-test")
    b.setup()
    assert b.call("p", 15000, 0.0).text is None
    assert calls[0]["max_tokens"] == 16000


def test_anthropic_drops_temperature_once_rejected(monkeypatch):
    calls = []

    def create(**kw):
        calls.append(dict(kw))
        if "temperature" in kw:
            raise FakeBadRequest("temperature is not supported for this model")
        return SimpleNamespace(content=[block("text", text="ok")])

    fake_anthropic(monkeypatch, create)
    b = AnthropicBackend("claude-test")
    b.setup()
    assert b.call("p", 10, 0.7).text == "ok"
    assert b.call("p", 10, 0.7).text == "ok"
    assert ["temperature" in c for c in calls] == [True, False, False]


def test_anthropic_other_bad_request_propagates(monkeypatch):
    def create(**kw):
        raise FakeBadRequest("prompt too long")

    fake_anthropic(monkeypatch, create)
    b = AnthropicBackend("claude-test")
    b.setup()
    with pytest.raises(FakeBadRequest):
        b.call("p", 10, 0.7)


def test_anthropic_call_before_setup():
    with pytest.raises(ModelBackendError, match="setup"):
        AnthropicBackend("m").call("p", 1, 0.0)


# ── vllm ─────────────────────────────────────────────────────────


def fake_vllm(monkeypatch):
    sdk = types.ModuleType("vllm")
    sdk.engines = []

    class LLM:
        def __init__(self, **kw):
            sdk.engines.append(kw)

        def generate(self, prompts, params):
            self.last = (prompts, params)
            out = SimpleNamespace(text=" done ", token_ids=[1, 2])
            return [SimpleNamespace(outputs=[out], prompt_token_ids=[5, 6, 7])]

    sdk.LLM = LLM
    sdk.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", sdk)
    return sdk


def test_vllm_setup_and_call(monkeypatch):
    sdk = fake_vllm(monkeypatch)
    b = VLLMBackend("some/model", params={"engine": {"max_model_len": 8192}, "top_p": 0.9})
    b.setup()
    assert sdk.engines[0]["model"] == "some/model"
    assert sdk.engines[0]["max_model_len"] == 8192
    assert sdk.engines[0]["dtype"] == "bfloat16"
    r = b.call("p", 64, 0.4)
    assert (r.text, r.input_tokens, r.output_tokens) == ("done", 3, 2)
    prompts, params = b.llm.last
    assert prompts == ["p"]
    assert (params.max_tokens, params.temperature, params.top_p) == (64, 0.4, 0.9)


def test_vllm_rejects_tools(monkeypatch):
    fake_vllm(monkeypatch)
    b = VLLMBackend("m")
    b.setup()
    with pytest.raises(ModelBackendError, match="tool"):
        b.call("p", 1, 0.0, tools=[TOOL])


# ── mlx ──────────────────────────────────────────────────────────


def fake_mlx(monkeypatch):
    sdk = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sdk.generations = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return f"<tmpl>{messages[0]['content']}"

    sdk.load = lambda path: (f"weights:{path}", Tokenizer())

    def generate(model, tokenizer, **kw):
        sdk.generations.append(kw)
        return " text "

    sdk.generate = generate
    sample_utils.make_sampler = lambda **kw: ("sampler", kw)
    monkeypatch.setitem(sys.modules, "mlx_lm", sdk)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    return sdk


def test_mlx_passes_temperature_through_sampler(monkeypatch):
    sdk = fake_mlx(monkeypatch)
    b = MLXBackend("local/model", params={"top_p": 0.9})
    b.setup()
    assert b.weights == "weights:local/model"
    assert b.call("p", 32, 0.8).text == "text"
    assert b.call("p", 32, 0.1).text == "text"
    kw = sdk.generations[0]
    assert kw["prompt"] == "<tmpl>p"
    assert kw["max_tokens"] == 32
    assert kw["sampler"] == ("sampler", {"temp": 0.8, "top_p": 0.9})
    assert sdk.generations[1]["sampler"] == ("sampler", {"temp": 0.1, "top_p": 0.9})


def test_mlx_rejects_tools_and_requires_setup(monkeypatch):
    with pytest.raises(ModelBackendError, match="setup"):
        MLXBackend("m").call("p", 1, 0.0)
    fake_mlx(monkeypatch)
    b = MLXBackend("m")
    b.setup()
    with pytest.raises(ModelBackendError, match="tool"):
        b.call("p", 1, 0.0, tools=[TOOL])


# ── end to end through build_models ──────────────────────────────


def test_build_models_mixes_backends_and_records_hosting(monkeypatch):
    fake_anthropic(monkeypatch, lambda **kw: SimpleNamespace(content=[block("text", text="j")]))
    fake_mlx(monkeypatch)
    models = ModelsSection(
        generator=ModelConfig(backend="mlx", model="local/model"),
        judge=ModelConfig(backend="anthropic", model="claude-test"),
    )
    built = build_models(models)
    assert built.backend("generator").call("p", 8, 0.2).text == "text"
    assert built.backend("judge").call("p", 8, 0.2).text == "j"
    assert built.external_endpoints() == [
        {
            "stage": "judge",
            "backend": "anthropic",
            "model": "claude-test",
            "hosting": "provider_api",
        }
    ]
