"""RetryingBackend: transient model errors (rate limits, 5xx, dropped connections) are
retried with backoff; anything else, or a transient error that outlasts the retries,
stops the call. No network: a scripted backend raises the errors."""

import urllib.error

import pytest

from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse
from sdgf.models.retry import RetryingBackend


def http(code):
    return urllib.error.HTTPError("https://api.example.invalid", code, "x", None, None)


class SdkStatusError(Exception):
    """Shaped like a provider SDK error (anthropic.APIStatusError has .status_code)."""

    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class Scripted(ModelBackend):
    name = "scripted"
    default_hosting = "local"

    def __init__(self, *outcomes):
        super().__init__("m", None)
        self.outcomes = list(outcomes)
        self.calls = 0

    def call(self, prompt, max_tokens, temperature, tools=None):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return ModelResponse(text=out)


def retrying(inner, **kw):
    sleeps = []
    return RetryingBackend(inner, sleep=sleeps.append, **kw), sleeps


def test_a_rate_limit_is_retried_after_a_backoff():
    backend, sleeps = retrying(Scripted(http(429), "ok"))
    assert backend.call("p", 10, 0.0).text == "ok"
    assert sleeps == [1.0]


def test_backoff_doubles_on_each_retry():
    backend, sleeps = retrying(Scripted(http(503), http(503), "ok"))
    assert backend.call("p", 10, 0.0).text == "ok"
    assert sleeps == [1.0, 2.0]


def test_a_dropped_connection_is_retried():
    backend, _ = retrying(Scripted(urllib.error.URLError("reset"), "ok"))
    assert backend.call("p", 10, 0.0).text == "ok"


def test_an_sdk_error_with_a_retryable_status_is_retried():
    backend, _ = retrying(Scripted(SdkStatusError(529), "ok"))
    assert backend.call("p", 10, 0.0).text == "ok"


def test_a_client_error_is_not_retried():
    inner = Scripted(http(401), "ok")
    backend, sleeps = retrying(inner)
    with pytest.raises(urllib.error.HTTPError):
        backend.call("p", 10, 0.0)
    assert inner.calls == 1 and sleeps == []


def test_a_rate_limit_that_outlasts_the_retries_stops_with_a_named_error():
    backend, sleeps = retrying(Scripted(http(429), http(429), http(429)), max_retries=2)
    with pytest.raises(ModelBackendError, match="after 2 retries") as e:
        backend.call("p", 10, 0.0)
    assert isinstance(e.value.__cause__, urllib.error.HTTPError)
    assert sleeps == [1.0, 2.0]


def test_it_reports_the_wrapped_backend_name_model_and_hosting():
    backend, _ = retrying(Scripted("ok"))
    assert (backend.name, backend.model, backend.hosting) == ("scripted", "m", "local")
