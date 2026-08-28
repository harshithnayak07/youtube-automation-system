"""Tests for the llm.client module — all use mocks, no real API calls."""

from __future__ import annotations

import pytest
import requests

from llm.client import (
    LLMClient,
    LLMResponse,
    LLMError,
    LLMConfigError,
    GroqProvider,
    OpenRouterProvider,
    _is_transient,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class StubProvider:
    """Deterministic LLM provider for testing."""

    def __init__(
        self,
        name: str = "stub",
        responses: list[str] | None = None,
        errors: list[Exception] | None = None,
    ):
        self._name = name
        self._responses = list(responses or [])
        self._errors = list(errors or [])
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def complete(self, prompt: str, *, model: str | None = None, max_tokens: int = 2048) -> str:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._errors):
            raise self._errors[idx]
        return self._responses.pop(0) if self._responses else "default response"


def _http_error(status: int, message: str = "error") -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    resp._content = message.encode()
    return requests.HTTPError(response=resp)


def _timeout_error() -> requests.Timeout:
    return requests.Timeout("request timed out")


def _connection_error() -> requests.ConnectionError:
    return requests.ConnectionError("connection refused")


# ---------------------------------------------------------------------------
# _is_transient classification
# ---------------------------------------------------------------------------

class TestIsTransient:
    def test_retryable_http_codes(self):
        for code in (408, 429, 500, 502, 503, 504):
            assert _is_transient(_http_error(code)) is True

    def test_non_retryable_http_codes(self):
        for code in (400, 401, 403, 404, 422):
            assert _is_transient(_http_error(code)) is False

    def test_timeout_is_transient(self):
        assert _is_transient(_timeout_error()) is True

    def test_connection_error_is_transient(self):
        assert _is_transient(_connection_error()) is True

    def test_unknown_exception_not_transient(self):
        assert _is_transient(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# LLMClient — primary success
# ---------------------------------------------------------------------------

class TestLLMClientPrimarySuccess:
    def test_returns_response_on_first_call(self):
        provider = StubProvider(responses=["Hello from Groq"])
        client = LLMClient(primary=provider, max_retries=2)

        result = client.complete("Say hello")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello from Groq"
        assert result.provider == "stub"

    def test_passes_model_through(self):
        provider = StubProvider(responses=["ok"])
        client = LLMClient(primary=provider)

        result = client.complete("test", model="custom-model")

        assert result.model == "custom-model"


# ---------------------------------------------------------------------------
# LLMClient — primary transient failure with recovery
# ---------------------------------------------------------------------------

class TestLLMClientTransientRecovery:
    def test_retries_after_transient_error(self):
        provider = StubProvider(
            responses=["recovered"],
            errors=[_timeout_error()],
        )
        client = LLMClient(primary=provider, max_retries=3, backoff_base=0.01)

        result = client.complete("retry this")

        assert result.text == "recovered"
        assert provider._call_count == 2  # 1 failure + 1 success

    def test_retries_on_429_rate_limit(self):
        provider = StubProvider(
            responses=["rate limited then ok"],
            errors=[_http_error(429)],
        )
        client = LLMClient(primary=provider, max_retries=3, backoff_base=0.01)

        result = client.complete("test")

        assert result.text == "rate limited then ok"

    def test_retries_on_503_service_unavailable(self):
        provider = StubProvider(
            responses=["service back"],
            errors=[_http_error(503)],
        )
        client = LLMClient(primary=provider, max_retries=2, backoff_base=0.01)

        result = client.complete("test")

        assert result.text == "service back"


# ---------------------------------------------------------------------------
# LLMClient — primary failure then fallback
# ---------------------------------------------------------------------------

class TestLLMClientFallback:
    def test_fallback_used_after_primary_exhausts_retries(self):
        primary = StubProvider(errors=[_timeout_error(), _timeout_error()])
        fallback = StubProvider(responses=["from fallback"])
        client = LLMClient(primary=primary, fallback=fallback, max_retries=2, backoff_base=0.01)

        result = client.complete("use fallback")

        assert result.text == "from fallback"
        assert result.provider == "stub"  # fallback's name

    def test_fallback_used_after_primary_non_retryable_error(self):
        primary = StubProvider(errors=[_http_error(401)])
        fallback = StubProvider(responses=["fallback ok"])
        client = LLMClient(primary=primary, fallback=fallback, max_retries=3)

        result = client.complete("auth fail")

        assert result.text == "fallback ok"


# ---------------------------------------------------------------------------
# LLMClient — both providers fail
# ---------------------------------------------------------------------------

class TestLLMClientBothFail:
    def test_raises_llm_error_when_both_fail(self):
        primary = StubProvider(errors=[_timeout_error()])
        fallback = StubProvider(errors=[_connection_error()])
        client = LLMClient(primary=primary, fallback=fallback, max_retries=1, backoff_base=0.01)

        with pytest.raises(LLMError, match="All LLM providers failed"):
            client.complete("doomed")

    def test_raises_llm_error_when_primary_fails_no_fallback(self):
        primary = StubProvider(errors=[_http_error(500)])
        client = LLMClient(primary=primary, max_retries=1, backoff_base=0.01)

        with pytest.raises(LLMError, match="All LLM providers failed"):
            client.complete("no fallback")


# ---------------------------------------------------------------------------
# Missing API keys — provider construction
# ---------------------------------------------------------------------------

class TestMissingAPIKeys:
    def test_groq_raises_config_error_without_key(self):
        with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
            GroqProvider(api_key="")

    def test_openrouter_raises_config_error_without_key(self):
        with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
            OpenRouterProvider(api_key="")


# ---------------------------------------------------------------------------
# LLMResponse model
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_response_fields(self):
        r = LLMResponse(text="hello", provider="groq", model="llama-3")
        assert r.text == "hello"
        assert r.provider == "groq"
        assert r.model == "llama-3"

    def test_response_is_frozen(self):
        r = LLMResponse(text="x", provider="y", model="z")
        with pytest.raises(AttributeError):
            r.text = "changed"  # type: ignore[misc]
