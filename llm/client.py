"""LLM client with primary/fallback provider routing and controlled retries."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised when all LLM providers fail."""


class LLMConfigError(LLMError):
    """Raised for missing or invalid configuration (non-retryable)."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """Interface every LLM backend must implement."""
    @property
    def name(self) -> str: ...
    def complete(self, prompt: str, *, model: str | None = None, max_tokens: int = 2048) -> str: ...


# ---------------------------------------------------------------------------
# Transient-error classification
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

def _is_transient(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    return False


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class GroqProvider:
    """Groq cloud LLM provider (OpenAI-compatible API)."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str, default_model: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise LLMConfigError("GROQ_API_KEY is not set")
        self._api_key = api_key
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "groq"

    def complete(self, prompt: str, *, model: str | None = None, max_tokens: int = 2048) -> str:
        model = model or self._default_model
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# OpenRouter provider
# ---------------------------------------------------------------------------

class OpenRouterProvider:
    """OpenRouter LLM provider (OpenAI-compatible API)."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, default_model: str = "meta-llama/llama-3.3-70b-instruct:free"):
        if not api_key:
            raise LLMConfigError("OPENROUTER_API_KEY is not set")
        self._api_key = api_key
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "openrouter"

    def complete(self, prompt: str, *, model: str | None = None, max_tokens: int = 2048) -> str:
        model = model or self._default_model
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# LLM client — routing + retries
# ---------------------------------------------------------------------------

class LLMClient:
    """Routes LLM requests through a primary provider with fallback.

    Retry behaviour:
    - Only transient errors (network, timeout, 429, 5xx) are retried.
    - Authentication / config errors (401, 403, 422) are never retried.
    - After exhausting retries on the primary, the fallback is tried once.
    - If both fail, ``LLMError`` is raised.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None = None,
        *,
        max_retries: int = 2,
        backoff_base: float = 1.0,
    ):
        self._primary = primary
        self._fallback = fallback
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # --- Try primary with retries ---
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                text = self._primary.complete(prompt, model=model, max_tokens=max_tokens)
                return LLMResponse(text=text, provider=self._primary.name, model=model or "")
            except Exception as exc:
                last_exc = exc
                if not _is_transient(exc):
                    logger.error("Primary provider %s failed (non-retryable): %s", self._primary.name, exc)
                    break
                if attempt < self._max_retries:
                    delay = self._backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "Primary provider %s attempt %d failed (transient), retrying in %.1fs: %s",
                        self._primary.name, attempt, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    logger.warning("Primary provider %s exhausted %d retries: %s", self._primary.name, self._max_retries, exc)

        # --- Try fallback (once, no retries) ---
        if self._fallback is not None:
            try:
                text = self._fallback.complete(prompt, model=model, max_tokens=max_tokens)
                return LLMResponse(text=text, provider=self._fallback.name, model=model or "")
            except Exception as exc:
                logger.error("Fallback provider %s also failed: %s", self._fallback.name, exc)
                last_exc = exc

        raise LLMError(
            f"All LLM providers failed. Last error: {last_exc}"
        ) from last_exc
