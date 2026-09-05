"""LLM module — bounded semantic-output retry for content generation.

Adds semantic-output validation/retry on top of ``llm.client.LLMClient``.
Transport retries and provider fallback in ``LLMClient`` are unchanged;
this module only re-prompts when a *successful* response is empty,
malformed, or fails structured validation.
"""

from __future__ import annotations

import logging
from typing import Protocol

from llm.client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Bounded number of complete() attempts per stage (initial + re-prompts).
#: A single attempt runs through the full transport retry/fallback in
#: ``LLMClient`` before it can even reach semantic validation, so the total
#: real API calls are potentially large; keep this small and bounded.
_MAX_SEMANTIC_ATTEMPTS = 3

_FEEDBACK_TEMPLATE = """
Note: your previous response was not usable because it was empty, malformed, or did not follow the required format.
Please respond again now, following the required format EXACTLY. Do not add any explanation.
"""


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class _SemanticRetry(Exception):
    """Signal that the model output was semantically invalid.

    Raised by stage adapters for empty/malformed/unusable output only.
    It is intentionally distinct from transport errors (``LLMError``) and
    is the ONLY signal ``run_retryable_llm_stage`` treats as retryable.
    """
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RetryableLLMStage(Protocol):
    """Adapter contract for a single LLM-response validation stage."""

    def retry(self, response: LLMResponse, prompt: str, attempt_number: int) -> object:
        """Return the parsed/validated result, or raise ``_SemanticRetry`` on bad output.

        - ``attempt_number`` is 1-based.
        - On success, returns the stage result.
        - On invalid output, raise ``_SemanticRetry`` (does NOT perform the retry).
        - Any other raised exception (e.g. ``LLMError``, unrelated ``RuntimeError``)
          propagates immediately.
        """
        ...


def run_retryable_llm_stage(
    llm_client: LLMClient,
    retryable: RetryableLLMStage,
    prompt: str,
    *,
    max_tokens: int,
    max_attempts: int = _MAX_SEMANTIC_ATTEMPTS,
) -> object:
    """Call ``llm_client`` and re-prompt on semantically invalid output.

    The existing ``LLMClient.complete`` transport retry/fallback is used
    on every attempt — it is not redesigned or replaced.

    Parameters
    ----------
    llm_client:
        The project-wide LLM router (unchanged behavior).
    retryable:
        Stage adapter that validates one response, raising ``_SemanticRetry``
        if the response is empty, malformed, or otherwise unusable.
    prompt:
        The base prompt for the stage (exactly as today).
    max_tokens:
        Upper bound on the LLM response length.
    max_attempts:
        Bounded number of complete() attempts (initial + re-prompts).

    Returns
    -------
    The stage's validated result from the first structurally valid response.

    Raises
    ------
    LLMError
        If the LLM call fails at the transport level after its own
        retries and fallback.  Propagates unchanged and is never retried here.
    RuntimeError
        If an ``_SemanticRetry`` is raised on every attempt.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    corrected_suffix = ""
    last_reason = "no valid output produced"

    for attempt in range(1, max_attempts + 1):
        attempt_prompt = f"{prompt}\n\n{corrected_suffix.strip()}" if corrected_suffix else prompt

        if attempt > 1:
            logger.warning(
                "Semantic retry attempt %d/%d for LLM stage",
                attempt, max_attempts,
            )

        response = llm_client.complete(attempt_prompt, max_tokens=max_tokens)
        last_reason = response.text

        try:
            result = retryable.retry(response, attempt_prompt, attempt)
        except _SemanticRetry as exc:
            last_reason = str(exc)
            corrected_suffix = _FEEDBACK_TEMPLATE.strip()
            continue

        if result is not None:
            return result

        last_reason = "stage produced a None result"
        corrected_suffix = _FEEDBACK_TEMPLATE.strip()

    raise RuntimeError(
        f"LLM stage failed after {max_attempts} semantic attempt(s): {last_reason}"
    )