"""Tests for bounded semantic-output retries across LLM content stages.

Covers, for each stage (narration / scenes / metadata):

A. valid first response -> no semantic retry
B. invalid first response -> valid second response -> success
C. invalid responses on all semantic attempts -> clear RuntimeError

Also covers the retry signal specifically:

D. unrelated RuntimeError from an adapter -> no retry, propagates immediately
E. LLMError from the client -> no semantic retry, propagates unchanged
"""
from __future__ import annotations

import json

import pytest

from content.metadata import VideoMetadata, generate_metadata
from content.scenes import Scene, plan_scenes
from content.script import generate_narration
from llm.client import LLMError, LLMResponse
from llm.retry import _SemanticRetry, run_retryable_llm_stage
from research.trending import Topic

MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_topic(
    title: str = "OpenAI Announces GPT-5",
    url: str = "https://example.com/gpt5",
    summary: str = "OpenAI released GPT-5 with major reasoning improvements.",
    source: str = "TechCrunch",
) -> Topic:
    return Topic(title=title, url=url, summary=summary, source=source)


NARRATION = (
    "OpenAI just announced GPT-5, their most advanced model yet. "
    "It features significant improvements in reasoning and coding. "
    "Developers are already building groundbreaking applications with it."
)

VALID_NARRATION_TEXT = (
    "OpenAI just released GPT-5 with major reasoning improvements. "
    "Developers are already building with it. This changes everything."
)

SCENES_JSON = json.dumps([
    {"scene": 1, "text": "OpenAI just announced GPT-5.", "visual_description": "OpenAI logo"},
    {"scene": 2, "text": "It improves reasoning and coding.", "visual_description": "AI code editor"},
    {"scene": 3, "text": "Developers are already building with it.", "visual_description": "Developer workstation"},
])

METADATA_JSON = json.dumps({
    "title": "GPT-5 Is Here: What You Need to Know",
    "description": "OpenAI has released GPT-5, featuring major reasoning improvements.",
    "tags": ["gpt-5", "openai", "ai news"],
})


class ScriptedLLMStub:
    """Returns a pre-configured sequence of raw texts from ``complete``."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedLLMStub exhausted its responses")
        text = self._responses.pop(0)
        self.calls.append(text)
        return LLMResponse(text=text, provider="stub", model="stub-model")


class FailingLLMStub:
    """LLMClient stub whose complete() always raises a transport LLMError."""

    def __init__(self, message: str = "simulated provider failure") -> None:
        self.calls = 0
        self.message = message

    def complete(self, prompt: str, *, model: str | None = None, max_tokens: int = 2048):
        self.calls += 1
        raise LLMError(self.message)


# ---------------------------------------------------------------------------
# Narration stage
# ---------------------------------------------------------------------------

class TestNarrationSemanticRetry:
    def test_A_valid_first_response_no_semantic_retry(self):
        client = ScriptedLLMStub([VALID_NARRATION_TEXT])

        result = generate_narration(client, _make_topic())

        assert result == VALID_NARRATION_TEXT
        assert len(client.calls) == 1

    def test_B_invalid_then_valid(self):
        # "**Only**" and "## Heading" both clean to empty after artefact stripping,
        # so the stub must fall back on a raw-truthiness-based invalid check.
        # Use a genuinely-empty first response, then a valid one.
        client = ScriptedLLMStub(["", VALID_NARRATION_TEXT])

        result = generate_narration(client, _make_topic())

        assert result == VALID_NARRATION_TEXT
        assert len(client.calls) == 2

    def test_C_all_attempts_invalid_raises(self):
        client = ScriptedLLMStub(["", "", ""])

        with pytest.raises(RuntimeError, match="empty narration content"):
            generate_narration(client, _make_topic())

        assert len(client.calls) == MAX_ATTEMPTS

    def test_c_prompt_includes_retry_feedback(self):
        # After an invalid first response, the second prompt must include
        # the format-correction instruction appended to the original prompt.
        client = ScriptedLLMStub(["", VALID_NARRATION_TEXT])

        generate_narration(client, _make_topic())

        assert len(client.calls) == 2
        assert "previous response was not usable" in client.prompts[1]
        assert "AI topic" in client.prompts[1]

    def test_first_response_empty_strips_to_false_also_raises(self):
        # Sanity: an all-whitespace response also triggers semantic retry.
        client = ScriptedLLMStub(["   \n  \n  ", VALID_NARRATION_TEXT])

        result = generate_narration(client, _make_topic())

        assert result == VALID_NARRATION_TEXT
        assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# Scenes stage
# ---------------------------------------------------------------------------

class TestScenesSemanticRetry:
    def test_A_valid_first_response_no_semantic_retry(self):
        client = ScriptedLLMStub([SCENES_JSON])

        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 3
        assert all(isinstance(s, Scene) for s in scenes)
        assert len(client.calls) == 1

    def test_B_invalid_then_valid(self):
        client = ScriptedLLMStub(["I cannot generate scenes right now.", SCENES_JSON])

        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 3
        assert len(client.calls) == 2

    def test_C_all_attempts_invalid_raises(self):
        client = ScriptedLLMStub(["bad", "[]", "[{broken"])

        with pytest.raises(RuntimeError, match="no valid scenes"):
            plan_scenes(client, NARRATION)

        assert len(client.calls) == MAX_ATTEMPTS

    def test_c_prompt_includes_retry_feedback(self):
        client = ScriptedLLMStub(["bad", SCENES_JSON])

        plan_scenes(client, NARRATION)

        assert len(client.calls) == 2
        assert "previous response was not usable" in client.prompts[1]
        assert "Split this narration into visual scenes" in client.prompts[1]


# ---------------------------------------------------------------------------
# Metadata stage
# ---------------------------------------------------------------------------

class TestMetadataSemanticRetry:
    def test_A_valid_first_response_no_semantic_retry(self):
        client = ScriptedLLMStub([METADATA_JSON])

        metadata = generate_metadata(client, _make_topic(), NARRATION)

        assert isinstance(metadata, VideoMetadata)
        assert len(client.calls) == 1

    def test_B_invalid_then_valid(self):
        client = ScriptedLLMStub(["I cannot generate metadata right now.", METADATA_JSON])

        metadata = generate_metadata(client, _make_topic(), NARRATION)

        assert isinstance(metadata, VideoMetadata)
        assert metadata.title == "GPT-5 Is Here: What You Need to Know"
        assert len(client.calls) == 2

    def test_C_all_attempts_invalid_raises(self):
        client = ScriptedLLMStub(["nope", '{"title": "T"}', "[not an object]"])

        with pytest.raises(RuntimeError, match="malformed or empty metadata"):
            generate_metadata(client, _make_topic(), NARRATION)

        assert len(client.calls) == MAX_ATTEMPTS

    def test_c_prompt_includes_retry_feedback(self):
        client = ScriptedLLMStub(["nope", METADATA_JSON])

        generate_metadata(client, _make_topic(), NARRATION)

        assert len(client.calls) == 2
        assert "previous response was not usable" in client.prompts[1]
        assert "Generate metadata for a YouTube Short" in client.prompts[1]


# ---------------------------------------------------------------------------
# Shared helper — bounds
# ---------------------------------------------------------------------------

class TestSemanticRetryBounds:
    def test_max_attempts_respected(self):
        class NeverValid:
            def retry(self, response, prompt, attempt_number):
                raise _SemanticRetry("always invalid")

        client = ScriptedLLMStub(["a", "b", "c", "d"])
        with pytest.raises(RuntimeError, match="after 3 semantic attempt"):
            run_retryable_llm_stage(client, NeverValid(), "prompt", max_tokens=100)
        assert len(client.calls) == 3

    def test_custom_attempt_count(self):
        class NeverValid:
            def retry(self, response, prompt, attempt_number):
                raise _SemanticRetry("always invalid")

        client = ScriptedLLMStub(["a", "b"])
        with pytest.raises(RuntimeError, match="after 2 semantic attempt"):
            run_retryable_llm_stage(client, NeverValid(), "p", max_tokens=100, max_attempts=2)
        assert len(client.calls) == 2

    def test_rejects_zero_attempts(self):
        class NeverValid:
            def retry(self, response, prompt, attempt_number):
                raise _SemanticRetry("always invalid")

        client = ScriptedLLMStub(["a"])
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            run_retryable_llm_stage(client, NeverValid(), "p", max_tokens=100, max_attempts=0)


# ---------------------------------------------------------------------------
# Retry signal — only _SemanticRetry may trigger a semantic retry
# ---------------------------------------------------------------------------

class TestRetrySignalIsPrecise:
    def test_unrelated_runtime_error_propagates_without_retry(self):
        class AdapterWithBustedHelpingCode:
            def retry(self, response, prompt, attempt_number):
                raise RuntimeError("internal stage bug")

        client = ScriptedLLMStub(["a", "b"])
        with pytest.raises(RuntimeError, match="internal stage bug"):
            run_retryable_llm_stage(client, AdapterWithBustedHelpingCode(), "p", max_tokens=100)
        assert len(client.calls) == 1

    def test_llm_error_propagates_without_semantic_retry(self):
        class AlwaysValid:
            def retry(self, response, prompt, attempt_number):
                return "ok"

        client = FailingLLMStub()
        with pytest.raises(LLMError, match="simulated provider failure"):
            run_retryable_llm_stage(client, AlwaysValid(), "p", max_tokens=100)
        assert client.calls == 1

    def test_narration_stage_llm_error_propagates_without_retry(self):
        client = FailingLLMStub()

        with pytest.raises(LLMError, match="simulated provider failure"):
            generate_narration(client, _make_topic())

        assert client.calls == 1